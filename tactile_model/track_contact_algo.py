import os
import joblib
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.nn import MSELoss
from tactile_model.hand_model import HandModel
from tactile_model.train.sensor_model import ForwardNet, InverseNet

from tactile_model.utils.math_utils import (
    euler_derivative_to_omega,
    pos_euler_to_rigid_transform,
    compute_matrix_inverse,
)
from tactile_model.utils.utils import dic_to_tensor, tensor_to_dic, rearrange_dic
from tactile_model.utils.math_utils import pinocchio_SE3_to_matrix


class ContactTracker(object):
    def __init__(self, nn_dir, xml_dir, cfg) -> None:
        self.hand_kin = HandModel(model_path=xml_dir)
        self.sensor_model = None

        self.nn_dir = nn_dir
        self.xml_dir = xml_dir

        self.cfg = cfg

        self.load_sensor_model()

    def load_sensor_model(self):
        """
            Load sensor force motion model
        """
        _nn_cfg = self.cfg["nn"]

        _forward_model = ForwardNet(
            input_dim=_nn_cfg["input_dim"],
            output_dim=_nn_cfg["output_dim"]
        )
        _inverse_model = InverseNet(
            input_dim=_nn_cfg["output_dim"],
            output_dim=_nn_cfg["input_dim"]
        )
        _forward_model.load_state_dict(torch.load(os.path.join(self.nn_dir, 'best_forward_model.pt')))
        _inverse_model.load_state_dict(torch.load(os.path.join(self.nn_dir, 'best_inverse_model.pt')))
        _forward_model.eval()
        _inverse_model.eval()

        self.sensor_model = {
            "forward": _forward_model,
            "inverse": _inverse_model
        }

        _scaler = joblib.load(os.path.join(self.nn_dir, 'scaler.pkl'))
        self.input_scaler = _scaler['X']
        self.output_scaler = _scaler['Y']

        self.loss_criterion = nn.MSELoss(reduction='none')

    def get_pose_from_taxel_data(self, taxel_data):
        """
            Get the taxel pose from sensor readings
            :param taxel_data: dic, knuckle_name -> tactile_array (4x4 numpy array)
        """
        with torch.no_grad():
            if isinstance(taxel_data, dict):
                _, taxel_data = dic_to_tensor(taxel_data)
            else:
                if len(taxel_data.shape) == 2:
                    taxel_data = np.expand_dims(taxel_data, axis=0)
                taxel_data = torch.from_numpy(taxel_data).to(torch.float32)
            taxel_data_unscaled = torch.tensor(self.output_scaler.transform(taxel_data.view(-1, 16)), dtype=torch.float)
            taxel_pose_unscaled = self.sensor_model["inverse"](taxel_data_unscaled)
            taxel_pose_scaled = self.input_scaler.inverse_transform(taxel_pose_unscaled)
            taxel_pose_scaled = torch.from_numpy(taxel_pose_scaled).to(torch.float32)
            taxel_pose = pos_euler_to_rigid_transform(z=taxel_pose_scaled[:, 0], rx=taxel_pose_scaled[:, 1], ry=taxel_pose_scaled[:, 2])

        return taxel_pose

    def compute_panel_base_transform(self, desired_tactile, current_tactile, panel_keys=None):
        """
            Get the panel base movement (correction) based on the
            desired and current tactile readings.
            desired/current_tactile: dic, knuckle_name -> tactile_array
        """
        desired_tactile = rearrange_dic(desired_tactile, panel_keys)
        current_tactile = rearrange_dic(current_tactile, panel_keys)

        n_panels = len(panel_keys)

        panel_transforms = {}

        _, desired_panel_data = dic_to_tensor(desired_tactile, panel_keys)
        desired_tac_tensor_unscaled = torch.tensor(self.output_scaler.transform(desired_panel_data.view(-1, 16)), dtype=torch.float)
        _, current_panel_data = dic_to_tensor(current_tactile, panel_keys)
        current_tac_tensor_unscaled = torch.tensor(self.output_scaler.transform(current_panel_data.view(-1, 16)), dtype=torch.float)
        
        with torch.no_grad():
            panel_pose_unscaled = self.sensor_model["inverse"](current_tac_tensor_unscaled)
            panel_pose_scaled = self.input_scaler.inverse_transform(panel_pose_unscaled)
            panel_pose_scaled = torch.from_numpy(panel_pose_scaled).to(torch.float32)
            # # convert pose (z, rx, ry) to rigid-transform
            # panel_rel_rtrans = pos_euler_to_rigid_transform(z=panel_pose_scaled[:, 0], rx=panel_pose_scaled[:, 1], ry=panel_pose_scaled[:, 2])
            # panel_transforms['T_P2FW'] = panel_rel_rtrans
            panel_transforms['T_P2FW'] = self.get_pose_from_taxel_data(current_tactile)

        panel_pose_unscaled.requires_grad = True

        current_tac_tensor_unscaled = self.sensor_model["forward"](panel_pose_unscaled)

        loss = self.loss_criterion(current_tac_tensor_unscaled, desired_tac_tensor_unscaled)
        loss = torch.mean(loss, dim=-1)
        # backward can be applied on scalar with gradient param.
        loss.backward(gradient=torch.ones_like(loss))

        panel_pose_dot_scaled = -torch.from_numpy(self.input_scaler.scale_) * panel_pose_unscaled.grad
        panel_transforms['T_P_delta'] = panel_pose_dot_scaled

        # convert delta pose to linvel and angvel (omega)
        panel_linvel = torch.zeros(n_panels, 3)
        panel_linvel[:, 2] = panel_pose_dot_scaled[:, 0]
        # TODO(yongpeng): debug
        panel_linvel = 1000 * panel_linvel
        panel_linvel[1:] *= 0

        panel_transforms['linvel'] = panel_linvel

        panel_euler = torch.cat((panel_pose_scaled[:, 1:], torch.zeros(panel_pose_scaled.shape[0], 1)), dim=1).to(torch.float)
        panel_euler_dot = torch.cat((panel_pose_dot_scaled[:, 1:], torch.zeros(panel_pose_dot_scaled.shape[0], 1)), dim=1).to(torch.float)
        panel_angvel = euler_derivative_to_omega(panel_euler, panel_euler_dot, seq="XYZ")
        panel_angvel = 0.1 * panel_angvel
        panel_angvel[1:] *= 0

        panel_transforms['angvel'] = panel_angvel.to(torch.float32)

        # desired taxel pose
        panel_transforms['d_T_P2FW'] = self.get_pose_from_taxel_data(desired_tactile)

        return panel_transforms, loss.detach().cpu()

    def get_hand_dof_movement(self, wrist_qpos, qpos, desired_tactile, current_tactile):
        """
            Get the hand dof movement (correction) based on the
            qpos and tactile readings
        """
        if not isinstance(wrist_qpos, torch.Tensor):
            wrist_qpos = torch.from_numpy(wrist_qpos)
        self.hand_kin.set_arm2wrist_offset(qpos=wrist_qpos)

        panel_fk = self.hand_kin.compute_forward_kinematics(qpos, panel_only=True)
        # convert Transforms3d to rotation matrix
        panel_rtrans = {k: v.get_matrix()[0] for k, v in panel_fk.items()}
        panel_keys, panel_rtrans = dic_to_tensor(panel_rtrans)

        panel_transforms, tactile_loss = self.compute_panel_base_transform(desired_tactile, current_tactile, panel_keys)
        print("delta linvel: ", panel_transforms['linvel'][0])
        print("delta angvel: ", panel_transforms['angvel'][0])

        # compute fake world for each panel (viewed as the imagined contact surface)
        panel_fake_world_rtrans = torch.matmul(panel_rtrans, torch.linalg.inv(panel_transforms['T_P2FW']))

        # compute the desired panel pose
        # import roma
        # panel_delta_rtrans = -torch.matmul(torch.linalg.inv(panel_transforms['T_P2FW']), panel_transforms['d_T_P2FW'])
        # panel_desired_velocity = torch.concat((
        #     panel_delta_rtrans[:, :3, 3],
        #     0 * roma.rotmat_to_euler('XYZ', panel_delta_rtrans[:, :3, :3])
        # ), dim=1)

        # TODO(yongpeng): debug
        import numpy as np
        import roma
        if np.any(current_tactile['ffdistal'] > 0) and 'ffdistal' not in self.hand_kin.panel_base_target:
            self.hand_kin.panel_base_target['ffdistal'] = panel_fake_world_rtrans[0] @ panel_transforms['d_T_P2FW'][0]

        """
            Compute DoF velocity to track velocity (computed using gradient of taxel loss)
        """
        # compute panel velocity (in world frame)
        panel_desired_velocity = torch.concat((
            torch.matmul(panel_fake_world_rtrans[:, :3, :3], panel_transforms['linvel'].unsqueeze(-1)).squeeze(-1),
            torch.matmul(panel_fake_world_rtrans[: ,:3, :3], panel_transforms['angvel'].unsqueeze(-1)).squeeze(-1)
        ), dim=1)
        print("panel desired velocity: ", panel_desired_velocity[0])

        # TODO(yongpeng): debug
        if 'ffdistal' in self.hand_kin.panel_base_target:
            delta_rtrans = -torch.matmul(torch.linalg.inv(panel_rtrans[0]), self.hand_kin.panel_base_target['ffdistal'])
            panel_desired_velocity[1:] *= 0
            panel_desired_velocity[0, :3] = delta_rtrans[:3, 3]
            panel_desired_velocity[0, 3:] = roma.rotmat_to_euler('XYZ', delta_rtrans[:3, :3])

        # jacobians (in panel_keys order)
        # panel_jacobian = self.hand_kin.compute_jacobian(qpos, panel_keys=panel_keys, return_dic=False)
        panel_jacobian = self.hand_kin.compute_jacobian_pinocchio(qpos, panel_keys=panel_keys, return_dic=False, body_jac=False, include_base=False)

        # jacobian (pseudo) inverse
        panel_jacobian_inv = compute_matrix_inverse(panel_jacobian, type="transpose")

        # compute hand movement
        hand_movement = torch.zeros(len(qpos),)
        for idx in range(len(panel_keys)):
            hand_movement += torch.matmul(panel_jacobian_inv[idx], panel_desired_velocity[idx])
        
        # TODO(yongpeng): handle jacobian inverse and compute hand movement
        infos = {
            "panel_fake_world_rtrans": tensor_to_dic(panel_keys, panel_fake_world_rtrans),
            "dof_movement": torch.clamp(hand_movement, -0.01, 0.01).numpy(),
            "tactile_loss": tactile_loss.numpy()
        }
        return infos
    
    def get_desired_panel_base_pose(self, qpos, frame, desired_tactile, current_tactile):
        """
            Compute the desired panel base pose based on current and desired tactile readings
        """
        import pinocchio as pin
        panel_name = frame.split('_')[0]

        taxel_pose = self.hand_kin.compute_frame_pose_pinocchio(qpos, frame, pk_order=True)
        taxel_pose = torch.from_numpy(pinocchio_SE3_to_matrix(taxel_pose)).to(torch.float)

        T_P2FW = self.get_pose_from_taxel_data(current_tactile[panel_name]).squeeze(0)
        T_P2DW = self.get_pose_from_taxel_data(desired_tactile[panel_name]).squeeze(0)

        desired_taxel_pose = taxel_pose @ np.linalg.inv(T_P2FW) @ T_P2DW
        desired_taxel_pose = pin.SE3(desired_taxel_pose[:3, :3].numpy(), desired_taxel_pose[:3, 3].numpy())

        return desired_taxel_pose
    
    def solve_frame_pose_with_diff_ik(self, qpos, frame, desired_pose):
        """
            Solve the frame pose using differential IK
        """
        from scipy.spatial.transform import Rotation as SciR
        import pinocchio as pin

        # get current pose
        cur_pose = self.hand_kin.compute_frame_pose_pinocchio(qpos, frame, pk_order=True)
        cur_pose = pinocchio_SE3_to_matrix(cur_pose)
        desired_pose = pinocchio_SE3_to_matrix(desired_pose)

        # compute delta pose in world
        delta_pose = np.eye(4)
        delta_pose[:3, :3] = cur_pose[:3, :3].T @ desired_pose[:3, :3]
        delta_pose[:3, 3] = desired_pose[:3, 3] - cur_pose[:3, 3]
        # delta_pose = np.linalg.inv(cur_pose) @ desired_pose
        
        delta_vel = np.zeros(6,)
        delta_vel[:3] = delta_pose[:3, 3]
        delta_vel[3:] = cur_pose[:3, :3] @ SciR.from_matrix(delta_pose[:3, :3]).as_rotvec()

        # project to J space
        jac = self.hand_kin.compute_frame_jacobian_pinocchio(qpos, frame, pk_order=True)

        # inverse does better than transpose
        # dq = jac.T @ delta_vel
        dq = np.linalg.pinv(jac) @ delta_vel

        # clamp actions
        dq = np.clip(dq, -0.003, 0.003)

        return dq


if __name__ == "__main__":
    cfg = {
        "nn": {
            "input_dim": 3,
            "output_dim": (4, 4)
        },
    }

    algo = ContactTracker(
        nn_dir="./model",
        xml_dir="./Adroit/Adroit_hand_kin_v2.xml",
        cfg=cfg
    )

    demo_tactile = pickle.load(open("/home/yongpeng/research/dexterous/hand_dapg/dapg/debug/demo_tactile.pkl", "rb"))
    real_tactile = pickle.load(open("/home/yongpeng/research/dexterous/hand_dapg/dapg/debug/real_tactile.pkl", "rb"))

    test_idx = 200
    breakpoint()

    qpos = torch.zeros(24, dtype=torch.float)
    algo.get_hand_dof_movement(qpos, demo_tactile[test_idx], real_tactile[test_idx])
