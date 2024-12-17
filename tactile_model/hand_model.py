"""
    This file includes the HandModel class, which is used to
    calculate fk and jacobian of ShadowHand.
    
    Currently, the pytorch_kinematics pkg do not support
    parallel chains, so we have to calculate jacobians for each taxel in a loop.

    Pinocchio does not have such problem, but we the conversion
    from mjcf to urdf is problematic.
"""

import os
import shutil
import numpy as np
import pytorch_kinematics as pk
from pytorch_kinematics.transforms import Transform3d
import pinocchio as pin
import torch
import roma

from tactile_model.utils.math_utils import transforms3d_to_xyz_quat


class HandModel(object):
    def __init__(self, model_path, lib="torch") -> None:
        self.model_path = model_path
        self.chain = pk.build_chain_from_mjcf(open(model_path).read())
        self.hand_dofs = len(self.chain.get_joints())
        self.panel_base_keys = []

        # TODO(yongpeng): debug
        self.panel_base_target = {}

        # if lib == "torch":
        self.load_via_torch()
        self.get_panel_base_keys()
        self.initialize_jacobian_calc()
        # elif lib == "pinocchio":
        self.load_via_pinocchio()
        self.get_panel_base_keys_and_ids_pinocchio()

        self.get_joint_map()

    def load_via_torch(self):
        try:
            self.arm_offset = self.chain.find_link("forearm").offset
        except:
            self.arm_offset = Transform3d(dtype=torch.float)
        self.wrist2arm_offset = Transform3d(dtype=torch.float)

    def load_via_pinocchio(self):
        """
            TODO(yongpeng): solve mjcf2urdf problems
        """
        if not self.model_path.endswith("urdf"):
            self.model_path = self.model_path.replace("xml", "urdf")
        asset_root = os.path.dirname(self.model_path)
        if not os.path.exists(self.model_path):
            os.system(f"mjcf2urdf {self.model_path.replace('urdf', 'xml')} --out {os.path.join(asset_root, 'urdf_tmp')}")
            shutil.move(os.path.join(asset_root, 'urdf_tmp/_forearm.urdf'), self.model_path)
            shutil.rmtree(os.path.join(asset_root, 'urdf_tmp'))
        self.pin_model = pin.buildModelFromUrdf(self.model_path)
        self.pin_data = self.pin_model.createData()

    def get_joint_map(self):
        """
            Get the mapping between pytorch_kinematics joint names and pinocchio joint names
        """
        # get pk & pin joint names
        pk_jnames = [j.name for j in self.chain.get_joints()]
        assert self.pin_model.names[0] == "universe"
        pin_jnames = self.pin_model.names[1:].tolist()
        self.pk_to_pin_map = [pk_jnames.index(j) for j in pin_jnames]
        self.pin_to_pk_map = [pin_jnames.index(j) for j in pk_jnames]

    def map_joints(self, qpos, pk2pin=True):
        if pk2pin:
            assert len(qpos) == len(self.pk_to_pin_map)
            return qpos[self.pk_to_pin_map]
        else:
            assert len(qpos) == len(self.pin_to_pk_map)
            return qpos[self.pin_to_pk_map]
    
    def map_matrix(self, mat, dim=0, pk2pin=True):
        if dim == 0:
            if pk2pin:
                assert mat.shape[0] == len(self.pk_to_pin_map)
                return mat[self.pk_to_pin_map]
            else:
                assert mat.shape[0] == len(self.pin_to_pk_map)
                return mat[self.pin_to_pk_map]
        elif dim == 1:
            if pk2pin:
                assert mat.shape[1] == len(self.pk_to_pin_map)
                return mat[:, self.pk_to_pin_map]
            else:
                assert mat.shape[1] == len(self.pin_to_pk_map)
                return mat[:, self.pin_to_pk_map]
        else:
            raise NotImplementedError

    def set_arm2wrist_offset(self, matrix=None, qpos=None):
        if matrix is not None:
            matrix = matrix.to(torch.float)
            self.wrist2arm_offset = Transform3d(matrix=matrix, dtype=torch.float)
        else:
            assert qpos is not None
            qpos = qpos.to(torch.float)
            pos = qpos[:3]
            rot = roma.euler_to_rotmat("XYZ", qpos[3:])
            self.wrist2arm_offset = Transform3d(rot=rot, pos=pos, dtype=torch.float)

    def get_panel_base_keys(self):
        """ Get the keys of tactile sensor panel """
        self.panel_base_keys.clear()
        fk = self.chain.forward_kinematics(np.zeros(self.hand_dofs))
        for key in fk.keys():
            if key.endswith("_panel_base"):
                # self.panel_base_keys.append(key.split('_')[0])
                self.panel_base_keys.append(key)
        print("Found {} tactile panels in the hand model".format(len(self.panel_base_keys)))

    def get_panel_base_keys_and_ids_pinocchio(self):
        """ Get the names and ids of tactile sensor panel """
        self.panel_base_keys.clear()
        self.panel_base_fids = {}
        for f in self.pin_model.frames:
            if f.name.endswith("_panel_base"):
                self.panel_base_keys.append(f.name)
                self.panel_base_fids[f.name] = self.pin_model.getFrameId(f.name)
        print("Found {} tactile panels in the hand model".format(len(self.panel_base_keys)))

    def initialize_jacobian_calc(self):
        assert len(self.panel_base_keys) > 0
        self.panel_serial_chains = {}
        for key in self.panel_base_keys:
            self.panel_serial_chains[key] = pk.SerialChain(self.chain, key)

    def compute_forward_kinematics(self, q, panel_only=False):
        fk = self.chain.forward_kinematics(q)
        new_wrist_trans = self.arm_offset.compose(self.wrist2arm_offset).compose(self.arm_offset.inverse())
        fk = {key:new_wrist_trans.compose(fk[key]) for key in fk.keys()}
        if not panel_only:
            return fk
        else:
            fk_panel = {k.split('_')[0]: fk[k] for k in self.panel_base_keys}
            return fk_panel

    def compute_jacobian(self, q, panel_keys=None, return_dic=False):
        """
            Return shape (N, 6, 24), N - number of panels
        """
        if return_dic:
            panel_jacobians = {}
        else:
            panel_jacobians = []

        if panel_keys is None:
            panel_keys = self.panel_base_keys
        panel_keys = [key+"_panel_base" if not key.endswith("panel_base") else key for key in panel_keys]
            
        for key in panel_keys:
            jac = self.panel_serial_chains[key].jacobian(q)
            if return_dic:
                panel_jacobians[key.split('_')[0]] = jac
            else:
                panel_jacobians.append(jac)
        
        if not return_dic:
            panel_jacobians = torch.vstack(panel_jacobians)

        return panel_jacobians
    
    def compute_forward_kinematics_pinocchio(self, q):
        if len(q) != self.pin_model.nq:
            assert len(q) == self.pin_model.nq - 7
            q_wrist = transforms3d_to_xyz_quat(self.wrist2arm_offset, quat="xyzw").numpy().flatten()
            q = np.concatenate((q_wrist, q))

        pin.forwardKinematics(self.pin_model, self.pin_data, q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        fk = {}
        for key in self.panel_base_keys:
            fid = self.panel_base_fids[key]
            fk[key] = self.pin_data.oMf[fid]
        return fk
    
    def compute_jacobian_pinocchio(self, q, panel_keys=None, return_dic=False, body_jac=False, include_base=False):
        """
            Return shape (N, 6, 24), N - number of panels
            If include_base is True, return shape (N, 6, 31), the base jacobian will be included
            If body_jac is True, will compute body Jacobian
        """
        if len(q) != self.pin_model.nq:
            assert len(q) == self.pin_model.nq - 7
            q_wrist = transforms3d_to_xyz_quat(self.wrist2arm_offset, quat="xyzw").numpy().flatten()
            q = np.concatenate((q_wrist, q))

        if return_dic:
            panel_jacobians = {}
        else:
            panel_jacobians = []

        if panel_keys is None:
            panel_keys = self.panel_base_keys
        panel_keys = [key+"_panel_base" if not key.endswith("panel_base") else key for key in panel_keys]

        for key in panel_keys:
            f_id = self.pin_model.getFrameId(key)
            if body_jac:
                jac = pin.computeFrameJacobian(self.pin_model, self.pin_data, q, f_id, pin.LOCAL)
            else:
                jac = pin.computeFrameJacobian(self.pin_model, self.pin_data, q, f_id, pin.LOCAL_WORLD_ALIGNED)
            jac = torch.from_numpy(jac).unsqueeze(0).to(torch.float)
            if not include_base:
                jac = jac[..., 6:]
            if return_dic:
                panel_jacobians[key.split('_')[0]] = jac
            else:
                panel_jacobians.append(jac)
        
        if not return_dic:
            panel_jacobians = torch.vstack(panel_jacobians)

        return panel_jacobians
    
    def random_frame_pose_pinocchio(self, frame):
        """
            Generate a random pose for a certain frame
        """
        lower, upper = self.pin_model.lowerPositionLimit, self.pin_model.upperPositionLimit
        q_random = np.random.uniform(lower, upper) * 0.05
        q_random[5] += np.pi/2
        
        return self.compute_frame_pose_pinocchio(q_random, frame)
    
    def compute_frame_pose_pinocchio(self, q, frame, pk_order=False):
        """
            Compute the pose of a certain frame
            If pk_order=True, input param q will be in the order of pytorch_kinematics,
            this function will do all the conversions (q & jac)
        """
        if pk_order:
            q = self.map_joints(q, pk2pin=True)

        pin.forwardKinematics(self.pin_model, self.pin_data, q)
        f_id = self.pin_model.getFrameId(frame)
        pose = pin.updateFramePlacement(self.pin_model, self.pin_data, f_id)

        return pose

    def compute_frame_jacobian_pinocchio(self, q, frame, pk_order=False):
        """
            Compute the jacobian of a certain frame
            If pk_order=True, input param q will be in the order of pytorch_kinematics,
            this function will do all the conversions (q & jac)
        """
        if pk_order:
            q = self.map_joints(q, pk2pin=True)

        f_id = self.pin_model.getFrameId(frame)
        # Space Jacobian
        jac = pin.computeFrameJacobian(self.pin_model, self.pin_data, q, f_id, pin.LOCAL_WORLD_ALIGNED)
        # Body Jacobian
        # jac = pin.computeFrameJacobian(self.pin_model, self.pin_data, q, f_id, pin.LOCAL)

        if pk_order:
            jac = self.map_matrix(jac, dim=1, pk2pin=False)

        return jac

if __name__ == "__main__":
    def euler_to_quat(euler):
        from scipy.spatial.transform import Rotation as SciR
        return SciR.from_euler("XYZ", euler).as_quat()

    # model = HandModel("Adroit/Adroit_hand_kin_v2.xml", lib="pinocchio")
    model = HandModel("Leap/Leap_hand_kin.xml", lib="pinocchio")
    # model.load_via_pinocchio()
    
    # test_wrist_q = np.array([0.05, -0.06, 0.07, -0.1, 0.05, -0.15])
    test_wrist_q = np.zeros(6,)

    # test_q = np.array([
    #     0, 0.183,                     # 0~1
    #     0.0, 0.4, 1.6, 0.288,         # 2~5
    #     0.0, 0.4, 1.6, 0.288,         # 6~9
    #     0.0, 0.4, 1.6, 0.288,         # 10~13
    #     0.0, 0.0, 0.4, 1.6, 0.288,    # 14~18
    #     1.0, 1.17, 0.0, 0.0, 0.0      # 19~23
    # ])
    test_q = np.random.uniform(-0.2, 0.2, (16,))

    model.set_arm2wrist_offset(qpos=torch.from_numpy(test_wrist_q))
    test_fk_pk = model.compute_forward_kinematics(test_q)

    # model.compute_jacobian(test_q)
    # test_q_full = np.concatenate((test_wrist_q[:3], euler_to_quat(test_wrist_q[3:]), \
    #                               test_q[0:6], test_q[14:19], test_q[6:10], test_q[10:14], test_q[19:]))
    test_q_full = np.concatenate((test_q[0:4], test_q[12:16], test_q[4:12]))
    test_fk_pin = model.compute_forward_kinematics_pinocchio(test_q_full)
    breakpoint()

    # pin.forwardKinematics(model.pin_model, model.pin_data, test_q)

    # breakpoint()
    # test_jac = model.chain.jacobian(test_q)

