"""
    This env is modified from relocate-v0, there is shadow hand
    interacting with a rigid wall to achieve desired finger
    contacts.
"""

import numpy as np
from gym import utils
from tactile_model.envs import mujoco3_env
import mujoco.viewer
import os
import re

ADD_BONUS_REWARDS = True
TAXEL_NAME_PATTERN = r"_T_r\d+c\d+$"

def get_actuator_id(sim, name):
    return mujoco.mj_name2id(sim["model"], mujoco.mjtObj.mjOBJ_ACTUATOR, name)

def get_site_id(sim, name):
    return mujoco.mj_name2id(sim["model"], mujoco.mjtObj.mjOBJ_SITE, name)

def get_mocap_id(sim, name):
    return sim["model"].body_mocapid[mujoco.mj_name2id(sim["model"], mujoco.mjtObj.mjOBJ_BODY, name)]

def get_all_taxel_joint_ids(sim):
    """
        The joints connecting taxels and knuckles contribute to nq.
        Thus their ids are needed to specifically reset the hand DoFs.
    """
    taxel_jpos_ids, taxel_jvel_ids = [], []
    taxel_sensor_names = []
    for i in range(sim["model"].nbody):
        body_name = mujoco.mj_id2name(sim["model"], mujoco.mjtObj.mjOBJ_BODY, i)
        if re.search(TAXEL_NAME_PATTERN, body_name):
            jid = sim["model"].body_jntadr[i]
            taxel_jpos_ids.append(sim["model"].jnt_qposadr[jid])
            taxel_jvel_ids.append(sim["model"].jnt_dofadr[jid])
            taxel_sensor_names.append(body_name.replace("T", "S"))
    taxel_jpos_ids.sort()
    taxel_jvel_ids.sort()
    print(f"Found {len(taxel_sensor_names)} taxel sensors in the hand model!")

    # parse taxel names
    knuckles = set()
    for name in taxel_sensor_names:
        knuckles.add(name.split("_")[0])
    knuckles = list(knuckles)
    print(f"Found {len(knuckles)} knuckles in the hand model!")

    taxel_meta = {}
    for name in knuckles:
        all_taxels = [n for n in taxel_sensor_names if n.startswith(name)]
        # _T_rxcx
        nrow = int(max([n.split("_", -1)[-1][1] for n in all_taxels]))+1
        ncol = int(max([n.split("_", -1)[-1][3] for n in all_taxels]))+1
        taxel_meta[name] = (nrow, ncol)

    return taxel_jpos_ids, taxel_jvel_ids, taxel_meta

def read_taxel_data(sim, taxel_meta):
    """
        taxel_meta contains knuckle_name:(nrow, ncol) as key:value
    """
    taxel_data = {}
    for knuckle, size in taxel_meta.items():
        nrow, ncol = size
        panel_readings = np.zeros((nrow, ncol))
        for i in range(nrow):
            for j in range(ncol):
                taxel_sensor = f"{knuckle}_S_r{i}c{j}"
                panel_readings[i, j] = sim["data"].sensor(taxel_sensor).data
        taxel_data[knuckle] = panel_readings
    return taxel_data

class ContactEnvV1(mujoco3_env.Mujoco3Env, utils.EzPickle):
    def __init__(self):
        self.target_obj_sid = 0
        self.obj_bid = 0
        self.taxel_meta = {}
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mujoco3_env.Mujoco3Env.__init__(self, os.path.join(curr_dir, '../Leap/env-v1.xml'), 5)

        utils.EzPickle.__init__(self)

        self.taxel_jpos_ids, self.taxel_jvel_ids, self.taxel_meta = get_all_taxel_joint_ids(self.sim)
        self.non_taxel_jpos_ids = [i for i in list(range(self.sim["model"].nq)) if i not in self.taxel_jpos_ids]
        self.non_taxel_jvel_ids = [i for i in list(range(self.sim["model"].nv)) if i not in self.taxel_jvel_ids]

    def step(self, a):
        a = a
        self.do_simulation(a, self.frame_skip)
        ob = self.get_obs()

        reward = 0.0

        return ob, reward, False, dict()

    def get_obs(self):
        # qpos for hand
        # xpos for obj
        # xpos for target
        qp = self.data.qpos.ravel()
        # taxel readings
        self.taxel_data = read_taxel_data(self.sim, self.taxel_meta)
        return np.concatenate([qp[:-6], np.zeros(3,), np.zeros(3,), np.zeros(3,)])
       
    def reset_model(self):
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)
        mujoco.mj_forward(self.sim["model"], self.sim["data"])
        return self.get_obs()

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qp = self.data.qpos.ravel().copy()
        qv = self.data.qvel.ravel().copy()
        hand_qpos = qp[self.non_taxel_jpos_ids]
        return dict(hand_qpos=hand_qpos, palm_pos=None,
            qpos=qp, qvel=qv)

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        qp = state_dict['qpos']
        qv = state_dict['qvel']

        qp_all = np.zeros(self.sim["model"].nq,)
        qp_all[self.non_taxel_jpos_ids] = qp
        qv_all = np.zeros(self.sim["model"].nv,)
        qv_all[self.non_taxel_jvel_ids] = qv
        
        self.set_state(qp_all, qv_all)

        mujoco.mj_forward(self.sim["model"], self.sim["data"])

    def mj_viewer_setup(self):
        self.viewer = mujoco.viewer.launch_passive(self.sim["model"], self.sim["data"])
        self.viewer.cam.azimuth = 90
        self.viewer.cam.distance = 1.5
        self.viewer.opt.sitegroup[5] = 1

    def evaluate_success(self, paths):
        num_success = 0
        num_paths = len(paths)
        # success if object close to target for 25 steps
        for path in paths:
            if np.sum(path['env_infos']['goal_achieved']) > 25:
                num_success += 1
        success_percentage = num_success*100.0/num_paths
        return success_percentage
    
    # ====== for debug purposes ======
    def render_panel_for_debug(self, infos):
        from scipy.spatial.transform import Rotation as SciR
        for key, rtrans in infos.items():
            mocap_id = get_mocap_id(self.sim, key+"_mocap")
            self.sim["data"].mocap_pos[mocap_id] = rtrans[:3, 3]
            self.sim["data"].mocap_quat[mocap_id] = SciR.from_matrix(rtrans[:3, :3]).as_quat()[[3, 0, 1, 2]]

    def render_marker_for_debug(self, part, rtrans):
        from scipy.spatial.transform import Rotation as SciR
        mocap_id = get_mocap_id(self.sim, part+'_marker')
        self.sim["data"].mocap_pos[mocap_id] = rtrans.translation
        self.sim["data"].mocap_quat[mocap_id] = SciR.from_matrix(rtrans.rotation).as_quat()[[3, 0, 1, 2]]
