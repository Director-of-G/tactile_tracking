TACTILE_MODEL_DIR = "/home/yongpeng/research/dexterous/contact_playground/tactile_model/tactile_model"

import click 
import os
import numpy as np
from tactile_model.envs.gym_env import GymEnv
from tactile_model.track_contact_algo import ContactTracker

DESC = '''
Helper script to visualize demonstrations.\n
USAGE:\n
    Visualizes demonstrations on the env\n
    $ python utils/visualize_demos --env_name relocate-v0\n
'''

# shadow hand
# INIT_WPOS = np.array([0]*6)
# INIT_QPOS = np.array([0, -0.2, 0.5, -np.pi/2, 0, 0]+[0]*24)
# INIT_QPOS[6:12] = np.array([0.0, 0.537, 0.0, 0.256, 0.288, 0.0356])
# INIT_STATE = {
#     "qpos": INIT_QPOS,
#     "qvel": np.zeros(6+24),
# }

# leap hand
INIT_WPOS = np.array([])
INIT_QPOS = np.array([0]*4+[0]*4+[0]*4+[0, np.pi/2, 0, 0])
INIT_QPOS = np.array([
    0.58, -1.05, 1.59, -0.0234,
    0.382, 0.0106, 1.24, 0.263,
    0.582, 1.05, 1.64, -0.0235,
    1.89, 0.185, -0.486, 1.06])
INIT_STATE = {
    "qpos": INIT_QPOS,
    "qvel": np.zeros(16),
}

DOF_VEL_SCALE = 0.1

# MAIN =========================================================
@click.command(help=DESC)
@click.option('--env_name', type=str, help='environment to load', required= True)
def main(env_name):
    if env_name is "":
        print("Unknown env.")
        return
    # render demonstrations
    demo_playback(env_name)

def demo_playback(env_name):
    # demo_tactile = joblib.load("/home/yongpeng/research/dexterous/hand_dapg/dapg/debug/demo_tactile.pkl")[0]
    # for key in demo_tactile:
    #     if key != "ffdistal":
    #         demo_tactile[key] = np.zeros_like(demo_tactile[key])
    #     else:
    #         demo_tactile[key] = 0.4 * np.ones_like(demo_tactile[key])
    # print("desired tactile: ", demo_tactile)

    demo_tactile = {}
    for key in ["fffingertip", "mffingertip", "rffingertip", "thfingertip"]:
        demo_tactile[key] = np.array([
            [0.1, 0.3, 0.3, 0.1],
            [0.3, 0.7, 0.7, 0.3],
            [0.3, 0.7, 0.7, 0.3],
            [0.1, 0.3, 0.3, 0.1],
        ])
    # demo_tactile["fffingertip"] = 0.4 * np.ones((4, 4))

    cfg = {
        "nn": {
            "input_dim": 3,
            "output_dim": (4, 4)
        },
    }

    # initialize ContactTracker
    origin_dir = os.getcwd()
    os.chdir(os.path.join(origin_dir, '../'))
    algo = ContactTracker(
        nn_dir="./model",
        # xml_dir="./Adroit/Adroit_hand_kin_v2.xml",
        xml_dir="./Leap/Leap_hand_kin.xml",
        cfg=cfg
    )
    os.chdir(origin_dir)

    e = GymEnv(env_name)
    e.reset()
    e.set_env_state(INIT_STATE)
    if hasattr(e.env.env, "reset_object_properties"):
        e.env.env.reset_object_properties()
    actions = np.repeat(np.expand_dims(INIT_QPOS, 0), 1000, axis=0)

    tactile_data = []
    tactile_loss = {}
    current_dof_target = INIT_QPOS

    # desired_pose_fftip = algo.hand_kin.random_frame_pose_pinocchio(f"{test_finger_part}_panel_base")
    # desired_pose_fftip = pickle.load(open('./debug/pose_target.pkl', 'rb'))
    # desired_pose_fftip = None
    # optimization_cnt = 0
    # e.env.env.render_marker_for_debug("fffingertip", desired_pose_fftip)

    ftip_target_info = {name:{"target":None, "cnt":0} for name in \
                        ["fffingertip", "mffingertip", "rffingertip", "thfingertip"]}
    ftip_target_info["fffingertip"]["indices"] = [1, 0, 2, 3]
    ftip_target_info["mffingertip"]["indices"] = [5, 4, 6, 7]
    ftip_target_info["rffingertip"]["indices"] = [9, 8, 10, 11]
    ftip_target_info["thfingertip"]["indices"] = [12, 13, 14, 15]

    e.env.env.sim["data"].ctrl = INIT_QPOS
    for t in range(2500):
        e.step(np.concatenate((INIT_WPOS, current_dof_target)))

        env_state = e.env.env.get_env_state()
        env_tactile = e.env.env.taxel_data
        tactile_data.append(env_tactile)
        if t % 125 == 0:
            # print("ffdistal activation: ", env_tactile[test_finger_part])
            print("ffdistal activation: ", env_tactile)

        for ftip_name in ftip_target_info:
            if ftip_name not in tactile_loss: tactile_loss[ftip_name] = []
            tactile_loss[ftip_name].append(
                np.linalg.norm(demo_tactile[ftip_name] - env_tactile[ftip_name]))
        
        """
        compute hand dof movement
        """
        # infos = algo.get_hand_dof_movement(
        #     # wrist_qpos=env_state['hand_qpos'][:6],
        #     # qpos=env_state['hand_qpos'][6:],
        #     wrist_qpos=INIT_WPOS,
        #     qpos=env_state['hand_qpos'],
        #     desired_tactile=demo_tactile,
        #     current_tactile=e.env.env.taxel_data
        # )
        # print("tactile loss: ", infos['tactile_loss'])

        """
        use vanilla policy until taxel makes contact (single fingertip)
        """
        # if (np.any(env_tactile[test_finger_part] > 0.05) and desired_pose_fftip is None) or \
        #     (optimization_cnt > 1000 and desired_pose_fftip is not None):
        #     print("re-compute desired pose")
        #     desired_pose_fftip = algo.get_desired_panel_base_pose(
        #         qpos=env_state['hand_qpos'],
        #         frame=f"{test_finger_part}_panel_base",
        #         desired_tactile=demo_tactile,
        #         current_tactile=env_tactile
        #     )
        #     e.env.env.render_marker_for_debug(test_finger_part, desired_pose_fftip)
        #     optimization_cnt = 0

        # if desired_pose_fftip is None:
        #     dof_target_delta = np.zeros(16,); dof_target_delta[12] = 0.01; dof_target_delta[14] = 0.005
        # else:
        #     dof_target_delta = algo.solve_frame_pose_with_diff_ik(
        #         qpos=env_state['hand_qpos'],
        #         frame=f"{test_finger_part}_panel_base",
        #         desired_pose=desired_pose_fftip,
        #     )

        """
        track contact of multiple fingertips
        """
        dof_target_delta = np.zeros(16,)
        for ftip_name in ftip_target_info:
            # re-compute fingertip targets
            if (np.any(env_tactile[ftip_name] > 0.05) and ftip_target_info[ftip_name]["target"] is None) or \
                (ftip_target_info[ftip_name]["cnt"] > 250 and ftip_target_info[ftip_name]["target"] is not None):
                print(f"re-compute desired pose for {ftip_name}")
                ftip_target_info[ftip_name]['target'] = algo.get_desired_panel_base_pose(
                    qpos=env_state['hand_qpos'],
                    frame=f"{ftip_name}_panel_base",
                    desired_tactile=demo_tactile,
                    current_tactile=env_tactile
                )
                e.env.env.render_marker_for_debug(ftip_name, ftip_target_info[ftip_name]['target'])
                ftip_target_info[ftip_name]["cnt"] = 0

            # compute dof actions
            if ftip_target_info[ftip_name]["target"] is not None:
                indices = ftip_target_info[ftip_name]["indices"]
                dof_target_delta[indices] = algo.solve_frame_pose_with_diff_ik(
                    qpos=env_state['hand_qpos'],
                    frame=f"{ftip_name}_panel_base",
                    desired_pose=ftip_target_info[ftip_name]["target"],
                )[indices]

        current_dof_target = current_dof_target + DOF_VEL_SCALE * dof_target_delta

        # optimization_cnt += 1
        for ftip_name in ftip_target_info:
            if ftip_target_info[ftip_name]["target"] is not None:
                ftip_target_info[ftip_name]["cnt"] += 1
        e.env.mj_render()

        # delayed start for video recording
        if t == 0:
            input('Press Enter to start')

    breakpoint()

if __name__ == '__main__':
    main()