import os
import numpy as np
from scipy.spatial.transform import Rotation as SciR

def load_dataset_from_disk(path, sel_dim=None, use_quat=True):
    # 绘制数据集的统计数据
    all_taxel_reading = np.zeros((0, 16))
    all_panel_state = np.zeros((0, 7))
    for seg_name in os.listdir(path):
        seg = np.load(os.path.join(path, seg_name))
        all_taxel_reading = np.vstack((all_taxel_reading, seg['taxel_reading']))
        all_panel_state = np.vstack((all_panel_state, seg['panel_state']))

    if not use_quat:
        all_panel_state = np.concatenate((all_panel_state[:, :3], 
                                          SciR.from_quat(all_panel_state[:, 3:]).as_euler('xyz', degrees=False)), axis=1)
        
    if sel_dim is not None:
        all_panel_state = all_panel_state[:, sel_dim]

    # dataset meta
    panel_state_range = all_panel_state.max(axis=0) - all_panel_state.min(axis=0)
    taxel_reading_range = all_taxel_reading.max(axis=0) - all_taxel_reading.min(axis=0)
    metadata = {
        "panel_state_range": panel_state_range,
        "taxel_reading_range": taxel_reading_range
    }
    
    return all_taxel_reading, all_panel_state, metadata
