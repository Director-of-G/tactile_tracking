import numpy as np
import roma
import torch
from scipy.spatial.transform import Rotation as SciR


def euler_derivative_to_omega(euler, euler_dot, seq="XYZ"):
    """
        Convert the derivative of euler angles to angular velocity
        @TODO(yongpeng): not fully tested
    """
    assert seq=="XYZ"   # currently, only support XYZ sequence

    batch_size = euler.shape[0]
    alpha, beta, gamma = euler[:, 0], euler[:, 1], euler[:, 2]
    sina, cosa = torch.sin(alpha), torch.cos(alpha)
    sinb, cosb = torch.sin(beta), torch.cos(beta)

    # T = np.array([
    #     [1, 0, sinb],
    #     [0, cosa, -sina*cosb],
    #     [0, sina, cosa*cosb]
    # ])
    row1 = torch.tensor([1, 0, 0], dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
    row2 = torch.stack([torch.zeros(batch_size), cosa, sina], dim=-1)
    row3 = torch.stack([sinb, -sina*cosb, cosa*cosb], dim=-1)
    T = torch.stack([row1, row2, row3], dim=-1)
    
    omega = torch.matmul(T, euler_dot.unsqueeze(-1)).squeeze(-1)

    return omega

def pos_euler_to_rigid_transform(**kwargs):
    """
        x, y, z
        rx, ry, rz in euler "XYZ" sequence
    """
    assert len(kwargs) > 0
    batch_size = kwargs[next(iter(kwargs))].shape[0]    # get the first key-value pair
    
    trans = torch.zeros((batch_size, 3))
    euler = torch.zeros((batch_size, 3))

    for pos_idx, key in enumerate(['x', 'y', 'z']):
        if key in kwargs:
            trans[:, pos_idx] = kwargs[key]
    
    for pos_idx, key in enumerate(['rx', 'ry', 'rz']):
        if key in kwargs:
            euler[:, pos_idx] = kwargs[key]

    matrix = torch.zeros((batch_size, 4, 4))
    matrix[:, :3, 3] = trans
    matrix[:, :3, :3] = roma.euler_to_rotmat('XYZ', euler)
    matrix[:, 3, 3] = 1.0

    return matrix

def transforms3d_to_xyz_quat(t3d, quat="xyzw"):
    assert quat in ["wxyz", "xyzw"]
    trans = torch.zeros((len(t3d), 7))
    trans[:, :3] = t3d.get_matrix()[:, :3, 3]
    trans[:, 3:] = roma.rotmat_to_unitquat(t3d.get_matrix()[:, :3, :3])
    if quat == "wxyz":
        trans[:, 3:] = roma.quat_xyzw_to_wxyz(trans[:, 3:])
    return trans

def compute_matrix_inverse(matrix, type="transpose"):
    """
        Compute the (pseudo) inverse of matrix
    """
    if type == "transpose":
        return matrix.transpose(-2, -1)
    elif type == "block_pinv":
        col_mask = ((matrix>0).to(torch.int).sum(dim=-2) > 0).to(torch.bool)
        block_pinv = torch.zeros_like(matrix.transpose(-2, -1))
        for i in range(matrix.shape[0]):
            # block_pinv[i] = torch.pinverse(matrix[i])
            block_pinv[i, col_mask[i], :] = torch.pinverse(matrix[i, :, col_mask[i]])
        return block_pinv
    
def pinocchio_SE3_to_matrix(se3):
    mat = np.eye(4)
    mat[:3, 3] = se3.translation
    mat[:3, :3] = se3.rotation

    return mat

if __name__ == "__main__":
    def quaternion_to_angular_velocity(q, q_dot):
        """
        由四元数及其导数计算角速度向量。
        q: 四元数 [q0, q1, q2, q3]
        q_dot: 四元数的导数 [dq0, dq1, dq2, dq3]
        返回值: 角速度向量 [ωx, ωy, ωz]
        """
        q0, q1, q2, q3 = q
        dq0, dq1, dq2, dq3 = q_dot
        
        omega_x = 2 * (q0 * dq1 - q1 * dq0 + q2 * dq3 - q3 * dq2)
        omega_y = 2 * (q0 * dq2 - q1 * dq3 + q2 * dq0 + q3 * dq1)
        omega_z = 2 * (q0 * dq3 + q1 * dq2 - q2 * dq1 + q3 * dq0)
        
        return np.array([omega_x, omega_y, omega_z])
    import roma
    
    # quat x, y, z, w

    euler = roma.unitquat_to_euler('ZYX', roma.random_unitquat(), degrees=False)
    # euler = np.array([0.0, 0.0, 0.0])
    euler_dot = 1e-9*np.random.randn(3)
    # euler_dot = np.array([1e-5, 0.0, 0.0])
    euler_1 = euler + euler_dot
    print("euler: ", euler)

    omega_ans = euler_derivative_to_omega(euler, euler_dot, seq="XYZ")
    print("omega (computed by matrix): ", omega_ans)

    quat = roma.quat_xyzw_to_wxyz(roma.euler_to_unitquat('ZYX', euler))
    quat_1 = roma.quat_xyzw_to_wxyz(roma.euler_to_unitquat('ZYX', euler_1))
    quat_dot = quat_1 - quat
    print("quat(dot): ", quat_dot)

    def make_quat_matrix(quat):
        q0, q1, q2, q3 = quat
        return np.array([
            [q0, -q1, -q2, -q3],
            [q1, q0, -q3, q2],
            [q2, q3, q0, -q1],
            [q3, -q2, q1, q0]
        ])
    
    # omega_fdiff = (2 * np.linalg.inv(make_quat_matrix(quat)).dot(quat_dot))[1:]
    omega_fdiff = quaternion_to_angular_velocity(quat, quat_dot)
    print("omega (computed by finite difference): ", omega_fdiff)
