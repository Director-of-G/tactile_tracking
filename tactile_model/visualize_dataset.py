import os
import plotly.subplots as sp
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as SciR

from dataset import load_dataset_from_disk

def create_cube(position, orientation, size=(1.0, 1.0, 1.0)):
    """ 创建立方体的顶点和面数据

    :param position: 立方体的中心位置，格式为(x, y, z)
    :param orientation: 旋转矩阵（3x3），定义立方体的朝向
    :param size: 立方体的尺寸，格式为(length_x, length_y, length_z)
    :return: 顶点坐标(x, y, z)和面索引(i, j, k, l)
    """
    # 立方体的初始顶点（中心位于原点）
    half_size_x, half_size_y, half_size_z = np.array(size) / 2.0
    vertices = np.array([[-half_size_x, -half_size_y, -half_size_z],
                         [ half_size_x, -half_size_y, -half_size_z],
                         [ half_size_x,  half_size_y, -half_size_z],
                         [-half_size_x,  half_size_y, -half_size_z],
                         [-half_size_x, -half_size_y,  half_size_z],
                         [ half_size_x, -half_size_y,  half_size_z],
                         [ half_size_x,  half_size_y,  half_size_z],
                         [-half_size_x,  half_size_y,  half_size_z]])

    # 旋转并平移顶点
    rotated_vertices = np.dot(vertices, orientation.T) + position

    # 立方体的面索引
    faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
             [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5]]

    return rotated_vertices, faces

def plot_cubes(cubes, colors=None, fig=None, rc=(1, 1)):
    """ 使用Plotly绘制立方体

    :param cubes: 包含立方体数据的列表，每个立方体数据为一个元组 (position, orientation, size)
    """
    if fig is None:
        fig = go.Figure()

    for idx, (position, orientation, size) in enumerate(cubes):
        vertices, faces = create_cube(position, orientation, size)

        # 将面展开为三角形绘制
        i, j, k = [], [], []
        x, y, z = [], [], []

        for face in faces:
            i.extend([face[0], face[0], face[0]])
            j.extend([face[1], face[2], face[3]])
            k.extend([face[2], face[3], face[1]])

            # 顶点坐标
            x.extend(vertices[face, 0])
            y.extend(vertices[face, 1])
            z.extend(vertices[face, 2])

        fig.add_trace(go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=i,
            j=j,
            k=k,
            opacity=0.5,
            color=colors[idx] if colors is not None else 'blue'
        ), row=rc[0], col=rc[1])

    return fig

def plot_taxel_debug_view(sensor_state, taxel_gt, sensor_pred=None, taxel_pred=None):
    """
        sensor_pred is useful for double (forward & inverse) model training
    """
    fig = sp.make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "scatter3d"}, {"type": "xy"}], # 第一行
            [{"type": "scatter3d"}, {"type": "xy"}]],                 # 第二行
        subplot_titles=("pose_gt", "taxel_gt", "pose_pred", "taxel_pred"),  # 标题
        column_widths=[0.5, 0.5],  # 左边和右边的列宽相等
        row_heights=[0.5, 0.5],    # 右边的两行高度相等
        horizontal_spacing=0.1,    # 控制左右子图之间的间距
        vertical_spacing=0.1       # 控制上下子图之间的间距
    )

    position1 = np.array([0.0, 0.0, -0.01])
    orientation1 = np.eye(3)
    size1 = (0.06, 0.06, 0.02)

    # 示例数据：立方体的位置和朝向
    position2 = sensor_state[:3]
    orientation2 = SciR.from_quat(sensor_state[3:]).as_matrix()
    size2 = (0.012, 0.012, 0.004)

    position3 = sensor_pred[:3]
    orientation3 = SciR.from_quat(sensor_pred[3:]).as_matrix()
    size3 = size2

    fig = plot_cubes([(position1, orientation1, size1), (position2, orientation2, size2)], colors=['mediumvioletred', 'lightgreen'], fig=fig, rc=(1, 1))

    fig = plot_cubes([(position1, orientation1, size1), (position3, orientation3, size3)], colors=['mediumvioletred', 'lightgreen'], fig=fig, rc=(2, 1))

    # 绘制tactile reading
    fig.add_trace(go.Heatmap(z=taxel_gt.reshape(4, 4), colorscale='gray'), row=1, col=2)
    fig.update_xaxes(scaleanchor="y", row=1, col=2)

    if taxel_pred is not None:
        fig.add_trace(go.Heatmap(z=taxel_pred.reshape(4, 4), colorscale='gray'), row=2, col=2)
        fig.update_xaxes(scaleanchor="y", row=2, col=2)        

    fig.show()


if __name__ == "__main__":
    dataset_path = './data/dataset_v1'
    seg_idx = 28
    data_idx = 1
    data = np.load(f'{dataset_path}/{str(seg_idx).zfill(3)}.npz')
    taxel_reading_data = data['taxel_reading']
    panel_state_data = data['panel_state']

    position1 = np.array([0.0, 0.0, -0.01])
    orientation1 = np.eye(3)
    size1 = (0.06, 0.06, 0.02)

    # 示例数据：立方体的位置和朝向
    position2 = panel_state_data[data_idx, :3]
    orientation2 = SciR.from_quat(panel_state_data[data_idx, 3:]).as_matrix()
    size2 = (0.012, 0.012, 0.004)

    fig = sp.make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'xy'}]],  # 左边为3D图，右边为2D图
        column_widths=[0.7, 0.3]  # 调整列的宽度比例
    )

    # 绘制两个立方体
    import pdb; pdb.set_trace()
    fig = plot_cubes([(position1, orientation1, size1), (position2, orientation2, size2)], colors=['mediumvioletred', 'lightgreen'], fig=fig, rc=(1, 1))

    # 绘制tactile reading
    fig.add_trace(go.Heatmap(
        z=taxel_reading_data[data_idx].reshape(4, 4),          # 图像数据
        colorscale='Greys',  # 选择颜色映射
        showscale=False,       # 不显示颜色条
        xgap=1,                # 增加色块之间的间距
        ygap=1
    ), row=1, col=2)
    fig.update_xaxes(scaleanchor="y", row=1, col=2)

    fig.show()

    all_taxel_reading, all_panel_state, _ = load_dataset_from_disk(dataset_path)

    print("dataset | taxel_reading shape:", all_taxel_reading.shape)
    print("dataset | panel_state shape:", all_panel_state.shape)

    # 绘制每个像素点的灰度分布箱线图
    plt.figure(figsize=(12, 6))
    plt.boxplot(all_taxel_reading, vert=True, patch_artist=True)

    # 设置x轴标签，表示每个像素点的位置
    plt.xticks(ticks=np.arange(1, 17), labels=[f'({i//4},{i%4})' for i in range(16)])
    plt.xlabel('Pixel Position')
    plt.ylabel('Gray Value')
    plt.title('Gray Value Distribution Across All Pixels')

    all_panel_state = np.concatenate((all_panel_state[:, :3], \
                                    SciR.from_quat(all_panel_state[:, 3:]).as_euler('xyz', degrees=False)), axis=-1)
    plt.figure(figsize=(12, 6))
    plt.boxplot(all_panel_state, vert=True, patch_artist=True)
    plt.xticks(ticks=np.arange(1, 7), labels=['x', 'y', 'z', 'rx', 'ry', 'rz'])
    plt.xlabel('SE(3) dim')
    plt.ylabel('Value')
    plt.title('SE(3) distribution')

    plt.show()
