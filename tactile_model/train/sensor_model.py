import torch
from torch import nn


"""
    We include two models here.
    The ForwardNet, maps panel pose to tactile reading.
    The InverseNet, maps tactile reading to panel pose
"""

class ForwardNet(nn.Module):
    def __init__(self, input_dim=6, output_dim=(4, 4)):
        super(ForwardNet, self).__init__()
        # 输入的SE(3)位姿是一个6维向量 (3D平移和3D欧拉角)，需要重塑为适合卷积层的形状
        num_taxels = output_dim[0] * output_dim[1]
        self.fc1 = nn.Linear(input_dim, 16 * num_taxels)  # 6维输入转换为 16个4x4的特征图
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 32个 4x4 的特征图
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 64个 4x4 的特征图
        self.fc2 = nn.Linear(64 * num_taxels, num_taxels)  # 将卷积层输出展平后，转换为 4x4的触觉阵列
        self.output_dim = output_dim
        
    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, 16, self.output_dim[0], self.output_dim[1])  # 将全连接层输出重塑为16个4x4的特征图
        x = torch.relu(self.conv1(x))  # 卷积层1
        x = torch.relu(self.conv2(x))  # 卷积层2
        x = x.view(x.size(0), -1)  # 展平卷积层输出
        x = self.fc2(x)  # 输出层
        return x

class InverseNet(nn.Module):
    def __init__(self, input_dim=(4, 4), output_dim=6):
        super(InverseNet, self).__init__()
        num_taxels = input_dim[0] * input_dim[1]
        self.fc1 = nn.Linear(num_taxels, 64 * num_taxels)  # 4x4的输入转换为64个4x4的特征图
        self.conv1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)  # 32个 4x4 的特征图
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)  # 16个 4x4 的特征图
        self.fc2 = nn.Linear(16 * num_taxels, output_dim)  # 将卷积层输出展平后，转换为6维位姿
        self.input_dim = input_dim
        
    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, 64, self.input_dim[0], self.input_dim[1])  # 将全连接层输出重塑为64个4x4的特征图
        x = torch.relu(self.conv1(x))  # 卷积层1
        x = torch.relu(self.conv2(x))  # 卷积层2
        x = x.view(x.size(0), -1)  # 展平卷积层输出
        x = self.fc2(x)  # 输出层，得到6维的位姿
        return x
    
if __name__ == "__main__":
    forward_model = ForwardNet(input_dim=3, output_dim=(4, 4))
    inverse_model = InverseNet(input_dim=(4, 4), output_dim=3)

    test_pose_in = torch.randn(64, 3)
    test_tactile_out = torch.randn(64, 16)

    print("------ Forward Model ------")
    print("Input shape: ", test_pose_in.shape)
    print("Output shape: ", forward_model(test_pose_in).shape)

    print("------ Inverse Model ------")
    print("Input shape: ", test_tactile_out.shape)
    print("Output shape: ", inverse_model(test_tactile_out).shape)

    print("------ Forward-Inverse Model ------")
    print("Input shape: ", test_pose_in.shape)
    print("Output shape: ", inverse_model(forward_model(test_pose_in)).shape)

    print("------ Inverse-Forward Model ------")
    print("Input shape: ", test_tactile_out.shape)
    print("Output shape: ", forward_model(inverse_model(test_tactile_out)).shape)
