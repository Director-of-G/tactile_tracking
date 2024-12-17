import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from torch.nn import MSELoss
import torch.optim as optim
from matplotlib import pyplot as plt
import numpy as np
import joblib
from scipy.spatial.transform import Rotation as SciR
from sklearn.model_selection import train_test_split
from dataset import load_dataset_from_disk
from sklearn.preprocessing import StandardScaler

from sensor_model import ForwardNet
from visualize_dataset import plot_taxel_debug_view


# 创建模型实例
model = ForwardNet(input_dim=3, output_dim=(4, 4))

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 生成一些示例数据
# 输入数据是6维（3D平移和3D欧拉角），假设为1000个样本
# 输出数据是4x4的触觉阵列读数
dataset_path = '../data/dataset_v1'
all_taxel_reading, all_panel_state, _ = load_dataset_from_disk(dataset_path, sel_dim=[2, 3, 4], use_quat=False)
valid_idx = all_panel_state[:, 0] < 0.02
print(f'Valid data: {np.sum(valid_idx)}')
all_taxel_reading = all_taxel_reading[valid_idx]
all_panel_state = all_panel_state[valid_idx]

input_data = all_panel_state.reshape(-1, 3).astype(np.float32)
output_data = all_taxel_reading.reshape(-1, 16).astype(np.float32)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2, random_state=42)

# 对输入和输出数据进行标准化
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

# 训练集和测试集都需要进行标准化
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
y_train = scaler_Y.fit_transform(y_train)
y_test = scaler_Y.transform(y_test)

# save the scaler (joblib works while pickle doe not)
joblib.dump({'X': scaler_X, 'Y': scaler_Y}, '../model/scaler.pkl')

# 转换为PyTorch张量
X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)
X_test = torch.tensor(X_test)
y_test = torch.tensor(y_test)

# 训练模型
def train_force_motion_model():
    num_epochs = 1000
    best_eval_loss = np.inf
    train_epoch, train_loss = [], []
    eval_epoch, eval_loss = [], []
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()

        train_epoch.append(epoch)
        train_loss.append(loss.item())
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        if epoch == 1 or (epoch + 1) % 25 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                test_loss = criterion(test_outputs, y_test)
                if test_loss < best_eval_loss:
                    best_eval_loss = test_loss
                    torch.save(model.state_dict(), '../model/best_model.pt')
                    print("Save best model with test loss: {:.4f}".format(test_loss.item()))
                eval_epoch.append(epoch)
                eval_loss.append(test_loss.item())
                print(f'Test Loss: {test_loss.item():.4f}')

    plt.figure()
    plt.plot(train_epoch, train_loss, label='Train Loss', color='blue')
    plt.plot(eval_epoch, eval_loss, label='Test Loss', color='orange')
    plt.legend()
    plt.grid("on")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

def test_force_motion_model():
    model.load_state_dict(torch.load('../model/best_model.pt'))
    model.eval()
    for _ in range(10):
        idx = np.random.choice(len(X_test))
        sample_input, sample_gt = X_test[idx].unsqueeze(0), y_test[idx].unsqueeze(0)
        sample_output = model(sample_input).detach().numpy()
        
        # scale
        sample_input = scaler_X.inverse_transform(sample_input)
        sample_gt = scaler_Y.inverse_transform(sample_gt)
        sample_output = scaler_Y.inverse_transform(sample_output)

        taxel_state = np.zeros(6,)
        taxel_state[[2, 3, 4]] = sample_input
        taxel_state = np.concatenate((taxel_state[:3], SciR.from_euler('xyz', taxel_state[3:]).as_quat()))
        
        plot_taxel_debug_view(
            sensor_state=taxel_state.squeeze(),
            taxel_gt=sample_gt.squeeze(),
            taxel_pred=sample_output.squeeze()
        )
        import pdb; pdb.set_trace()

def test_optimization():
    model.load_state_dict(torch.load('../model/best_model.pt'))
    model.eval()

    criterion = MSELoss()

    idx = np.random.choice(len(X_test), size=2)
    idx_target, idx_sample = idx
    target_state, target_tactile = X_test[idx_target], y_test[idx_target]
    sample_state, sample_tactile = X_test[idx_sample], y_test[idx_sample]
    print("x0: ", sample_state)

    sample_state.requires_grad = True
    optimizer = optim.Adam([sample_state], lr=0.1)

    num_steps = 1000
    for idx in range(num_steps):
        optimizer.zero_grad()
        pred_tactile = model(sample_state)

        loss = criterion(pred_tactile.squeeze(), target_tactile)
        print("loss: ", loss.item())

        if loss.item() < 1e-3:
            breakpoint()

        loss.backward()

        if idx == 0:
            print("grad: ", sample_state.grad)

        optimizer.step()


if __name__ == "__main__":
    # train_force_motion_model()
    # test_force_motion_model()
    test_optimization()
