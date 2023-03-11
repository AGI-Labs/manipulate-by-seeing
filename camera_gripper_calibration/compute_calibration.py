import numpy as np
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F
import pdb
from scipy.spatial.transform import Rotation as R 

# senity check pass
# T1 = torch.tensor([[np.sqrt(2)/2, np.sqrt(2)/2, 0, 0],[-np.sqrt(2)/2, np.sqrt(2)/2, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]], dtype=torch.float32)
# T = torch.tensor([[ 0.6965,  0.6512,  0.3015,  0.1061],
#         [ 0.7176, -0.6284, -0.3003, -0.1021],
#         [-0.0061,  0.4255, -0.9049,  0.0678],
#         [ 0.0000,  0.0000,  0.0000,  1.0000]])

# print(torch.matmul(torch.inverse(T1), T))


x = torch.tensor([[0.7071],[0.7071],[0.]], requires_grad = True)
y = torch.tensor([[0.672],[-0.6725],[0.3090]], requires_grad = True)
p = torch.tensor([[0.0990],[-0.0990],[0.0750]], requires_grad = True)

class CalibrationDataset(Dataset):

    def __init__(self):
        data = np.load('calibration.npz')
        self.robot_poses = data['robot_poses']
        self.tool_poses = data['tool_poses']

    def __len__(self):
        return len(self.robot_poses)  

    def __getitem__(self, idx):
        pos = self.robot_poses[idx][:3]
        quat = self.robot_poses[idx][3:]

        T_srbr0 = np.eye(4).astype(np.float32)
        T_srbr0[:3,:3] = R.from_quat(quat).as_matrix()
        T_srbr0[:3,3] = pos

        pos = self.tool_poses[idx][:3]
        quat = self.tool_poses[idx][3:]

        T_stbt0 = np.eye(4).astype(np.float32)
        T_stbt0[:3,:3] = R.from_quat(quat).as_matrix()
        T_stbt0[:3,3] = pos

        id1 = np.random.randint(len(self.robot_poses), size=1)[0]
        
        pos = self.robot_poses[id1][:3]
        quat = self.robot_poses[id1][3:]

        T_srbr1 = np.eye(4).astype(np.float32)
        T_srbr1[:3,:3] = R.from_quat(quat).as_matrix()
        T_srbr1[:3,3] = pos

        pos = self.tool_poses[id1][:3]
        quat = self.tool_poses[id1][3:]

        T_stbt1 = np.eye(4).astype(np.float32)
        T_stbt1[:3,:3] = R.from_quat(quat).as_matrix()
        T_stbt1[:3,3] = pos

        return T_srbr0, T_stbt0, T_srbr1, T_stbt1

cal_dataset = CalibrationDataset()
train_data = DataLoader(cal_dataset, batch_size=64, shuffle=True)

criterion = nn.MSELoss()
optimizer = SGD(params=[x, y, p], lr=1e-3)

for i in range(10000):
    for T_srbr0, T_stbt0, T_srbr1, T_stbt1 in train_data:
        
        x_n = F.normalize(x, dim=0)
        z = torch.cross(x_n,y)
        z_n = F.normalize(z, dim=0)
        y_n = torch.cross(z_n, x_n)

        T_br_bt = torch.eye(4)

        T_br_bt[:3,0] = x_n[:,0]
        T_br_bt[:3,1] = y_n[:,0]
        T_br_bt[:3,2] = z_n[:,0]
        T_br_bt[:3,3] = p[:,0]

        T_br0br1 = torch.matmul(torch.inverse(T_srbr0), T_srbr1)
        T_br0bt1 = torch.matmul(T_br0br1, T_br_bt)

        T_bt0bt1 = torch.matmul(torch.inverse(T_stbt0), T_stbt1)
        T_br0bt1_p = torch.matmul(T_br_bt, T_bt0bt1)

        loss = criterion(T_br0bt1, T_br0bt1_p)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)

print(x)
print(y)
print(p)
print(T_br_bt)