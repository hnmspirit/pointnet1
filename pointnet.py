import torch
import torch.nn as nn
import torch.nn.functional as F

class STN3d(nn.Module):
    def __init__(self, cpu=False):
        self.cpu = cpu
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU(True)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]

        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        eye = torch.tensor([1,0,0, 0,1,0, 0,0,1], dtype=torch.float32)
        if not self.cpu:
            eye = eye.cuda()
        x = x + eye
        x = x.view(-1, 3, 3)
        return x


class PointNetFeat(nn.Module):
    def __init__(self, cpu=False):
        super().__init__()
        self.stn = STN3d(cpu=cpu)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        # X = B,3,npoint
        trans = self.stn(x)
        x = torch.bmm(trans, x)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2)[0]
        return x

class PointNetCls(nn.Module):
    def __init__(self, k, cpu=False):
        super().__init__()
        self.feat = PointNetFeat(cpu=cpu)
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),

            nn.Linear(512, 256),
            nn.Dropout(p=0.5),
            nn.BatchNorm1d(256),
            nn.ReLU(True),

            nn.Linear(256, k)
        )

    def forward(self, x):
        # x = x.transpose(2,1) # to B,3,N
        x = self.feat(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    x = torch.randn(2,3,1024).cuda()
    net = PointNetCls(16).cuda()
    out = net(x)
    print('out: ', out.shape)
