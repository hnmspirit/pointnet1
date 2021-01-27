import os
import argparse
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from data.shapenet import ShapeNet
from pointnet import PointNetCls

seed = 20
blue = lambda x: '\033[94m' + x + '\033[0m'

def parse_args():
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--dataset', type=str, default='shapenetcore_partanno_segmentation_benchmark_v0')
    parser.add_argument('--nclass', type=int, default=16)
    parser.add_argument('--outf', type=str, default='ckpt')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--nepoch', type=int, default=5)
    parser.add_argument('--pretrain_epoch', type=int, default=-1)
    parser.add_argument('--npoint', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_l2', type=float, default=1e-4)
    parser.add_argument('--neval', type=int, default=5)
    return parser.parse_args()

# init
opt = parse_args()
random.seed(seed)
torch.manual_seed(seed)
os.makedirs(opt.outf, exist_ok=True)

# data
trn_set = ShapeNet(opt.dataset, opt.npoint)
tst_set = ShapeNet(opt.dataset, opt.npoint, 'test')

trn_loader = DataLoader(trn_set, batch_size=opt.batch_size, shuffle=True, num_workers=4)
tst_loader = DataLoader(tst_set, batch_size=opt.batch_size, shuffle=False, num_workers=4)

nbatch_trn = len(trn_loader)
nbatch_tst = len(tst_loader)
ntest = len(tst_set)

print('data loaded.')
print('#train: %5d, #btrain: %4d' % (len(trn_set), nbatch_trn))
print('#test:  %5d, #btest:  %4d' % (len(tst_set), nbatch_tst))


# model
net = PointNetCls(k=opt.nclass)
if opt.pretrain_epoch > 0: net.load_state_dict(torch.load('%s/net_%06d.pth' % (opt.outf, opt.pretrain_epoch)))
net = net.cuda()

print('\nmodel loaded.')

crit = nn.CrossEntropyLoss()
optim = Adam(net.parameters(), lr=opt.lr, weight_decay=opt.weight_l2)
schdl = lr_scheduler.StepLR(optim, step_size=10, gamma=0.7)

def evaluate(net, ntest, tst_loader):
    nbatch_tst = len(tst_loader)
    net = net.eval()
    correct = 0.
    tst_loss = 0.
    with torch.no_grad():
        for (x,y) in tst_loader:
            x = x.cuda()
            y = y.cuda()

            out = net(x)
            loss = crit(out, y)
            tst_loss += loss.item()

            pred = out.argmax(1)
            correct += (pred == y).sum().item()
    print('[%d] %s loss: %f - acc: %f' % (epoch, blue('test'), tst_loss / nbatch_tst, correct / ntest))

# train
print('\nstart training ...')
for epoch in range(opt.pretrain_epoch+1, opt.nepoch+1):
    trn_loss = 0.
    for i, (x,y) in enumerate(trn_loader, 0):
        net = net.train()
        x = x.cuda()
        y = y.cuda()
        optim.zero_grad()

        out = net(x)
        loss = crit(out, y)
        loss.backward()
        optim.step()

        trn_loss += loss.item()

    print('[%d] train loss: %f' % (epoch, trn_loss / nbatch_trn))
    schdl.step()

    # eval
    if epoch % opt.neval == 0:
        evaluate(net, ntest, tst_loader)
        torch.save(net.state_dict(), '%s/net_%06d.pth' % (opt.outf, epoch))
