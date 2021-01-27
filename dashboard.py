import os
from os.path import basename, dirname, join
import json
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import pyvista as pv
import panel as pn
from data.shapenet import normalize
from pointnet import PointNetCls

np.random.seed(2)
net = PointNetCls(k=16, cpu=True)
net.load_state_dict(torch.load('ckpt/net_000095.pth', map_location=torch.device('cpu')))
net = net.eval()
print('net loaded.')

def get_paths(root, partition='test'):
    listpath = join(root, 'train_test_split', 'shuffled_%s_file_list.json'%partition)
    with open(listpath) as f:
        paths = json.load(f)

    sampled_paths = []
    choice = np.random.choice(len(paths), 20, False)
    for i in choice:
        _, fid, uid = paths[i].split('/')
        xpath = join(root, fid, 'points', uid+'.pts')
        sampled_paths.append(xpath)
    return sorted(sampled_paths)

def get_cats(root):
    catpath = join(root, 'synsetoffset2category.txt')
    fid2cat = {}
    id2cat = []
    with open(catpath) as f:
        for line in f:
            x = line.split()
            id2cat.append(x[0])
            fid2cat[x[1]] = x[0]
    return fid2cat, id2cat

def inference(x, k=3):
    x = normalize(x).T[None,]
    x = torch.from_numpy(x)
    with torch.no_grad():
        prob = F.softmax(net(x)[0]).numpy()
    ids = np.argsort(prob)[::-1][:k]
    return ids, prob

def btn_run(event):
    # panel
    camera = [(0.0, 1.5, 1.5),
             (0.0, 0.0, 0.0),
             (0.0, 1.0, 0.0)]
    path = select.value
    xyz = np.loadtxt(path, dtype=np.float32)

    pl.clear()
    cloud = pv.PolyData(xyz)
    cloud['fields'] = xyz[:,2]
    pl.add_mesh(cloud, cmap='cool')
    pl.camera_position =  camera
    pan.object = pl.ren_win

    # text
    metabox.value = 'visualizing {} point'.format(len(xyz))

    # gt
    gt = fid2cat[basename(dirname(dirname(path)))]
    gtbox.value = gt

    # pred
    ids, prob = inference(xyz, 3)
    res = [(id2cat[i], prob[i]) for i in ids]

    for i in range(3):
        resboxs[i].value = '{}: {:.2f}'.format(res[i][0], res[i][1])


root = '../../repo/dat3d/shapenetcore/'
paths = get_paths(root)
print('#paths:', len(paths))
fid2cat, id2cat = get_cats(root)


select = pn.widgets.Select(name='select pointcloud here', options=paths)
button = pn.widgets.Button(name='run', button_type='primary')
button.on_click(btn_run)
metabox = pn.widgets.TextInput(value='')
metabox.disabled = True
gtbox = pn.widgets.TextInput(value='')
gtbox.disabled = True
resboxs = [pn.widgets.TextInput(value='') for _ in range(3)]
for resbox in resboxs:
    resbox.disabled = True
pl = pv.Plotter()
pan = pn.panel(pl.ren_win, sizing_mode='stretch_both', orientation_widget=True)
dashb = pn.Row(
    pn.Column('### Shape Classification',
              '**input**', select, button, metabox,
              '**groundtruth**', gtbox,
              '**prediction**', resboxs[0], resboxs[1], resboxs[2]),
    pan,
    sizing_mode='stretch_both',
)
dashb.show()