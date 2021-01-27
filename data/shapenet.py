import os, glob
from os.path import basename, dirname, isdir, isfile, join
import json
import random
import numpy as np
from numpy import pi, sin, cos
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset

def normalize(x):
    mu = np.mean(x, 0)
    x = x - mu # center
    rad = np.max(np.sqrt(np.sum(x**2, 1)))
    x = x / rad # unit rad
    return x

def load_data(root, partition):
    cache_dir = root + '_cache'
    os.makedirs(cache_dir, exist_ok=True)

    cache_file = join(cache_dir, partition+'.pkl')
    if isfile(cache_file):
        print('loading data from cache ...')
        with open(cache_file, 'rb') as f:
            all_data = pickle.load(f)
    else:
        print('loading data and caching ...')
        all_data = save_cache(root, partition)
    return all_data

def save_cache(root, partition):
    listpath = join(root, 'train_test_split', 'shuffled_%s_file_list.json'%partition)
    with open(listpath) as f:
        paths = json.load(f)

    fids = [basename(x) for x in sorted(glob.glob(join(root, '0*')))]
    fid2id = {v:k for k,v in enumerate(fids)}

    all_data = []
    for p in tqdm(paths):
        _, fid, uid = p.split('/')
        xpath = join(root, fid, 'points', uid+'.pts')
        x = np.loadtxt(xpath, dtype=np.float32)
        y = fid2id[fid]
        all_data.append((x,y))
    cache_file = join(root + '_cache', partition+'.pkl')
    with open(cache_file, 'wb') as f:
        pickle.dump(all_data, f)
    return all_data

# augment
def random_point_dropout(x, low=0.1, high=0.8):
    coef = np.random.random(x.shape[0])
    drop_ratio = random.uniform(low, high)
    drop_idx = coef <= drop_ratio
    x[drop_idx] = x[0]
    return x

def random_point_scale(x, low=0.7, high=1.3):
    scale_ratio = random.uniform(low, high)
    x = x * scale_ratio
    return x

def random_point_shift(x, low=-0.5, high=0.5):
    shift_val = np.random.uniform(low, high, 3).astype(np.float32)
    x = x + shift_val
    return x

def random_point_jitter(x, mu=0, sigma=0.02):
    shift_val = np.random.normal(mu, sigma, x.shape).astype(np.float32)
    x = x + shift_val
    return x

def random_point_rot(x, low=0, high=2*pi):
    phi = random.uniform(low, high)
    M = np.array([[cos(phi), -sin(phi)],[sin(phi), cos(phi)]])
    x[:,[0,2]] = x[:,[0,2]].dot(M) # xz coord
    return x

class ShapeNet(Dataset):
    def __init__(self, root, npoint, partition='train'):
        self.data = load_data(root, partition)
        self.partition = partition
        self.npoint = npoint

    def __getitem__(self, idx):
        x,y = self.data[idx]
        choice = np.random.choice(len(x), self.npoint, replace=True)
        x = x[choice]
        x = normalize(x)

        if self.partition == 'train':
            x = random_point_dropout(x)
            x = random_point_rot(x)
            x = random_point_scale(x)
            x = random_point_jitter(x)
            x = random_point_shift(x)

        return x.T, y

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    tst = ShapeNet('shapenetcore_partanno_segmentation_benchmark_v0', 1024, 'test')
    print(len(tst))
    x, y = tst[0]
    print(x.shape, y)