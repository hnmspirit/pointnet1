# pointnet pytorch implementation
+ task: Shape Classification
+ dataset: ShapeNetSubset
+ paper: https://arxiv.org/abs/1612.00593
+ source: https://github.com/fxia22/pointnet.pytorch

## data
```
wget https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip --no-check-certificate
unzip shapenetcore_partanno_segmentation_benchmark_v0
```

## train
```
python train_cls.py
```

## inference and visualization
```
python dashboard.py
```