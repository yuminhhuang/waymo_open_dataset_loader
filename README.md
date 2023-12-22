# Pytorch dataset loader for Waymo Open Dataset

# Waymo Open Dataset
[dataset](https://waymo.com/open/) [download](https://waymo.com/open/download/) [documentation](https://github.com/waymo-research/waymo-open-dataset)

## This tools use the v2.0.0 version of dataset
[v2 documentation](https://github.com/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial_v2.ipynb)

# Requirements
- torch
- dask
- waymo_open_dataset
- numba

# Usage
## Dataset Download
Downloads [waymo_open_dataset_v_2_0_0](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_2_0_0) or use the scripts below for a mini version of waymo_open_dataset_v_2_0_0.
```bash
cd waymo-mini
./run.sh
```
You may need to init [gsutils](https://cloud.google.com/storage/docs/discover-object-storage-gcloud?hl=zh_CN&_ga=2.213582103.-2027393445.1701001832) at first.

## Dataset Preprocess
Change `data_path` in `preprocess.py` to your location, and
```bash
python preprocess.py
```
This will create a preprocessed dataset in the directory where your dataset locates.

## Load pointclouds and labels for training
```python
from waymo_open_dataset_loader.dataset import Waymo

data_path = '/home/yuminghuang/dataset/waymo-mini'
dataset = Waymo(root=data_path, version='v2.0.0', split='train', has_image=False, has_label=True)
pointcloud, sem_label, inst_label = dataset.loadDataByIndex(index=0)
print("pointcloud, sem_label, inst_label", pointcloud, sem_label, inst_label)

import open3d as o3d
import numpy as np
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.asarray(pointcloud[:, :3], dtype=np.float32))
pcd.colors = o3d.utility.Vector3dVector(np.asarray(dataset.sem_color_lut[sem_label], dtype=np.float32))
o3d.visualization.draw_geometries([pcd])
```

## Project pointclouds to RANGE images
```python
from waymo_open_dataset_loader.dataset import Waymo
from waymo_open_dataset_loader.projection import RangeProjection

data_path = '/home/yuminghuang/dataset/waymo-mini'
dataset = Waymo(root=data_path, version='v2.0.0', split='train', has_image=False, has_label=False)
pointcloud, sem_label, inst_label = dataset.loadDataByIndex(0)
extrinsic, inclination = dataset.loadCalibrationByIndex(0)
proj_pointcloud, proj_range, proj_idx, proj_mask = RangeProjection(64, 2650).doProjection(pointcloud, extrinsic, inclination)
import matplotlib.pyplot as plt
plt.imshow(proj_range)
plt.show()
```
