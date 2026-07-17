# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, cluster, metrics
import torch
import torch.nn as nn
from score_matching import train, langevin_dynamic, sm_cluster_predict, langevin_dynamic_improved
from torchsummary import summary

import random
import os

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # nếu dùng nhiều GPU
    # Đảm bảo các thuật toán của cuDNN luôn nhất quán
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42) # Số 42 là số may mắn kinh điển, bạn có thể chọn số bất kỳ


# %% Load Wine Dataset
wine = datasets.load_wine()
data = wine.data    # Chứa các thuộc tính hóa học của rượu (13 features)
label = wine.target # Chứa nhãn của 3 loại rượu khác nhau (classes)

# Kiểm tra kích thước dữ liệu
print(f"Kích thước dữ liệu Wine: {data.shape}") # Sẽ trả về (178, 13)
print(f"Số lượng nhãn: {np.unique(label)}")    # Sẽ trả về [0, 1, 2]

# from sklearn.preprocessing import StandardScaler
# data = StandardScaler().fit_transform(data)

#%%
model = nn.Sequential(
    nn.Linear(13, 150), nn.Softplus(),
    nn.Linear(150, 150), nn.Softplus(),
    nn.Linear(150, 13)
).cuda()

dataset = torch.tensor(data).float().cuda()
train(model, dataset, epoch=15000, method='sm')

#%% +step
dataset = dataset.detach()
cluster_dataset = langevin_dynamic(model, dataset, step=300000)
#%% 平方距離 沒開根號 

sm_y = sm_cluster_predict(cluster_dataset,eps=130)
print("SM result:", sm_y)
print("SM Cluster rand_score:", metrics.rand_score(label, sm_y))
print("SM Cluster adjusted rand_score:", metrics.adjusted_rand_score(label, sm_y))

#%%
dbscan = cluster.DBSCAN(eps=0.15)
dbscan.fit(data)
dbscan_cluster_y = dbscan.labels_.astype(int)
print("dbscan result:", dbscan_cluster_y)
print("dbscan rand_score:", metrics.rand_score(label, dbscan_cluster_y))
print("dbscan adjusted rand_score:", metrics.adjusted_rand_score(label, dbscan_cluster_y))
#%%
agg = cluster.AgglomerativeClustering(
        linkage="single",
        n_clusters=3
    )
agg.fit(data)
agg_cluster_y = agg.labels_.astype(int)
print("agg result:", agg_cluster_y)
print("agg rand_score:", metrics.rand_score(label, agg_cluster_y))
print("agg adjusted_rand_score:", metrics.adjusted_rand_score(label, agg_cluster_y))
#%%
kmeans =  cluster.MiniBatchKMeans(n_clusters=3)
kmeans.fit(data)
kmeans_cluster_y = kmeans.labels_.astype(int)
print("kmeans result:", kmeans_cluster_y)
print("kmeans rand_score:", metrics.rand_score(label, kmeans_cluster_y))
print("kmeans adjusted_rand_score:", metrics.adjusted_rand_score(label, kmeans_cluster_y))
#%%
# c=0
# for i in range(150):
#     if(2==cluster_y[i]):
#         c+=1