# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, cluster, metrics
import torch
import torch.nn as nn
from score_matching import train, langevin_dynamic, sm_cluster_predict
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

seed_everything(1234) # Số 42 là số may mắn kinh điển, bạn có thể chọn số bất kỳ


iris = datasets.load_iris()
data = iris.data
label = iris.target



#%%
model = nn.Sequential(
    nn.Linear(4, 150), nn.Softplus(),
    nn.Linear(150, 150), nn.Softplus(),
    nn.Linear(150, 4)
).cuda()

dataset = torch.tensor(data).float().cuda()
train(model, dataset, epoch=5000)

#%% +step
dataset = dataset.detach()
cluster_dataset = langevin_dynamic(model, dataset, step=300000)
#%% 平方距離 沒開根號 

sm_y = sm_cluster_predict(cluster_dataset,eps=0.1) 

print("SM result:", sm_y)
print("SM Cluster rand_score:", metrics.rand_score(label, sm_y))
print("SM Cluster adjusted rand_score:", metrics.adjusted_rand_score(label, sm_y))

#%%
dbscan = cluster.DBSCAN(eps=0.5)
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