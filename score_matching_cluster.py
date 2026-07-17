# %%
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from sklearn import datasets, cluster, metrics
from scipy.spatial.distance import pdist, squareform
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from itertools import cycle, islice
from score_matching import train, sm_cluster_predict, plot_gradients
from torchsummary import summary
# plt.axis('off')

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


swiss_roll, _ = datasets.make_swiss_roll(n_samples=1000, noise=1.0) 
swiss_roll = swiss_roll[:,[0,2]] / 10

noisy_circles, _ = datasets.make_circles(n_samples=1000, factor=0.5, noise=0.05)

noisy_moons, _ = datasets.make_moons(n_samples=1000, noise=0.05)

blobs,_ = datasets.make_blobs(n_samples=1000, random_state=8)

no_structure = np.random.rand(1000, 2)

varied,_ = datasets.make_blobs(n_samples=1000, cluster_std=[1.0, 2.5, 0.5], random_state=1700) 

X, y = datasets.make_blobs(n_samples=1000, random_state=170)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso, _ = (X_aniso, y)

# pathbased = np.loadtxt("E:\Dataset\cluster\\pathbased.txt")[:,:2]
# s1 = np.loadtxt("E:\Dataset\cluster\\s3.txt")

# spiral = (np.loadtxt("E:\Dataset\cluster\\spiral.txt")[:,:2] - 17 )/ 10
# b,x = spiral.shape
# s = spiral
# t = 10
# for i in range(t):
#     noise = np.random.uniform(-0.1,0.1,(b,x)) + s
#     spiral = np.concatenate((spiral, noise))

# spiral.reshape(b*(t+1), x )

data = swiss_roll.T 
# data = varied.T 
plt.scatter(*data, color='blue', edgecolor='white', alpha=0.99, s=30)
plt.show()

#%%
model = nn.Sequential(
    nn.Linear(2, 150), nn.Softplus(),
    nn.Linear(150, 150), nn.Softplus(),
    nn.Linear(150, 2)
).cuda()

dataset = torch.tensor(data.T).float().cuda()

train(model, dataset, 3000, 0.001)

#%%
def plot_x_alogn_grad():
    def sample_simple(model, x, n_steps=4, eps=0.2):
        x_sequence = [x.unsqueeze(0)]
        for s in range(n_steps):
            score =  model(x)
            normalize_grad = (score.T / torch.norm(score.T, dim=0)).T
            x = x + eps * normalize_grad
            x_sequence.append(x.unsqueeze(0))
        return torch.cat(x_sequence)

    x = torch.Tensor([1.5, 1.5]).cuda()
    samples = sample_simple(model, x).detach().cpu().numpy()
    plot_gradients(model=model, data=data,max=1.8,min=-1.8, eps=0.0001)

    plt.scatter(samples[:, 0], samples[:, 1], color='green', edgecolor='white', s=200)
    # draw arrows for each  step
    deltas = (samples[1:] - samples[:-1])
    deltas = deltas - deltas / np.linalg.norm(deltas, keepdims=True, axis=-1) * 0.04
    for i, arrow in enumerate(deltas):
        plt.arrow(samples[i,0], samples[i,1], arrow[0], arrow[1], width=1e-4, head_width=4e-2, color="blue", linewidth=3)
    plt.show()
plot_x_alogn_grad()

plot_gradients(model, data,-2,2)
plt.show()

#%% +step
dataset = dataset.detach()
arr = []
fig = plt.figure()

def smld_subplot(dataset, cluster_step=3001, eps=0.01, show_at=[0, 10, 20, 3000]):
    _dataset = dataset.data.clone()
    
    # Khởi tạo figure với 2 hàng và 2 cột
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # Trải phẳng mảng axes để dễ dàng truy cập theo chỉ số (0, 1, 2, 3)
    axes_flat = axes.flatten()
    plot_idx = 0

    for s in range(cluster_step):
        # Kiểm tra nếu bước hiện tại cần vẽ và chúng ta còn chỗ trên đồ thị (tối đa 4 plots)
        if s in show_at and plot_idx < 4:
            ax = axes_flat[plot_idx]
            
            # Vẽ dữ liệu gốc làm nền (màu xám mờ)
            ax.scatter(*data, alpha=0.1, color='gray', s=10, label='Original')
            
            # Vẽ vị trí các samples hiện tại
            current_points = _dataset.cpu().detach().numpy().T
            ax.scatter(*current_points, color='blue', edgecolor='white', s=30, label=f'Step {s}')
            
            ax.set_title(f"Iteration Step: {s}")
            ax.legend(loc='upper right')
            plot_idx += 1

        with torch.no_grad():
            # Cập nhật vị trí
            grad_direction = model(_dataset)
            # Normalize giúp bước đi ổn định
            _dataset += eps * torch.nn.functional.normalize(grad_direction)

    plt.tight_layout() # Tự động căn chỉnh khoảng cách giữa các đồ thị
    plt.show()

    return _dataset

# Gọi hàm thực hiện (Lưu ý: cluster_step phải lớn hơn giá trị cuối cùng trong show_at)
cluster_dataset = smld_subplot(dataset, cluster_step=3001, eps=0.01, show_at=[0, 10, 20, 3000])

# %%

#%% sgd 不行 adam ok ??? SGD learning rate  要比較大
# def bkpg_cluster(dataset, step=20000, learning_rate=100):
#     _dataset = dataset.data.clone()
#     clustering_optimizer = optim.SGD([_dataset], lr=learning_rate)
#     # clustering_optimizer = optim.SGD([_dataset], lr=learning_rate)

#     for i in range(step):
#         loss = score_matching(model, _dataset)
#         clustering_optimizer.zero_grad()
#         loss.backward()
#         clustering_optimizer.step()

#     plt.scatter(*data, alpha=0.5, color='blue', edgecolor='white',s = 50)
#     plt.scatter(*_dataset.cpu().detach().numpy().T, color='black', edgecolor='white', s=80)
#     plt.show()

#     return _dataset

# bkpg_cluster_dataset = bkpg_cluster(dataset)



#%%

sm_y = sm_cluster_predict(cluster_dataset, eps=0.01)



#%%
def draw_cluster_color(center, y, draw_center=True):
    if type(cluster_dataset) == torch.Tensor and center is not None:
        center = center.cpu().detach().numpy()
    colors = np.array(
                list(
                    islice(
                        cycle(
                            [
                                "#377eb8","#ff7f00","#4daf4a","#f781bf","#a65628","#984ea3","#e41a1c",
                                "#dede00","#006000","#000093","#707038","#FFE4CA","#FF0080","#28004D",
                                "#B7FF4A","#3D7878","#DEFFAC","#DDDDFF","#D9FFFF","#DEDEBE","#28004D",
                                "#000079","#FF00FF","#CCFF00","#8FBC8F","#00FA9A","#6A5ACD","#FFF0F5",
                                "#EE82EE","#FFB7DD",'#FF88C2','#FF44AA','#FF0088','#C10066','#A20055',
                                '#8C0044','#FFCCCC','#FF8888','#FF3333','#FF0000','#CC0000','#AA0000',
                                '#880000','#FFC8B4',"#668800","#DDDDDD","#AAAAAA","#888888","#666666",
                                "#444444",'#FFDDAA','#FFEE99','#FFFFBB','#EEFFBB','#CCFF99','#99FF99',
                                '#BBFFEE','#AAFFEE',
                         ]
                        ),
                        int(max(y) + 1),
                    )
                )
            )
            # add black color for outliers (if any)
    colors = np.append(colors, ["#000000"])
    data_T = data.T
    plt.scatter(data_T[:,0], data_T[:,1], alpha=0.5, color=colors[y], edgecolor='white',s = 50)
    if center is not None and draw_center:
        plt.scatter(center[:,0], center[:,1], color=colors[y], edgecolor='black', s=90)
    plt.show()


draw_cluster_color(cluster_dataset, sm_y)
# print("SM Cluster silhouette_score:", metrics.silhouette_score(data.T, sm_y, metric='euclidean'))

# #%%
# dbscan = cluster.DBSCAN(eps=0.8)
# dbscan_data = data.T
# dbscan.fit(dbscan_data)
# y_pred = dbscan.labels_.astype(int)
# draw_cluster_color(None, y_pred)
# print("DBSCAN Cluster silhouette_score:", metrics.silhouette_score(dbscan_data, y_pred, metric='euclidean'))


#%%
#%% minimal spaning tree

# def minimum_spanning_tree(X, copy_X=True):
#     """X are edge weights of fully connected graph"""
#     if copy_X:
#         X = X.copy()

#     if X.shape[0] != X.shape[1]:
#         raise ValueError("X needs to be square matrix of edge weights")
#     n_vertices = X.shape[0]
#     spanning_edges = []
    
#     # initialize with node 0:                                                                                         
#     visited_vertices = [0]                                                                                            
#     num_visited = 1
#     # exclude self connections:
#     diag_indices = np.arange(n_vertices)
#     X[diag_indices, diag_indices] = np.inf
    
#     while num_visited != n_vertices:
#         new_edge = np.argmin(X[visited_vertices], axis=None)
#         # 2d encoding of new_edge from flat, get correct indices                                                      
#         new_edge = divmod(new_edge, n_vertices)
#         new_edge = [visited_vertices[new_edge[0]], new_edge[1]]                                                       
#         # add edge to tree
#         spanning_edges.append(new_edge)
#         visited_vertices.append(new_edge[1])
#         # remove all edges inside current tree
#         X[visited_vertices, new_edge[1]] = np.inf
#         X[new_edge[1], visited_vertices] = np.inf                                                                     
#         num_visited += 1
#     return np.vstack(spanning_edges)

# X = squareform(pdist(d.T))
# edge_list  = minimum_spanning_tree(X)
# fig = plt.figure(figsize=(16, 12))
# for edge in edge_list:
#     i, j = edge
#     plt.plot([d.T[i, 0], d.T[j, 0]], [d.T[i, 1], d.T[j, 1]], c='g')
# plt.show()


#%% animation tree
# X = squareform(pdist(d.T))
# edge_list  = minimum_spanning_tree(X)
# fig = plt.figure(figsize=(16, 12))
# plt.scatter(*d, alpha=0.5, color='red', edgecolor='white', s=80)

# def animate(x):
#     [i, j] = edge_list[x]
#     plt.xlim(-1.5, 2.0)
#     plt.ylim(-1.5, 2.0)
#     plt.plot([d.T[i, 0], d.T[j, 0]], [d.T[i, 1], d.T[j, 1]], c='b')


# ani = animation.FuncAnimation(fig, animate, frames=len(edge_list), interval=30, repeat_delay=1000)
# ani.save("tree.gif",writer='pillow')