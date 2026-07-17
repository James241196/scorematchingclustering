#%% Sample
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from helper_plot import hdr_plot_style
hdr_plot_style()

sample_size = 30
x = np.random.rand(sample_size,2,1)
x = x / 5
x = x + [[-0.9],[0.8]]




# Train
W = np.random.rand(2,2) 
eps = 0.01 
print(W)


def train(W, eps):
    for _ in range(10000):
        for _x in x:
            W -= np.asarray(eps * (np.mat(W) * np.mat(_x) * np.mat(_x.T)),dtype=np.float64)
            W[0,0] += - eps 
            W[1,1] += - eps

train(W, eps)

#plot gradient
xx = np.stack(np.meshgrid(np.linspace(-1.5, 1.5, 50), np.linspace(-1.5, 1.5, 50)), axis=-1).reshape(-1, 2)
scores = np.asarray([np.asarray(np.mat(W) * np.mat(xx[i]).T) for i in range(2500)]).reshape(2500,2)
scores_norm = np.linalg.norm(scores, axis=-1, ord=2, keepdims=True)
scores_log1p = scores / (scores_norm + 1e-9) * np.log1p(scores_norm)
plt.figure(figsize=(16,16))
for i in range(sample_size):
    plt.scatter(x[i][0],x[i][1], alpha=0.5, color='red', edgecolor='white', s=40)
plt.quiver(*xx.T, *scores_log1p.T, width=0.002, color='black')

plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)

plt.show()

print(W)

