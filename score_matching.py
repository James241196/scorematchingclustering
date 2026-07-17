# %%
import numpy as np
import torch
import torch.optim as optim
import torch.autograd as autograd
import matplotlib.pyplot as plt
from datetime import datetime
import os
from pathlib import Path

today = datetime.today().strftime('%m%d')
# ckpt_path = 'E:\JyunYi\Score Matching\\ckpt\\{}'.format(today)
ckpt_path = 'C:\Score matching\\ckpt\\{}'.format(today)

def jacobian(f, x):
    """Computes the Jacobian of f w.r.t x.
    :param f: function R^N -> R^N
    :param x: torch.tensor of shape [B, N]
    :return: Jacobian matrix (torch.tensor) of shape [B, N, N]
    """
    B, N = x.shape
    y = f(x)
    jacobian = list()
    for i in range(N):
        v = torch.zeros_like(y)
        v[:, i] = 1.
        dy_i_dx = autograd.grad(y, x, grad_outputs=v, retain_graph=True, create_graph=True, allow_unused=True)[0]  # shape [B, N]
        jacobian.append(dy_i_dx)
    jacobian = torch.stack(jacobian, dim=2).requires_grad_()
    return jacobian

def score_matching(model, samples, train=False):
    samples.requires_grad_(True)
    logp = model(samples)
    norm_loss = torch.norm(logp, dim=-1) ** 2 / 2.
    jacob_mat = jacobian(model, samples)
    tr_jacobian_loss = torch.diagonal(jacob_mat, dim1=-2, dim2=-1).sum(-1)
    return (tr_jacobian_loss + norm_loss).mean(-1)

def denoising_score_matching(scorenet, samples, sigma=0.01):
    perturbed_samples = samples + torch.randn_like(samples) * sigma
    target = - 1 / (sigma ** 2) * (perturbed_samples - samples)
    scores = scorenet(perturbed_samples)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1).mean(dim=0)
    return loss

def sliced_score_matching(model, samples):
    samples.requires_grad_(True)
    # Construct random vectors
    vectors = torch.randn_like(samples)
    vectors = vectors / torch.norm(vectors, dim=-1, keepdim=True)
    # Compute the optimized vector-product jacobian
    logp, jvp = autograd.functional.jvp(model, samples, vectors, create_graph=True)
    # Compute the norm loss
    norm_loss = (logp * vectors) ** 2 / 2.
    # Compute the Jacobian loss
    v_jvp = jvp * vectors
    jacob_loss = v_jvp
    loss = jacob_loss + norm_loss
    return loss.mean(-1).mean(-1)

def train(model, dataset, epoch=3000, learning_rate=1e-3, ckpt_name='', method='sm', print_itr=1000):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for t in range(1, epoch):
        if method == 'ssm':
            loss = sliced_score_matching(model, dataset)
        elif method == 'dsm':
            loss = denoising_score_matching(model, dataset)
        else:
            loss = score_matching(model, dataset)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ((t % print_itr) == 0):
            print(loss)
            if ckpt_name != '':
                Path(ckpt_path).mkdir(parents=True, exist_ok=True)
                torch.save({
                'model_state_dict': model.state_dict(),
                }, '{}\{}_{}.ckpt'.format(ckpt_path, ckpt_name, t))
# 



#%% +step
def langevin_dynamic(model, dataset, step=100000, eps = 1e-3):
    _dataset = dataset.data.clone()
    for s in range(step):
        with torch.no_grad():
            _dataset += eps * model(_dataset)

    return _dataset

#%%
def langevin_dynamic_improved(model, dataset, step=100000, eps=0.5):
    _dataset = dataset.clone()
    for s in range(step):
        with torch.no_grad():
            # Thử bỏ normalize hoặc dùng lr scheduler cho eps
            _dataset += eps * model(_dataset) 
    return _dataset
#%%
def sm_cluster_predict(dataset, eps = 1):
    _dataset = dataset.cpu().detach().numpy()
    category = 0
    y = np.ones_like(_dataset[:,0], dtype=np.int16)* -1
    b, _ = _dataset.shape
    for i in range(b):
        if(y[i] == -1):
            y[i] = category
            category += 1
        for j in range (i,b):
            if(y[j] != -1):
                continue
            is_pass = (np.square(_dataset[i] - _dataset[j])).mean(axis=-1) < eps
            if(np.all(is_pass)):
                y[j] = y[i]
    return y

def plot_gradients(model, data,max, min, eps =0.1):
    xx = np.stack(np.meshgrid(np.linspace(min, max, 20), np.linspace(min,max, 20)), axis=-1).reshape(-1, 2)
    score = model(torch.tensor(xx).cuda().float())
    normalize_grad = (score.T / torch.norm(score.T, dim=0)).T
    normalize_grad = eps * normalize_grad.detach().cpu().numpy()
    plt.figure(figsize=(10,10))
    plt.scatter(*data, alpha=0.2, color='red', edgecolor='white', s=120)
    plt.quiver(*xx.T, *normalize_grad.T, width=0.005, color='black')
    # plt.xlim(-1.5, 2.0)
    # plt.ylim(-1.5, 2.0)

def normalize(score, batch, c):
    norm = torch.norm(score,dim=-1).reshape([batch, 1]).expand(-1,c)
    norm = torch.where(norm == 0., torch.tensor(0.00000001 ,dtype=torch.float32).to('cuda:0'), norm)
    return score / norm