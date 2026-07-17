#%%
from matplotlib.cbook import flatten
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from score_matching import train, sm_cluster_predict
from datetime import datetime
from torchsummary import summary
import glob
import os
from datetime import datetime
from pathlib import Path

today = datetime.today().strftime('%m%d')

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



path = "C:\Score matching"
type="jpg"
style_name = 'test1'
style_image = np.asarray(Image.open('{}\{}.{}'.format(path, style_name,type))) / 255
l, w, c = style_image.shape
style_5d = np.zeros([l,w,5])

result_path = 'C:\Score matching\\results\\5D_pixel\{}\{}'.format(today, style_name)
Path(result_path).mkdir(parents=True, exist_ok=True)

for i in range(l):
    for j in range(w):
        style_5d[i,j] = np.append(style_image[i,j],[i/l,j/w])

style_dataset = torch.tensor(style_5d).float().to('cuda:0')
style_dataset = style_dataset.reshape([l*w,5])
#%%
content_name = 'test1'
content_image = np.asarray(Image.open('{}\{}.{}'.format(path, content_name,type))) / 255
content_5d = np.zeros([l,w,5])
for i in range(l):
    for j in range(w):
        content_5d[i,j] = np.append(content_image[i,j],[i/l,j/w])
content_dataset = torch.tensor(content_5d).float().to('cuda:0')
# step_size = 5e-4

#%%
model = nn.Sequential(
    nn.Linear(5, 300), nn.GELU(),
    nn.Linear(300, 300), nn.GELU(),
     nn.Linear(300 , 5)
).to('cuda:0')


train(model, style_dataset,epoch=5000, method='sm')

#%%
step_size = 4e-4
flaten_image_3d = np.zeros([l*w,3])

def cluster(content_dataset):
    ims = []
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(projection='3d')


    for s in range(4201):
        # im = plt.imshow(content_dataset.cpu().detach().numpy(), animated=True)
        # ims.append([im])

        score =  model(content_dataset)
        normalize_grad = score / torch.norm(score,dim=-1).reshape([l*w,1]).expand(-1,5)
        # print(normalize_grad)

        with torch.no_grad():
            content_dataset = content_dataset +  step_size * normalize_grad
            content_dataset = torch.where(content_dataset >= 1., torch.tensor(1 ,dtype=torch.float32).to('cuda:0'), content_dataset)
            content_dataset = torch.where(content_dataset <= 0., torch.tensor(0,dtype=torch.float32).to('cuda:0'), content_dataset)

            if s % 600 == 0:
                ax.set_xlim3d(0.0, 1.0)
                ax.set_ylim3d(0.0, 1.0)
                ax.set_zlim3d(0.0, 1.0)
                flatten_image = content_dataset.cpu().detach().numpy().reshape([l*w, 5])
                flaten_image_3d = flatten_image[:,0:3]

                ax.scatter(*flaten_image_3d.T, s=1200, alpha=0.7,c=[*flaten_image_3d],edgecolors='black',linewidths=0.1)
                plt.savefig("{}\\5D_{}.png".format(result_path,s))
                ax.clear()
                im = Image.fromarray((flaten_image_3d.reshape(l,w,3) *255 ).astype(np.uint8))
                im.save("{}\\pixel_{}.png".format(result_path,s))
            # ims.append([sd_scatter])
    # flatten_image = content_dataset.cpu().detach().numpy().reshape([l*w, c])
    # ax.scatter(*flatten_image.T, s=10, alpha=0.1,c=[*flatten_image])
    # plt.show()
cluster(content_dataset.reshape([l*w, 5]))
