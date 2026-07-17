#%%
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from score_matching import train, sm_cluster_predict
from datetime import datetime
from torchsummary import summary

from pathlib import Path


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



today = datetime.today().strftime('%m%d')
path = "C:\Score matching"
type="jpg"
style_name = 'fig10'

# path = "E:\Dataset"
# type="jpg"
# style_name = 'mountain'

# path = "E:\Dataset\\tmp"
# type="JPEG"
# style_name = 'ILSVRC2012_val_00017396'

style_image = np.asarray(Image.open('{}\{}.{}'.format(path, style_name,type))) / 255
# style_image = np.asarray(Image.open("E:\Dataset\M10909121 (1).jpg")) / 255
result_path = 'C:\Score matching\\results\{}\{}'.format(today, style_name)

l, w, c = style_image.shape

style_dataset = torch.tensor(style_image).float().to('cuda:0')
style_dataset = style_dataset.reshape([l*w,3])

content_name = 'fig10'
content_image = np.asarray(Image.open('{}\{}.{}'.format(path, content_name,type))) 
# content_image = np.asarray(Image.open("E:\Dataset\M10909121 (1).jpg")) 
content_dataset = torch.tensor(content_image).float().to('cuda:0') / 255
l, w, c = content_image.shape

#%%

model = nn.Sequential(
    nn.Linear(3, 200), nn.GELU(),
    nn.Linear(200, 200), nn.GELU(),
     nn.Linear(200 , 3)
).to('cuda:0')


train(model, style_dataset,epoch=3000, method='sm')

#%%
Path(result_path).mkdir(parents=True, exist_ok=True)
step_size = 4e-4
'''for grad nomarlize to one'''
# step_size = 1e-4


def cluster(content_dataset):
    ims = []
    # Khởi tạo figure một lần duy nhất để tiết kiệm tài nguyên
    fig = plt.figure(figsize=(20, 20))
    
    # Thiết lập số bước chạy
    total_steps = 4200
    print_every = 600

    for s in range(total_steps + 1):  # +1 để bao gồm cả bước 4200
        score = model(content_dataset)
        
        # Tránh lỗi chia cho 0 khi tính norm
        score_norm = torch.norm(score, dim=-1).reshape([l*w, 1]).expand(-1, c)
        score_norm = torch.where(score_norm == 0, torch.tensor(1e-9).to('cuda:0'), score_norm)
        normalize_grad = score / score_norm

        with torch.no_grad():
            content_dataset = content_dataset + step_size * normalize_grad
            # Giữ giá trị pixel trong khoảng [0, 1]
            content_dataset = torch.clamp(content_dataset, 0.0, 1.0)

            # Chỉ thực hiện vẽ và lưu file tại các mốc 0, 600, 1200, ...
            if s % print_every == 0:
                print(f"Processing step: {s}")
                
                # 1. Vẽ đồ thị 3D trong không gian màu RGB
                ax = fig.add_subplot(projection='3d')
                ax.set_xlim3d(0.0, 1.0)
                ax.set_ylim3d(0.0, 1.0)
                ax.set_zlim3d(0.0, 1.0)
                ax.set_xlabel('Red')
                ax.set_ylabel('Green')
                ax.set_zlabel('Blue')
                
                flatten_image = content_dataset.cpu().detach().numpy()
                # Vẽ scatter plot với màu sắc thực của pixel
                ax.scatter(*flatten_image.T, s=120, alpha=0.5, c=flatten_image)
                
                plt.savefig(f"{result_path}\\{content_name}_3D_step_{s}.png")
                ax.clear() # Xóa nội dung cũ để vẽ bước tiếp theo
                plt.close() # Đóng figure hiện tại để giải phóng RAM

                # 2. Lưu ảnh kết quả (Pixel Visualization)
                # Chuyển tensor về dạng [Height, Width, Channels] và nhân 255
                output_img = content_dataset.reshape([l, w, c]).cpu().detach().numpy()
                im = Image.fromarray((output_img * 255).astype(np.uint8))
                im.save(f"{result_path}\\{content_name}_pixel_step_{s}.png")
                
                # Khởi tạo lại figure cho lần lặp sau (tránh lỗi ax.clear)
                fig = plt.figure(figsize=(20, 20))

    return content_dataset

# Gọi hàm với dataset đã reshape
cluster(content_dataset.reshape([l*w, c]))
#%%

# now = datetime.now() # current date and time
# date_time = now.strftime("%Y_%m_%d %H_%M_%S")
# ani = animation.ArtistAnimation(fig, ims, interval=60, repeat_delay =1, blit=True)
# ani.save("E:\JyunYi\Score Matching\\result\color transfer\{}_{}_{}.gif".format(content_name, style_name, date_time),writer='pillow')

# plt.imshow(content_dataset.cpu().detach().numpy(), animated=True)


#%%
# x = content_dataset.reshape([w*l, 3])
# y = sm_cluster_predict(x, eps = 0.01)
# print(max(y))
