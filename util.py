#%% GIF
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
import matplotlib.animation as animation
'''5000張 delay300'''
'''10000張 delay150??'''

type = "3D"
path = "E:\JyunYi\Score Matching\\result\論文素材\pixel unnormalize\色彩轉換\\tmp\*"
# path ='E:\JyunYi\Score Matching\\result\mse_vgg_test\*'
for folder in glob.glob(path):
    fig = plt.figure(figsize=(20, 20))
    ims = []
    for filename in sorted(glob.glob('{}\*{}*'.format(folder, type)), key=os.path.getmtime):
        i=Image.open(filename)
        plt.xticks(())
        plt.yticks(())
        im = plt.imshow(i)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=300, repeat_delay =3000, blit=True)
    ani.save("{}\{}.gif".format(folder, type),writer='pillow')
#%% GIF
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
from skimage.segmentation import mark_boundaries
from skimage.measure import label
from skimage import color
import numpy as np

path = "E:\JyunYi\Score Matching\\result\\5D_pixel\\1208\*"
for folder in glob.glob(path):
    for filename in sorted(glob.glob('{}\pixel*20000*'.format(folder)), key=os.path.getmtime):
        black_i = np.asarray(Image.open(filename).convert('L'))
        seg_i =np.asarray(Image.open(filename))
        black_i = (np.floor(black_i/10) * 10).astype(int)
        black = np.zeros_like(black_i)
        l = label(black_i,connectivity=1)
        seg = color.rgb2gray(mark_boundaries(black, l))
        prediction = np.where(seg>0,1,0)
        plt.imshow(prediction,cmap=plt.cm.gray)
        plt.axis('off')

        plt.savefig("{}\\seg.png".format(folder))
        # im = Image.fromarray(prediction * 255)
        # im.save()

#%% SSIM MSE PSNR
from skimage.metrics import structural_similarity as ssim, mean_squared_error,peak_signal_noise_ratio
import numpy as np
from PIL import Image


original_image = np.asarray(Image.open('E:\Dataset\\tmp\\00000002_(4).jpg')) / 255
segment_image = np.asarray(Image.open('E:\JyunYi\Score Matching\\result\論文素材\\5D_pixel\\00000002_(4).jpg\pixel_10000.png')) / 255
ssim_score = ssim(original_image, segment_image,multichannel=True)
mse = mean_squared_error(original_image, segment_image)
# me = np.mean(np.abs(original_image-segment_image))
psnr = peak_signal_noise_ratio(original_image, segment_image)

print('ssim: {}'.format(ssim_score))
print('mean_squared_error: {}'.format(mse))
print('peak_signal_noise_ratio: {}'.format(psnr))
#%% image to 3d 
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
import matplotlib.animation as animation
import numpy as np

type = "pixel"
path ='E:\JyunYi\Score Matching\\result\論文素材\\3Dpixel normalize\\flower'
fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(projection='3d')

count=0
for filename in sorted(glob.glob('{}\*{}*.png'.format(path, type)), key=os.path.getmtime):
    i = np.asarray(Image.open(filename)) /255
    [l,w,c] = i.shape
    ax.set_xlim3d(0.0, 1.0)
    ax.set_ylim3d(0.0, 1.0)
    ax.set_zlim3d(0.0, 1.0)
    flaten_image_3d = i.reshape([l*w, c])

    ax.scatter(*flaten_image_3d.T, s=1200, alpha=0.7,c=[*flaten_image_3d],edgecolors='black',linewidths=0.1)
    plt.savefig("{}\\3D_{}.png".format(path,count))
    ax.clear()
    count+=100
   