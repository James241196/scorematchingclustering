#%%
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.measure import label
from PIL import Image

mat = scipy.io.loadmat('E:\Dataset\BSD500\groundTruth\\train\\3063.mat')

ground_ruth = mat['groundTruth'][0][0][0][0][1]
plt.imshow(ground_ruth,cmap=plt.cm.gray)
plt.axis('off')
plt.show()

import math
from skimage import color

# 100080
# 302003
# 246053
# 323016
# 189011
# 181018
# 124084

ORIGINAL_IMAGE_PATH = 'E:\Dataset\edge\\1208\\3063.jpg'
SEG_IMAGE_PATH = 'E:\JyunYi\Score Matching\\result\論文素材\\3Dpixel normalize\Bliss\Bliss_pixel_6000.png'

rgb_i = np.asarray(Image.open(ORIGINAL_IMAGE_PATH))
h,w,c = rgb_i.shape
black_i = np.asarray(Image.open(SEG_IMAGE_PATH).convert('L'))
seg_i =np.asarray(Image.open(SEG_IMAGE_PATH))
black_i = (np.floor(black_i/10) * 10).astype(int)
black = np.zeros_like(black_i)
l = label(black_i,connectivity=1)

plt.imshow(rgb_i)
plt.axis('off')
plt.show()

plt.imshow(seg_i)
plt.axis('off')

plt.show()

plt.imshow(mark_boundaries(seg_i, l))
plt.axis('off')

plt.show()

seg = color.rgb2gray(mark_boundaries(black, l))
prediction = np.where(seg>0,1,0)
plt.imshow(prediction,cmap=plt.cm.gray)
plt.axis('off')

plt.show()







#%%
from sklearn.cluster import KMeans, MeanShift

image_5d = np.zeros([h,w,5])
for i in range(h):
    for j in range(w):
        image_5d[i,j] = np.append(rgb_i[i,j],[i/h,j/w])
kmeans = KMeans(n_clusters=6)
kmeans.fit(image_5d.reshape([h*w,5]))
centroid =kmeans.cluster_centers_
y_kmeans = kmeans.predict(image_5d.reshape([h*w,5]))
result = np.zeros([h*w,c])
for i in range(h*w):
    result[i] = centroid[y_kmeans[i]][:3]

result = result.reshape(h,w,c)

kmeans_l = label(color.rgb2gray(result),connectivity=1)
kmeans_seg = color.rgb2gray(mark_boundaries(black, kmeans_l))
kmenas_prdiction = np.where(kmeans_seg>0,1,0)
plt.imshow(kmeans_seg,cmap=plt.cm.gray)
plt.axis('off')

plt.show()

from sklearn.metrics import f1_score,accuracy_score
print("SM F1: {}".format(f1_score(ground_ruth, prediction, average='micro')))
print("kmeans F1: {}".format(f1_score(ground_ruth, kmenas_prdiction, average='micro')))

print("SM accuracy_score: {}".format(accuracy_score(ground_ruth, prediction)))
print("kmeans accuracy_score: {}".format(accuracy_score(ground_ruth, kmenas_prdiction)))

#%%

import numpy as np
import matplotlib.pyplot as plt

from skimage import filters
from skimage.data import camera
from skimage.util import compare_images
from sklearn.metrics import f1_score,accuracy_score


image = camera()
edge_roberts = filters.roberts(np.asarray(Image.open(ORIGINAL_IMAGE_PATH).convert('L')))
edge_sobel = filters.sobel(np.asarray(Image.open(ORIGINAL_IMAGE_PATH).convert('L')))


edge_roberts = np.where(edge_roberts>0.1,1,0)
edge_sobel = np.where(edge_sobel>0.15,1,0)

fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True,
                         figsize=(8, 4))

axes[0].imshow(edge_roberts, cmap=plt.cm.gray)
axes[0].set_title('Roberts Edge Detection')

axes[1].imshow(edge_sobel, cmap=plt.cm.gray)
axes[1].set_title('Sobel Edge Detection')

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()

print("Roberts accuracy_score: {}".format(f1_score(ground_ruth, edge_roberts,average='micro')))
print("Sobel accuracy_score: {}".format(f1_score(ground_ruth, edge_sobel,average='micro')))
