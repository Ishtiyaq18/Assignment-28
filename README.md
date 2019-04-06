# Assignment-28


In this assignment students have to compress racoon grey scale image into 5 clusters. In the end, visualize both raw and compressed image and look for quality difference. The raw image is available in spicy.misc package with the name face. Hint: import numpy as np from sklearn import cluster, datasets from scipy import misc
In [1]:
import numpy as np
from scipy import misc
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

face = misc.face(gray=True)
In [2]:
n_clusters = 5
np.random.seed(0)

X = face.reshape((-1, 1))
k_means = KMeans(n_clusters=n_clusters, n_init=4)
k_means.fit(X)
values = k_means.cluster_centers_.squeeze()
labels = k_means.labels_

# create an array from labels and values
face_compressed = np.choose(labels, values)
face_compressed.shape = face.shape

vmin = face.min()
vmax = face.max()

# original face
plt.figure(1)
plt.imshow(face, cmap=plt.cm.gray, vmin=vmin, vmax=256)

# compressed face
plt.figure(2)
plt.imshow(face_compressed, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
Out[2]:
<matplotlib.image.AxesImage at 0x2164c7d3390>

 

 
