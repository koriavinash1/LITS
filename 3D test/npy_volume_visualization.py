import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

d = np.load("../../DataSet/NewDataSet/train/volume-103_size_32.npy")

# z,x,y = d.nonzero()

# ax.scatter(x, y, -z, zdir='z', c= 'red')
# plt.show("demo.png")
for img in d:
	plt.imshow(d[0])
plt.			show()