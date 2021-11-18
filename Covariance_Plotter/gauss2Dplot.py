import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

basePath = "..\\Covariance_Project\\"
path = basePath + "cloudGauss.txt"

cloudData = np.loadtxt(path)
cloudData = np.transpose(cloudData)

X = cloudData[0]
Y = cloudData[1]
Z = cloudData[2]

X = np.reshape(X,(2000,1000))
Y = np.reshape(Y,(2000,1000))
Z = np.reshape(Z,(2000,1000))

fig= plt.figure()
ax = plt.axes(projection='3d')

ax.plot_surface(X,Y,Z, cmap='viridis', edgecolor='none')
ax.set_title('Surface plot')

plt.show()
