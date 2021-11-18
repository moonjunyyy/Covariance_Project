import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

basePath = "..\\Covariance_Project\\"
path = basePath + "cloud2D.txt"

cloudData = np.loadtxt(path)
cloudData = np.transpose(cloudData)

plt.scatter(cloudData[0],cloudData[1])

plt.show()