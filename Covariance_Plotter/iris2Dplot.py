import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

basePath = "..\\Covariance_Project\\"
path = basePath + "iris2D.txt"

irisData = np.loadtxt(path)

iris1 = irisData[irisData[:,2] == 0]
iris2 = irisData[irisData[:,2] == 1]
iris3 = irisData[irisData[:,2] == 2]

iris1 = np.transpose(iris1)
iris2 = np.transpose(iris2)
iris3 = np.transpose(iris3)

plt.scatter(iris1[0],iris1[1])
plt.scatter(iris2[0],iris2[1])
plt.scatter(iris3[0],iris3[1])

plt.show()
