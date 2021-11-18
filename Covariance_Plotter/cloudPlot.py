import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

basePath = "..\\Covariance_Project\\"
path = basePath + "planeCloud.txt"

cloudData = np.loadtxt(path)
cloudData = np.transpose(cloudData)

path = basePath + "CloudEigenValue.txt"
cloudEV = np.loadtxt(path)
cloudEV = np.transpose(cloudEV)

path = basePath + "CloudEigenVectors.txt"
cloudEigen = np.loadtxt(path)
cloudEigen = np.transpose(cloudEigen)

path = basePath + "CloudMeans.txt"
cloudMean = np.loadtxt(path)

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(cloudData[0], cloudData[1], cloudData[2], marker="o")

ax.quiver(
    cloudMean[0], cloudMean[1], cloudMean[2], 
    cloudEV[0]/15*cloudEigen[0][0], 
    cloudEV[0]/15*cloudEigen[0][1], 
    cloudEV[0]/15*cloudEigen[0][2], color="red") 
ax.quiver(
    cloudMean[0], cloudMean[1], cloudMean[2], 
    cloudEV[1]/15*cloudEigen[1][0], 
    cloudEV[1]/15*cloudEigen[1][1], 
    cloudEV[1]/15*cloudEigen[1][2], color="red") 
ax.quiver(
    cloudMean[0], cloudMean[1], cloudMean[2], 
    cloudEV[2]/15*cloudEigen[2][0], 
    cloudEV[2]/15*cloudEigen[2][1], 
    cloudEV[2]/15*cloudEigen[2][2], color="red") 

plt.show()

