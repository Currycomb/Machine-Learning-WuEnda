import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.io import loadmat
from sklearn.metrics import classification_report  # 这个是评价报告


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost_function(theta, X, y, learning_rate):
    theta = np.mat(theta)
    X = np.mat(X)
    y = np.mat(y)
    h_theta = sigmoid(X * theta.T)
    first = -y * np.log(h_theta)
    second = -(1 - y) * np.log(1 - h_theta)
    reg = (learning_rate / (2 * len(X))) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))    # 注意这里不包含θ0
    return np.sum(first + second) / len(X) + reg


def  



# 读取数据
path = 'D:\Study\Coding\Machine Learning WuEnda\homework\ex3\ex3data1.mat'
data = loadmat(path)
# print(data['X'].shape, data['y'].shape)

# 数据可视化
sample_index = np.random.choice(np.arange(data['X'].shape[0]), 100)  # 从5000行中选取100行作为样本
sample_images = data['X'][sample_index, :]
# print(sample_image)

fig, ax_array = plt.subplots(nrows=10, ncols=10, sharex='all', sharey='all', figsize=(12, 12))
for r in range(10):
    for c in range(10):
        ax_array[r, c].matshow(np.array(sample_images[10 * r + c].reshape(20, 20)).T, cmap=matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
plt.show()
