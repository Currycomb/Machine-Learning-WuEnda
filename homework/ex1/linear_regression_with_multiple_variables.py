import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linear_regression import gradient_descent
from linear_regression import compute_cost

path = 'D:\Study\Coding\Machine Learning WuEnda\homework\ex1\ex1data2.txt'
data = pd.read_csv(path, header=None, names=['Size', 'Bedroom_nums', 'Price'])
# print(data.head())

# 归一化处理
# data.mean()处理平均值,data.std() (short for 'standard')处理标准偏差(max-min)
data = (data - data.mean()) / data.std()
data.insert(0, 'Ones', 1)
print(data.head())

# 初始化x,y
cols = data.shape[1]    # 返回一个元组(行, 列)
x = data.iloc[:, :cols-1]
y = data.iloc[:, cols-1:cols]

# 转换成矩阵
x = np.mat(x.values)
y = np.mat(y.values)
theta = np.mat(np.array([0, 0, 0]))

alpha = 0.01
iters = 1500
g2, cost = gradient_descent(x, y, theta, alpha, iters)
print(g2)
