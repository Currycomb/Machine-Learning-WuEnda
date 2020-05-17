import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.metrics import classification_report  # 这个是评价报告


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost_function(theta, X, y, learning_rate):
    theta = np.mat(theta)
    X = np.mat(X)
    y = np.mat(y)
    h_theta = sigmoid(X * theta.T)

    first = np.multiply(-y, np.log(h_theta))
    second = np.multiply(-(1 - y), np.log(1 - h_theta))

    reg = (learning_rate / (2 * len(X))) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))    # 注意这里不包含θ0
    return np.sum(first + second) / len(X) + reg


def gradient(theta, X, y, learning_rate):
    theta = np.mat(theta)
    X = np.mat(X)
    y = np.mat(y)

    parameters = int(theta.ravel().shape[1])
    error = sigmoid(X @ theta.T) - y

    grad = ((X.T @ error) / len(X)).T + ((learning_rate / len(X)) * theta)

    # intercept gradient is not regularized
    grad[0, 0] = np.sum(np.multiply(error, X[:, 0])) / len(X)

    return np.array(grad).ravel()


def one_vs_all(X, y, num_labels, learning_rate):
    rows = X.shape[0]
    params = X.shape[1]

    # k * (n + 1) array for the parameters of each of the k classifiers
    all_theta = np.zeros((num_labels, params + 1))

    # insert a column of ones at the beginning for the intercept term
    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    # labels are 1-indexed instead of 0-indexed
    for i in range(1, num_labels + 1):
        theta = np.zeros(params + 1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))

        # minimize the objective function
        fmin = minimize(fun=cost_function, x0=theta, args=(X, y_i, learning_rate), method='TNC', jac=gradient)
        all_theta[i - 1, :] = fmin.x

    return all_theta


def predict_all(X, all_theta):
    rows = X.shape[0]
    params = X.shape[1]
    num_labels = all_theta.shape[0]

    # same as before, insert ones to match the shape
    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    # convert to matrices
    X = np.mat(X)
    all_theta = np.mat(all_theta)

    # compute the class probability for each class on each training instance
    h = sigmoid(X @ all_theta.T)

    # create array of the index with the maximum probability
    h_argmax = np.argmax(h, axis=1)

    # because our array was zero-indexed we need to add one for the true label prediction
    h_argmax = h_argmax + 1

    return h_argmax



# 读取数据
path = 'D:\Study\Coding\Machine Learning WuEnda\homework\ex3\ex3data1.mat'
data = loadmat(path)
# print(data['X'], '\n', data['y'])
# print(data['X'].shape, data['y'].shape)

# 数据可视化
sample_index = np.random.choice(np.arange(data['X'].shape[0]), 100)  # 从5000行中选取100行作为样本
sample_images = data['X'][sample_index, :]
# print(sample_images)

fig, ax_array = plt.subplots(nrows=10, ncols=10, sharex='all', sharey='all', figsize=(12, 12))
for r in range(10):
    for c in range(10):
        ax_array[r, c].matshow(np.array(sample_images[10 * r + c].reshape(20, 20)).T, cmap=matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
plt.show()

rows = data['X'].shape[0]
params = data['X'].shape[1]

all_theta = np.zeros((10, params + 1))

X = np.insert(data['X'], 0, values=np.ones(rows), axis=1)

theta = np.zeros(params + 1)

y_0 = np.array([1 if label == 0 else 0 for label in data['y']])
y_0 = np.reshape(y_0, (rows, 1))

# print(X.shape, y_0.shape, theta.shape, all_theta.shape)
# 去除数组中的重复数字，并进行排序之后输出
# print(np.unique(data['y']))

# 训练开始
# all_theta = one_vs_all(data['X'], data['y'], 10, 0.1)
# print(all_theta)

# y_pred = predict_all(data['X'], all_theta)
# print(classification_report(data['y'], y_pred))


# 实现前馈神经网络预测
weight = loadmat('D:\Study\Coding\Machine Learning WuEnda\homework\ex3\ex3weights.mat')
theta1, theta2 = weight['Theta1'], weight['Theta2']
# print(theta1.shape, theta2.shape)

X2 = np.mat(np.insert(data['X'], 0, values=np.ones(X.shape[0]), axis=1))
y2 = np.mat(data['y'])

a1 = X2
z2 = a1 * theta1.T
# print(z2.shape)

a2 = sigmoid(z2)
# print(a2.shape)

a2 = np.insert(a2, 0, values=np.ones(a2.shape[0]), axis=1)
z3 = a2 * theta2.T
# print(z3.shape)

a3 = sigmoid(z3)
# print(a3)

y_pred2 = np.argmax(a3, axis=1) + 1    # 取出a3中元素最大值所对应的索引 (axis=0是行，1是列)
# print(y_pred2.shape)

print(classification_report(y2, y_pred2))

