import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.io import loadmat
from scipy.optimize import minimize

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report    # 这个包是评价报告

from tqdm import tqdm


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_gradient(z):
    g_z = sigmoid(z)
    return np.multiply(g_z, 1-g_z)

def forward_propagate(X, theta1, theta2):    # only one hidden unit
    m = X.shape[0]
    a1 = np.insert(X, 0, values=np.ones(m), axis=1)
    z2 = np.dot(a1, theta1.T)
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)
    z3 = np.dot(a2, theta2.T)
    h = sigmoid(z3)

    return a1, z2, a2, z3, h

def cost_function(X, y, theta1, theta2, learning_rate=1, reg=True):
    h = forward_propagate(X, theta1, theta2)[-1]
    m = X.shape[0]

    part_one = np.multiply(y, np.log(h)) + np.multiply((1-y), np.log(1-h))
    all_one = - np.sum(part_one) / m

    if reg:    # 判断是否需要正则化
        part_two = np.power(theta1[:, 1:], 2).sum() + np.power(theta2[:, 1:], 2).sum()    # 正则化项
        all_two = part_two * learning_rate / (2 * m)
        J = all_one + all_two
    else:
        J = all_one

    return J

def back_propagation(params, input_size, hidden_size, num_labels, X, y, learning_rate=1, reg=True):
    m = X.shape[0]

    # reshape the parameter array into parameter matrices for each layer
    # 将输入的params分成theta11,2
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    # initializations
    J = 0
    delta1 = np.zeros(theta1.shape)  # (25, 401)
    delta2 = np.zeros(theta2.shape)  # (10, 26)

    # compute the cost
    for i in range(m):
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        J += np.sum(first_term - second_term)

    J = J / m

    # add the cost regularization term
    if reg:
        J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))

    # perform backpropagation
    for t in tqdm(range(m)):
        a1t = a1[t, :]  # (1, 401)
        z2t = z2[t, :]  # (1, 25)
        a2t = a2[t, :]  # (1, 26)
        ht = h[t, :]  # (1, 10)
        yt = y[t, :]  # (1, 10)

        d3t = ht - yt  # (1, 10)

        z2t = np.insert(z2t, 0, values=np.ones(1))  # (1, 26)
        d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2t))  # (1, 26)

        delta1 = delta1 + (d2t[:, 1:]).T * a1t
        delta2 = delta2 + d3t.T * a2t

    delta1 = delta1 / m
    delta2 = delta2 / m

    # add the gradient regularization term
    # 除去bias项都加上正则化
    delta1[:, 1:] = delta1[:, 1:] + (theta1[:, 1:] * learning_rate) / m
    delta2[:, 1:] = delta2[:, 1:] + (theta2[:, 1:] * learning_rate) / m

    # unravel the gradient matrices into a single array
    grad = np.hstack((np.ravel(delta1), np.ravel(delta2)))

    # just a show


    return J, grad



def gradient_check(X, y, theta1, theta2, e=0.001):
    print('Start checking gradient: ')
    theta_all = np.hstack((np.ravel(theta1), np.ravel(theta2))).T.reshape(-1, 1)    # (10285, 1)
    f_theta = np.zeros(theta_all.shape).reshape(-1, 1)

    for i in tqdm(range(theta_all.shape[0])):
        e_add = np.zeros(theta_all.shape).reshape(-1, 1)
        e_add[i, 0] = e
        theta_minus = theta_all - e_add
        theta_plus = theta_all + e_add

        theta_minus_1 = theta_minus[:theta1.shape[0] * theta1.shape[1], 0].reshape(theta1.shape)
        theta_minus_2 = theta_minus[theta1.shape[0] * theta1.shape[1]:, 0].reshape(theta2.shape)
        theta_plus_1 = theta_plus[:theta1.shape[0] * theta1.shape[1], 0].reshape(theta1.shape)
        theta_plus_2 = theta_plus[theta1.shape[0] * theta1.shape[1]:, 0].reshape(theta2.shape)

        f_theta[i, 0] = (cost_function(X, y, theta_minus_1, theta_minus_2) - cost_function(X, y, theta_plus_1, theta_plus_2)) / (2 * e)

    return f_theta

# 读取数据
data = loadmat('D:/Study/MachineLearning/Machine Learning WuEnda/homework/ex4/ex4data1.mat')

X = data['X']
y = data['y']
print(X.shape, y.shape)

encoder = OneHotEncoder(sparse=False)    # 对y进行独热编码
y_onehot = encoder.fit_transform(y)
print(y_onehot.shape)

weight = loadmat('D:/Study/MachineLearning/Machine Learning WuEnda/homework/ex4/ex4weights.mat')
theta1, theta2 = weight['Theta1'], weight['Theta2']
print(theta1.shape, theta2.shape)

# 数据可视化
sample_index = np.random.choice(np.arange(data['X'].shape[0]), 100, replace=False)  # 从5000行中选取100行作为样本,replace=False表示不能取相同的数字
sample_images = data['X'][sample_index, :]    # sample_index行,所有列
# print(sample_images.shape)

fig, ax_array = plt.subplots(nrows=10, ncols=10, sharex='all', sharey='all', figsize=(12, 12))
for r in range(10):
    for c in range(10):
        # matshow是按色块画图
        ax_array[r, c].matshow(np.array(sample_images[10 * r + c].reshape(20, 20)).T, cmap=matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
plt.show()


# 对theta做随机初始化
e1 = 0.12
e2 = 0.414

# 初始化设置
input_size = 400
hidden_size = 25
num_labels = 10
learning_rate = 1

cost = cost_function(X, y_onehot, theta1, theta2, learning_rate=1)
# print(cost)

# np.random.random(size) 返回size大小的0-1随机浮点数, 初始化theta
params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.24
print(params.shape)    # (10285,)

# print(back_propagation(params, input_size, hidden_size, num_labels, X, y_onehot, learning_rate=1))

# 使用工具库计算参数最优解
# minimize the objective function
'''
 minimize里面参数解释:
 fun : 要计算的函数
 x0 : 函数里面传入的第一个参数,通常是自变量或者要优化的值的初始值
 args : 函数里面传入的其他的参数，要用tuple打包
 method : 优化方法
 jac : 
 options : 其他设置
'''
fmin = minimize(fun=back_propagation, x0=params, args=(input_size, hidden_size, num_labels, X, y_onehot, learning_rate),
                method='TNC', jac=True, options={'maxiter': 250})
# print(fmin)

X = np.matrix(X)
theta1_final = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2_final = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))


# 检测back propagation是否正常运行，记得检查一次后关闭gradient_check(),这里只计算第一次训练(套用5000个数据)的theta
# params_check = np.hstack((theta1.ravel(), theta2.ravel()))
# print(params_check.shape)
# gradient_check = gradient_check(X, y_onehot, theta1, theta2)
# gradient_train = back_propagation(params_check, input_size, hidden_size, num_labels, X, y_onehot)[1]
# 求誤差
# error = np.fabs(gradient_check - gradient_train).mean()
# print('-------------------------------------------')
# print(error)
# print('-------------------------------------------')


# 计算使用优化后的θ得出的预测
a1, z2, a2, z3, h = forward_propagate(X, theta1_final, theta2_final)
y_pred = np.array(np.argmax(h, axis=1) + 1)
print(y_pred)

# 预测值与实际值比较
print(classification_report(y, y_pred))    # 完美 √

# 可视化hidden layer
hidden_layer = theta1_final[:, 1:]
fig, ax_array = plt.subplots(nrows=5, ncols=5, sharey='all', sharex='all', figsize=(12, 12))
for r in range(5):
    for c in range(5):
        ax_array[r, c].matshow(np.array(hidden_layer[5 * r + c].reshape((20, 20))), cmap=matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
plt.show()


