import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import Logistic_Regression as lr
import scipy.optimize as opt

# 数据可视化
path = 'D:\Study\Coding\Machine Learning WuEnda\homework\ex2\ex2data2.txt'
data_init = pd.read_csv(path, header=None, names=['Microchip Test 1', 'Microchip Test 2', 'Accepted'])

positive = data_init[data_init['Accepted'].isin([1])]
negative = data_init[data_init['Accepted'].isin([0])]
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['Microchip Test 1'], positive['Microchip Test 2'], alpha=0.5, s=50, c='b', marker='o', label='Accepted')
ax.scatter(negative['Microchip Test 1'], negative['Microchip Test 2'], alpha=0.5, s=50, c='r', marker='x', label='Rejected')
ax.legend()
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')
plt.show()

# 发现不适用于直线拟合，所以我们要创造更多的特征
data_init.insert(0, 'Ones', 1)
print(data_init.head())

loc_nums = 3
for i in range(6):    # 控制x1的幂
    for j in range(6):    # 控制x2的幂
        cols_name = 'F' + str(i+1) + str(j+1)
        X1 = data_init.loc[:, ['Microchip Test 1']]
        X2 = data_init.loc[:, ['Microchip Test 2']]
        F_ij = np.multiply(np.power(X1, i+1), np.power(X2, j+1))
        data_init.insert(loc=loc_nums, column=cols_name, value=F_ij)
        loc_nums += 1
        # print(data_init.head())

# 初始化数据
cols = data_init.shape[1]
X = data_init.iloc[:, :-1]
y = data_init.iloc[:, cols-1:cols]
# print(X.head(), '\n', y.head())

X = np.mat(X.values)
y = np.mat(y.values)
theta = np.mat(np.array([0.0509,  -0.1138,  0.1981,  -0.4741,  -0.2382,  0.0609,  -0.1437,  0.1020,  -0.0419,  -0.3915,  -1.1065,  -0.4496,  -0.6370,  -0.3913,  -0.4575,  0.1050,  -0.1164,  0.1048,  -0.0310,  0.0660,  0.0071,  -0.2492,  -0.5434,  -0.1493,  -0.2261,  -0.1088,  -0.1304,  0.0727,  -0.1306,  0.0462,  -0.0369,  0.0198,  -0.0085,  -0.1664,  -0.3108,  -0.0552,  -0.1013,  -0.0361,  -0.0482]))
# theta = np.mat(np.zeros(39))

alpha = 0.0001
iters = 1000000
lamda = 1


def cost_function_regularized(theta, X, y, lamda):    # correct √
    plus = (lamda / (2 * X.shape[0])) * np.sum(np.power(theta, 2))
    return lr.cost_function(X=X, y=y, theta=theta) + plus


print(cost_function_regularized(theta, X, y, lamda))    # correct √


def gradient_decent_regularized(theta, X, y, alpha, lamda, vistualize=True):
    temp = np.mat(np.zeros(theta.shape))    # temp暂存theta的值
    parameters = int(theta.shape[1])    # parameters控制循环次数
    cost = np.zeros(iters)    # cost用来存放每次迭代后的代价函数

    for i in range(iters):
        subtraction = lr.sigmoid(X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(subtraction, X[:, j])
            if not j:
                temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term, axis=0))    # len()返回矩阵的行数
            else:
                temp[0, j] = np.multiply(theta[0, j], 1 - alpha * lamda / len(X)) - ((alpha / len(X)) * np.sum(term, axis=0))  # len()返回矩阵的行数

        theta = temp
        cost[i] = cost_function_regularized(X, y, theta, lamda)

        if vistualize:
            print('这是第{}次循环, 代价函数的值为{:.4f}' .format(i+1, cost[i]), end=' ')
            for k in range(parameters):
                print(', Ѳ{:d}为{:.4f}' .format(k, theta[0, k]), end=' ')
            print('\n')
        for k in range(parameters):
            print('{:.4f}, '.format(theta[0, k]), end=' ')
        print('\n')
        if i != 0 and cost[i] > cost[i-1]:    # 检查梯度下降算法是否正常工作
            raise ValueError('The value of cost_function becomes larger')

    return cost

# print(lr.ensure_alpha(X, y, theta, alpha, iters))    # 确定学习速率
# print(gradient_decent_regularized(X, y, theta, lamda))


# theta_new, cost = gradient_decent_regularized(theta, X, y, alpha, lamda, vistualize=False)
# expected_result = lr.predict(theta_new, X)
expected_result = opt.fmin_tnc(func=cost_function_regularized, x0=theta, fprime=gradient_decent_regularized, args=(X, y, lamda))
print(expected_result)
lr.compare(expected_result, y)

