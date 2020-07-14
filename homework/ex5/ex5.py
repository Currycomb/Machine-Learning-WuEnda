import numpy as np
import scipy.io as sio
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = sio.loadmat('D:\Study\MachineLearning\Machine Learning WuEnda\homework\ex5\ex5data1.mat')
# map() 会根据提供的函数对指定序列做映射
X, y, X_val, y_val, X_test, y_test = map(lambda x: np.reshape(x, (-1, 1)), [data['X'], data['y'], data['Xval'], data['yval'], data['Xtest'], data['ytest']])
print(X.shape, '\n', y.shape)    # (12, 1)     (12, 1)
print(type(X))
print(y)

plt.figure(figsize=(12, 8))
ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=1, rowspan=1)
ax1.scatter(X, y, label='Training data', alpha=0.7)
ax1.set_xlabel('water_level')
ax1.set_ylabel('flow')

def gradient(theta, X, y, reg):
    theta = np.mat(theta)
    m = len(X)
    inner = np.dot(X, theta.T) - y
    grad = np.zeros(theta.shape)

    for j in range(theta.shape[1]):
        grad_first = np.sum(np.multiply(inner, X[:, [j]])) / m
        if j == 0:
            grad[0, j] = grad_first
        else:
            grad_reg = (reg * theta[0, j]) / m
            grad[0, j] = grad_first + grad_reg

    return grad

def linear_regression(X, y, l=1.0):
    """linear regression
    args:
        X: feature matrix, (m, n+1) # with intercept x0=1
        y: target vector, (m, )
        l: lambda constant for regularization

    return: trained parameters
    """
    # init theta
    _theta = np.ones(X.shape[1])

    # train it
    result = opt.minimize(fun=cost_function,    # target minimize function
                          x0=_theta,
                          args=(X, y, l),
                          method='TNC',
                          jac=gradient,    # a function that calculate the gradient
                          options={'disp': True})
    return result

X, X_val, X_test = [np.insert(x, 0, np.ones(x.shape[0]), axis=1) for x in (X, X_val, X_test)]

# 数据处理
def cost_function(theta, X, y, reg=0):
    X = np.mat(X)
    y = np.mat(y)
    theta = np.mat(theta)
    m = len(X)

    h_theta = np.dot(X, theta.T)    # (m, 2) @ (2, 1) -> (m, 1)
    J_first = np.sum(np.power(h_theta - y, 2)) / (2 * m)
    J_second = reg * np.sum(np.power(theta, 2)) / (2 * m)

    J = J_first + J_second
    return J

theta = np.ones(X.shape[1])
theta_final = opt.minimize(fun=cost_function, x0=theta, args=(X, y, 0), method='TNC', jac=gradient, options={'disp': True}).x
print(theta_final)
print(cost_function(theta_final, X, y, reg=0))
# print(gradient(theta_final, X, y, reg=1))

theta_0 = theta_final[0]
theta_1 = theta_final[1]
x_plot1 = np.linspace(-50, 50)
y_plot1 = theta_0 + theta_1 * x_plot1
ax1.plot(x_plot1, y_plot1, color='red', label='Prediction')
ax1.legend()


# plot the learning curves
def cv_train_cost_compute(X, y, X_val, y_val, l=1.0):
    training_cost, cv_cost = [], []
    m = X.shape[0]
    # 用循环记录下不同training dataset下的cost，然后append到列表里，画出learning curve
    for i in range(1, m+1):
        # 用X, y训练模型, 然后分别算出训练集和交叉验证集上的J(θ)
        model = linear_regression(X[:i, :], y[:i, :], l)

        train_error = cost_function(model.x, X[:i, :], y[:i, :])
        cv_error = cost_function(model.x, X_val, y_val)
        training_cost.append(train_error)
        cv_cost.append(cv_error)

    return training_cost, cv_cost


training_cost, cv_cost = cv_train_cost_compute(X, y, X_val, y_val)
ax2 = plt.subplot2grid((2, 3), (0, 1), colspan=1, rowspan=1)
ax2.plot(np.arange(1, len(X)+1), training_cost, label='training cost')
ax2.plot(np.arange(1, len(X)+1), cv_cost, label='cross validation cost')
ax2.legend()
ax2.set_xlabel('numbers of training data')
ax2.set_ylabel('J(θ)')

# 此时发现模型underfit, 于是增加多项式特征
def add_polynomial(X, num):
    """add polynomial features to X
    args:
        X: a matrix that to be add  (*, 2) (already add cols x0=1)
        num: number of polynomial
        normal: whether to do normalization
    return: a new X
    """
    orgin = X[:, 1]
    for i in range(num):
        add_col = np.power(orgin, i+2)
        X = np.insert(X, obj=X.shape[1], values=add_col, axis=1)

    return X

X_new, X_val, X_test = map(add_polynomial, [X, X_val, X_test], [5, 5, 5])    ####

# 将X扩展到8阶并使用归一化
def normalize_feature(X, method='Z'):
    """to normalize feature
    :param X: type() --> np.ndarry, X already have a col, namely x0 = 1
    :param method: how to normalize, include 'Z'(Z-score), '01'((0,1)normalize), 'sigmoid'
    :return: a new X
    """
    for i in range(1, X.shape[1]):
        each_col = X[:, i]
        if method == 'Z':    # (X - mu) / sigma
            each_col = (each_col - np.average(each_col)) / np.std(each_col)

        elif method == '01':    # (X - max) / (max - min)
            each_col = (each_col - np.max(each_col)) / (np.max(each_col) - np.min(each_col))

        elif method == 'sigmoid':    # 1 / (1 + e^(-x))
            each_col = 1 / (1 + np.exp(-each_col))

        X[:, i] = each_col

    return X

X_new, X_val, X_test = map(normalize_feature, [X_new, X_val, X_test])    # (12, 10)
# print(X_new)

# 下一步，画出learning curve，判断是underfit还是overfit
poly_training_cost, poly_cv_cost = cv_train_cost_compute(X_new, y, X_val, y_val, l=0.55)   ####
ax3 = plt.subplot2grid((2, 3), (1, 1), colspan=1, rowspan=1)
ax3.plot(np.arange(1, len(X_new)+1), poly_training_cost, label='training cost')
ax3.plot(np.arange(1, len(X_new)+1), poly_cv_cost, label='cross validation cost')
ax3.set_xlabel('numbers of training data')
ax3.set_ylabel('J(θ)')
ax3.legend()

# 训练θ
theta_new = linear_regression(X_new, y, l=0.55).x.reshape(-1, 1)    ####
print(theta_new)

ax4 = plt.subplot2grid((2, 3), (1, 0), colspan=1, rowspan=1)
x_temp = np.insert(np.linspace(-50, 50).reshape(-1, 1), 0, np.ones(np.linspace(-50, 50).reshape(-1, 1).shape[0]), axis=1)
x_temp = normalize_feature(add_polynomial(x_temp, 5))    ####
y_plot2 = np.dot(x_temp, theta_new)

ax4.plot(np.linspace(-50, 50), y_plot2, color='red', label='Prediction')
ax4.scatter(np.delete(X, 0, axis=1), y, label='Training data', alpha=0.7)
ax4.set_xlabel('water_level')
ax4.set_ylabel('flow')
ax4.legend()

# 选取最适合的λ
l_candidate = [0, 0.01, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.8]
training_cost, cv_cost = [], []
for l in l_candidate:
    theta_plot = linear_regression(X_new, y, l).x
    training_error = cost_function(theta_plot, X_new, y, reg=l)
    cv_error = cost_function(theta_plot, X_val, y_val, reg=l)

    training_cost.append(training_error)
    cv_cost.append(cv_error)

ax5 = plt.subplot2grid((2, 3), (0, 2), rowspan=1, colspan=1)
ax5.plot(l_candidate, training_cost, label='training cost')
ax5.plot(l_candidate, cv_cost, label='cross validation cost')
ax5.set_xlabel('λ')
ax5.set_ylabel('J(θ)')
ax5.legend()
plt.show()

# 在测试集上进行测试
# use test data to compute the cost
for l in l_candidate:
    theta = linear_regression(X_new, y, l).x
    print('test cost(l={}) = {}'.format(l, cost_function(theta, X_test, y_test)))

# 计算测试集上的准确率
y_pred = np.dot(X_test, theta_new)
mse = np.sum(np.square(y_test - y_pred)) / len(y_pred)
print("MSE为: ", mse)

'''
最终参数：
        λ=0.55
        增添到x^6系数项
        最终MSE: 14.86687
'''

