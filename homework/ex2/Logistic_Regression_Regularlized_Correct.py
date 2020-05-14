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
# plt.show()

# 发现不适用于直线拟合，所以我们要创造更多的特征
data_init.insert(0, 'Ones', 1)
print(data_init.head())

degree = 20
loc_nums = 3
for i in range(degree):    # 控制x1的幂
    for j in range(degree):    # 控制x2的幂
        cols_name = 'F' + '('+str(i+1) + ')'+'('+str(j+1)+')'
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
print(X.head(), '\n', y.head())

X = np.mat(X.values)
y = np.mat(y.values)
theta = np.zeros(cols-1)

learningRate = 1.5


def cost_function_regularized(theta, X, y, learningRate):    # correct √
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(lr.sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - lr.sigmoid(X * theta.T)))
    reg = (learningRate / (2 * len(X))) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
    return np.sum(first - second) / len(X) + reg


print(cost_function_regularized(theta, X, y, learningRate))    # correct √


def gradient_decent_regularized(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = lr.sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:, i])

        if (i == 0):
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = (np.sum(term) / len(X)) + ((learningRate / len(X)) * theta[:, i])

    return grad


def predict(theta, X):
    probability = lr.sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]


expected_result = opt.fmin_tnc(func=cost_function_regularized, x0=theta, fprime=gradient_decent_regularized, args=(X, y, learningRate))
print(expected_result)

theta_min = np.matrix(expected_result[0])
predictions = predict(theta_min, X)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print('accuracy = {0}%'.format(accuracy))



def hfunc2(theta, x1, x2):
    temp = theta[0][0]
    place = 0
    for i in range(1, degree+1):
        for j in range(0, i+1):
            temp+= np.power(x1, i-j) * np.power(x2, j) * theta[0][place+1]
            place+=1
    return temp


def find_decision_boundary(theta):
    t1 = np.linspace(-1, 1.5, 1000)
    t2 = np.linspace(-1, 1.5, 1000)

    cordinates = [(x, y) for x in t1 for y in t2]
    x_cord, y_cord = zip(*cordinates)
    h_val = pd.DataFrame({'x1':x_cord, 'x2':y_cord})
    h_val['hval'] = hfunc2(theta, h_val['x1'], h_val['x2'])

    decision = h_val[np.abs(h_val['hval']) < 2 * 10**-3]
    return decision.x1, decision.x2


fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['Microchip Test 1'], positive['Microchip Test 2'], s=50, c='b', marker='o', label='Accepted')
ax.scatter(negative['Microchip Test 1'], negative['Microchip Test 2'], s=50, c='r', marker='x', label='Rejected')
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')

x, y = find_decision_boundary(expected_result)
plt.scatter(x, y, c='y', s=10, label='Prediction')
ax.legend()
plt.show()
