import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt


# define sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def gradient_descent(X, y, theta, alpha, iters, vistualize=True):
    temp = np.mat(np.zeros(theta.shape))    # temp暂存theta的值
    parameters = int(theta.shape[1])    # parameters控制循环次数
    cost = np.zeros(iters)    # cost用来存放每次迭代后的代价函数

    for i in range(iters):
        subtraction = sigmoid(X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(subtraction, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term, axis=0))    # len()返回矩阵的行数

        theta = temp
        cost[i] = cost_function(X, y, theta)

        if vistualize:
            print('这是第{}次循环, 代价函数的值为{:.4f}' .format(i+1, cost[i]), end=' ')
            for k in range(parameters):
                print(', Ѳ{:d}为{:.4f}' .format(k, theta[0, k]), end=' ')
            print('\n')

        if i != 0 and cost[i] > cost[i-1]:    # 检查梯度下降算法是否正常工作
            raise ValueError('The value of cost_function becomes larger')

    return theta, cost


def cost_function(theta, X, y):    # correct  √
    hypothesis_function = sigmoid(X * theta.T)
    each = np.multiply(-y, np.log(hypothesis_function)) - np.multiply((1 - y), np.log(1 - hypothesis_function))

    return np.sum(each, axis=0) / len(X)


def ensure_alpha(X, y, theta, alpha, iters):    # 选择学习速率
    alpha_first = alpha
    subtraction = 0
    while alpha / alpha_first >= subtraction:
        try:
            gradient_descent(X, y, theta, alpha, iters, vistualize=False)
            alpha *= 1.2
            subtraction = 0.2 * alpha

        except:
            alpha /= 1.1
            subtraction = 0.1 * alpha
        print(alpha)
    return alpha


# 评价录取结果
def func1(theta, X):
    return sigmoid(X * theta.T)


def predict(theta, X):    # 预测模型对训练集的数据准确率
    result = []
    a = sigmoid(X * theta.T)
    for each in a:
        if each >= 0.5:
            result.append(1)
        else:
            result.append(0)

    result = np.mat(result)
    result = result.reshape(result.shape[1], 1)
    return result


def compare(matrix, y):
    temp = (matrix == y)
    correct_nums = 0
    rows = temp.shape[0]
    for each in range(rows):
        if temp[each, 0]:
            correct_nums += 1
        else:
            pass

    accuracy = correct_nums / temp.shape[0]
    print('accuracy = {:.2f}%' .format(accuracy*100))
    return accuracy


if __name__ == '__main__':
    # 首先进行数据可视化(Visualizing the data)
    # 读取数据
    path = 'D:\Study\Coding\Machine Learning WuEnda\homework\ex2\ex2data1.txt'
    data = pd.read_csv(path, header=None, names=['exam1', 'exam2', 'result'])
    # print(data)
    data.insert(0, 'Ones', 1)
    yes = data[data['result'] == 1]
    no = data[data['result'] == 0]
    # print(yes, '\n', no)

    # 开始画图
    plt.figure(figsize=(8, 6))
    dot_yes = plt.scatter(x=yes['exam1'], y=yes['exam2'], s=40, alpha=0.5, label='Admitted', color='blue')
    dot_no = plt.scatter(x=no['exam1'], y=no['exam2'], s=40, alpha=0.5, label='Not admitted', color='red', marker='x')
    plt.xlabel('exam1 score')
    plt.ylabel('exam2 score')

    ax = plt.gca()
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')

    plt.legend(loc='upper right')
    # plt.show()

    # 初始化X, y, theta
    # data.shape返回一个元组(行, 列)
    # rows = data.shape[0]
    cols = data.shape[1]
    X = data.iloc[:, :-1]
    y = data.iloc[:, cols-1:cols]
    print(X.head(), '\n', y.head())

    X = np.mat(X.values)
    y = np.mat(y.values)
    theta = np.mat(np.array([-25.1537, 0.2062, 0.2014]))    # 将theta初始化为[0, 0, 0]即将Ѳ0，Ѳ1，Ѳ2均置零
    # print(X.shape, theta.shape, y.shape)
    #
    # print(cost_function(X, y, theta))

    alpha = 0.0009817
    iters = 1000
    theta_new, cost = gradient_descent(X, y, theta, alpha, iters)
    print(theta_new)
    print(func1(theta_new, np.mat([1, 45, 85])))
    # print(ensure_alpha(X, y, theta, alpha, iters))

    # 可视化拟合结果
    theta_0 = theta_new[0, 0]
    theta_1 = theta_new[0, 1]
    theta_2 = theta_new[0, 2]
    x_data = np.linspace(25, 100, 1024)
    y_data = -(theta_0 + np.multiply(theta_1, x_data)) / theta_2
    plt.plot(x_data, y_data, color='black', linewidth=1.0)
    plt.show()

    expected_result = predict(theta_new, X)
    compare(expected_result, y)

