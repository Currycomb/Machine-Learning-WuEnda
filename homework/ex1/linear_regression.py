import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 这个部分计算J(Ѳ)，X是矩阵
def compute_cost(X, y, theta):
    inner = np.power((X*theta.T - y), 2)
    return np.sum(inner, axis=0) / (2*len(X))


def gradient_descent(X, y, _theta, alpha, iters):
    # temp用来暂存theta的值，因为要同时改变theta1和2
    temp = np.mat(np.zeros(_theta.shape))

    # parameters用来控制循环次数
    parameters = int(_theta.ravel().shape[1])

    # cost用来存放每次迭代后的代价函数
    cost = np.zeros(iters)

    # 外循环控制迭代次数
    for i in range(iters):
        error = (X * _theta.T) - y

        # 内循环对每个theta进行处理，即有几个theta就循环几次
        for j in range(parameters):

            term = np.multiply(error, X[:, j])
            temp[0, j] = _theta[0, j] - ((alpha / len(X)) * np.sum(term, axis=0))

        _theta = temp
        cost[i] = compute_cost(X, y, _theta)

        print('这是第{}次循环, 代价函数的值为{:.4f}' .format(i+1, cost[i]), end=' ')
        for k in range(parameters):
            print(', Ѳ{:d}为{:.4f}' .format(k, _theta[0, k]), end=' ')
        print('\n')
    return _theta, cost


def normal_equation(X, y):
    _theta = (X.T*X).I*X.T*y
    return _theta


if __name__ == '__main__':
    # 几维矩阵一般是指有多少列的矩阵而不是行数
    # 因为做数据分析的时候有时样本行数是几万几万的来
    # 但是降维分析不是缩减样本体积而是去掉不必要的列数来构建特征向量

    # 读取数据
    show_predict_result = False
    show_3d_plot = True
    show_2d_plot = True

    path = 'D:\Study\Coding\Machine Learning WuEnda\homework\ex1\ex1data1.txt'
    data = pd.read_csv(path, header=None, names=['Population', 'Profits'])

    plt.figure(figsize=(8, 6))
    dot = plt.scatter(x=data['Population'], y=data['Profits'], s=40, alpha=0.6, label='Training data')
    # plt.xlim((0, 23))
    plt.ylim((-5, 25))
    plt.xlabel('Population')
    plt.ylabel('Profits')

    ax = plt.gca()
    ax.spines['top'].set_color('None')
    ax.spines['right'].set_color('None')
    # plt.show()

    # 加入一列x，用于更新Ѳ
    data.insert(0, 'Ones', 1)
    # print(data)

    # 初始化X和y
    # data.shape返回(97, 3)即97行3列
    cols = data.shape[1]
    X = data.iloc[:, :-1]    # X是data里的除最后列
    y = data.iloc[:, cols-1:cols]    # y是data最后一列
    # print(X.head(), '\n', y.head())

    X = np.mat(X.values)
    y = np.mat(y.values)
    theta = np.mat(np.array([0, 0]))
    # print(normal_equation(X, y))
    # print('----------------------------------------------------------------')
    # print(X.shape, '\n', y.shape, '\n',  theta.shape, theta.ravel())

    # 计算J(Ѳ)
    # print(compute_cost(X, y, theta))

    # np.shape显示(矩阵的行数, 矩阵的列数)
    # np.zeros((x, y))输出一个x行y列的都是0的矩阵
    # np.ravel(np.array([1, 2], [3, 4]))输出[1 2 3 4]实现多维度矩阵降维

    # 开始进行梯度下降
    # 这个部分实现了Ѳ的更新
    # alpha为学习速率, iters为迭代次数
    alpha = 0.01
    iters = 1500
    g, cost = gradient_descent(X, y, theta, alpha, iters)
    # print(g)

    # 构造预测数据,即人口为3.5w和7w时餐馆的利润
    predict1 = np.mat([1, 3.5]) * g.T
    print('predict1: {:.3f}' .format(predict1[0, 0]))    # 保留三位小数
    predict2 = np.mat([1, 7]) * g.T
    print('predict2: {:.3f}' .format(predict2[0, 0]))

    # 可视化处理
    # 产生数据,图例在plot,scatter里面画
    x = np.linspace(2, data.Population.max(), 100)
    f_predict = g[0, 0] + (g[0, 1] * x)
    predict_line = plt.plot(x, f_predict, linewidth=1.7, linestyle='-', label='Predict line')
    plt.scatter(x=[3.5, 7], y=[predict1, predict2], s=35, alpha=0.9, color='red', label='Predict profits')    # 产生预测点
    plt.title('Predicted Profit vs. Population Size')
    plt.legend(loc='upper left')
    plt.show()

    # 可视化J(θ) 暂且不会



