import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint
from tqdm import tqdm

import scipy.io as sio
import scipy.optimize as opt
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score

raw_data = sio.loadmat('D:\Study\MachineLearning\Machine Learning WuEnda\homework\ex6\ex6data1.mat')
# pprint(raw_data)
data = pd.DataFrame(raw_data.get('X'), columns=['X1', 'X2'])
data['y'] = raw_data.get('y')
# print(data.head())

# 进行划分数据集，train, cross_validation, test
def divide_dataset(data, train_size=0.6, cv_size=0.3, test_size=0.1, nums=3):
    """
    分为三类, 训练集, 交叉验证集, 测试集, 默认比例6: 3: 1
    """
    # 数据集的行数
    m = data.shape[0]

    # 打乱重排
    data = data.sample(frac=1.0)

    train_pro, cv_pro, test_pro = map(lambda x: round(m * x), [train_size, cv_size, test_size])

    # 检查是否有0出现，若有则替换为1
    alist = [train_pro, cv_pro, test_pro]
    for i in range(3):
        if not alist[i]:
            alist[i] = 1

    data_train = data[: train_pro].reset_index(drop=True)
    data_cv = data[train_pro: train_pro + cv_pro].reset_index(drop=True)
    data_test = data[-test_pro:].reset_index(drop=True)

    return data_train, data_cv, data_test

data_train, data_cv, data_test = divide_dataset(data)

# 先进行可视化
plt.figure(figsize=(12, 8))
ax1 = plt.subplot2grid((2, 2), loc=(0, 0), rowspan=1, colspan=1)

def plot_init_data(data, ax, title=None):
    positive = data[data['y'].isin([1])]
    negative = data[data['y'].isin([0])]

    ax.scatter(positive['X1'], positive['X2'], s=20, marker='x', label='Positive')
    ax.scatter(negative['X1'], negative['X2'], s=20, marker='o', label='Negative')
    ax.legend()
    ax.set_title(title)

plot_init_data(data, ax1, title='SVM (C=1) Decision Boundary')

# 开始训练
svc = svm.LinearSVC(C=1, loss='hinge', max_iter=10000)
svc.fit(data[['X1', 'X2']], data['y'])
print(svc.score(data[['X1', 'X2']], data['y']))

# 可视化分类边界
x = np.linspace(0, 5)

def find_decision_boundary(svc, x1min, x1max, x2min, x2max, diff):
    x1 = np.linspace(x1min, x1max, 1000)
    x2 = np.linspace(x2min, x2max, 1000)

    # 先绘制网格，也就是全平面的所有点
    cordinates = [(x, y) for x in x1 for y in x2]
    x_cord, y_cord = zip(*cordinates)
    c_val = pd.DataFrame({'x1': x_cord, 'x2': y_cord})

    # decision_function(X)返回样本到超平面的符号距离
    c_val['c_val'] = svc.decision_function(c_val[['x1', 'x2']])

    # 将到超平面距离小于diff的样本返回, 也就是调整线宽
    decision = c_val[np.abs(c_val['c_val']) < diff]

    return decision.x1, decision.x2

x1, x2 = find_decision_boundary(svc, 0, 4, 1.5, 5, 1 * 10**-3)
ax1.scatter(x1, x2, s=10, c='r', label='Boundary')
ax1.legend()

# 然后在第26行改变C的值, 观察曲线变化




# 开始使用具有高斯核函数内核的SVM
def gaussian_kernel(x1, x2, sigma):
    return np.exp(-(np.sum((x1 - x2) ** 2) / (2 * (sigma ** 2))))

# 测试高斯函数
x1 = np.array([1.0, 2.0, 1.0])
x2 = np.array([0.0, 4.0, -1.0])
sigma = 2
print(gaussian_kernel(x1, x2, sigma))


# 开始使用dataset2进行拟合，注意变量的替换
raw_data = sio.loadmat('D:\Study\MachineLearning\Machine Learning WuEnda\homework\ex6\ex6data2.mat')

data = pd.DataFrame(raw_data['X'], columns=['X1', 'X2'])
data['y'] = raw_data['y']

# 划分为三个数据集
data_train, data_cv, data_test = divide_dataset(data)

# 进行参数筛选
def select_params(data, gamma_list, c_list, method='SVC'):
    best_score = 0
    best_parameters = {}
    for gamma in tqdm(gamma_list):
        for c in c_list:

            # 对于每种参数可能的组合，进行一次训练
            if method == 'SVC':
                svc = svm.SVC(gamma=gamma, C=c)
            elif method == 'LinearSVC':
                svc = svm.LinearSVC(gamma=gamma, C=c)
            else:
                svc = None

            # 5 折交叉验证
            scores = cross_val_score(svc, data[['X1', 'X2']], data[['y']].values.ravel(), cv=10)
            score = scores.mean()
            # print('gamma={}, C={}, score is {}' .format(gamma, c, score))

            # 找到表现最好的参数
            if score > best_score:
                best_parameters = {'gamma': gamma, "C": c}
                best_score = score

    print('THE BEST : gamma={}, C={}, score is {}' .format(best_parameters['gamma'], best_parameters['C'], best_score))

    return best_parameters

best_parameters = select_params(data=data, gamma_list=range(50, 70, 5), c_list=[0.1, 1, 10, 50, 100])

# 使用最佳参数，构建新的模型
svc2 = svm.SVC(**best_parameters)

# 使用训练集和验证集进行训练
svc2.fit(data[['X1', 'X2']], data['y'])

ax2 = plt.subplot2grid((2, 2), (0, 1), rowspan=1, colspan=1)
plot_init_data(data, ax2, title='SVM With Gaussian_Kernel')

# 开始训练模型
# svc2 = svm.SVC(C=100, kernel='rbf', gamma=10, probability=True)
# print(svc2)

# svc2.fit(data[['X1', 'X2']], data['y'])
print(svc2.score(data[['X1', 'X2']], data['y']))

# 超平面边界可视化
x1, x2 = find_decision_boundary(svc2, 0, 1.1, 0.35, 1.1, 7 * 10**-3)
ax2.scatter(x1, x2, s=5, c='r', label='Boundary', alpha=0.4)
ax2.legend()



# 进行dataset3的数据分类
raw_data = sio.loadmat('D:\Study\MachineLearning\Machine Learning WuEnda\homework\ex6\ex6data3.mat')
# pprint(raw_data)

data = pd.DataFrame(raw_data.get('X'), columns=['X1', 'X2'])
data['y'] = raw_data.get('y')
data_val = pd.DataFrame(raw_data.get('Xval'), columns=['X1', 'X2'])
data_val['y'] = raw_data.get('yval')
# print(data.head(), data_val.head())

# 原始数据可视化
ax3 = plt.subplot2grid((2, 2), (1, 0), rowspan=1, colspan=1)
plot_init_data(data, ax3, title='plot3')

# 开始筛选参数
best_parameters = select_params(data=data, gamma_list=[0.1, 1, 10, 50, 100], c_list=[0.1, 1, 10, 50, 100], method='SVC')

svc3 = svm.SVC(**best_parameters)
svc3.fit(data[['X1', 'X2']], data['y'])
print(svc3.score(data_val[['X1', 'X2']], data_val['y']))

x1, x2 = find_decision_boundary(svc3, -0.6, 0.3, -0.7, 0.6, 2 * 10**-3)
ax3.scatter(x1, x2, s=3, c='r', label='Boundary', alpha=0.4)
ax3.legend()

plt.show()
