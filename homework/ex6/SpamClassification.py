import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint
from tqdm import tqdm

import scipy.io as sio
import scipy.optimize as opt
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score

# 引用一些自定义的函数
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

# 进行参数筛选
def select_params(data, gamma_list=None, c_list=None, method='SVC', max_iter=1000):
    best_score = 0
    best_parameters = {}
    for c in tqdm(c_list):

        if method == 'SVC':
            for gamma in gamma_list:

                # 对于每种参数可能的组合，进行一次训练
                svc = svm.SVC(gamma=gamma, C=c, max_iter=max_iter)

                # 5 折交叉验证
                scores = cross_val_score(svc, data.iloc[:, :-1], data[['y']].values.ravel(), cv=10)
                score = scores.mean()
                # print('gamma={}, C={}, score is {}' .format(gamma, c, score))

                # 找到表现最好的参数
                if score > best_score:
                    best_parameters = {'gamma': gamma, "C": c}
                    best_score = score

        elif method == 'LinearSVC':

            # 对于每种参数可能的组合，进行一次训练
            svc = svm.LinearSVC(C=c, max_iter=max_iter)

            # 5 折交叉验证
            scores = cross_val_score(svc, data.iloc[:, :-1], data[['y']].values.ravel(), cv=10)
            score = scores.mean()
            # print('C={}, score is {}' .format(gamma, c, score))

            # 找到表现最好的参数
            if score > best_score:
                 best_parameters = {"C": c}
                 best_score = score

    if method == 'SVC':
        print('THE BEST : gamma={}, C={}, score is {}' .format(best_parameters['gamma'], best_parameters['C'], best_score))
    elif method == 'LinearSVC':
        print('THE BEST : C={}, score is {}' .format(best_parameters['C'], best_score))

    return best_parameters

# 读取数据
raw_data = sio.loadmat('D:\Study\MachineLearning\Machine Learning WuEnda\homework\ex6\spamTrain.mat')
# pprint(raw_data)

data = pd.DataFrame(raw_data.get('X'), columns=['X' + str(x) for x in range(1, 1900)])
data['y'] = raw_data.get('y')
# print(data.shape)

raw_data_test = sio.loadmat('D:\Study\MachineLearning\Machine Learning WuEnda\homework\ex6\spamTest.mat')
data_test = pd.DataFrame(raw_data_test.get('Xtest'), columns=['X' + str(x) for x in range(1, 1900)])
data_test['y'] = raw_data_test.get('ytest')
# print(data_test.head())
# print(data_test.shape)

# 读取词频表
word_list = []
with open('vocab.txt', 'r') as f:
    for each_line in f.readlines():
        word_list.append(each_line.split('\t')[1][:-1])
# print(word_list)

# 选取训练参数
# best_params = select_params(data=data, c_list=np.arange(0.017, 0.025, 0.001), method='LinearSVC', max_iter=30000)

# 开始训练模型, 选用线性svm
# svc = svm.LinearSVC(**best_params)
svc = svm.LinearSVC(C=0.021)
svc.fit(data.iloc[:, :-1], data[['y']].values.ravel())

print('Training accuracy = {0}%'.format(np.round(svc.score(data.iloc[:, :-1], data[['y']].values.ravel()) * 100, 3)))
print('Test accuracy = {0}%'.format(np.round(svc.score(data_test.iloc[:, :-1], data_test[['y']].values.ravel()) * 100, 3)))

# 可视化拟合结果
kw = np.eye(1899)
spam_val = pd.DataFrame({'idx': range(1899)})    # 创建1899维的单位矩阵
spam_val['is_spam'] = svc.decision_function(kw)
# print(spam_val['is_spam'].describe())
# print(spam_val.head())

decision = spam_val[spam_val['is_spam'] > 0].sort_values(by=['is_spam'], ascending=False).reset_index(drop=True)
decision['word'] = [word_list[i] for i in decision['idx']]
# print(decision.head(10))

# 读取输入的邮件
path = 'D:\Study\MachineLearning\Machine Learning WuEnda\homework\ex6\spamSample2.txt'
with open(input('请输入邮件的本地地址: '), 'r') as f:
    all_words = []
    symbol = '!@#$%^&*(),.-?><;:|\"\\\'*`~'
    for each_line in f.readlines():
        for each_word in each_line.split():
            # print(each_word)
            each_word = each_word.casefold()

            if each_word[-1] in symbol:
                each_word = each_word[:-1]
            elif each_word[0] in symbol:
                if len(each_word) == 1:
                    continue
                else:
                    each_word = each_word[1:]
            elif each_word == '\n':
                continue


            if not each_word:
                continue

            elif each_word.isalpha() and each_word in word_list:
                all_words.append(each_word)

            elif each_word[-1] == 's' and each_word not in word_list:
                each_word = each_word[:-1]
                if each_word in word_list:
                    all_words.append(each_word)

            elif each_word[-3:] == 'ing':
                each_word = each_word[:-3]
                if each_word in word_list:
                    all_words.append(each_word)

# print(all_words)

all_words_loc = [word_list.index(each) for each in all_words]
X_test = np.zeros((1, 1899))

for each in all_words_loc:
    X_test[0, each] = 1

predict = svc.predict(X_test)
if predict:
    print('是垃圾邮件')
else:
    print('不是垃圾邮件')

