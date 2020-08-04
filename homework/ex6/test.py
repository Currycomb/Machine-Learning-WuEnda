import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint

import scipy.io as sio
import scipy.optimize as opt
from sklearn import svm
from sklearn.model_selection import train_test_split

data = pd.DataFrame({'x1': [i for i in range(10)], 'x2': [i**2 for i in range(10)],
                     'x3': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]})
print(data)

data_train, data_cv = train_test_split(data, test_size=0.4, random_state=1, stratify=data['x3'])
data_left, data_test = train_test_split(data.drop(data_train), test_size=0.25, random_state=1, stratify=data['x3'])
print(data_cv.reset_index(drop=True))
print(data_train.reset_index(drop=True))










