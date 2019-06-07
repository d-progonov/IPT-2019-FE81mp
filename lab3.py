from pandas import *
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, HuberRegressor
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, r2_score
import random as rand
import os
import numpy as np
import scipy as sp
from scipy import stats
import pandas as pd
from PIL import Image, ImageDraw
import statistics as st


RGB = {'red': 0, 'green': 1, 'blue': 2}

def create_list_files(target_format_in, path_in):

    file_list = list()
    for root, _, files in os.walk(path_in):
        for curr_file in files:
            if target_format_in in curr_file:
                file_list.append(root + curr_file)
    return file_list
if __name__ == "__main__":
    file_list_rand = create_list_files(target_format_in='.jpg',
                                  path_in='H:\\test\\')
    mean_R = []
    mean_G = []
    mean_B = []

    var_R = []
    var_G = []
    var_B = []

    skew_R = []
    skew_G = []
    skew_B = []

    kurt_R = []
    kurt_G = []
    kurt_B = []
    data = {}
    for name, num in RGB.items():
        data[name] = pd.DataFrame()
        for image_name in file_list_rand:
            image = np.array(Image.open(image_name))
            a = image[ :, num].ravel()
            df = {'name': image_name,
                 'mean': np.mean(a),
                 'var': np.var(a),
                 'skewness': sp.stats.skew(a),
                 'kurtosis': sp.stats.kurtosis(a)}
            data[name] = pd.concat([data[name], pd.DataFrame(pd.DataFrame(df, index=[0, ]))], ignore_index=True)
            if num == 0:
                mean_R.append(round(df['mean'], 3))
                var_R.append(round(df['var'], 3))
                skew_R.append(round(df['skewness'], 3))
                kurt_R.append(round(df['kurtosis'], 3))
            elif num == 1:
                mean_G.append(round(df['mean'], 3))
                var_G.append(round(df['var'], 3))
                skew_G.append(round(df['skewness'], 3))
                kurt_G.append(round(df['kurtosis'], 3))
            else:
                mean_B.append(round(df['mean'], 3))
                var_B.append(round(df['var'], 3))
                skew_B.append(round(df['skewness'], 3))
                kurt_B.append(round(df['kurtosis'], 3))
VMEAN = np.array((mean_R, mean_G, mean_B))
VVAR=np.array((var_R, var_G, var_B))
VSKEW=np.array((skew_R, skew_G, skew_B))
VKURT=np.array((kurt_R, kurt_G, kurt_B))


#SPAM

set_option('display.max_columns', 50)
set_option('display.width', 500)

data_tune = read_csv('H:/lab/SPAM_train_641.csv') #Набор тестовых данных
data_test = read_csv('H:/lab/SPAM_test_641.csv') # Набор контрольных данных

x = data_tune.iloc[:,:-1] # Набор признаков
y = data_tune.iloc[:,-1] # Метки классов

x_train = x
x_test = data_test.iloc[:,:-1] #Вся матрица кроме последнего столбца с метками

y_train = y
y_test = data_test.iloc[:,-1] # последний столбец

d = DataFrame(index=y_test.index) # Набор данных для проверки модели
d["Actual"] = y_test # Реальные значения тегов

#SPAM lin_regression
COUNT_TRUE_LIN_REG_SPAM = []
COUNT_1_ERROR_LIN_REG_SPAM = []
COUNT_2_ERROR_LIN_REG_SPAM = []
model1 = LinearRegression()
model1.fit(x_train, y_train)
for i in np.arange(10):
    d['Predict_lin_reg_SPAM'] = model1.predict(x_test) # Тест
    d['Predict_lin_reg_SPAM'] = round(d['Predict_lin_reg_SPAM'])

    count_lin_reg1 = 0
    count_lin_reg2 = 0
    count_true = 0

    d["Error_lin_reg_SPAM"] = (d["Actual"]-d['Predict_lin_reg_SPAM'])

    for i in d["Error_lin_reg_SPAM"]:
            if i == -1:
                count_lin_reg2 += 1
            elif i == 1:
                count_lin_reg1 += 1
            elif i == 0:
                count_true += 1
    COUNT_TRUE_LIN_REG_SPAM.append(count_true)
    COUNT_1_ERROR_LIN_REG_SPAM.append(count_lin_reg1)
    COUNT_2_ERROR_LIN_REG_SPAM.append(count_lin_reg2)
COUNT_TRUE_LIN_REG_MEAN_SPAM = np.mean(COUNT_TRUE_LIN_REG_SPAM)
COUNT_1_ERROR_LIN_REG_MEAN_SPAM = np.mean(COUNT_1_ERROR_LIN_REG_SPAM)
COUNT_2_ERROR_LIN_REG_MEAN_SPAM = np.mean(COUNT_2_ERROR_LIN_REG_SPAM)
print("Percent of correct lin_reg_SPAM: %.1f%%" % ((COUNT_TRUE_LIN_REG_MEAN_SPAM / len(d["Error_lin_reg_SPAM"]) * 100)))
print("Error 1 - %.1f%%" % (COUNT_1_ERROR_LIN_REG_MEAN_SPAM / len(d["Error_lin_reg_SPAM"])*100))
print("Error 2 - %.1f%%" % (COUNT_2_ERROR_LIN_REG_MEAN_SPAM / len(d["Error_lin_reg_SPAM"])*100))

#robust regression
COUNT_TRUE_ROBUST_SPAM = []
COUNT_1_ERROR_ROBUST_SPAM = []
COUNT_2_ERROR_ROBUST_SPAM = []
model2 = HuberRegressor()
model2.fit(x_train, y_train)
for i in np.arange(10):
    count_robust1 = 0
    count_robust2 = 0
    count_true = 0

    d['Predict_robust_SPAM'] = model2.predict(x_test) # Тест
    d['Predict_robust_SPAM'] = round(d['Predict_robust_SPAM'])
    d["Error_robust_SPAM"] = (d["Actual"]-d['Predict_robust_SPAM'])

    for i in d["Error_robust_SPAM"]:
            if i == -1:
                count_robust2 += 1
            elif i == 1:
                count_robust1 += 1
            elif i == 0:
                count_true += 1
    COUNT_TRUE_ROBUST_SPAM.append(count_true)
    COUNT_1_ERROR_ROBUST_SPAM.append(count_robust1)
    COUNT_2_ERROR_ROBUST_SPAM.append(count_robust2)
COUNT_TRUE_ROBUST_MEAN_SPAM = np.mean(COUNT_TRUE_ROBUST_SPAM)
COUNT_1_ERROR_ROBUST_MEAN_SPAM = np.mean(COUNT_1_ERROR_ROBUST_SPAM)
COUNT_2_ERROR_ROBUST_MEAN_SPAM = np.mean(COUNT_2_ERROR_ROBUST_SPAM)
print("Percent o correct robust_SPAM: %.1f%%" % ((COUNT_TRUE_ROBUST_MEAN_SPAM / len(d["Error_robust_SPAM"]) * 100)))
print("Error 1 - %.1f%%" % (COUNT_1_ERROR_ROBUST_MEAN_SPAM / len(d["Error_robust_SPAM"])*100))
print("Error 2 - %.1f%%" % (COUNT_2_ERROR_ROBUST_MEAN_SPAM / len(d["Error_robust_SPAM"])*100))

#SPAM log_regression
COUNT_TRUE_LR_SPAM = []
COUNT_1_ERROR_LR_SPAM = []
COUNT_2_ERROR_LR_SPAM = []
model3 = LogisticRegression()
model3.fit(x_train, y_train)
for i in np.arange(10):
    d['Predict_LR_SPAM'] = model3.predict(x_test) # Тест

    count_logistic1 = 0
    count_logistic2 = 0
    count_true = 0

    d["Error_LR_SPAM"] = (d["Actual"]-d['Predict_LR_SPAM'])
    for i in d["Error_LR_SPAM"]:
        if i == -1:
            count_logistic2 += 1
        elif i == 1:
            count_logistic1 += 1
        elif i == 0:
            count_true += 1
    COUNT_TRUE_LR_SPAM.append(count_true)
    COUNT_1_ERROR_LR_SPAM.append(count_logistic1)
    COUNT_2_ERROR_LR_SPAM.append(count_logistic2)
COUNT_TRUE_LR_MEAN_SPAM = np.mean(COUNT_TRUE_LR_SPAM)
COUNT_1_ERROR_LR_MEAN_SPAM = np.mean(COUNT_1_ERROR_LR_SPAM)
COUNT_2_ERROR_LR_MEAN_SPAM = np.mean(COUNT_2_ERROR_LR_SPAM)
print("Percent of logistic regression for SPAM: %.1f%%" % ((COUNT_TRUE_LR_MEAN_SPAM / len(d["Error_LR_SPAM"]) * 100)))
print("Error 1 - %.1f%%" % (COUNT_1_ERROR_LR_MEAN_SPAM / len(d["Error_LR_SPAM"])*100))
print("Error 2 - %.1f%%" % (COUNT_2_ERROR_LR_MEAN_SPAM / len(d["Error_LR_SPAM"])*100))

#SPAM metod SVM (opornih vectorov)
COUNT_TRUE_SVM_SPAM = []
COUNT_1_ERROR_SVM_SPAM = []
COUNT_2_ERROR_SVM_SPAM = []
model4 = SVC()
model4.fit(x_train, y_train)
for i in np.arange(10):
    count_svm1 = 0
    count_svm2 = 0
    count_true = 0

    d['Predict_SVM_SPAM'] = model4.predict(x_test) # Тест
    d["Error_SVM_SPAM"] = (d["Actual"]-d['Predict_SVM_SPAM'])

    for i in d["Error_SVM_SPAM"]:
            if i == -1:
                count_svm2 += 1
            elif i == 1:
                count_svm1 += 1
            elif i == 0:
                count_true += 1
    COUNT_TRUE_SVM_SPAM.append(count_true)
    COUNT_1_ERROR_SVM_SPAM.append(count_svm1)
    COUNT_2_ERROR_SVM_SPAM.append(count_svm2)
COUNT_TRUE_SVM_MEAN_SPAM = np.mean(COUNT_TRUE_SVM_SPAM)
COUNT_1_ERROR_SVM_MEAN_SPAM = np.mean(COUNT_1_ERROR_SVM_SPAM)
COUNT_2_ERROR_SVM_MEAN_SPAM = np.mean(COUNT_2_ERROR_SVM_SPAM)
print("Percent o correct SVM_SPAM: %.1f%%" % ((COUNT_TRUE_SVM_MEAN_SPAM / len(d["Error_SVM_SPAM"]) * 100)))
print("Error 1 - %.1f%%" % (COUNT_1_ERROR_SVM_MEAN_SPAM / len(d["Error_SVM_SPAM"])*100))
print("Error 2 - %.1f%%" % (COUNT_2_ERROR_SVM_MEAN_SPAM / len(d["Error_SVM_SPAM"])*100))

################################################################################################################## CCPEV

set_option('display.max_columns', 50)
set_option('display.width', 500)

data_tune = read_csv('H:/lab/CCPEV_train_641.csv') #Набор тестовых данных
data_test = read_csv('H:/lab/CCPEV_test_641.csv') # Набор контрольных данных

x = data_tune.iloc[:, :-1] # Набор признаков
y = data_tune.iloc[:, -1] # Метки классов

x_train = x
x_test = data_test.iloc[:, :-1]

y_train = y
y_test = data_test.iloc[:, -1]

count_true_actual = 0
count_false_actual = 0

d = DataFrame(index=y_test.index) # Набор данных для проверки модели
d["Actual"] = y_test # Реальные значения тегов

# CCPEV lin_regression
COUNT_TRUE_LIN_REG_CCPEV = []
COUNT_1_ERROR_LIN_REG_CCPEV = []
COUNT_2_ERROR_LIN_REG_CCPEV = []
model5 = LinearRegression()
model5.fit(x_train, y_train)
for i in np.arange(10):

    d['Predict_lin_reg_CCPEV'] = model5.predict(x_test)  # Тест
    d['Predict_lin_reg_CCPEV'] = round(d['Predict_lin_reg_CCPEV'])

    count_lin_reg1 = 0
    count_lin_reg2 = 0
    count_true = 0

    d["Error_lin_reg_CCPEV"] = (d["Actual"] - d['Predict_lin_reg_CCPEV'])

    for i in d["Error_lin_reg_CCPEV"]:
        if i == -1:
            count_lin_reg2 += 1
        elif i == 1:
            count_lin_reg1 += 1
        elif i == 0:
            count_true += 1
    COUNT_TRUE_LIN_REG_CCPEV.append(count_true)
    COUNT_1_ERROR_LIN_REG_CCPEV.append(count_lin_reg1)
    COUNT_2_ERROR_LIN_REG_CCPEV.append(count_lin_reg2)
COUNT_TRUE_LIN_REG_MEAN_CCPEV = np.mean(COUNT_TRUE_LIN_REG_CCPEV)
COUNT_1_ERROR_LIN_REG_MEAN_CCPEV = np.mean(COUNT_1_ERROR_LIN_REG_CCPEV)
COUNT_2_ERROR_LIN_REG_MEAN_CCPEV = np.mean(COUNT_2_ERROR_LIN_REG_CCPEV)
print("Percent of correct lin_reg_CCPEV: %.1f%%" % ((COUNT_TRUE_LIN_REG_MEAN_CCPEV / len(d["Error_lin_reg_CCPEV"]) * 100)))
print("Error 1 - %.1f%%" % (COUNT_1_ERROR_LIN_REG_MEAN_CCPEV / len(d["Error_lin_reg_CCPEV"]) * 100))
print("Error 2 - %.1f%%" % (COUNT_2_ERROR_LIN_REG_MEAN_CCPEV / len(d["Error_lin_reg_CCPEV"]) * 100))

# robust regression
COUNT_TRUE_ROBUST_CCPEV = []
COUNT_1_ERROR_ROBUST_CCPEV = []
COUNT_2_ERROR_ROBUST_CCPEV = []
model6 = HuberRegressor()
model6.fit(x_train, y_train)
for i in np.arange(10):
    count_robust1 = 0
    count_robust2 = 0
    count_true = 0
    d['Predict_robust_CCPEV'] = model6.predict(x_test)  # Тест
    d['Predict_robust_CCPEV'] = round(d['Predict_robust_CCPEV'])
    d["Error_robust_CCPEV"] = (d["Actual"] - d['Predict_robust_CCPEV'])

    for i in d["Error_robust_CCPEV"]:
        if i == -1:
            count_robust2 += 1
        elif i == 1:
            count_robust1 += 1
        elif i == 0:
            count_true += 1
    COUNT_TRUE_ROBUST_CCPEV.append(count_true)
    COUNT_1_ERROR_ROBUST_CCPEV.append(count_robust1)
    COUNT_2_ERROR_ROBUST_CCPEV.append(count_robust2)
COUNT_TRUE_ROBUST_MEAN_CCPEV = np.mean(COUNT_TRUE_ROBUST_CCPEV)
COUNT_1_ERROR_ROBUST_MEAN_CCPEV = np.mean(COUNT_1_ERROR_ROBUST_CCPEV)
COUNT_2_ERROR_ROBUST_MEAN_CCPEV = np.mean(COUNT_2_ERROR_ROBUST_CCPEV)
print("Percent o correct robust_CCPEV: %.1f%%" % ((COUNT_TRUE_ROBUST_MEAN_CCPEV / len(d["Error_robust_CCPEV"]) * 100)))
print("Error 1 - %.1f%%" % (COUNT_1_ERROR_ROBUST_MEAN_CCPEV / len(d["Error_robust_CCPEV"]) * 100))
print("Error 2 - %.1f%%" % (COUNT_2_ERROR_ROBUST_MEAN_CCPEV / len(d["Error_robust_CCPEV"]) * 100))

# CCPEV log_regression
COUNT_TRUE_LR_CCPEV = []
COUNT_1_ERROR_LR_CCPEV = []
COUNT_2_ERROR_LR_CCPEV = []
model7 = LogisticRegression()
model7.fit(x_train, y_train)
for i in np.arange(10):
    d['Predict_LR_CCPEV'] = model7.predict(x_test)  # Тест

    count_logistic1 = 0
    count_logistic2 = 0
    count_true = 0

    d["Error_LR_CCPEV"] = (d["Actual"] - d['Predict_LR_CCPEV'])
    for i in d["Error_LR_CCPEV"]:
        if i == -1:
            count_logistic2 += 1
        elif i == 1:
            count_logistic1 += 1
        elif i == 0:
            count_true += 1
    COUNT_TRUE_LR_CCPEV.append(count_true)
    COUNT_1_ERROR_LR_CCPEV.append(count_logistic1)
    COUNT_2_ERROR_LR_CCPEV.append(count_logistic2)
COUNT_TRUE_LR_MEAN_CCPEV = np.mean(COUNT_TRUE_LR_CCPEV)
COUNT_1_ERROR_LR_MEAN_CCPEV = np.mean(COUNT_1_ERROR_LR_CCPEV)
COUNT_2_ERROR_LR_MEAN_CCPEV = np.mean(COUNT_2_ERROR_LR_CCPEV)
print("Percent of logistic regression for CCPEV: %.1f%%" % ((COUNT_TRUE_LR_MEAN_CCPEV / len(d["Error_LR_CCPEV"]) * 100)))
print("Error 1 - %.1f%%" % (COUNT_1_ERROR_LR_MEAN_CCPEV / len(d["Error_LR_CCPEV"]) * 100))
print("Error 2 - %.1f%%" % (COUNT_2_ERROR_LR_MEAN_CCPEV / len(d["Error_LR_CCPEV"]) * 100))

# CCPEV metod SVM (opornih vectorov)
COUNT_TRUE_SVM_CCPEV = []
COUNT_1_ERROR_SVM_CCPEV = []
COUNT_2_ERROR_SVM_CCPEV = []
model8 = SVC()
model8.fit(x_train, y_train)
for i in np.arange(10):
    count_svm1 = 0
    count_svm2 = 0
    count_true = 0

    d['Predict_SVM_CCPEV'] = model8.predict(x_test)  # Тест
    d["Error_SVM_CCPEV"] = (d["Actual"] - d['Predict_SVM_CCPEV'])

    for i in d["Error_SVM_CCPEV"]:
        if i == -1:
            count_svm2 += 1
        elif i == 1:
            count_svm1 += 1
        elif i == 0:
            count_true += 1
    COUNT_TRUE_SVM_CCPEV.append(count_true)
    COUNT_1_ERROR_SVM_CCPEV.append(count_svm1)
    COUNT_2_ERROR_SVM_CCPEV.append(count_svm2)
COUNT_TRUE_SVM_MEAN_CCPEV = np.mean(COUNT_TRUE_SVM_CCPEV)
COUNT_1_ERROR_SVM_MEAN_CCPEV = np.mean(COUNT_1_ERROR_SVM_CCPEV)
COUNT_2_ERROR_SVM_MEAN_CCPEV = np.mean(COUNT_2_ERROR_SVM_CCPEV)
print("Percent o correct SVM_CCPEV: %.1f%%" % ((COUNT_TRUE_SVM_MEAN_CCPEV / len(d["Error_SVM_CCPEV"]) * 100)))
print("Error 1 - %.1f%%" % (COUNT_1_ERROR_SVM_MEAN_CCPEV / len(d["Error_SVM_CCPEV"]) * 100))
print("Error 2 - %.1f%%" % (COUNT_2_ERROR_SVM_MEAN_CCPEV / len(d["Error_SVM_CCPEV"]) * 100))