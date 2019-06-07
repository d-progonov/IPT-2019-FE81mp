import random as rand
import os
import matplotlib.pyplot as plt
from numpy.linalg import matrix_power

import numpy as np
import scipy as sp
from scipy import stats
import pandas as pd
from PIL import Image, ImageDraw

RGB = {'red': 0, 'green': 1, 'blue': 2}

def create_list_files(target_format_in, path_in):

    file_list = list()
    for root, _, files in os.walk(path_in):
        for curr_file in files:
            if target_format_in in curr_file:
                file_list.append(root + curr_file)
    return file_list

if __name__ == "__main__":
    file_list = create_list_files(target_format_in='.jpg',
                                   path_in='H:\\mirflickr25k\\')

    file_list_rand = rand.sample(file_list, 100)
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
            d = {'name': image_name,
                 'mean': np.mean(a),
                 'var': np.var(a),
                 'skewness': sp.stats.skew(a),
                 'kurtosis': sp.stats.kurtosis(a)}
            data[name] = pd.concat([data[name], pd.DataFrame(pd.DataFrame(d, index=[0, ]))], ignore_index=True)
            if num == 0:
                mean_R.append(round(d['mean'], 3))
                var_R.append(round(d['var'], 3))
                skew_R.append(round(d['skewness'], 3))
                kurt_R.append(round(d['kurtosis'], 3))
            elif num == 1:
                mean_G.append(round(d['mean'], 3))
                var_G.append(round(d['var'], 3))
                skew_G.append(round(d['skewness'], 3))
                kurt_G.append(round(d['kurtosis'], 3))
            else:
                mean_B.append(round(d['mean'], 3))
                var_B.append(round(d['var'], 3))
                skew_B.append(round(d['skewness'], 3))
                kurt_B.append(round(d['kurtosis'], 3))

    VMEAN = np.array((mean_R, mean_G, mean_B))
    VVAR=np.array((var_R, var_G, var_B))
    VSKEW=np.array((skew_R, skew_G, skew_B))
    VKURT=np.array((kurt_R, kurt_G, kurt_B))
    VARM=((mean_R, mean_G, mean_B, var_R, var_G, var_B))
    SKEWMV=((mean_R, mean_G, mean_B, var_R, var_G, var_B, skew_R, skew_G, skew_B))
    KURTMVS=((mean_R, mean_G, mean_B, var_R, var_G, var_B, skew_R, skew_G, skew_B, kurt_R, kurt_G, kurt_B))

    print('Mean matrix')
    print(VMEAN)

    COV_MEAN = np.cov(np.vstack((mean_R, mean_G, mean_B)))
    print('Cov mean matrix')
    print(COV_MEAN)

    print('Mean_var matrix')
    print(VARM )

    COV_MEAN_VAR = np.array(np.cov(np.vstack((mean_R, mean_G, mean_B, var_R, var_G, var_B))))
    print('Cov mean_var matrix')
    print((COV_MEAN_VAR))

    print('Mean_var_skew matrix')
    print(SKEWMV)

    COV_MEAN_VAR_SKEW = np.array(np.cov(np.vstack((mean_R, mean_G, mean_B, var_R, var_G, var_B, skew_R, skew_G, skew_B))))
    print('Cov mean_var_skew matrix')
    print(COV_MEAN_VAR_SKEW )

    print('Mean_var_skew_kurt matrix')
    print(KURTMVS)

    COV_MEAN_VAR_SKEW_KURT = np.cov(np.vstack((mean_R, mean_G, mean_B, var_R, var_G, var_B, skew_R, skew_G, skew_B, kurt_R, kurt_G, kurt_B)))
    print('Cov mean_var_skew_kurt matrix')
    print(COV_MEAN_VAR_SKEW_KURT)

    path = "H:\\mirflickr25k\\im32.jpg"
    image = Image.open(path)
    width = image.size[0]
    height = image.size[1]
    draw = ImageDraw.Draw(image)
    pix = image.load()

    r_new = [[0 for i in range(height)] for i in range(width)]
    for i in range(width):
        for j in range(height):
            r_new[i][j] = pix[i, j][0]

    U, s, V = np.linalg.svd(r_new, full_matrices=True)
    print('Матрица U')
    print(np.array(U))
    print('Список s')
    print(np.array(s))
    print('Матрица V')
    print(np.array(V))
    comp=[]
    MSE=[]
    for number in range(len(s)):
        singular_matrix = np.zeros((width, height))
        for k in range(width):
            for l in range(height):
                if k == l:
                    singular_matrix[k][l] = s[k]
        for i in range(number,width):
            for j in range(number,height):
                singular_matrix[i][j] = 0
        US = np.dot(U, singular_matrix)
        restored_matrix = np.dot(US, V)


    for i in range(width):
        for j in range(height):
            a = int(restored_matrix[i][j])
            b = pix[i, j][1]
            c = pix[i, j][2]
            draw.point((i, j), (a, b, c))
    image.save("ans.jpg", "JPEG")
    del draw

    total_error = 0
    for i in range(width):
        for j in range(height):
            diff = (r_new[i][j] - restored_matrix[i][j])**2
            total_error += diff
    EPS = (total_error/(width*height))**(1/2)
    MSE.append(EPS)
    comp.append(number)
    #print('Ошибка исходной матрици и полученой: ', MSE, '%')
    #print()
plt.figure()
plt.plot(comp, MSE)
plt.ylabel('MSE')
plt.xlabel('Components')
plt.grid(True)
plt.show()

path = "H:\\mirflickr25k\\im1.jpg"
image = Image.open(path)
width = image.size[0]
height = image.size[1]
draw = ImageDraw.Draw(image)
pix = image.load()

r_new = [[0 for i in range(height)] for i in range(width)]
for i in range(width):
    for j in range(height):
        r_new[i][j] = pix[i, j][0]

Stochastic_matrix = np.zeros((256, 256))
sum_row = 0
print('Стохастическая матрица L -> R: ')
for i in range(height):
    for j in range(width - 1):
        k1 = r_new[i][j]
        k2 = r_new[i][j + 1]
        Stochastic_matrix[k1-1][k2-1] += 1

for i in range(256):
    for j in range(256):
        sum_row += Stochastic_matrix[i][j]
    Stochastic_matrix[i] /= sum_row
    sum_row = 0
print(Stochastic_matrix)

r1=np.array(r_new)
r1=r1[:, 1:len(r1)]
P = np.zeros((256, 256))
sum_row = 0
print('P')
for i in range(len(r1)):
    for j in range(len(r1[0]) - 1):
        k1 = r1[i][j]
        k2 = r1[i][j + 1]
        P[k1-1][k2-1] += 1

for i in range(256):
    for j in range(256):
        sum_row += P[i][j]
    if sum_row == 0:
        P[i]=0
    else:
        P[i] /= sum_row
    sum_row = 0
print(P)

Stochastic_matrix2=np.dot(Stochastic_matrix, P)
Stochastic_matrix2_=np.dot(P, P)

print('Стохастическая матрица второго порядка:')
print(Stochastic_matrix2)
print(Stochastic_matrix2_)

Stochastic_matrix = np.zeros((256, 256))
suma_row = 0
print('Стохастическая матрица R -> L: ')
for i in range(height):
    for j in range(width - 1):
        k1 = r_new[i][width - j - 1]
        k2 = r_new[i][width - j - 2]
        Stochastic_matrix[k1-1][k2-2] += 1

for i in range(256):
    for j in range(256):
        suma_row += Stochastic_matrix[i][j]
    Stochastic_matrix[i] /= suma_row
    suma_row = 0
print(Stochastic_matrix)

P = np.zeros((256, 256))
sum_row = 0
print('P')
for i in range(len(r1)):
    for j in range(len(r1[0]) - 1):
        k1 = r1[i][j]
        k2 = r1[i][j + 1]
        P[k1-1][k2-1] += 1


for i in range(256):
    for j in range(256):
        sum_row += P[i][j]
    if sum_row == 0:
        P[i] = 0
    else:
        P[i] /= sum_row
    sum_row = 0
print(P)

Stochastic_matrix2=np.dot(Stochastic_matrix, P)
Stochastic_matrix2_=np.dot(P, P)

print('Стохастическая матрица второго порядка:')
print(Stochastic_matrix2)
print(Stochastic_matrix2_)

count=0
for k in range(6):
    reg=matrix_power(Stochastic_matrix, k)
    for i in range(256):
            if reg[i][i] == 0:
                count += 1
if count == 0:
    print('Свойство регулярности выполняется')
    print('Свойство рекуррентности выполняется')
else:
    print('Свойство регулярности не выполняется')
    print('Свойство рекуррентности не выполняется')

count=0
for i in range(256):
    for j in range(256):
        if (Stochastic_matrix[i][i] == 1) or (Stochastic_matrix[i][j] == 0):
            count +=1
if count == 0:
    print('Свойство необратимости выполняется')
else:
    print('Свойство необратимости не выполняется')


count=0
for k in range(6):
    reg=matrix_power(Stochastic_matrix2_, k)
    for i in range(256):
            if reg[i][i] == 0:
                count += 1
if count == 0:
    print('Свойство регулярности выполняется')
    print('Свойство рекуррентности выполняется')
else:
    print('Свойство регулярности не выполняется')
    print('Свойство рекуррентности не выполняется')

count=0
for i in range(256):
    for j in range(256):
        if (Stochastic_matrix2_[i][i] == 1) or (Stochastic_matrix2_[i][j] == 0):
            count +=1
if count == 0:
    print('Свойство необратимости выполняется')
else:
    print('Свойство необратимости не выполняется')

print("All is OK")
