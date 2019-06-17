import os
from collections import defaultdict

import numpy as np
import scipy as sp
from scipy import misc, stats
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import plotly.plotly as py
import seaborn as sns
from pylab import *
import sys
import threading
from datetime import datetime
from itertools import zip_longest
from PIL import Image, ImageDraw
import random

DIRNAME = 'E:/pr/m/'
COLOR = {'red': 0,
         'green': 1,
         'blue': 2} 


with open('E:/pr/random.txt') as f:
    image_names = ['im'+ x.strip()+'.jpg' for x in f.readlines()]


print (len(image_names))

MEAN_VECTOR_R = []
MEAN_VECTOR_G = []
MEAN_VECTOR_B = []

VAR_VECTOR_R = []
VAR_VECTOR_G = []
VAR_VECTOR_B = []

SKEW_VECTOR_R = []
SKEW_VECTOR_G = []
SKEW_VECTOR_B = []

KURT_VECTOR_R = []
KURT_VECTOR_G = []
KURT_VECTOR_B = []
data = {}
for name, num in COLOR.items():
    data[name] = pd.DataFrame()
    for image_name in image_names[:100]:
        image = np.array(Image.open(DIRNAME+image_name))
        a = image[:, :, num].ravel()
        d = {'col': name,
             'n': image_name,
             'mean': np.mean(a),
             'var': np.var(a),
             'skewness': sp.stats.skew(a),
             'kurtosis': sp.stats.kurtosis(a)}
        data[name] = pd.concat([data[name], pd.DataFrame(pd.DataFrame(d, index=[0,]))], ignore_index=True)
        if d['col']=='red':
            MEAN_VECTOR_R.append(round(d['mean']))
        elif d['col']=='green':
            MEAN_VECTOR_G.append(round(d['mean']))
        else :
            MEAN_VECTOR_B.append(round(d['mean']))

        if d['col']=='red':
            VAR_VECTOR_R.append(round(d['var'],3))
        elif d['col']=='green':
            VAR_VECTOR_G.append(round(d['var'],3))
        else :
            VAR_VECTOR_B.append(round(d['var'],3))

        if d['col']=='red':
            SKEW_VECTOR_R.append(round(d['skewness'],3))
        elif d['col']=='green':
            SKEW_VECTOR_G.append(round(d['skewness'],3))
        else :
            SKEW_VECTOR_B.append(round(d['skewness'],3))

        if d['col']=='red':
            KURT_VECTOR_R.append(round(d['kurtosis'],3))
        elif d['col']=='green':
            KURT_VECTOR_G.append(round(d['kurtosis'],3))
        else :
            KURT_VECTOR_B.append(round(d['kurtosis'],3))

MATRICA_MEAN_ARRAY = np.array((MEAN_VECTOR_R, MEAN_VECTOR_G, MEAN_VECTOR_B))
MATRICA_VAR_ARRAY = np.array((MEAN_VECTOR_R, MEAN_VECTOR_G, MEAN_VECTOR_B, VAR_VECTOR_R, VAR_VECTOR_G, VAR_VECTOR_B))
MATRICA_SKEW_ARRAY = np.array((MEAN_VECTOR_R, MEAN_VECTOR_G, MEAN_VECTOR_B, VAR_VECTOR_R, VAR_VECTOR_G, VAR_VECTOR_B, SKEW_VECTOR_R, SKEW_VECTOR_G, SKEW_VECTOR_B))
MATRICA_KURT_ARRAY = np.array((MEAN_VECTOR_R, MEAN_VECTOR_G, MEAN_VECTOR_B, VAR_VECTOR_R, VAR_VECTOR_G, VAR_VECTOR_B, SKEW_VECTOR_R, SKEW_VECTOR_G, SKEW_VECTOR_B, KURT_VECTOR_R, KURT_VECTOR_G, KURT_VECTOR_B))


print('Матрица Мат.Ожидания')
print(MATRICA_MEAN_ARRAY)

MM=np.array([np.mean(MATRICA_MEAN_ARRAY[0]),np.mean(MATRICA_MEAN_ARRAY[1]),np.mean(MATRICA_MEAN_ARRAY[2])])
print('Мат. Ожидание матрицы  Мат.Ожидания')
print(MM)
MATRICA_MEAN_ARRAY_COV = np.cov(np.vstack((MEAN_VECTOR_R, MEAN_VECTOR_G, MEAN_VECTOR_B)))
print('Матрица ковариации Мат.Ожидания')
print(MATRICA_MEAN_ARRAY_COV)



print('Матрица Мат.Ожидания и дисперсии')
print(MATRICA_VAR_ARRAY)

MATRICA_MATRICA_VAR_COV = np.array(np.cov(np.vstack((MEAN_VECTOR_R, MEAN_VECTOR_G, MEAN_VECTOR_B, VAR_VECTOR_R, VAR_VECTOR_G, VAR_VECTOR_B))))
print('Матрица ковариации Мат.Ожидания и дисперсии')
print((MATRICA_MATRICA_VAR_COV))


print('Матрица Мат.Ожидания ,дисперсии и ексцесса')
print(MATRICA_SKEW_ARRAY)

MATRICA_SKEW_ARRAY_COV = np.array(np.cov(np.vstack((MEAN_VECTOR_R, MEAN_VECTOR_G, MEAN_VECTOR_B, VAR_VECTOR_R, VAR_VECTOR_G, VAR_VECTOR_B, SKEW_VECTOR_R, SKEW_VECTOR_G, SKEW_VECTOR_B))))
print('Матрица ковариации Мат.Ожидания ,дисперсии и ексцесса')
print(MATRICA_SKEW_ARRAY_COV)



print('Матрица Мат.Ожидания ,дисперсии, ексцесса и куртузиса')
print(MATRICA_KURT_ARRAY)

MATRICA_KURT_ARRAY_COV = np.cov(np.vstack((MEAN_VECTOR_R, MEAN_VECTOR_G, MEAN_VECTOR_B, VAR_VECTOR_R, VAR_VECTOR_G, VAR_VECTOR_B, SKEW_VECTOR_R, SKEW_VECTOR_G, SKEW_VECTOR_B, KURT_VECTOR_R, KURT_VECTOR_G, KURT_VECTOR_B)))
print('Матрица ковариации Мат.Ожидания ,дисперсии, ексцесса и куртузиса')
print(MATRICA_KURT_ARRAY_COV)

for image_name in image_names[2:3]:
        image = np.array(Image.open(DIRNAME+image_name))
        print(len(image)) 
        print(len(image[0])) 
        r = image[:, :, 0].ravel()
        g = image[:, :, 1].ravel()
        b = image[:, :, 2].ravel()

print(r)

r=r.reshape(500,341)
print (r)


U, s, V = np.linalg.svd(r)
UU=U.shape
VV=V.shape
SS = len(s)
print('Матрица U'',', UU)
print (np.array(U))
print('Список s'',',SS)
print (np.array(s))
print('Матрица V'',',VV)
print (np.array(V))


nulevaya_matriz = np.zeros((len(r),len(r[0])))
for i in range(len(r)):
    for j in range(len(r[0])):
        if i==j:
            nulevaya_matriz[i][j] = s[i]


el = 0
for i in range(el,len(r)):
    for j in range(el,len(r[0])):
        nulevaya_matriz[i][j] = 0
US = np.dot(U, nulevaya_matriz)
restored_matrix = np.dot(US, V)
print (restored_matrix)

rm=restored_matrix.transpose()
image = Image.open('E:/pr/m/im6.jpg')
draw = ImageDraw.Draw(image)
pix = image.load()
for i in range(len(r[0])):
    for j in range(len(r)):
        a = int(rm[i][j])
        b = pix[i, j][1]
        c = pix[i, j][2]
        draw.point((i, j), (a,b,c))
image.save("E:/laba2(0).jpg", "JPEG")
del draw

Q=np.array([len(r[0]),len(r)])
QQ=min(Q)

def Metric_MSE(C, S):
    EPS=np.power(C - S, 2).sum() / C.size
    return EPS
EPS = []
Error = []
el=0
while el < QQ:
    for i in range(len(r)):
        for j in range(len(r[0])):
            if i==j:
                nulevaya_matriz[i][j] = s[i]
    for i in range(el,len(r)):
        for j in range(el,len(r[0])):
            nulevaya_matriz[i][j] = 0
        
    US = np.dot(U, nulevaya_matriz)
    restored_matrix = np.dot(US, V)
    print ('MSE:',Metric_MSE(r,restored_matrix))
    Error.append(Metric_MSE(r,restored_matrix))
    el +=1

X = np.arange(0,QQ,1)
plt.figure()
plt.semilogy(X, Error)
plt.ylabel('MSE')
plt.xlabel('Components')
plt.grid(True)
plt.show()

for image_name in image_names[2:3]:
        image = np.array(Image.open(DIRNAME+image_name))
        print(len(image)
        print(len(image[0])) 
        r = image[:, :, 0].ravel()
        g = image[:, :, 1].ravel()
        b = image[:, :, 2].ravel()
print(r.shape)
def transition_matrix(transitions):
    n = 1+ max(transitions) 
    M = [[0]*n for _ in range(n)]
    for (i,j) in zip(transitions,transitions[1:]):
        M[i][j] += 1
    
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
    return M

def transition_matrix_arabic(transitions):
    n = 1+ max(transitions) 
    M = [[0]*n for _ in range(n)]
    for (i,j) in zip(transitions,transitions[1:] - 1):
        M[i][j - 1] += 1
   
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
    return M
b=r.reshape(500,341)
def transition_matrix_up(transitions):
    n=256
    M = [[0]*n for _ in range(n)]
    for i in range(len(transitions)):
        if i!=len(transitions)-1:
            for j in range(len(transitions[0])):
                k1 = transitions[i][j]
                k2 = transitions[i+1][j ]
                M[k1-1][k2-1] += 1
                
        else:
            j=0
            for j in range(len(transitions[0])-1):
                k1 = transitions[i][j]
                k2 = transitions[0][j+1 ]       
                M[k1-1][k2-1] += 1
                
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
                
    return M
def transition_matrix_down(transitions):
    n=256
    M = [[0]*n for _ in range(n)]
    for i in range(len(transitions)):
        if i!=len(transitions)-1:
            for j in range(len(transitions[0])):
                k1 = transitions[len(transitions)-1-i][j]
                k2 = transitions[len(transitions)-2-i][j ]
                M[k1-1][k2-1] += 1
                
        else:
            j=0
            for j in range(len(transitions[0])-1):
                k1 = transitions[len(transitions)-1-i][j]
                k2 = transitions[len(transitions)-i][j+1 ]       
                M[k1-1][k2-1] += 1
            
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
                
    return M


stohastic_marix = np.array(transition_matrix(r))
print('Стохастическая матрица с право на лево' )
print(stohastic_marix)


stohastic_marix_arabic = np.array(transition_matrix_arabic(r))
print('Стохастическая матрица с лево на право')
print(stohastic_marix_arabic)
      
stohastic_marix_up = np.array(transition_matrix_up(b))
print('Стохастическая матрица снизу вверх' )
print(stohastic_marix_up)

stohastic_marix_down = np.array(transition_matrix_down(b))
print('Стохастическая матрица сверху вниз' )
print(stohastic_marix_down)


def rekurentnost(x):  
    d = [x[i][i] for i in range(len(x))]
    if 0 not in d:
            print('Матрица реккурентная')
    else:
        print('В диагонале матрицы есть 0')

rekurentnost(stohastic_marix)

def neobratiomost(x):
    nol = 0;
    count = 0
    for i in range(len(x)):
        for j in range(len(x[0])):
            if x[i][j] == nol:
                count += 1
                 
    if count == 0:
        print('Матрица не обратимая во времени')       
    else:
        print('Матрица обратимая во времени')
        
neobratiomost(stohastic_marix)

def reguliarnost(x):  
    y = listmerge1(x)
    if 0 in y:
        print('В матрице есть нулевые елементы')
    else:
        print('в матрице нет нулевых елементов')

reguliarnost(stohastic_marix)

Rr=np.dot(stohastic_marix,stohastic_marix)
rekurentnost(Rr)
neobratiomost(Rr)
reguliarnost(Rr)

Rrr=np.dot(Rr,stohastic_marix)
rekurentnost(Rrr)
neobratiomost(Rrr)
reguliarnost(Rrr)

Rrrr=np.dot(Rrr,stohastic_marix)
rekurentnost(Rrrr)
neobratiomost(Rrrr)
reguliarnost(Rrrr)

Rrrrr=np.dot(Rrrr,stohastic_marix)
rekurentnost(Rrrrr)
neobratiomost(Rrrrr)
reguliarnost(Rrrrr)
