import os
from functions import count_markov_matrixes,basic_analysis, gaussian_model, read, im_PCA
import numpy as np
from Markov import MarkovChain
from PIL import Image
import cv2
_MARKOV_IM_NUMBER = 6
_TO_SHOW_GAUSS_MODELS = True

def check_matrix(matrix):
    print("Has Reverse Matrix?")
    print(np.linalg.det(matrix) == 0)
    print("Is min element > 0?")
    print(np.min(matrix) > 0)
    print("Is diag > 0?")
    print(np.min(np.diagonal(matrix)) > 0)

def multiply_matrixes(X,Y):
    result = X.dot(Y)
    return result

#all_errors = []
#best_errors = {'Norm': 0, 'Gamma': 0, 'Beta': 0, 'Uniform': 0}
results = []
directory = '/Users/mariatuchkova/PycharmProjects/Images_project 2/images_1/'
images = []
flatten_images = []
heights = []
widths = []

for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        print(filename)
        im = read(directory+filename)
        images.append(im)
        heights.append(len(im))
        widths.append(len(im[0]))

h = min(heights)
w = min(widths)
l = min(h,w)

new_images = []
for image in images:
    new_images.append(image[:l,:l,:])


for im in new_images:
    res, flatten = basic_analysis(im, 2)
    if _TO_SHOW_GAUSS_MODELS:
        [gaussian_model(vector) for vector in res['param_vectors']]
    print(res)
    for color in flatten:
        flatten_images.append(color)
    results.append(res)


im_PCA(flatten_images, height = l, length=l)



counter = 0
path = "/Users/mariatuchkova/PycharmProjects/Images_project 2/markov_matrixes/"
image = images[_MARKOV_IM_NUMBER]
colors = cv2.split(image)
for color in colors:
    res = count_markov_matrixes(color)
    for matrix in res:
        np.savetxt(path+ "matrix_" + str(counter) + ".csv",matrix.astype(float), delimiter = ",")
        print("Iteration 1")
        check_matrix(matrix)

        print("Iteration 2")
        matrix2 = multiply_matrixes(matrix,matrix)
        check_matrix(matrix2)

        print("Iteration 3")
        matrix3 = multiply_matrixes(matrix2,matrix)
        check_matrix(matrix3)

        print("Iteration 4")
        matrix4 = multiply_matrixes(matrix3,matrix)
        check_matrix(matrix4)

        print("Iteration 5")
        matrix5 = multiply_matrixes(matrix4,matrix)
        check_matrix(matrix5)
        print("#################")

        counter+=1
    print("#######################################################################################")


