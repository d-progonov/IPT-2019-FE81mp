from math import sqrt, pi, exp,log1p
import numpy as np
import scipy.stats as st
from scipy import linalg
from pylab import *
from PIL import Image, ImageDraw


DIRNAME = '/home/alex/Стільниця/mirflickr/'


COLOR = {'red': 0,
         'green': 1,
         'blue': 2}  # RGB

number_of_image = 100
image_names = []
np.seterr(divide='ignore', invalid='ignore')
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

for i in range(number_of_image):
    image_names.append('/home/alex/Стільниця/mirflickr/im'+str(i+1)+'.jpg')


textfile = open('laba4.txt', 'w')

MEAN_VECTOR_R = []
MEAN_VECTOR_G = []
MEAN_VECTOR_B = []

#DISPERS
VAR_VECTOR_R = []
VAR_VECTOR_G = []
VAR_VECTOR_B = []

#ASSYMETRY
SKEW_VECTOR_R = []
SKEW_VECTOR_G = []
SKEW_VECTOR_B = []

#EKSCCES
KURT_VECTOR_R = []
KURT_VECTOR_G = []
KURT_VECTOR_B = []


def get_array(n):
    array = [[0 for i in range(n)] for i in range(n)]
    return array


def get_RGB(image):
    width = image.size[0]  # Определяем ширину.
    height = image.size[1]  # Определяем высоту.
    pix = image.load()  # Выгружаем значения пикселей.
    RGB = [[], [], []]
    for i in range(width):
        for j in range(height):
            a = pix[i, j][0]
            b = pix[i, j][1]
            c = pix[i, j][2]
            RGB[0].append(a)
            RGB[1].append(b)
            RGB[2].append(c)
    return RGB


def descript(x):
    result = []
    result.append(np.mean(x))  # среднее
    result.append(np.var(x))  # дисперсия
    result.append(st.skew(x))  # асимметрия
    result.append(st.kurtosis(x))  # эксцесс
    #print('result',result)
    textfile.write('\nMean  {}:  \nVariance  {}   \nAssymetry  {}  \nKurtosis  {} \n'.format(str(result[0]),str(result[1]),str(result[2]),str(result[3])))
    return tuple(result)


def main():
    #textfile = open('laba2.txt', 'w')
    # --------Gauss------------
    print('------------------Task 1------------------')
    for j in range(1, number_of_image+1):
        image = Image.open('/home/alex/Стільниця/mirflickr/im{}.jpg'.format(j))
        RGB = get_RGB(image)
        textfile.write(str("\n"))
        textfile.write('Image number :  {}: \n  '.format(str(j)))
        mean0, var0, skew0, kurt0 = descript(RGB[0])
        mean1, var1, skew1, kurt1 = descript(RGB[1])
        mean2, var2, skew2, kurt2 = descript(RGB[2])

        MEAN_VECTOR_R.append(mean0)
        MEAN_VECTOR_G.append(mean1)
        MEAN_VECTOR_B.append(mean2)

        VAR_VECTOR_R.append(var0)
        VAR_VECTOR_G.append(var1)
        VAR_VECTOR_B.append(var2)

        SKEW_VECTOR_R.append(skew0)
        SKEW_VECTOR_G.append(skew1)
        SKEW_VECTOR_B.append(skew2)

        KURT_VECTOR_R.append(kurt0)
        KURT_VECTOR_G.append(kurt1)
        KURT_VECTOR_B.append(kurt2)

    M = np.cov(np.vstack((MEAN_VECTOR_R, MEAN_VECTOR_G, MEAN_VECTOR_B)))
    #M1 = np.vstack((MEAN_VECTOR_R, MEAN_VECTOR_G, MEAN_VECTOR_B))
    M_D = np.cov(np.vstack((MEAN_VECTOR_R, MEAN_VECTOR_G, MEAN_VECTOR_B, VAR_VECTOR_R, VAR_VECTOR_G, VAR_VECTOR_B)))
    M_D_Skew = np.cov(np.vstack((MEAN_VECTOR_R, MEAN_VECTOR_G, MEAN_VECTOR_B, VAR_VECTOR_R, VAR_VECTOR_G, VAR_VECTOR_B,
                                 SKEW_VECTOR_R, SKEW_VECTOR_G, SKEW_VECTOR_B)))
    M_D_Skew_Kurt = np.cov(np.vstack((MEAN_VECTOR_R, MEAN_VECTOR_G, MEAN_VECTOR_B, VAR_VECTOR_R, VAR_VECTOR_G,
                                      VAR_VECTOR_B, SKEW_VECTOR_R, SKEW_VECTOR_G, SKEW_VECTOR_B, KURT_VECTOR_R,
                                      KURT_VECTOR_G, KURT_VECTOR_B)))
    print('Матрица ковариации для векторов М: ')
    print(M)
    print()

    print('Матрица ковариации для векторов М и D: ')
    print(M_D)
    print()

    print('Матрица ковариации для векторов М, D, Skew: ')
    print(M_D_Skew)
    print()

    print('Матрица ковариации для векторов М, D, Skew, Kurt: ')
    print(M_D_Skew_Kurt)
    print()
    print(M_D_Skew_Kurt.shape)

    print('------------------Task 2------------------')
    image = Image.open('/home/alex/Стільниця/mirflickr/im6.jpg')
    width = image.size[0]  # 341
    height = image.size[1]  # 500
    print('wiiiiiiiiddddddddtttttttthhhhhhhhhhhh',width,height)
    draw = ImageDraw.Draw(image)
    pix = image.load()
    R_ch_RGB = [[0 for i in range(height)] for i in range(width)]  # Создаем пустой список нужного размера
    for i in range(width):
        for j in range(height):
            R_ch_RGB[i][j] = pix[i, j][0]

    U, S, Vt = linalg.svd(R_ch_RGB) #метод главн компонент
    print('uuuuuuuuuuuuuuuuuuu',S.shape)
    # restored_matrix = np.dot(U, S, Vt)
    S_add = np.zeros((width, height))  # Создаем пустую матрицу для сингулярных чисел
    for i in range(width):
        for j in range(height):
            if i == j:
                S_add[i][j] = S[i]
    print('SS[iii]',S_add)

    number = 341
    for i in range(number, width):
        for j in range(number, height):
            S_add[i][j] = 0
    US = np.dot(U, S_add)
    restored_matrix = np.dot(US, Vt)

    # Восстанавливаем картинку
    for i in range(width):
        for j in range(height):
            a = int(restored_matrix[i][j])
            b = pix[i, j][1]
            c = pix[i, j][2]
            draw.point((i, j), (a, b, c))

    image.save("Restored3.jpg", "JPEG")
    del draw

    error_count=[]
    error_count2=[]

    for number in range(min(width, height)+1):
        S_add = np.zeros((width, height))  # Create an empty matrix for singular numbers
        for i in range(width):
            for j in range(height):
                if i == j:
                    S_add[i][j] = S[i]
        for i in range(number, width):
            for j in range(number, height):
                S_add[i][j] = 0
        US = np.dot(U, S_add)
        restored_matrix = np.dot(US, Vt)  # dot of 2 array
        #textfile.write("Restored matrix :\n{}\n".format(str(restored_matrix)))
        total_error = 0
        for i in range(width):
            for j in range(height):
                diff = abs(R_ch_RGB[i][j] - restored_matrix[i][j])
                total_error += diff
        EPS = total_error / (width * height) * 100
        error_count.append(log1p(EPS))
        error_count2.append(EPS)
        print('Ошибка исходной матрици и полученой: ',EPS)
    x_number_list = list(range(0, min(width, height)+1))
    plt.plot(x_number_list, error_count, linewidth=2)
    plt.grid()
    plt.show()
    plt.plot(x_number_list, error_count2, linewidth=2)
    plt.grid()
    plt.show()

    print('------------------Task 3------------------')
    # Стохастическая матрица с лева на право
    Stochastic_matrix = np.zeros((256, 256))
    suma_row = 0
    sum1 = 0
    print('Стохастическая матрица L -> R: ')
    for i in range(width):
        for j in range(height - 1):
            k1 = R_ch_RGB[i][j]
            k2 = R_ch_RGB[i][j + 1]
            Stochastic_matrix[k1][k2] += 1

        # Преобразование в нормальный вид
    for i in range(256):
        for j in range(256):
            suma_row += Stochastic_matrix[i][j]
        Stochastic_matrix[i] /= suma_row
        suma_row = 0
    print(Stochastic_matrix)
    print()

    # Сума элементов разделенная на количество элементов
    for i in range(256):
        for j in range(256):
            sum1 += Stochastic_matrix[i][j]
    print('Сума элементов разделенная на количество строк = ', sum1 / (256))
    print()

    # Стохастическая матрица с права на лево
    Stochastic_matrix = np.zeros((256, 256))
    suma_row = 0
    sum1 = 0
    print('Стохастическая матрица R -> L: ')
    for i in range(width):
        for j in range(height - 1):
            k1 = R_ch_RGB[i][height - j - 2]
            k2 = R_ch_RGB[i][height - j - 1]
            Stochastic_matrix[k1][k2] += 1

        # Преобразование в нормальный вид
    for i in range(256):
        for j in range(256):
            suma_row += Stochastic_matrix[i][j]
        Stochastic_matrix[i] /= suma_row
        suma_row = 0
    print(Stochastic_matrix)
    # Сума элементов разделенная на количество элементов
    for i in range(256):
        for j in range(256):
            sum1 += Stochastic_matrix[i][j]
    print('Сума элементов разделенная на количество строк = ', sum1 / (256))

    textfile.write('Матрица ковариации для векторов М {}: \n'.format(str(M)))
    textfile.write('\n')
    textfile.write('Матрица ковариации для векторов М_D {} \n: '.format(str(M_D)))
    textfile.write('\n')
    textfile.write('Матрица ковариации для векторов М_D_Skew:{}  \n'.format(str(M_D_Skew)))
    textfile.write('\n')
    textfile.write('Матрица ковариации для векторов М_D_Skew_KUrt:{} \n'.format(str(M_D_Skew_Kurt)))
    textfile.write('\n')
    textfile.write('\n LAst task with stohastil matrix \n')
    textfile.write(str(Stochastic_matrix)+'\n')
    Stochastic_matrix5=np.linalg.matrix_power(Stochastic_matrix,5)
    textfile.write('Stochastic matrix after 5 intergration:\n{}\n'.format(str(Stochastic_matrix5)))

    print('-------------------Irreducible--------------------')
    new_matrix=Stochastic_matrix5.flatten()
    l = new_matrix.min()
    print('minimum value all matrix element ',l)


    print('-------------------Recurrent--------------------')
    matr_list2=[]
    for i in range(256):
        matr_list2.append(Stochastic_matrix5[i][i])
    print('matrix diagonal minimum value',min(matr_list2))
    print('matrix list value',matr_list2)


if __name__ == '__main__':
    main()