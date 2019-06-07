import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.stats import kurtosis as kurt, skew
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import pandas as pd

_TO_SHOW_COMPRESSED_IMAGES = False

def approximate_hist(data, to_show = True):
    error = {}

    # plot normed histogram
    y, x = np.histogram(data, bins=np.arange(-0.5,256.5))

    # find minimum and maximum of xticks, so we know
    # where we should compute theoretical distribution
    #xt = plt.xticks()[0]
    #xmin, xmax = np.min(xt), np.max(xt)
    lnspc = np.linspace(0, 255, 256)

    y = y/len(data)
    print("linspace done")

    # lets try the normal distribution first
    m, s = stats.norm.fit(data)  # get mean and standard deviation
    pdf_g = stats.norm.pdf(lnspc, m, s)  # now get theoretical values in our interval
    print("norm done")

    # error['Norm'] = np.sum(np.power(y - pdf_g, 2.0))

    # exactly same as above
    ag, bg, cg = stats.gamma.fit(data)
    pdf_gamma = stats.gamma.pdf(lnspc, ag, bg, cg)
    print("gamma done")


    ab, bb, cb, db = stats.beta.fit(data)
    pdf_beta = stats.beta.pdf(lnspc, ab, bb, cb, db)
    print("beta done")


    mu, std = stats.uniform.fit(data)
    pdf_uniform = stats.uniform.pdf(lnspc, mu, std)


    print("uniform done")
    error = {'Norm' : mean_squared_error(y, pdf_g),
             'Gamma': mean_squared_error(y, pdf_gamma),
             'Beta': mean_squared_error(y, pdf_beta),
             'Uniform' : mean_squared_error(y, pdf_uniform)}


    if to_show:
        lnspc = np.linspace(0, 255, 10000)

        y = y / len(data)
        print("linspace done")

        # lets try the normal distribution first
        m, s = stats.norm.fit(data)  # get mean and standard deviation
        pdf_g = stats.norm.pdf(lnspc, m, s)  # now get theoretical values in our interval
        print("norm done")



        ag, bg, cg = stats.gamma.fit(data)
        pdf_gamma = stats.gamma.pdf(lnspc, ag, bg, cg)
        print("gamma done")


        ab, bb, cb, db = stats.beta.fit(data)
        pdf_beta = stats.beta.pdf(lnspc, ab, bb, cb, db)
        print("beta done")


        mu, std = stats.uniform.fit(data)
        pdf_uniform = stats.uniform.pdf(lnspc, mu, std)
        plt.hist(data, normed=True, bins=40)
        plt.plot(lnspc, pdf_g, label="Norm")  # plot it
        plt.plot(lnspc, pdf_gamma, label="Gamma")
        plt.plot(lnspc, pdf_beta, label="Beta")
        plt.plot(lnspc, pdf_uniform, label="Uniform")
        plt.plot(y, color = "black")

        # error['Uniform'] = np.sum(np.power(y - pdf_uniform, 2.0))

        # print error

        plt.show()
    return error


def basic_analysis(img, mode = 1):
    res = {}


    colors = cv2.split(img)
    res['colors'] = ["blue","green","red"]
    res['std'] = [np.std(colors[0]), np.std(colors[1]), np.std(colors[2])]
    res['mean'] = [np.mean(colors[0]), np.mean(colors[1]), np.mean(colors[2])]
    flatten = [[item for sublist in colors[0] for item in sublist],
               [item for sublist in colors[1] for item in sublist],
               [item for sublist in colors[2] for item in sublist]]

    res['skewness'] = [skew(flatten[0]), skew(flatten[1]), skew(flatten[2])]
    res['kurtosis'] = [kurt(flatten[0]), kurt(flatten[1]), kurt(flatten[2])]

    if mode == 1:
        res['min'] = [np.min(colors[0]), np.min(colors[1]), np.min(colors[2])]
        res['max'] = [np.max(colors[0]), np.max(colors[1]), np.max(colors[2])]
        res['median'] = [np.median(colors[0]), np.median(colors[1]), np.median(colors[2])]
        res['percentile25'] = [[np.percentile(colors[0], 25), np.percentile(colors[0], 50), np.percentile(colors[0], 75)],
                               [np.percentile(colors[1], 25), np.percentile(colors[1], 50), np.percentile(colors[1], 75)],
                               [np.percentile(colors[2], 25), np.percentile(colors[2], 50), np.percentile(colors[2], 75)]]
        return res, flatten


    if mode == 2:
        res['param_vectors'] = []
        for i in [0,1,2]:
            param_vector = {}
            param_vector['mean'] = res['mean'][i]
            param_vector['mean_std'] = [res['mean'][i], res['std'][i]]
            param_vector['mean_std_skew'] = [res['mean'][i], res['std'][i], res['skewness'][i]]
            param_vector['mean_std_skew_kurt'] = [res['mean'][i], res['std'][i], res['skewness'][i], res['kurtosis'][i]]
            res['param_vectors'].append(param_vector)
        return res, flatten

    if mode == 3:
        return res

def read(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)


def gaussian_model(params):
    mu = params['mean_std'][0]
    sigma = params['mean_std'][1]
    skew = params['mean_std_skew'][2]
    x1 = 0
    x2 = 255

    lnspc = np.linspace(-100, 500, 6000)
    x = np.linspace(0, 255, 2550)
    y = stats.norm.pdf(x, mu, sigma)
    y2 = stats.norm.pdf(lnspc, mu, sigma)

    #y_skew = stats.skewnorm.pdf(lnspc, mu, sigma, skew)





    plt.plot(lnspc,y2,color = 'blue')
    plt.plot(x,y,color = 'black')
    #plt.plot(lnspc,y_skew,color = 'green')

    plt.show()

def im_PCA(images, height, length):
    components = range(2,202,4)

    # print(images[0])
    images = pd.DataFrame(images)
    # print(images)
    MSEs1 = []
    MSEs = []
    X_norm = normalize(images)

    for PCA_length in components:
        pca = PCA(PCA_length)
        # Run PCA on normalized image data
        lower_dimension_data = pca.fit_transform(X_norm)
        #print(lower_dimension_data.shape)
        approximation = pca.inverse_transform(lower_dimension_data)
        #print(approximation.shape)

        error = mean_squared_error(approximation, X_norm)
        print("first picture first color error:")
        print(mean_squared_error(approximation[0], X_norm[0]))
        MSEs1.append(mean_squared_error(approximation[0], X_norm[0]))
        print("general error:")
        print(error)
        MSEs.append(error)

        if _TO_SHOW_COMPRESSED_IMAGES:
            approximation_ = approximation.reshape(-1, height, length)
            X_norm_ = X_norm.reshape(-1, height, length)
            for i in range(0, X_norm_.shape[0]):
                X_norm_[i,] = X_norm_[i,].T
                approximation_[i,] = approximation_[i,].T
            fig4, axarr = plt.subplots(3, 2, figsize=(8, 8))

            axarr[0, 0].imshow(X_norm_[0,], cmap='gray')
            axarr[0, 0].set_title('Original Image')
            axarr[0, 0].axis('off')
            axarr[0, 1].imshow(approximation_[0,], cmap='gray')
            axarr[0, 1].set_title('XX% Variation')
            axarr[0, 1].axis('off')

            axarr[1, 0].imshow(X_norm_[3,], cmap='gray')
            axarr[1, 0].set_title('Original Image')
            axarr[1, 0].axis('off')
            axarr[1, 1].imshow(approximation_[3,], cmap='gray')
            axarr[1, 1].set_title('XX% Variation')
            axarr[1, 1].axis('off')

            axarr[2, 0].imshow(X_norm_[6,], cmap='gray')
            axarr[2, 0].set_title('Original Image')
            axarr[2, 0].axis('off')
            axarr[2, 1].imshow(approximation_[6,], cmap='gray')
            axarr[2, 1].set_title('XX% variation')
            axarr[2, 1].axis('off')

            plt.show()



    print("single image")
    plt.plot(components,MSEs1)
    plt.show()f
    print("all images")
    plt.plot(components,MSEs)
    plt.show()


    """
    approximation = approximation.reshape(-1, height, length)
    X_norm = X_norm.reshape(-1, height, length)
    for i in range(0,X_norm.shape[0]):
        X_norm[i,] = X_norm[i,].T
        approximation[i,] = approximation[i,].T
    fig4, axarr = plt.subplots(3,2,figsize=(8,8))
    axarr[0,0].imshow(X_norm[0,],cmap='gray')
    axarr[0,0].set_title('Original Image')
    axarr[0,0].axis('off')
    axarr[0,1].imshow(approximation[0,],cmap='gray')
    axarr[0,1].set_title('99% Variation')
    axarr[0,1].axis('off')
    axarr[1,0].imshow(X_norm[1,],cmap='gray')
    axarr[1,0].set_title('Original Image')
    axarr[1,0].axis('off')
    axarr[1,1].imshow(approximation[1,],cmap='gray')
    axarr[1,1].set_title('99% Variation')
    axarr[1,1].axis('off')
    axarr[2,0].imshow(X_norm[2,],cmap='gray')
    axarr[2,0].set_title('Original Image')
    axarr[2,0].axis('off')
    axarr[2,1].imshow(approximation[2,],cmap='gray')
    axarr[2,1].set_title('99% variation')
    axarr[2,1].axis('off')
    plt.show()
    """

def count_markov_matrixes(image):
    res = [0,0,0,0]
    for i in [0,1,2,3]:
        res[i] = np.zeros((256,256))
    #res = [np.zeros(256,256),np.zeros(256,256),np.zeros(256,256),np.zeros(256,256)]
    for i in range(image.shape[0]-1):
        for j in range(image.shape[1]-1):
            left_up = image[i, j]
            right_up = image[i, j+1]
            left_down = image[i+1, j]
            res[0][left_up, right_up] +=1
            res[1][right_up, left_up] +=1
            res[2][left_up, left_down] +=1
            res[3][left_down, left_up] +=1
    for matrix in res:
        for i in range(256):
            line = matrix[:,i]
            normalized_line = line / np.sum(line)
            matrix[:,i] = normalized_line

    return(res)
