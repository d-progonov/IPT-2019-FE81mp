import numpy as np
import scipy as sp
from scipy import stats
from PIL import Image
import matplotlib.pyplot as plt
from fitter import Fitter
import os

DIRNAME = '/home/dimas/workspace/my/python/img'
COLOR = {'red': 0,'green': 1,'blue': 2}
number_of_image = 15

image_names = []

for image in os.listdir(DIRNAME):
    print(image)
    image_names.append(DIRNAME+'/'+image)
print(image_names)

for image_name in image_names[36:37]:
    for name, number in COLOR.items():
        image = np.array(Image.open(image_name))
        a = image[:, :, number].ravel()
        d = {'name': image_name, 'min': np.min(a), 'max': np.max(a), 'mean_val': np.mean(a),
             'variance': np.var(a), 'mediana': np.median(a), 'inter0quartile ': sp.stats.iqr(a, rng=(25, 75)),
             'skewness': sp.stats.skew(a), 'kurtosis': sp.stats.kurtosis(a), 'summmm':sum(a)}

        print('channel: {} channel and extracting data {}'.format(name,d))
        plt.figure(figsize=(10,5))
        f = Fitter(a, distributions=['beta', 'gamma', 'laplace', 'norm', 'uniform'],bins=256,verbose=True)
        f.fit()
        f.summary()
        f.hist()
    plt.show()

