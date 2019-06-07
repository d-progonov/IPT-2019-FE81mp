import random as rand
import os
from collections import defaultdict

import numpy as np
import scipy as sp
from scipy import stats
import pandas as pd
from PIL import Image
from fitter import Fitter


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


    print('Значения RED')
    print(data['red'])

    print('Значения GREEN')
    print(data['green'])

    print('Значения BLUE')
    print(data['blue'])

    hist_min = {'red': defaultdict(int), 'green': defaultdict(int), 'blue': defaultdict(int), }
    hist_max = {'red': defaultdict(int), 'green': defaultdict(int), 'blue': defaultdict(int), }

    for name, num in RGB.items():
        for image_name in file_list_rand:
            plt.figure()
            image = np.array(Image.open(os.path.join(image_name)))
            a = image[ :, num].ravel()
            f = Fitter(a, distributions=['beta', 'gamma', 'laplace', 'norm', 'pareto'], bins=256)
            f.fit()
            f.summary()
            f.hist();
            hist_min[name][f.df_errors['sumsquare_error'].idxmin()] += 1
            hist_max[name][f.df_errors['sumsquare_error'].idxmax()] += 1
            plt.show()
print("All is OK")
