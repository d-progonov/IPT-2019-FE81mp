#!/usr/bin/env python
# coding: utf-8

import csv
import os

import numpy as np
import scipy.stats

from matplotlib import pyplot as plt
from PIL import Image


DISTRIBUTIONS = ["beta", "norm", "expon", "uniform", "laplace"]
RGB = ["r", "g", "b"]
fieldnames = [
    "r_max", "g_max", "b_max",
    "r_min", "g_min", "b_min",
    "r_mean", "g_mean", "b_mean",
    "r_var", "g_var", "b_var",
    "r_median", "g_median", "b_median",
    "r_iqr", "g_iqr", "b_iqr",
    "r_skewness", "g_skewness", "b_skewness",
    "r_kurtosis", "g_kurtosis", "b_kurtosis",
    "mse_norm", "mse_uniform", "mse_beta", "mse_laplace"
]


class Img(object):

    def __init__(self, array):
        self.img = array

    @property
    def im_max(self):
        return [np.amax(self.img[:, :, ch]) for ch in range(self.img.shape[-1])]

    @property
    def im_min(self):
        return [np.amin(self.img[:, :, ch]) for ch in range(self.img.shape[-1])]

    @property
    def im_mean(self):
        return [np.mean(self.img[:, :, ch]) for ch in range(self.img.shape[-1])]

    @property
    def im_var(self):
        return [np.var(self.img[:, :, ch]) for ch in range(self.img.shape[-1])]

    @property
    def im_median(self):
        return [np.median(self.img[:, :, ch]) for ch in range(self.img.shape[-1])]

    @property
    def im_iqr(self):
        return [scipy.stats.iqr(self.img[:, ch]) for ch in range(self.img.shape[-1])]

    @property
    def im_skewness(self):
        return [scipy.stats.skew(self.img[:, :, ch], axis=None) for ch in range(self.img.shape[-1])]

    @property
    def im_kurtosis(self):
        return [scipy.stats.kurtosis(self.img[:, :, ch], axis=None) for ch in range(self.img.shape[-1])]


def get_properties(img):
    result = dict()
    for prop in dir(img):

        if prop.startswith("im_"):
            r_res, g_res, b_res = getattr(img, prop)
            result.update({f"r_{prop[3:]}": r_res, f"g_{prop[3:]}": g_res, f"b_{prop[3:]}": b_res})

    return result


def get_distributions(img, plotting=False):
    result = dict()
    if plotting:
        fig = plt.figure()
    for ch in range(img.img.shape[-1]):
        flatten = img.img[ch].flatten()
        lnspc = np.linspace(0, 255, 256)
        hist = np.histogram(flatten, bins=lnspc, density=True)
        if plotting:
            plt.subplot(1, 3, ch + 1)
            plt.hist(flatten, lnspc, density=True, color=RGB[ch])
            plt.ylim(0, 0.025)

    for dist in DISTRIBUTIONS:
        if plotting:
            fig = plt.figure()
        mse = list()
        for ch in range(img.img.shape[-1]):
            flatten = img.img[ch].flatten()
            lnspc = np.linspace(0, 255, 256)
            hist = np.histogram(flatten, bins=lnspc, density=True)
            if plotting:
                plt.subplot(1, 3, ch + 1)
                plt.hist(flatten, lnspc, density=True, color=RGB[ch])

            # lets try the normal distribution first
            distribution = getattr(scipy.stats, dist)

            args = distribution.fit(flatten) # get mean and standard deviation
            for b in range(len(hist[0])):
                y = hist[0][b]
                expected = distribution.pdf(hist[1][b], *args)
                err = np.power(y - expected, 2.0)
                mse.append(err)


            pdf_g = distribution.pdf(lnspc, *args) # now get theoretical values in our interval

            if plotting:
                plt.plot(lnspc, pdf_g, label="Norm") # plot it
                plt.suptitle(dist)
                plt.ylim((0, 0.025))
        result.update({f"mse_{dist}": np.mean(np.sum(mse))})

    if plotting:
        plt.show()
    return result

images_dir = "../Images/"
test_image = "im24880.jpg"

image = np.array(Image.open(f"{images_dir}{test_image}"))
img = Img(image)
prop = get_properties(img)
for p in prop.keys():
    print(f"{p}: {prop[p]}")
print(get_distributions(img, True))

if __name__ == "__main__":
    distributions = {"norm": 0, "expon": 0, "uniform": 0, "laplace": 0, "beta": 0}
    with open("bohaichuk_lr1.csv", "w") as f:
        csvfile = csv.DictWriter(f, fieldnames=fieldnames)
        csvfile.writeheader()
        for image in os.listdir(images_dir):
            result = dict()
            img = Img(np.array(Image.open(f"{images_dir}{image}")))
            result.update(get_properties(img))
            d = get_distributions(img)

            l = [d["mse_norm"], d["mse_expon"], d["mse_beta"], d["mse_laplace"], d["mse_uniform"]]
            if np.nan in l:
                result.update(d)
                csvfile.writerow(result)
                continue
            ind = l.index(min(l))

            print(f"{l} {ind}")
            if ind == 0:
                distributions["norm"] += 1
            elif ind == 1:
                distributions["expon"] += 1
            elif ind == 2:
                distributions["beta"] += 1
            elif ind == 3:
                distributions["laplace"] += 1
            else:
                distributions["uniform"] += 1
            result.update(d)
            csvfile.writerow(result)
    print(distributions)
