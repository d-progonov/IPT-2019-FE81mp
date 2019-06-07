from PIL import Image
from scipy.stats import skew, kurtosis
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import sys
import statistics as stt
import matplotlib.pyplot as plt
import glob
import fitter

COLOR = {'Red': 0,
         'Green': 1,
         'Blue': 2}

# Main Function solve problem
def Solve():
    g = open('OutPut.txt', "w")
    pp = PdfPages("AllHistogram.pdf")
    count_dis={'beta':0, 'gamma':0, 'uniform':0, 'norm':0, 'laplace':0}
    np.seterr(divide='ignore', invalid='ignore')
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")  # ignore some warnings from system
    cnt = 0  # count how many images had taken
    for filename in glob.glob('D:/10semester/Progonov/Лабораторные работы/mirflickr/*.jpg') :
        photo = Image.open(filename)
        photo = photo.convert('RGB')
        g.write('Output for Image number {}\n\n'.format(cnt + 1))
        Red = []
        Green = []
        Blue = []

        width, height = photo.size  # define W and H
        for y in range(0, height):  # each pixel has coordinates
            for x in range(0, width):
                RGB = photo.getpixel((x, y))
                R, G, B = RGB  # now we can use the RGB value
                Red.append(R)
                Green.append(G)
                Blue.append(B)

        sorted(Red)
        sorted(Green)
        sorted(Blue)

        g.write('Max and min values of Red channel of image {} are: {}, {}\n'.format(cnt + 1, max(Red), min(Red)))
        g.write('Max and min values of Green channel of image {} are: {}, {}\n'.format(cnt + 1, max(Green), min(Green)))
        g.write('Max and min values of Blue channel of image {} are: {}, {}\n\n'.format(cnt + 1, max(Blue), min(Blue)))

        # for Red channel

        g.write('Sum of Red channel is : {}\n'.format(sum(Red)))
        g.write('Median of Red channel is : {}\n'.format(stt.median(Red)))
        g.write('Lower and Upper quantile of Red channel are : {} {}\n'.format(np.quantile(Red, 0.25),
                                                                               np.quantile(Red, 0.75)))
        g.write('Mean value is : {}\n'.format(stt.mean(Red)))
        g.write('Skewness and Kurtosis are : {} {}\n'.format(skew(np.array(Red)), kurtosis(Red)))
        g.write('Average value of Red channel is : {}\n'.format(sum(Red) / (width * height)))
        g.write('The Variance of Red channel is  : {}\n\n'.format(stt.variance(Red)))
        # ================================================
        # for Green channel

        g.write('Sum of Green channel is : {} \n'.format(sum(Green)))
        g.write('Median of Green channel is : {}\n'.format(stt.median(Green)))
        g.write('Lower and Upper quantile of Green channel are : {} {}\n'.format(np.quantile(Green, 0.25),
                                                                                 np.quantile(Green, 0.75)))
        g.write('Mean value is : {}\n'.format(stt.mean(Green)))
        g.write('Skewness and Kurtosis are : {} {}\n'.format(skew(np.array(Green)), kurtosis(Green)))
        g.write('Average value of Green channel is : {}\n'.format(sum(Green) / (width * height)))
        g.write('The Variance of Green channel is  : {}\n\n'.format(stt.variance(Green)))
        # =====================================================
        # for Blue channel

        g.write('Sum of Blue channel is : {}\n'.format(sum(Blue)))
        g.write('Median of Blue channel is : {}\n'.format(stt.median(Blue)))
        g.write('Lower and Upper quantile of Blue channel are : {} {}\n'.format(np.quantile(Blue, 0.25),
                                                                                np.quantile(Blue, 0.75)))
        g.write('Mean value is : {}\n'.format(stt.mean(Blue)))
        g.write('Skewness and Kurtosis are : {} {}\n'.format(skew(np.array(Blue)), kurtosis(Blue)))
        g.write('Average value of Blue channel is : {}\n'.format(sum(Blue) / (width * height)))
        g.write('The Variance of Blue channel is  : {}\n\n'.format(stt.variance(Blue)))
        photo.close()

        for name, num in COLOR.items():
            plt.figure()
            photo = np.array(Image.open(filename))
            a = photo[:, :, num].ravel()
            f = fitter.Fitter(a, distributions=['beta', 'gamma', 'uniform', 'norm', 'laplace'], bins=256, verbose=False)
            f.fit()
            g.write("Fitted errors for "+name+" channel:\n\n")
            for k, v in f._fitted_errors.items():
                g.write(str(k) + ' >>> ' + str(v) + '\n')
            for k in f.get_best().keys():
                print(str(k))
                count_dis[k]+=1
            g.write("\n")
            f.summary()
            f.hist()
            plt.title(str(name+" channel of Image number "+str(cnt+1)))
            pp.savefig()
        plt.close('all')
        g.write('=================================================\n\n')
        cnt += 1
        if cnt >= 20:
            break
    print(count_dis)
    pp.close()
    g.close()

Solve()
