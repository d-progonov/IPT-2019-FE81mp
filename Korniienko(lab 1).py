import os
from collections import defaultdict

import numpy as np
import scipy as sp
from scipy import misc, stats
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly
import seaborn as sns
from fitter import Fitter



%matplotlib inline

DIRNAME = 'E:/pr/m/'
COLOR = {'red': 0,
         'green': 1,
         'blue': 2}  # 

with open('E:/pr/random.txt') as f:
    image_names = ['im'+ x.strip()+'.jpg' for x in f.readlines()]
import os
from collections import defaultdict

import numpy as np
import scipy as sp
from scipy import misc, stats
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly
import seaborn as sns
from fitter import Fitter



%matplotlib inline

DIRNAME = 'E:/pr/m/'
COLOR = {'red': 0,
         'green': 1,
         'blue': 2}  # 

with open('E:/pr/random.txt') as f:
    image_names = ['im'+ x.strip()+'.jpg' for x in f.readlines()]


data = {}
for name, num in COLOR.items():
    data[name] = pd.DataFrame()
    for image_name in image_names[:100]:
        image = np.array(Image.open(DIRNAME+image_name))
        a = image[:, :, num].ravel()
        d = {'name': image_name,
             'min': np.min(a),
             'max': np.max(a),
             'mean': np.mean(a),
             'var': np.var(a),
             'median': np.median(a),
             'interquartile': sp.stats.iqr(a),
             'skewness': sp.stats.skew(a),
             'kurtosis': sp.stats.kurtosis(a)}
        data[name] = pd.concat([data[name], pd.DataFrame(pd.DataFrame(d, index=[0,]))], ignore_index=True)







DISTRIBUTIONS = ['beta', 'gamma', 'uniform', 'norm']
BINS = 256

hist = {'red': defaultdict(int),
        'green': defaultdict(int),
        'blue': defaultdict(int),}

for name, num in COLOR.items():
    for image_name in image_names[:100]:
        image = np.array(Image.open(os.path.join(DIRNAME, image_name)))
        a = image[:, :, num].ravel()
        f = Fitter(a, distributions=DISTRIBUTIONS, bins=BINS)
        f.fit()
        hist[name][f.df_errors['sumsquare_error'].idxmin()] += 1
