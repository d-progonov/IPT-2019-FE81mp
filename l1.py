from functions import basic_analysis, approximate_hist, read
import os

all_errors = []
best_errors = {'Norm': 0, 'Gamma': 0, 'Beta': 0, 'Uniform': 0}
results = []
directory = 'C:\\Users\Vitalii\PycharmProjects\Images_project\images_1\\'

for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        img = read(directory+filename)
        res, flatten = basic_analysis(img, 1)
        print(res)
        results.append(res)
        for color in flatten:
            err = approximate_hist(color, to_show=False)
            print(err)
            best_error = min(err.values())
            for distr in err.keys():  # for name, age in dictionary.iteritems():  (for Python 2.x)
                if err[distr] == best_error:
                    best_errors[distr] +=1
                    break
            print(best_errors)
            all_errors.append(err)
    else:
        continue

import matplotlib.pyplot as plt
to_plot = []
to_plot.append(best_errors['Norm'])
to_plot.append(best_errors['Gamma'])
to_plot.append(best_errors['Beta'])
to_plot.append(best_errors['Uniform'])

plt.plot(to_plot)
plt.show()
