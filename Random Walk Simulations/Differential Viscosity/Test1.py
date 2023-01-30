import os
import numpy as np
import pathlib
from matplotlib import pyplot as plt
import multiprocessing as mp
from sklearn.neighbors import KNeighborsClassifier as knc
import math
import re

type = input('Which file do you want?>>>  ')

csv_dir = pathlib.Path(__file__).parent.resolve()
csv_lst_disp = [file for file in os.listdir(csv_dir) if file.endswith('_disp.csv')]
csv_lst_frac = [file for file in os.listdir(csv_dir) if file.endswith('_fractals.csv')]



pattern = re.compile('\A' + type + ".+")
plt.figure()
count = 0
data_total_x = []
data_total_t = []
d = []
f = []
for i in range(len(csv_lst_frac)):
    if re.search(pattern, csv_lst_frac[i]):
        data_frac = np.genfromtxt(csv_lst_frac[i], delimiter = ',')
        f.append(data_frac[1])
    if re.search(pattern, csv_lst_disp[i]):
        data_disp = np.genfromtxt(csv_lst_disp[i], delimiter = ',')
        d.append(data_disp[1])

vel = [];
fra = [];
for i in range(len(d)):
    for j in range(d[i].shape[0]):
        if ((j-4)%5==0):
            v = (d[i][j] - d[i][j-9])/10;
            vel.append(v)
            fra.append(np.mean(f[i][j-4:j]))

nonneg_v = [vel[i] for i in range(len(vel)) if vel[i] > 0];
nonneg_f = [fra[i] for i in range(len(vel)) if vel[i] > 0];
# nonneg_v = np.array(vel);
# nonneg_v[nonneg_v <= 0] = 0.00001;
# nonneg_f = np.array(fra)
plt.semilogx(nonneg_v, nonneg_f, 'o')

m,b = np.polyfit(np.log(nonneg_v), nonneg_f, 1)
print(m)
print(b)
plt.show()
