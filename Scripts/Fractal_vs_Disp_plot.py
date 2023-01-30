import os
import numpy as np
import pathlib
from matplotlib import pyplot as plt
import multiprocessing as mp
from sklearn.neighbors import KNeighborsClassifier as knc
import math
import re

type = input('Which file do you want? Input xxx if all>>>  ')



csv_dir = pathlib.Path(__file__).parent.resolve()
csv_lst_disp = [file for file in os.listdir(csv_dir) if file.endswith('_disp.csv')]
csv_lst_frac = [file for file in os.listdir(csv_dir) if file.endswith('_fractals.csv')]

def get_velocity(data):
    data_result= np.zeros(data.shape)
    data_result[0] = 0
    for i in range(data.shape[0]-1):
        data_result[i+1] = (data[i+1] - data[i]) * 2
    return data_result


pattern = re.compile('\A' + type + ".+")
plt.figure()
count = 0
f=[]
d=[]
for i in range(len(csv_lst_disp)):
    if re.search(pattern, csv_lst_frac[i]):
        print(csv_lst_frac[i])
        data_frac = np.genfromtxt(csv_lst_frac[i], delimiter = ',')
        f.append(data_frac[1])
    if re.search(pattern, csv_lst_disp[i]):
        data_disp = np.genfromtxt(csv_lst_disp[i], delimiter = ',')
        data_vel = get_velocity(data_disp)
        d.append(data_vel[1])

print(f[0])

plt.plot(f[0],d[0],'o')

# def FindMaxLength(lst):
#     maxLength = max(x.shape[0] for x in lst )
#     return maxLength
# def getAverage(arr):
#     sum = np.sum(arr, axis = 0)
#     nzc = np.count_nonzero(arr, axis = 0)
#     return(np.divide(sum, nzc))
#
# def getStdErr(arr):
#     std = np.zeros([arr.shape[1]])
#     for j in range(arr.shape[1]):
#         std[j] = np.nanstd(np.where(np.isclose(arr[:,j],0), np.nan, arr[:,j]))
#     N2 = np.sqrt(np.count_nonzero(arr, axis = 0))
#
#     return np.divide(std, N2)
#
# if average_response == 1:
#     maxLen = FindMaxLength(data_total_x)
#     T = np.zeros([count, maxLen])
#     X = np.zeros([count, maxLen])
#     for i in range(X.shape[0]):
#         for j in range(data_total_x[i].shape[0]):
#             X[i,j] = data_total_x[i][j] *3
#             T[i,j] = data_total_t[i][j] /2
#
#     avgX = getAverage(X)
#     avgT = getAverage(T)
#     errX = getStdErr(X)
# # data_total_x = np.stack(data_total_x, axis = 1)
# # average_x = np.average(data_total_x, axis = 0)
# if average_response == 1:
#     plt.plot(avgT, avgX, 'k-')
#     plt.fill_between(avgT, avgX - errX, avgX + errX)
#     plt.ylim([0, 70])

plt.show()
