import os
import numpy as np
import pathlib
from matplotlib import pyplot as plt
import multiprocessing as mp
from sklearn.neighbors import KNeighborsClassifier as knc
import math
import re

type = input('Which file do you want?>>>  ')
average = input('Do you want an average? Y if yes, N if no>>>  ')

if average == 'Y':
    average_response = 1
elif average == 'N':
    average_response = 0

csv_dir = pathlib.Path(__file__).parent.resolve()
csv_lst = [file for file in os.listdir(csv_dir) if file.endswith('_fractals.csv')]



pattern = re.compile('\A' + type + ".+")
fig = plt.figure()
count = 0
data_total_x = []
data_total_t = []
for i in csv_lst:
    if re.search(pattern, i):
        data = np.genfromtxt(i, delimiter = ',')
        #label = i[0:3]
        if average_response == 0:
            plt.plot(data[0],data[1])
            plt.ylim([1.1, 1.4])
            print(i)
        else:
            data_total_t.append(data[0])
            data_total_x.append(data[1])
        count+= 1


def FindMaxLength(lst):
    maxLength = max(x.shape[0] for x in lst )
    return maxLength
def getAverage(arr):
    sum = np.sum(arr, axis = 0)
    nzc = np.count_nonzero(arr, axis = 0)
    return(np.divide(sum, nzc))

def getStdErr(arr):
    std = np.zeros([arr.shape[1]])
    for j in range(arr.shape[1]):
        std[j] = np.nanstd(np.where(np.isclose(arr[:,j],0), np.nan, arr[:,j]))
    N2 = np.sqrt(np.count_nonzero(arr, axis = 0))

    return np.divide(std, N2)

if average_response == 1:
    maxLen = FindMaxLength(data_total_x)
    T = np.zeros([count, maxLen])
    X = np.zeros([count, maxLen])
    for i in range(X.shape[0]):
        for j in range(data_total_x[i].shape[0]):
            X[i,j] = data_total_x[i][j]
            T[i,j] = data_total_t[i][j] /2

    avgX = getAverage(X)
    avgT = getAverage(T)
    errX = getStdErr(X)
# data_total_x = np.stack(data_total_x, axis = 1)
# average_x = np.average(data_total_x, axis = 0)
if average_response == 1:
    plt.plot(avgT, avgX, 'k-')
    plt.fill_between(avgT, avgX - errX, avgX + errX)
    plt.ylim([1.0, 1.4])

plt.xticks(fontsize=16, fontname = "Arial")
plt.yticks(fontsize=16, fontname = "Arial")
plt.show()
