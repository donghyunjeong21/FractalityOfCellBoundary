import os
import numpy as np
import math
import pathlib

listOfTypes = ['BB','LL','LB']

def getStdErr(arr):
    std = np.zeros([arr.shape[1]])
    for j in range(arr.shape[1]):
        std[j] = np.nanstd(np.where(np.isclose(arr[:,j],0), np.nan, arr[:,j]))
    N2 = np.sqrt(np.count_nonzero(arr, axis = 0))

    return np.divide(std, N2)

def getAvg(csv_lst):
    end_time_avg = np.zeros(len(csv_lst))
    index = 0
    for i in csv_lst:
        data = np.genfromtxt(i, delimiter = ',')
        length = data.shape[1]
        end_time_avg[index] = data[1,length-3:length].mean()
        index += 1
    return(end_time_avg.mean())

def getAvgStdErr(csv_lst):
    data_total_x = []
    index = 0
    for i in csv_lst:
        data = np.genfromtxt(i, delimiter = ',')
        length = data.shape[1]
        data_total_x.append(data[1])
        index += 1
    X = np.zeros([index, length])
    for i in range(X.shape[0]):
        for j in range(data_total_x[i].shape[0]):
            X[i,j] = data_total_x[i][j]

    return(getStdErr(X).mean())

csv_dir = pathlib.Path(__file__).parent.resolve()

csv_lst1 = [file for file in os.listdir(csv_dir) if file.startswith(listOfTypes[0])]
# csv_lst2 = [file for file in os.listdir(csv_dir) if file.startswith(listOfTypes[1])]
# csv_lst_same = csv_lst1 + csv_lst2
csv_lst_same = csv_lst1

csv_lst_diff = [file for file in os.listdir(csv_dir) if file.startswith(listOfTypes[2])]

csv_lst_total = [file for file in os.listdir(csv_dir) if file.endswith('.csv')]

average_same = getAvg(csv_lst_same)
average_diff = getAvg(csv_lst_diff)
avg_stdErr = getAvgStdErr(csv_lst_total)

print('Square difference in final fractal values is ', average_same - average_diff)
print('Average standard errors is ', avg_stdErr)
