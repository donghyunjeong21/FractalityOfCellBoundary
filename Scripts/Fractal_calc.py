import os
import numpy as np
import pathlib
from joblib import Parallel, delayed
import multiprocessing as mp
from sklearn.neighbors import KNeighborsClassifier as knc
import math

###Functions required to analyze the fractal geometry
def fractal_dimension(Z):
  #Z must be a binary, 2D array, with the boundary pixels in 1 and rest in 0
  def boxcount(Z, k):
    S = np.add.reduceat(
            np.add.reduceat(                                    #This creates a series of 1D arrays, each containing sum of i:i+k pixels
                Z,
                np.arange(0, Z.shape[0], k),                    #np.arange creates a linearly spaced array of spacing k
                axis=0
              ),
            np.arange(0, Z.shape[1], k), axis=1                 #Repeats reduceat but in the other direction, resulting in series of boxes with sums
      )
    return len(np.where(S > 0)[0])

  # Calculating the box sizes
  p = min(Z.shape)                                              #number of pixels in the shorter direction of the image. The box can't be bigger than this
  sizes = np.arange(1, p//2, 2)                                    #np array containing the length of each box, ranging from 2 to p

  # Actual box counting with decreasing size
  counts = []
  for size in sizes:
      counts.append(boxcount(Z, size))
  # Fit the successive log(sizes) with log (counts)
  coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
  return -coeffs[0]

def get_boundary(zz):
  boundary = np.zeros(zz.shape)
  for i in range(1,zz.shape[0]-1):
    for j in range(1,zz.shape[1]-1):
      sum = zz[i-1,j] + zz[i+1,j] + zz[i,j-1] + zz[i,j+1]
      if ((sum != 0 and sum != 4) and zz[i,j] == 1):
        boundary[i,j] = 1
    sum = zz[i-1,j] + zz[i+1,j] + zz[i,1]
    if ((sum != 0 and sum != 3) and zz[i,j] == 1):
      boundary[i,0] = 1

  return boundary

def get_regions(data, n_neighbor = 3, grid_step = 2, plot_result=False):
  xgrid = np.arange(data[0,:].min(), data[0,:].max(), grid_step)
  ygrid = np.arange(data[1,:].min(), data[1,:].max(), grid_step)
  xx, yy = np.meshgrid(xgrid, ygrid)
  xinput, yinput = xx.flatten(), yy.flatten()
  data_X = np.array([xinput, yinput]).T

  X = data[0:2, :].T
  Y = data[2,:]

  model = knc(n_neighbors = int(n_neighbor))
  model.fit(X,Y)

  label_predict = model.predict(data_X)
  zz = label_predict.reshape(xx.shape)
  if plot_result:
    plt.figure(figsize=(3,7))
    plt.contourf(xx,yy,zz, cmap = 'gray' )
  return zz

def read_csv_to_numpy(csv_dir, filename):
    data_path = os.path.join(csv_dir, filename)
    return np.genfromtxt(data_path, delimiter = ',')

def run_entire_pipeline(csv_dir, filename, parameters, verbose = False):
    #Parameter 1 = num of neighbors in KNN
    #Parameter 2 = grid spacing
    #Parameter 3 = t_min, ignores timepoints before this one
    data_raw = read_csv_to_numpy(csv_dir, filename)

    # tmax = data_raw[3,:].max()
    tmax = 34
    index = 0
    tmin = parameters[2]
    x = np.zeros(int(tmax - tmin + 1))
    t = np.zeros(int(tmax - tmin + 1))

    for i in range(int(tmin), int(tmax) + 1):
        if verbose:
            print(i)

        data = data_raw[:,np.where(data_raw[3,:]==i)[0]]
        region = get_regions(data, n_neighbor = parameters[0], grid_step = parameters[1])
        x[index] = fractal_dimension(get_boundary(region))
        t[index] = i
        index = index + 1
    output = np.array([t,x])
    return output

Num_of_neighbors = int(input('Select number of neighbor parameter, must be an odd int>>>  '))
Grid_pixel_size = float(input('Select pixel size, default is 2>>>  '))
Min_time = int(input('Select minimum time cutoff, recommended is 10>>>  '))
parameters = np.array([Num_of_neighbors, Grid_pixel_size, Min_time])
def processInput(filename):
    output = run_entire_pipeline(csv_dir, filename, parameters)
    filename_csv = filename[:-4] + '_fractals.csv'
    np.savetxt(filename_csv, output, delimiter = ', ')

csv_dir = pathlib.Path(__file__).parent.resolve()
csv_lst = [file for file in os.listdir(csv_dir) if file.endswith('.csv')]


num_cores = mp.cpu_count()
print(num_cores)

results = Parallel(n_jobs=num_cores)(delayed(processInput)(filename) for filename in csv_lst)
