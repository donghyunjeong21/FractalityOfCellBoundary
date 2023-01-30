import os
import numpy as np
import pathlib
from sklearn.neighbors import KNeighborsClassifier as knc
import math
from joblib import Parallel, delayed
import multiprocessing as mp

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

def get_min_y(lst_of_boundaries, tmax, tmin):
    #Due to some minor issues with imaging/post-processing, each frame in the video
    #may vary in the num of pixel count by 1 or 2 pixels. This isn't significant enough
    #to affect the results, but must be addressed to eliminate errors
    y_size = np.zeros(int(tmax-tmin)+1)
    for i in range(0, int(tmax-tmin)+1):

      y_size[i] =  lst_of_boundaries[i].shape[0]
    return(int(np.min(y_size)))

def get_init_boundary(lst_of_boundaries, min_y):
    init_position = np.zeros(min_y)

    for j in range(0, min_y):
      lst_of_boundaries[0][j,0] = 0
      lst_of_boundaries[0][j,-1] = 0
      loc_bound = 0
      for k in range(0,  lst_of_boundaries[0].shape[1]):
        if lst_of_boundaries[0][j,k] == 1:
          loc_bound = k
      init_position[j]=loc_bound
    return init_position

def get_displacement(lst_of_boundaries, init_position, min_y):
    disp = np.zeros(len(lst_of_boundaries))
    disp[0]=0

    for i in range(1, len(lst_of_boundaries)):
      boundary = lst_of_boundaries[i]
      loc_position = np.zeros(min_y)

      for j in range(0, min_y):
        check_max = 0
        loc_bound = init_position[j]
        boundary[j,0] = 0
        boundary[j,-1] = 0

        for k in range(0, boundary.shape[1]):
          if boundary[j,k] == 1:
            if abs(k-init_position[j]) >= check_max:
              loc_bound = k
              check_max = abs(k - init_position[j])
        loc_position[j] = loc_bound
      disp[i] = np.mean(abs(loc_position - init_position))
    return disp

def run_entire_pipeline(csv_dir, filename, parameters, verbose = False):
    #Parameter 1 = num of neighbors in KNN
    #Parameter 2 = grid spacing
    #Parameter 3 = t_min, ignores timepoints before this one
    data_raw = read_csv_to_numpy(csv_dir, filename)

    # tmax = data_raw[3,:].max()
    tmax = 34
    tmin = 15
    index = 0
    x = []
    t = np.zeros(int(tmax - tmin + 1))

    for i in range(int(tmin), int(tmax) + 1):
        data = data_raw[:,np.where(data_raw[3,:]==i)[0]]
        boundary = get_boundary(get_regions(data, n_neighbor = parameters[0], grid_step = parameters[1]))
        t[index] = i
        x.append(boundary)
        index = index + 1
    min_y = get_min_y(x, tmax, tmin)
    init_position = get_init_boundary(x, min_y)
    disp = get_displacement(x, init_position, min_y)

    return np.array([t,disp])

Num_of_neighbors = int(input('Select number of neighbor parameter, must be an odd int>>>  '))
Grid_pixel_size = float(input('Select pixel size, default is 2>>>  '))
parameters = np.array([Num_of_neighbors, Grid_pixel_size])

def processInput(filename):
    output = run_entire_pipeline(csv_dir, filename, parameters)
    filename_csv = filename[:-4] + '_disp.csv'
    np.savetxt(filename_csv, output, delimiter = ', ')

csv_dir = pathlib.Path(__file__).parent.resolve()
csv_lst = [file for file in os.listdir(csv_dir) if file.endswith('.csv')]

num_cores = mp.cpu_count()
print(num_cores)

results = Parallel(n_jobs=num_cores)(delayed(processInput)(filename) for filename in csv_lst)
