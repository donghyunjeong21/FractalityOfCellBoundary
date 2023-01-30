import os
import matplotlib.pyplot as plt
import pathlib
import numpy as np
from skimage import io
from cellpose import models
from cellpose import plot

#Set the path where images are stored
img_dir = pathlib.Path(__file__).parent.resolve()

images_lst = [file for file in os.listdir(img_dir)
            if file.endswith(".tif")]

def run_cellpose(vid, model, diameter, channel):
    vid_1channel = vid[:, channel, :, :]
    images = []
    for i in range(vid_1channel.shape[0]):
        images.append(vid_1channel[i,:,:])
    masks, flows, styles, diams = model.eval(images, diameter = diameter, flow_threshold=None, channels=[0,0])
    return masks

def get_cell_loc(vid, model, diameter):
    mask_0 = run_cellpose(vid, model, diameter, 0)
    mask_1 = run_cellpose(vid, model, diameter, 1)

    loc_all_time = []
    for time in range(vid.shape[0]):
        num_of_cells_0 = np.max(np.unique(mask_0[time]))
        num_of_cells_1 = np.max(np.unique(mask_1[time]))
        num_of_cells_t = num_of_cells_0 + num_of_cells_1
        x_coord, y_coord, label_array = np.zeros(num_of_cells_t), np.zeros(num_of_cells_t), np.zeros(num_of_cells_t)

        for i in range(1,num_of_cells_0 + 1):
            y_coord[i-1] = np.mean(np.where(mask_0[time]==i)[0])
            x_coord[i-1] = np.mean(np.where(mask_0[time]==i)[1])
            label_array[i-1] = 0
        for i in range(1,num_of_cells_1 + 1):
            y_coord[i-1 + num_of_cells_0] = np.mean(np.where(mask_1[time]==i)[0])
            x_coord[i-1 + num_of_cells_0] = np.mean(np.where(mask_1[time]==i)[1])
            label_array[i-1 + num_of_cells_0] = 1

        loc_all_time.append(np.array([x_coord, y_coord, label_array]))

    return loc_all_time



use_GPU = True
model = models.Cellpose(gpu=use_GPU, model_type='cyto')

for filenames in images_lst:
    data_path = os.path.join(img_dir,filenames)
    im = io.imread(data_path)
    data = get_cell_loc(im, model, 10)

    n = 0
    for i in range(len(data)):
      n = n + data[i].shape[1]

    data_arr = np.zeros([4, n])

    count = 0
    for i in range(len(data)):
      m = data[i].shape[1]
      data_arr[0:3, count:count+m] = data[i]
      data_arr[3, count:count+m] = i
      count = count + m

    filename_csv = filenames[:-4] + '.csv'
    np.savetxt(filename_csv, data_arr, delimiter = ', ')
