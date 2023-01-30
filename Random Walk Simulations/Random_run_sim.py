import numpy as np
import random
from matplotlib import pyplot as plt
import math
import copy
from matplotlib import animation
import cv2

#Basic parameters
n_cells = 400; #keep in mind, if n_cells/y_len isn't an integer, number of cells wouldn't equal this number

x_len = 30;
y_len = 50;

# #User prompts
# cell_1_mobility = int(input('Select number of neighbor parameter, must be an odd int>>>  '))
# Grid_pixel_size = float(input('Select pixel size, default is 2>>>  '))
# Min_time = int(input('Select minimum time cutoff, recommended is 10>>>  '))

#Nonrandom uniform initialization
def nonrandom_initialization(n_cells, x_len, y_len):
  max_x = n_cells // y_len
  cell_1_init = np.zeros([max_x*y_len,2]);
  cell_2_init = np.zeros([max_x*y_len,2]);
  index = 0;
  for i in range(max_x):
    for j in range(y_len):
      cell_1_init[index,0] = i
      cell_1_init[index,1] = j
      cell_2_init[index,0] = x_len-1-i;
      cell_2_init[index,1] = j
      index = index + 1
  return cell_1_init, cell_2_init

cell_1_init, cell_2_init = nonrandom_initialization(n_cells, x_len, y_len)

#Likeliness calculator for Random walk
def get_likeliness(x, y, cell_type, cell_1, cell_2, parameters):
  #cell_type should be an integer of 0 or 1, 0 for cell 1, 1 for cell 2
  p_move = parameters[int(cell_type)];
  like_prob1 = parameters[2];
  like_prob2 = parameters[3];
  dslk_prob = parameters[4];

  if cell_type==0:
    like_prob = like_prob1;
  elif cell_type == 1:
    like_prob = like_prob2;

  prob = np.array([1.0,1.0,1.0,1.0]);
  index = 0;
  for i in range(4):
    dx = [-1, 1, 0, 0];
    dy = [0, 0, -1, 1];
    x_check = x + dx[i];
    y_check = y + dy[i];
    adj_count = check_adjCell(x_check, y_check, cell_1, cell_2);
    like_count = adj_count[cell_type]-1;
    if like_count < 0:
      like_count = 0

    dislike_count = np.sum(adj_count) - adj_count[cell_type];

    prob[index] = math.pow(like_prob, like_count) * math.pow(dslk_prob, dislike_count);

    if (check_cellloc(x_check, y_check, cell_1, cell_2) != 0) or check_edge(x_check,y_check):
      prob[index] = 0;
    index = index + 1;
  prob_sum = np.sum(prob);
  prob = np.cumsum((prob/prob_sum) * p_move);
  W,E,S,N = prob
  if prob_sum == 0:
    W,E,S,N = np.array([0,0,0,0])
  return W,E,S,N

def check_cellloc(x, y, cell_1, cell_2):
  is_it_there = 0;
  for i in range(cell_1.shape[0]):
    if (x == cell_1[i,0]) and (y == cell_1[i,1]):
      is_it_there = 1;
    if (x == cell_2[i,0]) and (y == cell_2[i,1]):
      is_it_there = 2;
  return is_it_there

def check_edge(x, y):
  if (x < 0) or (x >= x_len):
    return True;
  elif (y < 0) or (y >= y_len):
    return True;
  else:
    return False;

def check_adjCell(x, y, cell_1, cell_2):
  count_1 = 0;
  count_2 = 0;
  for i in range(4):
    dx = [-1, 1, 0, 0];
    dy = [0, 0, -1, 1];
    x_check = x + dx[i];
    y_check = y + dy[i];
    if check_cellloc(x_check,y_check, cell_1, cell_2) == 1:
      count_1 = count_1 + 1;
    if check_cellloc(x_check,y_check,cell_1, cell_2) == 2:
      count_2 = count_2 + 1;
  return np.array([count_1, count_2])

#Updating the locations function
def update_loc(cell_1, cell_2, parameters):
  np.random.shuffle(cell_1)
  np.random.shuffle(cell_2)
  for i in range(n_cells):
    p = random.random()
    likeliness = get_likeliness(cell_1[i,0],cell_1[i,1], 0, cell_1, cell_2, parameters)
    selected_direction = 4
    for j in range(4):
      if p < likeliness[j]:
        selected_direction = j

        break
    # print(selected_direction)
    x_ind = [-1,1,0,0,0]
    y_ind = [0,0,-1,1,0]
    cell_1[i,0] = cell_1[i,0] + x_ind[selected_direction]
    cell_1[i,1] = cell_1[i,1] + y_ind[selected_direction]

  for i in range(n_cells):
    p = random.random()
    likeliness = get_likeliness(cell_2[i,0],cell_2[i,1], 1, cell_1, cell_2, parameters)
    selected_direction = 4

    for j in range(4):
      if p < likeliness[j]:
        selected_direction = j
        break

    x_ind = [-1,1,0,0,0]
    y_ind = [0,0,-1,1,0]
    cell_2[i,0] = cell_2[i,0] + x_ind[selected_direction]
    cell_2[i,1] = cell_2[i,1] + y_ind[selected_direction]
  return cell_1, cell_2

parameters = [0.99, 0.4, 2.5, 2.5, 0.2]
#Parameter[0] = mobility of cell 1. =0 indicates no movement, =1 indicates always moving
#Parameter[1] = mobility of cell 2. =0 indicates no movement, =1 indicates always moving
#Parameter[2] = cell1-cell1 affinity, <1 indicates repulsion
#Parameter[3] = cell2-cell2 affinity, <1 indicates repulsion
#Parameter[4] = cell1-cell2 affinity, <1 indicates repulsion

tmax = 80
cell_1 = copy.copy(cell_1_init)
cell_2 = copy.copy(cell_2_init)
cell_all = np.zeros([cell_1.shape[0]*2, cell_1.shape[1]+1, tmax])
print('Simulating cells...')
for j in range(tmax):
  if j%5==0:
      print('Currently working on timepoint ' + str(j))
  cell_1, cell_2 = update_loc(cell_1,cell_2,parameters)

  all_data = np.zeros([n_cells*2,3])
  for i in range(n_cells):
    all_data[i,0:2] = cell_1[i,:]
    all_data[i,2] = 0
    all_data[i+n_cells,0:2] = cell_2[i,:]
    all_data[i+n_cells,2] = 1
  cell_all[:,:,j] = all_data

prompt = input('filename? >>>  ')

def generate_figures(x,y,c):
  fig = plt.figure(figsize=[6,10])

  plt.scatter(x,y,c = c)
  fig.canvas.draw()
  img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='');
  img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,));

  img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
  img = cv2.resize(img, (300,500))
  return img

def generate_vid(cell_all):
    img = []
    for i in range(tmax):
      img.append(generate_figures(cell_all[:,0,i],cell_all[:,1,i],cell_all[:,2,i]))
    frameSize = (300,500)
    filename_avi = prompt + '.avi'
    out = cv2.VideoWriter(filename_avi,cv2.VideoWriter_fourcc(*'DIVX'), 3, frameSize)
    for images in img:
      out.write(images)
    out.release()

generate_vid(cell_all)



def export_cell_loc(cell_all):
    cell_reshape = np.zeros([cell_all.shape[0]*cell_all.shape[2],4])
    cellcount = cell_all.shape[0]

    for i in range(cell_all.shape[2]):
      cell_reshape[cellcount*i:cellcount*(i+1),0:2] = cell_all[:,0:2,i]
      cell_reshape[cellcount*i:cellcount*(i+1),2] = cell_all[:,2,i]
      cell_reshape[cellcount*i:cellcount*(i+1),3] = i
    return cell_reshape.T

reshaped_cell = export_cell_loc(cell_all)


filename_csv = prompt + '.csv'
np.savetxt(filename_csv, reshaped_cell, delimiter = ', ')
