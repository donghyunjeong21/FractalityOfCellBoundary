import numpy as np
import pathlib
import os
from matplotlib import pyplot as plt

neigh = ['1','3','5']
grids = ['0.5','1','2','3','5','10']

csv_dir = pathlib.Path(__file__).parent.resolve()
csv_lst = [file for file in os.listdir(csv_dir) if file.endswith('average.csv')]

data = np.genfromtxt(csv_lst[0], dtype = float, delimiter = ',')

fig, ax = plt.subplots()
im = ax.imshow(data,cmap = 'copper')

ax.set_xticks(np.arange(len(neigh)),labels = neigh)
ax.set_yticks(np.arange(len(grids)), labels = grids)

plt.setp(ax.get_xticklabels(), rotation=45, ha='right',rotation_mode='anchor')

for i in range(len(grids)):
    for j in range(len(neigh)):
        text = ax.text(j,i, data[i,j],ha='center',va='center',color='w')

ax.set_title('Average Final Fractal Values')
fig.tight_layout()
plt.show()
