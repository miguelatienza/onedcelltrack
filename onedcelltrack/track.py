# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 11:02:33 2021

@author: miguel.Atienza
"""
import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt
import pandas as pd

masks = imread('Y:/project/ag-moonraedler/MAtienza/cyto_masks_0-3000.tif', key=range(0,500))

#Import lines from csv
lines_file = 'Y:\\project\\ag-moonraedler\\JHeyn\\211117_UNikon_FN-lines_30s\\Export\\lines\\lines09.csv'
header = ['x_0', 'x_f', 'y_0', 'y_f']
df = pd.read_csv(lines_file, names=header)

i = 5
y_0 = df.y_0[i]
y_f = df.y_f[i]
x_0 = df.x_0[i]
x_f = df.x_f[i]
m = (y_f-y_0)/(x_f-x_0)
x_values = np.round(np.arange(x_0, x_f)).astype(int)
y_values = (y_0 + m*x_values).astype(int)

kymograph = masks[:, y_values, x_values].T
plt.imshow(kymograph)