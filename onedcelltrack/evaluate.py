# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 09:32:32 2022

@author: miguel.Atienza
"""
from celltracker import stack
import numpy as np
from tifffile import TiffFile, imwrite, imread
import os

def segment(cyto_file, nucleus_file, data_path, path_out, image_indices=None, cyto_contrast=1, cyto_brightness=0, invert_cyto=False, nucleus_contrast=1, nucleus_brightness=0, max_memory=None, save=False, pretrained_model=None, nucleus_diameter=None, cyto_diameter=None, nucleus_flow_threshold=3, nucleus_mask_threshold=-3, cyto_flow_threshold=0.8, cyto_mask_threshold=-2, gpu=True):
    
    tif = TiffFile(os.path.join(data_path + cyto_file))
    n_images = len(tif.pages)
    height, width = tif.pages[0].shape
    
    if max_memory is None:
        max_stack = n_images
    max_stack = max_memory
    
    if image_indices is not None:
        index_values = np.append(np.arange(image_indices[0], image_indices[1], max_stack), n_images)
    else:
        index_values = np.append(np.arange(0, n_images, max_stack), n_images)
        
    for j in range(index_values.size -1):
    
        image_indices = [index_values[j], index_values[j+1]]
        path_out_index = path_out + str(image_indices[0]) + '-' + str(image_indices[1]) + '_'
        cells = stack.Stack(cytoplasm_file=cyto_file, nucleus_file=nucleus_file, data_path=data_path, image_indices=image_indices)
        
        cells.preprocess(cyto_contrast=cyto_contrast, cyto_brightness=cyto_brightness, invert_cyto=False,
                          nucleus_contrast=nucleus_contrast, nucleus_brightness=nucleus_brightness)

        cells.segment(nucleus_diameter=nucleus_diameter, cyto_diameter=cyto_diameter, nucleus_flow_threshold=nucleus_flow_threshold, nucleus_mask_threshold=nucleus_mask_threshold, cyto_mask_threshold=cyto_mask_threshold, cyto_flow_threshold=cyto_flow_threshold, gpu=gpu, save=save, path_out=path_out_index)


