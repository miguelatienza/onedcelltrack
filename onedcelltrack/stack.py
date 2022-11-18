# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 11:11:50 2021

@author: miguel.Atienza
"""

from celltracker import functions
import numpy as np
import os

class Stack:
    
    def __init__(self, cytoplasm_file=None, nucleus_file=None,
                 lanes_file=None, image_indices=None,
                 x_range=None,y_range=None, data_path=None):
        
        self.data_path = data_path
        self.cytoplasm_file = cytoplasm_file
        self.nucleus_file = nucleus_file
        self.lanes_file = lanes_file
        self.image_indices = image_indices
        
        cytoplasm, nucleus, lanes = functions.extract_from_tif(cytoplasm_file=cytoplasm_file, nucleus_file=nucleus_file, lanes_file=lanes_file, image_indices=image_indices, x_range=x_range, y_range=y_range, data_path=data_path)
        
        self.cytoplasm = cytoplasm
        self.nucleus = nucleus
        self.lanes = lanes
        
        if cytoplasm_file is not None:
            self.n_images = self.cytoplasm.shape[0]
            self.height = self.cytoplasm.shape[1]
            self.width = self.cytoplasm.shape[2]
            self.image_shape = [self.height, self.width]
            self.stack_shape = self.cytoplasm.shape
                  
        elif nucleus_file is not None:
            self.n_images = self.nucleus.shape[0]
            self.height = self.cytoplasm.shape[1]
            self.width = self.cytoplasm.shape[2]
            self.image_shape = [self.height, self.width]
            self.stack_shape = self.nucleus.shape
            
    def preprocess(self, cyto_contrast=1.5, cyto_brightness=0.5, invert_cyto=False, nucleus_contrast=4, nucleus_brightness=0.8, lanes_contrast=1, lanes_brightness=0.4, bottom_percentile=0, top_percentile=100):
        
        if self.cytoplasm is not None:
            self.cytoplasm = functions.preprocess(self.cytoplasm, c=cyto_contrast, b=cyto_brightness, invert=invert_cyto, bottom_percentile=0, top_percentile=100)
            
        if self.nucleus is not None:
            self.nucleus = functions.preprocess(np.log(self.nucleus), c=nucleus_contrast, b=nucleus_brightness, invert=False, bottom_percentile=0, top_percentile=100)
            
        if self.lanes is not None:
            self.lanes = functions.preprocess(self.lanes, c=lanes_contrast, b=lanes_brightness, bottom_percentile=0, top_percentile=100)
        
            
    def get_rgb_image(self, image_index, x_range=None, y_range=None, 
                  view_nucleus=True, view_cyto=True, view_lanes=True, lanes_alpha=1):
       
        empty = np.zeros(self.image_shape)
        
        if view_nucleus and view_cyto and view_lanes:
            if self.lanes is not None:
                return functions.create_rgb(self.lanes, self.cytoplasm[image_index], self.nucleus[image_index], r_alpha=lanes_alpha)
                
        elif view_cyto and view_nucleus:
            return functions.create_rgb(
                empty, self.cytoplasm[image_index], self.nucleus[image_index])
    
        elif view_cyto and view_lanes:
            return functions.create_rgb(
                self.lanes, self.cytoplasm[image_index], empty, r_alpha=lanes_alpha)
        elif view_nucleus and view_lanes:
            return functions.create_rgb(
                self.lanes, empty, self.nucleus[image_index], r_alpha=lanes_alpha)
        elif view_nucleus:
            return functions.create_rgb(
                empty, empty, self.nucleus[image_index], r_alpha=lanes_alpha)
        elif view_cyto:
            return functions.create_rgb(
                empty, self.cytoplasm[image_index], empty)
        
       
    def segment_nuclei(self, index, gpu=False, model_type='nuclei', channels=None, diameter=None, flow_threshold=0.4, mask_threshold=0):
        
        from cellpose import models
        model = models.Cellpose(gpu=gpu, model_type=model_type)
        
        self.nucleus_mask, self.nucleus_flow, self.nucleus_style, self.nucleus_diam = model.eval(self.nucleus[index], diameter=diameter, channels=channels, flow_threshold=flow_threshold, mask_threshold=mask_threshold, normalize=False)
        
        # return model.eval(self.nucleus[index], diameter=diameter, channels=channels, flow_threshold=flow_threshold, mask_threshold=mask_threshold, normalize=False)
        
    def segment_cytoplasm(self, index, gpu=False, model_type='cyto', channels=[1,2], diameter=None, flow_threshold=0.4, mask_threshold=0):
        
        from cellpose import models
        model = models.Cellpose(gpu=gpu, model_type=model_type)
        
        images = np.stack((self.cytoplasm, self.nucleus), axis=1)
        self.cyto_mask, self.cyto_flow, self.cyto_style, self.cyto_diam = model.eval(images[index], diameter=diameter, channels=channels, flow_threshold=flow_threshold, mask_threshold=mask_threshold, normalize=False)
        
        # return  model.eval(images[index], diameter=diameter, channels=channels, flow_threshold=flow_threshold, mask_threshold=mask_threshold, normalize=False)
        
    def get_contours_image(self, cytoplasm, nucleus, cyto_mask, nucleus_mask):
        
        #from cellpose import plot
        from skimage.segmentation import find_boundaries
    
        cyto_contours = find_boundaries(cyto_mask)
        nucleus_contours = find_boundaries(nucleus_mask)
        
        out = np.stack((cytoplasm, cytoplasm, cytoplasm), axis=-1)
        out[cyto_contours[:,:]!=0] = [0,1,0]
        out[nucleus_contours[:,:]!=0] = [0,0,1]
    
        return out
    
    def segment(self, path_out, gpu=False, channels=[1,2], nucleus_diameter=None, cyto_diameter=None, nucleus_flow_threshold=0.8, nucleus_mask_threshold=0, cyto_flow_threshold=0.8, cyto_mask_threshold=0, save=False, pretrained_model=None):
        
        from cellpose import models
        from tqdm import tqdm
        from tifffile import imwrite
        
        #Start by segmenting the nuclei
        model = models.Cellpose(gpu=gpu, model_type='nuclei')
        
        self.nucleus_masks = np.zeros((self.n_images, self.height, self.width))
        #nucleus_segmentation_path = os.path.join('../output', os.path.splitext(self.nucleus_file)[0]) + '_segmented.tif'
        nucleus_segmentation_path = path_out + 'nucleus_masks.tif'
        
        for i in tqdm(range(self.n_images)):
            
            self.nucleus_masks[i] = model.eval(self.nucleus[i], diameter=nucleus_diameter, channels=[0,0], flow_threshold=nucleus_flow_threshold, mask_threshold=nucleus_mask_threshold, normalize=False)[0]
            print(f'Found {np.max(self.nucleus_masks[i])} nuclei on the {i}th image.')
            if save:
                imwrite(nucleus_segmentation_path, (self.nucleus_masks).astype('uint8'))
                
        print(f'Finished segmentation of the nuclei for the {self.n_images}')
        
        #Now segment the cytoplasm
        if pretrained_model is None:
            model = models.Cellpose(gpu=gpu, model_type='cyto')
        else:
            model = models.CellposeModel(gpu=gpu, pretrained_model=pretrained_model)

        cyto_segmentation_path = path_out + 'cyto_masks.tif'
        images = np.stack((self.cytoplasm, self.nucleus), axis=1)
        
        self.cyto_masks = model.eval(images, diameter=cyto_diameter, channels=[1,2], flow_threshold=cyto_flow_threshold, mask_threshold=cyto_mask_threshold, normalize=False)[0]

        if save:
            imwrite(cyto_segmentation_path, (self.cyto_masks).astype('uint8'))
        
        #Finally combine the two into an rgb image
        #contours_path = os.path.join('../output', os.path.splitext(self.cytoplasm_file)[0]) + '_rgb_outlines.tif'
        contours_path = path_out + 'contours.tif'
        
        self.contours_image = np.zeros((self.n_images, self.height, self.width, 3))
        for i in range(self.n_images):
            self.contours_image[i] = self.get_contours_image(self.cytoplasm[i], self.nucleus[i], self.cyto_masks[i], self.nucleus_masks[i])
        
        if save:
            imwrite(contours_path, (self.contours_image*255).astype('uint8'))
            
            
            
            
            
        
           
           
           
           
           
       
       
        
        
    
        
        