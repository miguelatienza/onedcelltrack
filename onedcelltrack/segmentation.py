# -*- coding: utf-8 -*-
import sys
sys.path.append('..')
from celltracker import functions
import numpy as np
import os
import json
from urllib import request
from cellpose import models
from cellpose.io import logger_setup 
from tqdm import tqdm
#logger_setup()

def segment(cytoplasm, nucleus, gpu=True, model_type='cyto', channels=[1,2], diameter=None, flow_threshold=0.4, mask_threshold=0, pretrained_model=None, nucleus_bottom_percentile=0.05, nucleus_top_percentile=99.95, cyto_bottom_percentile=0.1, cyto_top_percentile=99.9, check_preprocessing=False, verbose=True):   
    
    from cellpose import models
    from cellpose.io import logger_setup 
    #logger_setup()
    
    if pretrained_model is None:
        model = models.Cellpose(gpu=gpu, model_type='cyto')
    
    else:
        path_to_models = os.path.join(os.path.dirname(__file__), 'models')
        with open(os.path.join(path_to_models, 'models.json'), 'r') as f:
            dic = json.load(f)
            
        if pretrained_model in dic.keys():
            path_to_model = os.path.join(path_to_models, dic[pretrained_model]['path'])
            if os.path.isfile(path_to_model):
                pretrained_model = path_to_model
            else: 
                url = dic[pretrained_model]['link']
                print('Downloading model from Nextcloud...')
                request.urlretrieve(url, os.path.join(path_to_models, path_to_model))
                pretrained_model = os.path.join(path_to_models,dic[pretrained_model]['path'])

        model = models.CellposeModel(gpu=gpu, pretrained_model=pretrained_model)
    
    images = np.stack((cytoplasm, nucleus), axis=1)
    
    print('Running Cellpose')
    masks =  model.eval(images, diameter=diameter, channels=[1,2], flow_threshold=flow_threshold, mask_threshold=mask_threshold, normalize=True, verbose=verbose)[0].astype('uint8')
    
    return masks

def segment_looped(cytoplasm, nucleus, gpu=True, model_type='cyto', channels=[1,2], diameter=None, flow_threshold=0.4, mask_threshold=0, pretrained_model=None, nucleus_bottom_percentile=0.05, nucleus_top_percentile=99.95, cyto_bottom_percentile=0.1, cyto_top_percentile=99.9, check_preprocessing=False, verbose=True):

    
    if pretrained_model is None:
        model = models.Cellpose(gpu=gpu, model_type='cyto')
    
    else:
        path_to_models = os.path.join(os.path.dirname(__file__), 'models')
        with open(os.path.join(path_to_models, 'models.json'), 'r') as f:
            dic = json.load(f)
            
        if pretrained_model in dic.keys():
            path_to_model = os.path.join(path_to_models, dic[pretrained_model]['path'])
            if os.path.isfile(path_to_model):
                pretrained_model = path_to_model
            else: 
                url = dic[pretrained_model]['link']
                print('Downloading model from Nextcloud...')
                request.urlretrieve(url, os.path.join(path_to_models, path_to_model))
                pretrained_model = os.path.join(path_to_models,dic[pretrained_model]['path'])

        model = models.CellposeModel(gpu=gpu, pretrained_model=pretrained_model)
    
    images = np.stack((cytoplasm, nucleus), axis=-1)
    
    print('Running Cellpose')
    masks = np.zeros(cytoplasm.shape, 'uint8')

    for frame in tqdm(range(masks.shape[0])):

        masks[frame] =  model.eval(images[frame], diameter=diameter, channels=[1,2], flow_threshold=flow_threshold, mask_threshold=mask_threshold, normalize=True, verbose=verbose)[0].astype('uint8')

    return masks
