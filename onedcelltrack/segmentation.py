# -*- coding: utf-8 -*-
import sys
from . import functions
import numpy as np
import os
import json
from urllib import request
#from cellpose.io import logger_setup 
from tqdm import tqdm


class Segmentation:
    """
    Class for segmentation of cells in a stack of images. At the moment using cellpose.
    Based on Pipeline class from onedcelltrack.
    
    Parameters
    ----------
    pipeline : onedcelltrack.Pipeline
        Pipeline object.
    """
    def __init__(self, pretrained_model=None, omni=False, model_type=None, gpu=True):
        """
        Initialize cellpose model.
        
        Parameters
        ----------
        pretrained_model : str
            Path to pretrained model. If None, use default model.
        omni : bool
            Use omni model.
        model_type : cellpose.models.CellposeModel
            Cellpose model. Overridden if pretrained_model is not None.
        gpu : bool

        """

        if omni:
            from cellpose_omni.models import CellposeModel
            self.model = CellposeModel(
            gpu=gpu, omni=True, nclasses=4, nchan=2, pretrained_model=pretrained_model)
            return

        elif pretrained_model is None:
            from cellpose import models
            self.model = models.Cellpose(gpu=gpu, model_type=model_type)

        else:
            from cellpose import models
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

            
            if not omni:
                
                self.model = models.CellposeModel(gpu=gpu, pretrained_model=pretrained_model)
            else:
                from cellpose_omni.models import CellposeModel
                self.model = CellposeModel(
                gpu=gpu, omni=True, nclasses=4, nchan=2, pretrained_model=pretrained_model)


    def segment(self, brightfield, nucleus, channels=[1,2], diameter=None, flow_threshold=0.4, cellprob_threshold=0, verbose=True):
        """
        Segment cells in a stack of images.

        Parameters
        ----------
        model : cellpose.models.CellposeModel
            Cellpose model.
        brightfield : np.ndarray
            Stack of brightfield images or single image.
        nucleus : np.ndarray
            Stack of nucleus images or single image.
        gpu : bool
            Use GPU.
        channels : list
            Channels to use for segmentation.
        diameter : int
            Diameter of cells.
        flow_threshold : float
            Flow threshold.
        cellprob_threshold : float
        verbose : bool
            Print progress.
        """
        ## Infer the number of time_frames in the image
        if len(brightfield.shape) == 2:
            time_frames = 1
        else:
            time_frames = brightfield.shape[0]

        ## Check that the brightfield and nucleus images have the same shape
        if brightfield.shape != nucleus.shape:
            raise ValueError('The shape of the brightfield and nucleus images do not match.')
        
        ## Check that the brightfield and nucleus images have the same dtype
        if brightfield.dtype != nucleus.dtype:
            raise ValueError('The dtype of the brightfield and nucleus images do not match.')
        
        if time_frames == 1:
                images = np.stack((brightfield, nucleus), axis=-1)
                masks = self.model.eval(images, diameter=diameter, channels=channels, flow_threshold=flow_threshold, normalize=True, cellprob_threshold=cellprob_threshold)[0].astype('uint8')

        else:
            masks = np.zeros(brightfield.shape, 'uint8')
            for frame in tqdm(range(time_frames), disable=(not verbose)):
                images = np.stack((brightfield[frame], nucleus[frame]), axis=-1)
                mask = self.model.eval(images, diameter=diameter, channels=channels, flow_threshold=flow_threshold, normalize=True, cellprob_threshold=cellprob_threshold)[0].astype('uint8')
                masks[frame] =  mask.astype('uint8')
        
        return masks



# def segment(cytoplasm, nucleus, gpu=True, model_type='cyto', channels=[1,2], diameter=None, flow_threshold=0.4, mask_threshold=0, pretrained_model=None, nucleus_bottom_percentile=0.05, nucleus_top_percentile=99.95, cyto_bottom_percentile=0.1, cyto_top_percentile=99.9, check_preprocessing=False, verbose=True):   
    
#     from cellpose import models
#     #from cellpose.io import logger_setup 
#     #logger_setup()
    
#     if pretrained_model is None:
#         model = models.Cellpose(gpu=gpu, model_type='cyto')
    
#     else:
#         path_to_models = os.path.join(os.path.dirname(__file__), 'models')
#         with open(os.path.join(path_to_models, 'models.json'), 'r') as f:
#             dic = json.load(f)
            
#         if pretrained_model in dic.keys():
#             path_to_model = os.path.join(path_to_models, dic[pretrained_model]['path'])
#             if os.path.isfile(path_to_model):
#                 pretrained_model = path_to_model
#             else: 
#                 url = dic[pretrained_model]['link']
#                 print('Downloading model from Nextcloud...')
#                 request.urlretrieve(url, os.path.join(path_to_models, path_to_model))
#                 pretrained_model = os.path.join(path_to_models,dic[pretrained_model]['path'])

#         model = models.CellposeModel(gpu=gpu, pretrained_model=pretrained_model)
    
#     images = np.stack((cytoplasm, nucleus), axis=1)
    
#     print('Running Cellpose')
#     masks =  model.eval(images, diameter=diameter, channels=[1,2], flow_threshold=flow_threshold, mask_threshold=mask_threshold, normalize=True, verbose=verbose)[0].astype('uint8')
    
#     return masks

# def segment_looped(cytoplasm, nucleus, gpu=True, model_type='cyto', channels=[1,2], diameter=None, flow_threshold=0.4, mask_threshold=0, pretrained_model=None, nucleus_bottom_percentile=0.05, nucleus_top_percentile=99.95, cyto_bottom_percentile=0.1, cyto_top_percentile=99.9, check_preprocessing=False, verbose=True):

    
#     if pretrained_model is None:
#         model = models.Cellpose(gpu=gpu, model_type='cyto')
    
#     else:
#         path_to_models = os.path.join(os.path.dirname(__file__), 'models')
#         with open(os.path.join(path_to_models, 'models.json'), 'r') as f:
#             dic = json.load(f)
            
#         if pretrained_model in dic.keys():
#             path_to_model = os.path.join(path_to_models, dic[pretrained_model]['path'])
#             if os.path.isfile(path_to_model):
#                 pretrained_model = path_to_model
#             else: 
#                 url = dic[pretrained_model]['link']
#                 print('Downloading model from Nextcloud...')
#                 request.urlretrieve(url, os.path.join(path_to_models, path_to_model))
#                 pretrained_model = os.path.join(path_to_models,dic[pretrained_model]['path'])

#         model = models.CellposeModel(gpu=gpu, pretrained_model=pretrained_model)
    
#     images = np.stack((cytoplasm, nucleus), axis=-1)
    
#     print('Running Cellpose')
#     masks = np.zeros(cytoplasm.shape, 'uint8')

#     for frame in tqdm(range(masks.shape[0])):
#         mask = model.eval(images[frame], diameter=diameter, channels=[1,2], flow_threshold=flow_threshold, cellprob_threshold=mask_threshold, normalize=True)[0]
#         masks[frame] =  mask.astype('uint8')

#     return masks
