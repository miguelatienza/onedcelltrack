"""
This is the main file to run the data pipeline for tracking single cells in one dimension. 
It performs the following steps:
1. Detect the lanes in the image
2. Segmentation of the cells in the image
3. Tracking of the cells' nuclei in the image
4. Joining the tracks of the cells' nuclei to cell contours to create a dataframe of the cell's trajectories (front, rear, nucleus)
5. Postprocessing of the tracks to filter out tracks not correspoindf to single cells and trajectories with flaws such as unrealistic jumps
6. Classifying the tracks into different dynamic states [SteadySpread, MovingSpread, SteadyOscillatory, MovingOscillatory]
"""
import sys
import time
import logging
import traceback
import pickle
from tqdm import tqdm
from . import functions
from .segmentation import segment, segment_looped
import numpy as np
from tifffile import TiffFile, imwrite, imread
import os
from . import functions
from .segmentation import segment, segment_looped
from . import tracking 
from . import lane_detection
import pandas as pd
from skvideo import io
import json
from nd2reader import ND2Reader
import glob
from trackpy import SubnetOversizeException

def run_pipeline(param_dict):
    """
    Function to run the pipeline on a full experiment containing many fields of view

    Parameters
    ----------
    param_dict : dict
        Dictionary containing all the parameters for the pipeline.
        
        General parameters:
        
        data_path : str
            Path to the data folder.
        image_file : str
            Name of the image file. Can be a .nd2 or .tif file.
        lanes_file : str
            Name of the lanes file.
        path_out : str
            Path to the output folder.
        frame_indices : list
            List of frame indices to be processed.
        fovs : list
            List of field of views to be processed.
        sql : bool 
            Whether to save the intermediate results to a sql database.
        tres : float
            Time resolution of the image file in seconds.
        run_cellpose : bool
            Whether to run cellpose segmentation.
        bf_channel : int
            Index of the brightfield channel in the image file.
        nuc_channel : int
            Index of the nuclear channel in the image file.
                
        Lane detection parameters:
        
        lane_distance : int
            Distance between the lanes in pixels. It should strictly shorter than the distance between the closest lanes.
        lane_low_clip : int
            Lower value for clipping the lanes image for the lane detection.
        lane_high_clip : int
            Higher value for clipping the lanes image for the lane detection.
        lane_threshold : float
            Threshold for the lane detection.
        min_mass : int
            Minimum mass for the detection of the nuclei. (see trackpy documentation)
        max_travel : int
            Maximum travel distance in pixels for a nucleis between subsequent frames. (see trackpy documentation)
        track_memory : int
            Number of frames a nucleus can be lost before it is considered a new nucleus. (see trackpy documentation)
        diameter : int
            Diameter of the nuclei in pixels. (see trackpy documentation)
        min_frames : int
            Minimum number of frames a nucleus should be tracked to be considered a cell.
        cyto_diameter : int
            Diameter of the cytoplasm in pixels for cellpose segmentation (see cellpose documentation: diameter)
        flow_threshold : int
            Threshold for the flow in the cellpose segmentation (see cellpose documentation: flow_threshold)
        cellprob_threshold : int
            Threshold for the cell probability in the cellpose segmentation (see cellpose documentation: cellprob_threshold)
        pretrained_model : str
            Path to the pretrained model for cellpose segmentation (see cellpose documentation: pretrained_model)
    """
    
    # Get the parameters from the dictionary
    # General parameters
    data_path = param_dict['data_path']
    image_file = param_dict['image_file']
    lanes_file = param_dict['lanes_file']
    path_out = param_dict['path_out']
    frame_indices = param_dict['frame_indices']
    fovs = param_dict['fovs']
    sql = param_dict['sql']
    tres = param_dict['tres']
    run_cellpose = param_dict['run_cellpose']
    bf_channel = param_dict['bf_channel']
    nuc_channel = param_dict['nuc_channel']
    # Lane detection parameters
    lane_distance = param_dict['lane_distance']
    lane_low_clip = param_dict['lane_low_clip']
    lane_high_clip = param_dict['lane_high_clip']
    lane_threshold = param_dict['lane_threshold']
    # Tracking parameters
    min_mass = param_dict['min_mass']
    max_travel = param_dict['max_travel']
    track_memory = param_dict['track_memory']
    diameter = param_dict['diameter']
    min_frames = param_dict['min_frames']
    # Cellpose parameters
    cyto_diameter = param_dict['cyto_diameter']
    flow_threshold = param_dict['flow_threshold']
    cellprob_threshold = param_dict['cellprob_threshold']
    pretrained_model = param_dict['pretrained_model']

    ## prepare the logging
    logging.basicConfig(filename=path_out + 'pipeline.log', filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)

    ## check if the files are present
    check_files(data_path, image_file, lanes_file)

    # Save the arguments to a json file
    with open(path_out + 'pipeline_arguments.json', 'w') as fp:
        json.dump(param_dict, fp)
    
    ## Lane detection
    logging.info('Starting lane detection')




def build_param_dict():

    defaults = os.path.join(os.path.dirname(__file__), 'default_pipeline_arguments.json')
    with open(defaults) as f:
        param_dict = json.load(f)
   
    return param_dict

def check_files(data_path, image_file, lanes_file):
    """
    Checks if the files are present in the data_path and if they are of the right type. Accepted types are .nd2 and .tif.

    Parameters
    ----------
    data_path : str
        Path to the data folder.
    image_file : str or list
        Name of the image file or list of names of the image files.
    lanes_file : str
        Name of the lanes file.  
    """
    if isinstance(image_file, list):
        n_fovs = len(image_file)
        image_file = image_file[0]
    if isinstance(lanes_file, list):
        assert len(lanes_file)==n_fovs, 'The number of lanes files should be equal to the number of image files'
        lanes_file = lanes_file[0]
    else:
        f = ND2Reader(os.path.join(data_path, lanes_file))
        n_fovs = f.sizes['v']

    if not image_file.endswith('.nd2') and not image_file.endswith('.tif'):
        raise ValueError(f'{image_file} is not a .nd2 or .tif file')
    if not lanes_file.endswith('.nd2') and not lanes_file.endswith('.tif'):
        raise ValueError(f'{lanes_file} is not a .nd2 or .tif file')

    if not os.path.isfile(os.path.join(data_path, image_file)):
        raise FileNotFoundError(f'Could not find {image_file} in {data_path}')
    if not os.path.isfile(os.path.join(data_path, lanes_file)):
        raise FileNotFoundError(f'Could not find {lanes_file} in {data_path}')
    
    if image_file.endswith('.tif'):
        assert imread(os.path.join(data_path, image_file)).ndim==4, f'{image_file} is not a 2D image stack with 2 channels'
        assert imread(os.path.join(data_path, image_file)).shape[-1]==2, 'The tif stack should be of the shape (n_frames, height, width, channels) with 2 channels'
    elif image_file.endswith('.nd2'):
            f = ND2Reader(os.path.join(data_path, image_file))
            assert f.sizes['c']==2, 'The nd2 file should have 2 channels'
            assert f.sizes['v']==n_fovs, 'The nd2 file should have the same number of fields of view as the number of lanes files'
    else:
        print('All files are present and of the right type')
    
    return 

def update_files_for_dict(data_path, param_dict, lanes_file, nd2_file=None, cyto_file=None, bf_file=None):
    """
    Updates the image_file and lanes_file in the param_dict if they are not present in the data_path.

    Parameters
    ----------
    data_path : str
        Path to the data folder.
    image_file : str
        Name of the image file.
    lanes_file : str
        Name of the lanes file.
    param_dict : dict
        Dictionary containing all the parameters for the pipeline.
    """
    check_files(data_path, nd2_file, lanes_file)

    param_dict['data_path'] = data_path
    param_dict['nd2_file'] = nd2_file
    param_dict['lanes_file'] = lanes_file

    return param_dict

def read_image(param_dict, frames=None, channel=0, fov=0):
    """
    Function to read the image from the image file.

    Parameters
    ----------
    frames : list
        List of frame indices to be read.
    channel : int
        Index of the channel to be read.
    """
    if param_dict:
        
        return functions.read_nd2(os.path.join(data_path, file_name), fov, frames, channel)

    if file_name.endswith('.mp4'):

        return functions.mp4_to_np(os.path.join(data_path, file_name), frames=frames)

    if file_name.endswith('.tif') or file_name.endswith('.tiff'):
        # if the file_name is a list
        if isinstance(file_name, list):
            return imread(os.path.join(data_path, file_name[fov]), key=frames)
        else:
            return imread(os.path.join(data_path, file_name), key=frames)


def detect_lanes(file_name, fovs, lane_distance=30, low_clip=300, high_clip=2000, v_rel=0.5):

    for fov in fovs:
        lanes_image = read_image(file_name, fov, channel=0)
        lanes_image = np.clip(lanes_image, low_clip, high_clip)

        lanes_mask, lanes_metric = lane_detection.get_lane_mask(
            lanes_image, kernel_width=5, line_distance=lane_distance,
            threshold=v_rel)
        n_lanes = lanes_mask.max()

        try:
            lanes_dir = os.path.join(self.path_out, 'lanes')
            os.mkdir(lanes_dir)
        except FileExistsError:
            pass

        imwrite(os.path.join(lanes_dir, 'lanes_mask.tif'), self.lanes_mask)
        imwrite(os.path.join(lanes_dir, 'lanes_metric.tif'), self.lanes_metric)

        return
