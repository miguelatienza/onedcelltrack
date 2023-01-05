# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 15:52:52 2021
General functions used for all parts of the code.
@author: miguel.Atienza
"""
import os
from tkinter import Y
import numpy as np
import sys
from numba import njit, prange
from tqdm import tqdm
from skimage.feature import peak_local_max
import cv2
import datetime
import multiprocessing as mp
try:
    import cupy as cp
except:
    pass
from scipy import signal as sg
from skimage.transform import rescale
import matplotlib.pyplot as plt
from pathlib import Path


def extract_from_tif(cytoplasm_file=None, nucleus_file=None, lanes_file=None, image_indices=None, x_range=None, y_range=None, data_path=None, get_metadata=True): 
    """
    
    Takes in filepaths for cytoplasm, nucleus and lanes as tif and converts them to numpy arrays.

    Parameters
    ----------
    cytoplasm_file : string
    file_name of tif containing cytoplasm images (n_images, height, width)
    nucleus_file : string
        file_name of tif containing nucleus images (n_images, height, width)
    lanes_file : string
        file_name of tif containing lanes image (height, width)
    image_indices : list, optional
        image indices to use. The default is None and uses the full set of images.
    x_range : TYPE, optional
        x zoom [x_0, x_1]. The default is None and uses the full x value range.
    y_range : TYPE, optional
        y zoom [y_0, y_1]. The default is None and uses the full y value range.
p.copy(x)
    TypeError
        DESCRIPTION.

    Returns
    -------
    cytoplasm: numpy array (image_index, x, y)
    nucleus: numpy array (image_index, x, y)
    lanes

    """
    
    from tifffile import imread, TiffFile
    
    if cytoplasm_file is None and nucleus_file is None:
        raise TypeError('cytoplasm_file and nucleus_file cannot be both empty')
        
    if type(x_range)!=type(y_range):
        raise TypeError('x_range and y_range should either both be default or both specified')
        
    if data_path is None:
        data_path = os.getcwd() + '/../data/'
    
    if cytoplasm_file is not None:
        #This is quite slow but is not memory hungry, should become faster with OMERO
        tif = TiffFile(data_path + cytoplasm_file)
        n_images = len(tif.pages)
        height, width = tif.pages[0].shape
 
    else:
        tif = TiffFile(data_path + nucleus_file)
        n_images = len(tif.pages)
        height, width = tif.pages[0]
    if image_indices is not None:
        i_0, i_f = image_indices
    else:
        i_0, i_f = [0, n_images]
    
    if x_range is not None:
        x_0, x_f = x_range
        y_0, y_f = y_range[0.0, 1022, 268.663662644234, 232.974636223668]

    else:
        x_0, x_f = [0, width]
        y_0, y_f = [0, height]
     
    if x_range is not None:
        x_0, x_f = x_range
        y_0, y_f = y_range
        
    if cytoplasm_file is not None:
        cytoplasm = imread(os.path.join(data_path, cytoplasm_file), key=range(i_0, i_f))#[:][y_0:y_f, x_0:x_f]
            
    else:
        cytoplasm=None
        
    if nucleus_file is not None:
        nucleus = imread(data_path + nucleus_file, key=range(i_0, i_f))#[:][y_0:y_f, x_0:x_f]
    
    else:
        nucleus=None    
    
    if lanes_file is not None:
        lanes = imread(data_path + lanes_file)[y_0:y_f, x_0:x_f]
    else:
        lanes=None
        
    return cytoplasm, nucleus, lanes
        
        
def normalise_image(image, bottom_percentile=1, top_percentile=99): 
    """
    
    Parameters
    ----------
    image : array (height, width)
        DESCRIPTION.
    bottom_percentile : int or float, optional
        The default is 1.
    top_percentile : TYPE, optional
        DESCRIPTION. The default is 99.

    Returns
    -------
    X : numpy array image
        normalised image.

    """
    
    if np.max(image)==np.min(image):
        return image
    X = image.copy()
    
    x_min = np.percentile(X, bottom_percentile)
    x_max = np.percentile(X, top_percentile)
    
    X = np.clip((X - x_min) / (x_max - x_min),0, 1).astype('float32')
    
    return X
    
def create_rgb(r, g, b, r_alpha=1, g_alpha=1, b_alpha=1):
    
    X = normalise_image(r, 0, 100)*r_alpha
    Y = normalise_image(g, 0, 100)*g_alpha
    Z = normalise_image(b, 0, 100)*b_alpha
    
    image = np.stack((X,Y,Z), axis=-1)
    
    return image  

def nb_percentile(x, percentile):
    y = np.zeros(x.shape[0])
    for i in prange(x.shape[0]): 
        y[i] = np.percentile(x[i], percentile)
    
    return y

def preprocess_old(x, c=1, b=0, bottom_percentile=0.01, top_percentile=99.99, log=False, return_type='float32'):
    
    print('Preprocessing...')

    try:
        t, h, w = x.shape
        
    except ValueError:
        return(preprocess_single_image(x, c, b, bottom_percentile, top_percentile, log, return_type))
    
    if log:
        x = np.clip(x.astype('float32'), 0.5, 2**16)
        x = np.log(x.reshape(t, int(h*w)), dtype='float32')
    elif x.dtype=='uint8':
        x = np.divide(x.reshape(t, int(h*w)), 255, dtype='float32')
    elif x.dtype=='uint16':
        x = np.divide(x.reshape(t, int(h*w)), 65535, dtype='float32')
        

    # lows = np.percentile(x, np.float32(bottom_percentile), axis=1).astype('float32')
    # highs = np.percentile(x, np.float32(top_percentile), axis=1).astype('float32')

    lows = nb_percentile(x, np.float32(bottom_percentile))
    highs = nb_percentile(x, np.float32(top_percentile))
    
    if np.sum(lows==highs)>0:
        raise Exception(f'Careful, the top percentile: {top_percentile} is the same as the bottom percentile: {bottom_percentile}!')
        
    highs = highs[:, np.newaxis]
    lows = lows[:, np.newaxis]
    x = np.clip(np.float32(c)*x+np.float32(b), lows, highs).astype('float32')
  
    x = ((x - lows[np.newaxis,:])/(highs-lows)[np.newaxis,:]).astype('float32')

    if return_type =='uint8':
        
        x = (x*255).astype('uint8').reshape((t,h,w))
    elif return_type=='uint16':
        x = (x*65535).astype('uint16').reshape((t,h,w))

    else:
        x = x.astype(return_type).reshape((t,h,w))
        
    return x


def preprocess(x, c=1, b=0, bottom_percentile=0.01, top_percentile=99.99, log=False, return_type='float32', max_memory=500):
    
    print('Preprocessing...')

    try:
        t, h, w = x.shape
        
    except ValueError:
        return(preprocess_single_image(x, c, b, bottom_percentile, top_percentile, log, return_type))
    
    if t <= max_memory:
        return preprocess_old(x, c=c, bottom_percentile=bottom_percentile, top_percentile=top_percentile, log=log, return_type=return_type)
    
    i_0, i_f = 0, t
    i_step = max_memory
    index_values = np.append(np.arange(i_0, i_f, i_step), i_f)

    X = np.zeros(x.shape, dtype=return_type)

    for j in tqdm(range(index_values.size -1)):

        indices = np.arange(index_values[j], index_values[j+1])
        y = x[indices]
        t=indices.size

        if log:
            
            if y.dtype=='uint16':
                y= np.clip(y.astype('float32'), 0.5, 65535)
            elif y.dtype=='uint8':
                y= np.clip(y.astype('float32'), 0.5, 255)

            y= np.log(y.reshape(t, int(h*w)), dtype='float32')
        elif y.dtype=='uint8':
            y= np.divide(y.reshape(t, int(h*w)), 255, dtype='float32')
        elif y.dtype=='uint16':
            y= np.divide(y.reshape(t, int(h*w)), 65535, dtype='float32')
            

        # lows = np.percentile(x, np.float32(bottom_percentile), axis=1).astype('float32')
        # highs = np.percentile(x, np.float32(top_percentile), axis=1).astype('float32')

        lows = nb_percentile(y, np.float32(bottom_percentile))
        highs = nb_percentile(y, np.float32(top_percentile))
        
        if np.sum(lows==highs)>0:
            raise Exception(f'Careful, the top percentile: {top_percentile} is the same as the bottom percentile: {bottom_percentile}!')
            
        highs = highs[:, np.newaxis]
        lows = lows[:, np.newaxis]
        y= np.clip(np.float32(c)*y+np.float32(b), lows, highs).astype('float32')
    
        y= ((y- lows[np.newaxis,:])/(highs-lows)[np.newaxis,:]).astype('float32')

        if return_type =='uint8':
            
            y= (y*255).astype('uint8').reshape((t,h,w))
        elif return_type=='uint16':
            y= (y*65535).astype('uint16').reshape((t,h,w))

        else:
            y= y.astype(return_type).reshape((t,h,w))
            
        X[indices]=y
    
    return X

def preprocess_single_image(x, c=1, b=0, bottom_percentile=0.1, top_percentile=99.9, log=False, return_type='float32'):
    
    if log:
        x = np.log(x, dtype='float32')
    elif x.dtype=='uint8':
        x = (x/255).astype('float32')
    elif x.dtype=='uint16':
        x = (x/65535).astype('float32')

    lows = np.percentile(x, bottom_percentile)
    highs = np.percentile(x, top_percentile)
    
    if np.sum(lows==highs)>0:
        raise Exception(f'Careful, the top percentile: {top_percentile} is the same as the bottom percentile: {bottom_percentile}!')

    x = np.clip(x, a_min=lows, a_max=highs)
    
    x = ((x - lows)/(highs-lows))
 
    if return_type =='uint8':
        
        x = (x*255).astype('uint8')
    elif return_type=='uint16':
        x = (x*65535).astype('uint16')
    else:
        x = x.astype(return_type)
        
    return x
    
def cellpose_input_image(cytoplasm, nucleus):
    
    X = np.stack((cytoplasm, nucleus), axis=-1)
    
    return X
    
    
def get_random_set(data_path, size, pathout=None, osys='linux'):
    
    from tifffile import imread, TiffFile, imwrite
    import pandas as pd
    from tqdm import tqdm
    
    directory = os.listdir(data_path)
    file_names = [file for file in directory if 'UNikon' in file ]
    fields_of_view = [file.split('XY')[1][:2] for file in file_names]
    fields_of_view.sort()
    nucleus_files = [file for file in file_names if 'texasRed' in file]
    nucleus_files.sort()
    cyto_files = [file for file in file_names if 'CY5filter' in file]
    cyto_files.sort()
    n_fov = len(cyto_files)
    
    tif = TiffFile(data_path + cyto_files[0])
    n_images = len(tif.pages)
    height, width = tif.pages[0].shape
    
    cyto_set = np.zeros((int(size*n_fov), height, width))
    nucleus_set = np.zeros((int(size*n_fov), height, width))
    meta_data = {'cyto_file':[], 'nucleus_file':[], 'index':[]}
    
    for fov in tqdm(range(n_fov)):
        
        indices = np.random.randint(0, n_images, size)
        
        cyto_set[fov*size:int((fov+1)*size)] = imread(data_path + cyto_files[fov], key=indices)
        
        nucleus_set[fov*size:int((fov+1)*size)] = imread(data_path + nucleus_files[fov], key=indices)
    
        meta_data['cyto_file'].extend(
            [cyto_files[fov]]*size)
        meta_data['nucleus_file'].extend(
            [nucleus_files[fov]]*size)
        meta_data['index'].extend(list(
            indices))
        
    if pathout is None:
        return cyto_set, nucleus_set
    
    else:
        
        imwrite(os.path.join(pathout, 'cyto_test_set.tif'), (cyto_set).astype('uint8'))
        imwrite(os.path.join(pathout, 'nucleus_test_set.tif'), (nucleus_set).astype('uint8'))
    
    
    df = pd.DataFrame.from_dict(meta_data)
    df.to_csv(os.path.join(pathout, 'meta_data.csv'))
    
    
def get_best_fitting_line(coordinates, x, y):
    
    x_0, x_f, y_0, y_f = coordinates
    m_array = (y_f - y_0)/(x_f- x_0)
    c_array = y_0 - m_array*x_0
    m, c = np.polyfit(x, y, 1)
    
    line_index = np.argmin(np.sqrt((m_array - m)**2 + (c_array -c)**2))
    line_coordinates = [x_0[line_index], x_f[line_index], y_0[line_index], y_f[line_index]]
    
    return line_coordinates, line_index


def get_lanes_for_kymograph(coordinates, line_width, image_shape):
    
    x_0, x_f, y_0, y_f = coordinates
    height, width = image_shape
    
    steep = (y_f-y_0)**2 > (x_f-x_0)**2

    #Note this approach has issues: some pixels are 'ignored' if the lane is not close to horizontal or close to vertical
    if not steep:
        
        x_range = np.floor(x_f-x_0).astype(int)
        x_lane = np.zeros((line_width, x_range)).astype(int)
        y_lane = np.zeros((line_width, x_range)).astype(int)
        m = (y_f-y_0)/(x_f-x_0)
        
        #Compute first x_values
        y_0_values = np.round(np.arange(y_0-line_width/2, y_0+line_width/2)).astype(int)
        x_0_values = np.round(x_0 + -(y_0_values-y_0)*m).astype(int)
        
        for line in range(line_width):
            
            x_0_line = x_0_values[line]
            y_0_line = y_0_values[line]
            x_f_line = x_0_line + x_range
            x_values = np.round(np.arange(x_0_line, x_f_line)).astype(int)
            y_values = np.round(y_0_line + m*(x_values-x_0_line)).astype(int)
            
            x_lane[line] = x_values
            y_lane[line] = y_values
            
        if not y_lane.max() < height:

            if m>0:
                last_valid_index = np.min(np.argwhere(y_lane==height-1)[:,1])
                x_lane = x_lane[:last_valid_index]
                y_lane = y_lane[:last_valid_index]
            else:
                first_valid_index = np.max(np.argwhere(y_lane==height-1)[:,1])
                x_lane = x_lane[:,first_valid_index:]
                y_lane = y_lane[:,first_valid_index:]
        elif not y_lane.min() >= 0:

            if m>0:
                first_valid_index = np.max(np.argwhere(y_lane==0)[:,1])
                x_lane = x_lane[first_valid_index:]
                y_lane = y_lane[first_valid_index:]
            else:
                last_valid_index = np.min(np.argwhere(y_lane==0)[:,1])
                x_lane = x_lane[:last_valid_index]
                y_lane = y_lane[:last_valid_index]
        
    else:
        y_range = np.ceil(y_f-y_0).astype(int)
        y_lane = np.zeros((line_width, y_range)).astype(int)
        x_lane = np.zeros((line_width, y_range)).astype(int)
        m = (x_f-x_0)/(y_f-y_0)

        #Compute first x_values
        x_0_values = np.round(np.arange(x_0-line_width/2, x_0+line_width/2)).astype(int)
        y_0_values = np.round(y_0 + -(x_0_values-x_0)*m).astype(int)
        
        for line in range(line_width):
            
            y_0_line = y_0_values[line]
            x_0_line = x_0_values[line]
            y_f_line = y_0_line + y_range
            y_values = np.round(np.arange(y_0_line, y_f_line)).astype(int)
            x_values = np.round(x_0_line + m*(y_values-y_0_line)).astype(int)
            
            y_lane[line] = y_values
            x_lane[line] = x_values
            
    return x_lane, y_lane


def get_kymograph(image_stack, coordinates, line_width, nuc_track=None):
    
    height, width = image_stack.shape[1:]
    x_lane, y_lane = get_lanes_for_kymograph(coordinates, line_width, [height,width])
 
    kymograph = image_stack[:, y_lane, x_lane]
    #del image_stack
    kymograph = np.max(kymograph, axis=1).T
    
    return kymograph



def get_edges(kymo):
    
    front = np.argmax(kymo, axis=0)
    rear = kymo.shape[0] - np.argmax((kymo)[::-1,:], axis=0) -1
    #center = np.argmin(kymo, axis=0)
    
    return front, rear
    

def np_to_mp4(x, out, crf=0, rate=10, vf=None):
        

    import skvideo.io
    from tqdm import tqdm    

    outputdict={
      '-vcodec': 'libx264',  #use the h.264 codec
      '-crf': str(crf),           #set the constant rate factor to 0, which is lossless
      #'-preset':'faster'   #the slower the better compression, in princple, try 
      '-r': str(rate)
      #other options see https://trac.ffmpeg.org/wiki/Encode/H.264
    }

    # inputdict={
    #   '-r': str(rate),
    #   'crf':str(crf)
    # }

    if vf is not None:
        outputdict['vf']=vf

    writer = skvideo.io.FFmpegWriter(out, 
        #inputdict=inputdict,
      outputdict=outputdict)

    print(f'Encoding array to {os.path.basename(out)}...')
    for frame in tqdm(x):
        writer.writeFrame(frame)
    writer.close()

    return
#     print(f'It took {round(time()-t_0, 1)}s to compress the tif stack of size {os.path.getsize(stack_in)*1e-9} Gb')
#     compression_factor = os.path.getsize(stack_in)/os.path.getsize(out)
# print(f'Compression factor was {round(compression_factor, 3)}')      

# "drawtext=fontfile=/path/to/font.ttf:text='Stack Overflow':fontcolor=white:fontsize=24:box=1:boxcolor=black@0.5:boxborderw=5:x=(w-text_w)/2:y=(h-text_h)/2"

def tifs_to_mp4(dir, crf=0):
    from tifffile import imread, TiffFile

    files = os.listdir(dir)
    files = [os.path.join(dir, file) for file in files if file.endswith('.tif')]
    for file in files:

        tif = TiffFile(file)
        n_images = len(tif.pages)
        height, width = tif.pages[0].shape
        x = imread(file).reshape(n_images, height, width)

        video_file = file.split('.')[0] + '.mp4'
        np_to_mp4(x, video_file, crf=crf)    

def mp4_to_np(file, frames=None, as_grey=True):
   
    filename = Path(file)
    #print(f'Reading {os.path.basename(file)} ...')
    if os.path.isfile(file+'.npz'):
        x = np.load(file+'.npz').f.arr_0
        return x
    elif os.path.isfile(filename.with_suffix('.npz')):
        loader = np.load(str(filename.with_suffix('.npz')))
        x = loader.f.arr_0
        del loader
        return x
    elif file.endswith('.npz'):
        x = np.load(file).f.arr_0
        return x
    
        
    import skvideo.io

    if frames is not None:
        if as_grey:
            x = skvideo.io.vread(file, as_grey=True)[frames,:,:,0]
   
        else:
            x = skvideo.io.vread(file, as_grey=False)[frames,:,:,:]
   
    else:
        if as_grey:
            x = skvideo.io.vread(file, as_grey=True)[:,:,:,0]
   
        else:
            x = skvideo.io.vread(file, as_grey=False)[:,:,:,:]
          
    #print('Done Reading')
    return x

    
def remove_peaks(x, max_step=5, max_peak_width=5):

    y = np.copy(x)
    
    def remove_peak(width, x):
        
        y= np.copy(x)

        for i in range(1,x.size-width):
            
            my_bool = ((x[i]-x[i-1])>max_step and (x[i]-x[i+width])>max_step) or ((x[i]-x[i-1])<-max_step and (x[i]-x[i+width])<-max_step)
            #my_bool is true if the point is greater than(by at least max step ) the point to its left and to its right.

            if my_bool:
                i_vals = np.arange(i, i+width)
                i_range = np.array((i-1, i+width))
                x_range = x[i_range]
                y[i:i+width] = np.interp(i_vals, i_range, x_range)
        
        return y
                
    for width in range(1, max_peak_width+1):
        
        y = remove_peak(width, x)
        
    return y


def get_lines(coordinates):
    """Get array for straight line on image

    Args:
        coordinates (tuple): x_0, x_f, y_0, y_f

    Returns:
        x_lane, y_lane: x and y coordinates of the line
    """

    x_0, x_f, y_0, y_f = coordinates

    x_range = x_f-x_0
    y_range = y_f-y_0

    dense_length = np.max((x_range, y_range))*10

    x_dense = np.round(np.linspace(x_0, x_f, dense_length)).astype(int)
    y_dense = np.round(np.linspace(y_0, y_f, dense_length)).astype(int)
   
    dense_indices = np.arange(1, dense_length)

    x_bool = x_dense[dense_indices]!=x_dense[dense_indices-1]

    y_bool = y_dense[dense_indices]!=y_dense[dense_indices-1]

    xy_bool = np.argwhere(np.logical_or(x_bool, y_bool)).flatten()

    x_lane = x_dense[xy_bool]
    y_lane = y_dense[xy_bool]
   
    return x_lane, y_lane


def get_lanes_for_kymograph_2(coordinates, lane_width, image_shape):
    """Take in coordinates for single pixel wide line and generate a lane of width lane_width

    Args:
        coordinates (tuple): [ x_0_center, x_f_center, y_0_center, y_f_center]
        lane_width (int): 
        image_shape (tuple): h, w

    Returns:
        x_lanes, y_lanes: x and y coordinates of the lane as numpy array
    """
    x_0_center, x_f_center, y_0_center, y_f_center = coordinates
    h, w = image_shape

    #Force the lane_width to be odd
    lane_width = lane_width + (not lane_width%2)

    #Calculate shifts in y direction from the central line
    shifts = np.arange(-int(lane_width/2), int(1+lane_width/2))
    
    x_range = x_f_center-x_0_center
    y_range = y_f_center-y_0_center

    #If y_range is greater, then shift along x and viceversa
    x_shifts = shifts*(y_range>x_range)
    y_shifts = shifts*(x_range>=y_range)


    x_lane_center, y_lane_center = get_lines(coordinates)

    x_lanes = np.zeros((lane_width, x_lane_center.size)).astype(int)
    y_lanes = np.zeros((lane_width, x_lane_center.size)).astype(int)
    
    # print('hi', x_lanes[np.arange(line_width)].shape)
    # print((x_lane_center[np.newaxis, :]+shifts[:, np.newaxis]).shape)


    x_lanes[np.arange(lane_width)] = x_lane_center[np.newaxis, :] + x_shifts[:, np.newaxis]
    y_lanes[np.arange(lane_width)] = y_lane_center[np.newaxis, :] + y_shifts[:, np.newaxis]

    #Make sure that no values are outside of the image
    x_lanes = np.clip(x_lanes, 0, w-1)
    y_lanes = np.clip(y_lanes, 0, h-1)

    return x_lanes, y_lanes 

def get_hough_space(args):

    y_0, delta_y, h, w, kernel_width, kernel, image = args

    y_f = y_0 + delta_y
#check if all coordinates are inside image, otherwise pass
    
    if (y_f > h-kernel_width/2) or (y_f < kernel_width/2):
        return 0

    coordinates = 0 , w, y_0, y_f
    x_lanes, y_lanes = get_lanes_for_kymograph_2(coordinates, kernel_width, image.shape)
        

    mask = np.zeros(image.shape, dtype='float32')
    mask[y_lanes, x_lanes] = kernel[:, np.newaxis]

    convolution = np.sum(mask*image)/mask.size
    return convolution

def hough(image, delta_y_max, kernel_width, multiprocess=False, debug=False, gpu=True):

    if gpu:
        try:
            foo = cp.array([1,2])
        except:
            print('Warning gpu is not available, using cpu...')
            gpu=False
    if gpu:
        return gpu_hough(image, delta_y_max, kernel_width, debug=False)
    
    h,w = image.shape

    y_0_array = np.arange(int(kernel_width/2),h-int(kernel_width/2))
    delta_y_array = np.arange(-delta_y_max, delta_y_max+1)
    i=0

    hough_space = np.zeros((delta_y_array.size, y_0_array.size))
    
    kernel_width = kernel_width + (not kernel_width%2) #Force odd kernel width
    kernel = np.arange(int(-kernel_width/2), int(1+kernel_width/2))

    for i in tqdm(range(y_0_array.size)):
        y_0 = y_0_array[i]
        
        if multiprocess:

            delta_y_list = [delta_y_array[j] for j in range(delta_y_array.size)]

            # def get_hough_space(delta_y):
            #     y_f = y_0 + delta_y
            # #check if all coordinates are inside image, otherwise pass
                
            #     if (y_f > h-kernel_width/2) or (y_f < kernel_width/2):
            #         return 0
            
            #     coordinates = 0 , w, y_0, y_f
            #     x_lanes, y_lanes = get_lanes_for_kymograph_2(coordinates, kernel_width, image.shape)
                    

            #     mask = np.zeros(image.shape, dtype='float32')
            #     mask[y_lanes, x_lanes] = kernel[:, np.newaxis]

            #     convolution = np.sum(mask*image)/mask.size
            #     return convolution

            args = [[y_0, delta_y_list[j], h, w, kernel_width, kernel.copy(), image] for j in range(delta_y_array.size)]
            pool = mp.Pool(processes=len(delta_y_list))
            convolutions = pool.map(get_hough_space, args)
            hough_space[:, i] = np.array(convolutions)
            pool.terminate()
            continue

        for j in range(delta_y_array.size):
            delta_y = delta_y_array[j]

            y_f = y_0 + delta_y
            #check if all coordinates are inside image, otherwise pass
            if (y_f > h-kernel_width/2) or (y_f < kernel_width/2):
                if debug:
                    pass
                    #print(f'here: {i, j}')
                continue
            
            coordinates = 0 , w, y_0, y_f
            x_lanes, y_lanes = get_lanes_for_kymograph_2(coordinates, kernel_width, image.shape)
                

            mask = np.zeros(image.shape, dtype='float32')
            mask[y_lanes, x_lanes] = kernel[:, np.newaxis]

            convolution = np.sum(mask*image)/mask.size
            hough_space[j, i] = convolution

    hough_space = hough_space/np.max(hough_space)

    return hough_space

def gpu_hough(image, delta_y_max, kernel_width, max_y_0_size=100, debug=False):

    max_y_0_size=100
    i=0
    
    def batch(image, delta_y_array, y_0_list, kernel_width):
        
        h, w = image.shape
        image = cp.array(image)
        y_0_size = sum(y_0.size for y_0 in y_0_list)
        y_0_size = y_0_list[0].size
        hough = cp.zeros((delta_y_array.size, y_0_size))
        
        i=0
        for delta_y in delta_y_array:
            
            for y_0 in y_0_list:

                x = cp.arange(w)
                y = cp.arange(h)

                long_enough = ((y_0 + delta_y) > kernel_width/2) & ((y_0 + delta_y) < h-kernel_width/2)
                y_0 = y_0[long_enough]
                Y_0, Y, X = cp.meshgrid(y_0, y, x, sparse=True, indexing='ij')

                # p1 = 0, Y_0
                # p2 = w-1, Y_0 + delta_y 

                kernel = Y-(Y_0 + delta_y*X/w)
                kernel = kernel*(cp.abs(kernel)<(kernel_width/2))

                current_hough = cp.sum(kernel*image[cp.newaxis, :, :], axis=(1,2))#/cp.sum(image[cp.newaxis, :,:]*(kernel>0), axis=(1,2))
                hough[i, long_enough] = current_hough

            i+=1
        
        return hough.get()
    
    scaling = 0.25
    print('rescaling')
    image_rescaled = rescale(image, scaling, anti_aliasing=True)
    h,w = image_rescaled.shape
    delta_y_array = np.arange(int(-h/2), int(h/2) + 1)
    kernel_width=5
    h,w = image_rescaled.shape
    min_y0, max_y0 = int(kernel_width/2), h-int(kernel_width/2)
    y_0_list = [cp.arange(min_y0, max_y0)]
    
    hough = batch(image_rescaled, delta_y_array, y_0_list, kernel_width)
    
    
    #delta_y_opt_index = np.unravel_index(np.argmax(hough), hough.shape)[0]
    #delta_y_opt = np.round(delta_y_array[delta_y_opt_index]/scaling).astype(int)
    #print(delta_y_opt)
    
    delta_y_opt_index = peak_local_max(hough, min_distance=10, num_peaks=7)[:, 0]
    delta_y_opt = np.round(delta_y_array[delta_y_opt_index]/scaling).astype(int)
    delta_y_opt.sort()
    delta_y_width = delta_y_opt[-1]-delta_y_opt[0]+2
    delta_y_opt = delta_y_opt[3]
    
    t = np.concatenate(y_0_list).get()
    signal = np.mean(hough[delta_y_opt_index.min()-1:delta_y_opt_index.max()+1], axis=0)

    f, fft = get_spectrum(t, signal)
    x = np.arange(signal.size)
    
    T_opt = int(1/f[np.argmax(np.abs(fft))])

    
    peaks = sg.find_peaks(signal, height=signal.max()*0.2, distance=T_opt*0.6)[0]/scaling
    npeaks = sg.find_peaks(-signal, height=signal.max()*0.2, distance=T_opt*0.6)[0]/scaling
    
    max_coordinates = np.zeros((npeaks.size, 2), dtype='int')
    min_coordinates = np.zeros((peaks.size, 2), dtype='int')
    
    delta_y_array = np.arange(delta_y_opt-int(delta_y_width/2), delta_y_opt+int(delta_y_width/2)+1)
    kernel_width=5
    h,w = image.shape
    min_y0, max_y0 = int(kernel_width/2), h-int(kernel_width/2)
    #plt.subplot(121)
    #plt.imshow(np.clip(image, 0, 3000))
    
    for i in range(peaks.size):
        
        y_0 = peaks[i]
        
        #plt.plot([0,w], [y_0, y_f], color='red')
        #continue
        
        
        peak = peaks[i]
        y_0_width = 15
        y_0_left = np.clip(peak-y_0_width, min_y0, max_y0)
        y_0_right = np.clip(peak+y_0_width+1, min_y0, max_y0)

        y_0_list = [cp.arange(y_0_left, y_0_right, dtype='int')]
        
        hough_current = batch(image, delta_y_array, y_0_list, kernel_width)
        #plt.imshow(hough_current)
        #plt.colorbar()
        #break
        #print(hough_current.shape)
        #dy, y_0 = peak_local_max(hough_current, num_peaks=1)[0]
        dy, y_0 = np.unravel_index(np.argmax(hough_current), hough_current.shape)
        dy = delta_y_array[dy]
        y_0 = peak-y_0_width + y_0
        min_coordinates[i] = np.array([dy, y_0])
        
        #plt.plot([0,w], [y_0, y_0+dy], color='red')
        
    for i in range(npeaks.size):
        
        peak = npeaks[i]
        y_0_width = 15
        y_0_list = [cp.arange(peak-y_0_width, peak+y_0_width+1, dtype='int')]
        
        hough_current = batch(image, delta_y_array, y_0_list, kernel_width)
        #plt.imshow(hough_current)
        #plt.colorbar()
        #break
        #print(hough_current.shape)
        #dy, y_0 = peak_local_max(hough_current, num_peaks=1)[0]
        dy, y_0 = np.unravel_index(np.argmin(hough_current), hough_current.shape)
        dy = delta_y_array[dy]
        y_0 = peak-y_0_width + y_0
        max_coordinates[i] = np.array([dy, y_0])
        
    return min_coordinates, max_coordinates



def distance_to_line(p1, p2, X, Y):
    """_summary_

    Args:
        p0 (_type_): _description_
        p1 (_type_): _description_
        X (_type_): _description_
        Y (_type_): _description_

    Returns:
        _type_: Distance between the point (X, Y) which can be an array of points and the line between p0 and p1.
    """
    x1, y1 = p1
    x2, y2 = p2 

    d = np.abs(
        (x2-x1)*(y1-Y) - (x1-X)*(y2-y1)
    )/ np.sqrt(
        (x2-x1)**2 + (y2-y1)**2
    )
    
    return d

def get_lane_mask(image, delta_y_max=20, kernel_width=5, line_distance=30, threshold=0.5, debug=False, gpu=True):
    """Function that takes in an image of a lines pattern, and returns a mask of the detected lanes. The algorithm assumes that the experimentator has tried to get the lanes to run as close to horizontal as possible.

    Args:
        image (_type_): _description_
        delta_y_max (_type_): _description_
        kernel_width (int, optional): _description_. Defaults to 5.
        line_distance (int, optional): _description_. Defaults to 30. Estimate of the distance between the lines in pixels, where one is still 100% sure that it is below the real distance.
    
    Returns:
        Mask image containing 0s where there is no lane, and a different integer for every separate line.
    """
    print(gpu)
    
    #print('Detecting lanes...')
    h, w = image.shape
    if not gpu:
        myhough = hough(image, delta_y_max, kernel_width, debug=debug, gpu=gpu)

        min_coordinates = peak_local_max(myhough, min_distance=line_distance, exclude_border=False, threshold_rel=threshold)
        max_coordinates = peak_local_max(-myhough, min_distance=line_distance, exclude_border=False, threshold_rel=threshold)
    
    if gpu:
        print('doing this')
        min_coordinates, max_coordinates = gpu_hough(image, delta_y_max, kernel_width)
        print('done')
    delta_y_array = np.arange(-delta_y_max, delta_y_max+1)
    # max_coordinates_sorted = max_coordinates.copy()
    # min_coordinates_sorted = min_coordinates.copy()
    
    # max_coordinates_sorted[:,1].sort()
    # min_coordinates_sorted[:,1].sort()
    max_coordinates = max_coordinates[tuple([max_coordinates[:, 1].argsort()])]
    min_coordinates = min_coordinates[tuple([min_coordinates[:, 1].argsort()])]

    if debug:
        #show the hough space
        #import matplotlib.pyplot as plt
        #plt.subplot(121)
        #plt.imshow(myhough)
        #plt.scatter(min_coordinates[:,1], min_coordinates[:,0], color='red')
        #plt.scatter(max_coordinates[:,1], max_coordinates[:,0], color='white')
        #print(min_coordinates[1, :])
        #plt.subplot(122)
        #plt.imshow(image)
      
        #x = [0, image.shape[1]]
        #for i in range(min_coordinates.shape[0]):
        #    plt.plot(x, [min_coordinates[i,1], sum(min_coordinates[i])])
        
        return min_coordinates, max_coordinates
        

    if not gpu:
        max_coordinates[:,0]=delta_y_array[max_coordinates[:,0]]
        min_coordinates[:,0]=delta_y_array[min_coordinates[:,0]]

    if max_coordinates.size == min_coordinates.size:

        if max_coordinates[0,1] > min_coordinates[0,1]: #All lanes are complete
            pass
        else: #The top lane has no top boundary, and the bottom lane has no bottom boundary
            max_coordinates = max_coordinates[1:,:]
            min_coordinates = min_coordinates[:-1,:]
    elif max_coordinates.size +2 == min_coordinates.size: #The bottom lane has no bottom boundary
        min_coordinates = min_coordinates[:-1,:]
    elif max_coordinates.size -2 == min_coordinates.size: #The top lane has no top boundary
        max_coordinates = max_coordinates[1:,:]
    else:
        raise Warning('The lane detection has not worked well...')

    lane_mask = np.zeros(image.shape, dtype='uint8')
    lane_metric = np.zeros(image.shape, dtype='float32')
    X, Y = np.meshgrid(np.arange(w), np.arange(h))

    if debug:
        plt.imshow(image, vmin=0, vmax=1000)
    #Loop through the top and bottom lines to create the mask
    for i in range(max_coordinates.shape[0]):
        
        top_coordinates = [0, w, max_coordinates[i,1], np.sum(max_coordinates[i])]
        bottom_coordinates = [0, w, min_coordinates[i,1], np.sum(min_coordinates[i])]
    
        top_x, top_y = get_lines(top_coordinates)
        bottom_x, bottom_y = get_lines(bottom_coordinates)

        y_0_mean = np.round(np.mean((top_y[0], bottom_y[0]))).astype(int)
        y_f_mean = np.round(np.mean((top_y[-1], bottom_y[-1]))).astype(int)

        lane_width = np.mean((top_y[0] - bottom_y[0], top_y[-1]- bottom_y[-1]))

        lane_width = np.round(lane_width).astype(int)
    
        coordinates = 0, w, y_0_mean, y_f_mean
        x_bool, y_bool = get_lanes_for_kymograph_2(coordinates, lane_width, lane_mask.shape)
        
        lane_mask[y_bool, x_bool]=i+1

        #Now create the lane_metric array
        

        #p1 = x_bool[0,0], y_bool[0,0]
        p2 = np.array((x_bool[-1,0], y_bool[-1,0]), dtype='float32')
        tan = (np.abs(y_f_mean-y_0_mean)/w).astype('float32')
        p1 = p2 + np.array([-tan*lane_width, -lane_width]).astype('float32')

        if debug:
            #print(p1,p2)
            #plt.scatter(p1[0], p1[1], color='blue')
            #plt.scatter(p2[0], p2[1], color='red')
        #p2 = x_bool[1,0], y_bool[1,0]  
            pass

        #Use a wider lane_width for the metric, to prevent nuclei slightly outside the lane from going undetected
        x_bool, y_bool = get_lanes_for_kymograph_2(coordinates, lane_width+10, lane_mask.shape)
        lane_metric[y_bool, x_bool] = distance_to_line(p1, p2, X[y_bool, x_bool], Y[y_bool, x_bool])

    return lane_mask, lane_metric

def get_foot_print(masks, out=None, crf=0, rate=10, write=False):
    print('Getting footprint...')
    from skvideo.io import FFmpegWriter

    masks = masks>0
    if write:
        if out is None:
            raise ValueError('An output path is needed if write=True is passed!')

        writer = FFmpegWriter(out, 
            outputdict={
        '-vcodec': 'libx264',  #use the h.264 codec
        '-crf': str(crf),           #set the constant rate factor to 0, which is lossless
        #'-preset':'faster'   #the slower the better compression, in princple, try 
        '-r': str(rate)
        #other options see https://trac.ffmpeg.org/wiki/Encode/H.264
        })

        
        # foot_print = np.zeros(masks.shape, dtype='uint16')
        for i in tqdm(range(0, masks.shape[0])):
            # foot_print[i] = np.sum(masks[:i], axis=0)

            writer.writeFrame(np.sum(masks[:i], axis=0), dtype='uint16')
        writer.close()

        return
    
    else:
        footprint = np.zeros(masks.shape, dtype='uint16')
        for i in tqdm(range(1, masks.shape[0])):

            footprint[i] = footprint[i-1] + masks[i-1]
        
        return footprint


def read_nd2(file, v, frames=None, c=None):

    from nd2reader import ND2Reader
    #print('Reading nd2')
    f = ND2Reader(file)
    
    if frames is None:
        if c is None:
            x = f.get_frame_2D(v=v)
            return x
        else:
            x = f.get_frame_2D(v=v, c=c)
            return x

    x = np.zeros((
        frames.size, f.sizes['y'], f.sizes['x']), dtype='uint16')

    i=0
    for frame in frames:
        x[i] = f.get_frame_2D(v=v, t=frame, c=c)
        i+=1
    
    #print('Done reading.')
    return x

def label_movie(stack, fpm=2):

    org = (0, stack.shape[1])

    for frame in range(stack.shape[0]):
        
        minutes = frame/fpm

        text = f'Frame: {frame} {str(datetime.timedelta(minutes=minutes))}'
        stack[frame] = cv2.putText(stack[frame], text, org,cv2.FONT_HERSHEY_SIMPLEX, 0.85, (250, 250, 250))
    
    return stack 


def get_FN_intensity(image, mask):
    pass

def get_spectrum(t, signal):

    signal = sg.detrend(signal)

    n = signal.shape[0]
    d = (t[-1]-t[0])
    f = np.arange(1+int(n/2))/d
    fft = np.fft.rfft(signal)/(len(f))

    return f, fft




