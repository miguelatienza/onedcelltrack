import numpy as np
import pandas as pd
from . import tools
import sys
import os
from tqdm import tqdm
#from astropy.convolution import Gaussian1DKernel, convolve
from skimage.segmentation import find_boundaries



# def gsmooth(x, stddev=30):
#     #print('gs,ooth')
#     g = Gaussian1DKernel(stddev=stddev)

#     interpsize = stddev*3
   
#     x_left = np.ones(int(interpsize))*x[0]
#     x_right = np.ones(int(interpsize))*x[1]
#     x = np.concatenate((x_left, x, x_right))
    
#     xsm = convolve(x, g)

#     return xsm[interpsize:-interpsize]

# def V(t, x, ws):

#     v = np.gradient(x, t)
#     v_smooth = gsmooth(v, ws, method='gkernel')
#     return v_smooth

def O(t, x, ws):

    o = x
    return

def oscillation(x, period, min_period, sm=60):
    
    #x = smooth(x, min_period)
    xf = smooth(x, period)
    #n = np.abs(x-xf).mean()
    dp = np.clip(x-xf, a_min=0, a_max=None)
    dn = -np.clip(x-xf, a_min=None, a_max=0)

    pcorr = dp[period:]*dp[:-period]
    ncorr = dn[period:]*dn[:-period]

    method=None
    corr = smooth(pcorr, period, method=method)*smooth(ncorr, period, method=method)
    #corr = pcorr[int(period/2):]*ncorr[:-int(period/2)]
    

    #corr = (xv[period:]*xv[:-period])/(xv[period:]**2)
    #corr = smooth(corr, period)
    #pcorr = np.clip(corr, a_min=0, a_max=None).copy()
    #pcorr = smooth(pcorr, period)
    #ncorr = np.clip(corr, a_min=None, a_max=0).copy()
    #corr = smooth(pcorr, period)-smooth(ncorr, period)
    #corr = np.clip(corr, a_min=0, a_max=None)
    
    return corr**0.25

def get_segments(dfp, invalid,  min_length=30):

    if invalid.mean()==1:
        return []
    elif invalid.mean()==0:
        if len(dfp)>min_length:
            return [np.ones(len(dfp), dtype=np.bool)]

    boundaries = find_boundaries(invalid, mode='outer', background=0)
    boundaries = np.argwhere(boundaries)
    
    t = dfp.frame.values
    segments=[]

    #Else search for valid segments
    boundaries = np.concatenate(([0], boundaries.flatten(), [t.size-1]))
    for i in range(boundaries.size-1):

        start, end = boundaries[i], boundaries[i+1]
        
        if (end-start)<min_length:
            continue

        if invalid[start:end].mean()>0.1:
            #This is a non valid cell
            continue

        segments.append((dfp.frame>=t[start]) & (dfp.frame<=t[end]))
    
    return segments


def segment_dfp(dfp, invalid, min_length=6):
 
    segments = get_segments(dfp, invalid, min_length=min_length)
    #print(len(segments))
    for segment_number, segment in enumerate(segments):

        #print(segment_number, len(dfp.loc[segment, 'segment']))
        dfp.loc[segment, 'segment'] = int(segment_number+1)

    return dfp


def get_v_segments_from_arrays(frames, nucleus, coarsen, min_length, tres, pixelperum, Lth=0.98, sm=3, findcps=False):
    """_summary_

    Args:
        frames (array): time frames
        nucleus (array): nucleus 1d position
        coarsen (int): number frames over which to coarsen the data
        min_length (int): minimum length of a track after change point separation
        tres (int): time step between frames
        pixelperum (_type_):
        Lth (float, optional): _description_. Defaults to 0.98.
        sm (int, optional): Length of the smoothing filter. Defaults to 3.
    """
    if not findcps:
        conv_kernel = np.ones(sm)/sm
        v= np.gradient(
            np.convolve(nucleus, conv_kernel, mode='valid'),
            np.convolve(frames, conv_kernel, mode='valid')).mean()
        v /= (tres*pixelperum)


    frame__0 = frame[0]
        
    coarse_frames = np.linspace(0, nucleus.size-1, round(nucleus.size/coarsen)).astype(int)
      
    v_nuc_coarse = np.gradient(nucleus[coarse_frames], frames[coarse_frames])
    #print(v_nuc_coarse)

    
    
    cp_indices = cp.find_cps(v_nuc_coarse, 1000, Lth, min_length)*coarsen

    cp_indices = np.concatenate(([0, frames.size], cp_indices))
    
    #Now classify the motion
    sorted_cps = cp_indices.copy().astype(int)
    sorted_cps.sort()
    #print('sorted_cps', sorted_cps)
    locators, vs = [], []
    for i in range(sorted_cps.size-1):
        cp_0, cp_1 = sorted_cps[i], sorted_cps[i+1]
        #print('cps', cp_0, cp_1)
        segment = (t-t_0>=cp_0) & (t-t_0<cp_1) 
   
# 
#         # print(cp_0, cp_1)
        t_current = t[segment]
        nucleus_current = nucleus[segment]
        

        conv_kernel = np.ones(sm)/sm
        v= np.gradient(
            np.convolve(nucleus_current, conv_kernel, mode='valid'),
            np.convolve(t_current, conv_kernel, mode='valid')).mean()
        v /= (tres*pixelperum)
 
        locators.append(segment)
    
    return vs, locators

def get_v_segments_from_df(dfp, coarsen, min_length, tres, pixelperum, Lth=0.98, sm=3):
    
    t = dfp.frame.values
    t_0 = t[0]
    nucleus = dfp.nucleus.values
        
    coarse_frames = np.linspace(0, nucleus.size-1, round(nucleus.size/coarsen)).astype(int)
    nuc_coarse = nucleus[coarse_frames]    
    v_nuc_coarse = np.gradient(nuc_coarse, t[coarse_frames])

    cp_indices = cp.find_cps(v_nuc_coarse, 1000, Lth, min_length)*coarsen
    boundaries = np.array([])

    cp_indices = np.concatenate(([0, t.size], cp_indices))
    
    #Now classify the motion
    sorted_cps = cp_indices.copy().astype(int)
    sorted_cps.sort()
    #print('sorted_cps', sorted_cps)
    locators, vs = [], []
    for i in range(sorted_cps.size-1):
        cp_0, cp_1 = sorted_cps[i], sorted_cps[i+1]
        #print('cps', cp_0, cp_1)
        segment = (t-t_0>=cp_0) & (t-t_0<cp_1)    

        t_current = t[segment]
        nucleus_current = nucleus[segment]
        

        conv_kernel = np.ones(sm)/sm
        v= np.gradient(
            np.convolve(nucleus_current, conv_kernel, mode='valid'),
            np.convolve(t_current, conv_kernel, mode='valid')).mean()
        v /= (tres*pixelperum)

        vs.append(v)
        locators.append(segment)
    
    return vs, locators

def get_average_vs(df, coarsen, min_length, tres, pixelperum, sm):

    ids = np.unique(dfv.particle.unique())
    ids.sort()
    