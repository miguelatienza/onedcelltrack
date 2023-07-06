# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 11:02:33 2021

@author: miguel.Atienza
"""

import numpy as np
import sys
from . import functions
import trackpy as tp
# from scipy import stats
# from numba import prange, njit
from tqdm import tqdm
from skimage.segmentation import find_boundaries
import pandas as pd
from .classify import tools
from .classify import cp
pd.options.mode.chained_assignment = None

def track(nuclei, diameter=19, minmass=None, track_memory=15, max_travel=5, min_frames=10, pixel_to_um=1, verbose=False):
    """
    Detect the nuclei positions from fluoresecence images and link them to create individual particle tracks using trackpy.

    Parameters
    ----------
    nuclei : numpy 3D array
        Images of fluorescent nuclei as numpy array.
    diameter : TYPE, optional
        Estimate of nuclei diameters, better to overestime than  underestimate. The default is 19.
    minmass : TYPE, optional
        Minimum mass of a point to be considered valid nucleus. The default is 3500.
    track_memory : TYPE, optional
        Maximum number of time frames where nucleus position is interpolated if it is not detected. The default is 15.
    max_travel : TYPE, optional
        Maximum . The default is 5.
    min_frames : TYPE, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    t : TYPE
        DESCRIPTION.

    """

    max_travel = np.round(max_travel) #Maximum distance between nuclei in subsequent frames
    diameter = int(diameter + (not diameter%2)) #Force odd diameter

    if minmass is None:
        #Set default minmass depending on dtype of image
        if nuclei.dtype=='uint8':
            minmass=3500
        elif nuclei.dtype=='uint16':
            minmass=2500*(2**16)/255
    if not verbose:
        tp.quiet()

    print('Tracking nuclei using trackpy...')
    #print(diameter, minmass, track_memory, max_travel)
    f = tp.batch(nuclei, diameter=diameter, minmass=minmass)
    t = tp.link(f, max_travel, memory=track_memory)
    t = tp.filter_stubs(t, min_frames)
    print('Tracking of nuclei completed.')
    
    return t


def get_tracking_data(df, cyto_masks, lanes_mask, lanes_metric, patterns, tres=30, min_duration=None):
    
    if min_duration is None:
        ##Set this to an hour
        min_duration=int(3600/tres)
    
    df['nucleus'] = np.zeros(len(df))
    df['front'] = np.zeros(len(df))
    df['rear'] = np.zeros(len(df))
    df['lane_index'] = np.zeros(len(df))
    df['foot_print'] = np.zeros(len(df))
    df['area'] = np.zeros(len(df))
    df['valid'] = np.ones(len(df))
    df['interpolated'] = np.zeros(len(df))
    df['cyto_locator'] = np.zeros(len(df))

    def update_df():
        
        #function to update the dataframe
        df.loc[df.particle==particle, 'lane_index']=lane_index
        df.loc[df.particle==particle, 'foot_print']=footprint
        df.loc[df.particle==particle, 'nucleus']=nucleus
        df.loc[df.particle==particle, 'front']=front
        df.loc[df.particle==particle, 'rear']=rear
        df.loc[df.particle==particle, 'area']=area
        df.loc[df.particle==particle, 'valid']=valid
        df.loc[df.particle==particle, 'interpolated']=interpolated
        df.loc[df.particle==particle, 'cyto_locator']=cyto_locator
        df.loc[df.particle==particle, 'FN_signal']=FN_signal

        return

    footprint_array = functions.get_foot_print(cyto_masks)
    uniques = df.particle.unique()
    ids = np.arange(uniques.size)

    print('Obtaininig cell tracks...')
    for i in tqdm(ids):

        particle = uniques[i]
        #print(f'particle: {particle}')
        
        p = df[df.particle==particle]
        p.index.name=None
        p.sort_values('frame')
        t = p.frame.values.astype(int)
        p_x = p.x.values
        p_y = p.y.values
        valid = p.valid.values
        interpolated = p.interpolated.values
        
        if p_x.size <min_duration:
            #print('This track is too short')
            df.loc[df.particle==particle, 'valid']=valid*0
            continue
        
        cyto_locator = cyto_masks[t, np.round(p_y).astype(int), np.round(p_x).astype(int)].astype('uint8')

        binary_cyto_mask = cyto_locator[:,np.newaxis, np.newaxis]==cyto_masks[t]

        lonely_nuclei = cyto_locator==0

        if np.mean(lonely_nuclei)>0.3:
            #If more than 30% of the tracked nuclei have no corresponding cells, leave this track out
            df.loc[df.particle==particle, 'valid']=valid*0
            continue

        if np.sum(binary_cyto_mask)<25:
            #print('This track has no corresponding cells')
            df.loc[df.particle==particle, 'valid']=valid*0
            continue
        
        #Now extract the positions
        
        rear, front, lane_index, two_lanes = get_cyto_positions(lanes_mask, lanes_metric, binary_cyto_mask, lonely_nuclei, min_overlap=0.5)
        
        if rear is None:
            #Cell touches more than one lane or no lanes or has too small of an overlap
            df.loc[df.particle==particle, 'valid']=valid*0
            continue
        
        if two_lanes:
            ##Set this particle to non valid, but still calculate front rear and so on in order to filter properly later one for double nuclei cells and cells too close to one another
            df.loc[df.particle==particle, 'valid']=valid*0
                      
        nucleus = lanes_metric[np.round(p_y).astype(int), np.round(p_x).astype(int)]

        #Calculate footprint
        footprint = np.mean(footprint_array[t]*binary_cyto_mask, axis=(1,2))

        #Calculate area
        area = np.sum(binary_cyto_mask, axis=(1,2))
        lane_area = np.sum(binary_cyto_mask*lanes_mask[np.newaxis, :, :], axis=(1,2))

        #Calculate FN signal only on the region that is on a lane
        FN_signal = np.sum(patterns[np.newaxis, :,:]*binary_cyto_mask*lanes_mask, axis=(1,2))/lane_area

        ##Interpolate values and cut out at beginning and end if necessary
        island_boundaries = np.argwhere(find_boundaries(lonely_nuclei, mode='outer', background=0)).astype(int).flatten() #Boundaries of islands of lonely nuclei

        if island_boundaries.size<1:
            #There are no lonely nuclei, so update the dataframe and continue
            update_df()
            continue

        #Check all values to the left of the first boundary are lonely nuclei
        if np.all(lonely_nuclei[:island_boundaries[0]]):
        #There is an island at the start
            left_island=True 
            valid[:island_boundaries[0]]=0

            if not lonely_nuclei[island_boundaries[0]+1]:
                #The value to the right of first boundary is not lonely, so remove this boundary
                island_boundaries = island_boundaries[1:]
       
                if island_boundaries.size<1:
                #There are no more lonely nuclei, so update the dataframe and continue.
                    update_df()
                    continue

        #Same process for the right side
        if np.all(lonely_nuclei[island_boundaries[-1]+1:]):
            right_island=True
            valid[island_boundaries[-1]:]=0

            if not lonely_nuclei[island_boundaries[-1]-1]:
                #The value to the left of last boundary is not lonely, so remove this boundary
                island_boundaries = island_boundaries[:-1]

                if island_boundaries.size<1:
                #There are no more lonely nuclei, so update the dataframe and continue.
                    update_df()
                    continue
            

        #Now remove islands by interpolating
        #First check for any boundaries that have lonely nuclei at both sides, and duplicate these
        surrounded_boundaries = []
        n_boundaries = island_boundaries.size
        for i in range(n_boundaries):
            boundary = island_boundaries[i]
        
            if boundary+1==lonely_nuclei.size or boundary==0:
                continue
            
            if lonely_nuclei[boundary-1]==lonely_nuclei[boundary+1]:
                #Check if this boundary has masks at both sides
                if not ((i==0 and left_island) or (i==n_boundaries-1 and right_island)):
                    #Check that it is not already bounding the right or left end of the array. If so duplicate it: 
                    surrounded_boundaries.append(boundary)
        
        surrounded_boundaries = np.array(surrounded_boundaries, dtype=island_boundaries.dtype)

        island_boundaries = np.concatenate((island_boundaries, surrounded_boundaries))
        island_boundaries.sort()

        try:
            #Now build pairs of boundaries between which to interpolate
            island_boundaries = island_boundaries.reshape(int(island_boundaries.size/2), 2)
        except ValueError:
            raise ValueError('Not even dimensions')

        #Finally interpolate
        for boundary in island_boundaries:

            width = np.abs(boundary[1]-boundary[0]) -1
            if width >5:
                break
            #print(f'Removing island of size {width}')
            
            x = np.arange(boundary[0]+1, boundary[1])#values to be interpolated
            
            interpolated[x] = 1
            front[x] = np.interp(x, boundary, front[boundary])
            rear[x] = np.interp(x, boundary, rear[boundary])
            area[x] = np.interp(x, boundary, area[boundary])
            footprint[x] = np.interp(x, boundary, area[boundary])
            FN_signal[x] = np.interp(x, boundary, FN_signal[boundary])

        if width>5:
            #print(cyto_locator[boundary[0]-1:boundary[1]+1])
            valid[boundary[0]:]=0
            df.loc[df.particle==particle, 'valid']=valid
            update_df()
            continue
            
        update_df()
        
    return df

def get_cyto_positions(lanes_mask, lanes_metric, binary_cyto_mask, lonely_nuclei, min_overlap=0.5):

    two_lanes = False
    n_frames = binary_cyto_mask.shape[0]
    img_size = int(binary_cyto_mask.shape[1]*binary_cyto_mask.shape[2])

    frames = np.arange(n_frames)[~lonely_nuclei]
    coarse_frames = frames[np.arange(0, frames.size, 5)]

    #Find which line this mask corresponds to
    lane_index_mask = binary_cyto_mask[coarse_frames]*lanes_mask[np.newaxis, :, :]
    
    lane_index, counts = np.unique(lane_index_mask[lane_index_mask>0], return_counts=True)
    ##normalize the counts
    counts = counts/counts.sum()
    
    if lane_index.size>1:
        #Cell touches two different lanes
        #print('cell touches more than one line')
        ##allow for only 1 percent of the track being 
        if np.any((counts>0.01) & (counts<0.99)):
            ##Some lane index is present more than 1 percent of the area and time.
            two_lanes = True
        
        lane_index = lane_index[np.argmax(counts)]
        
    elif lane_index.size<1:
        #Cell does not touch any lines
        #print('cell touches no lines')
        return None, None, None, None
    else:
        lane_index = lane_index[0]
    
    lane_bool = (lanes_mask==lane_index).reshape(img_size)
    
    lanes_metric = lanes_metric.reshape(img_size)   
    binary_cyto_mask = binary_cyto_mask.reshape(n_frames, img_size)

    total_area=np.sum(binary_cyto_mask)

    lanes_metric = lanes_metric[lane_bool]
    binary_cyto_mask = binary_cyto_mask[:, lane_bool]

    overlap = total_area/np.sum(binary_cyto_mask)

    if overlap < min_overlap:
    #This cell does not overlap very well with the lane
         return None, None, None, None
    
    #binary_cyto_mask[binary_cyto_mask==0]=np.nan
    distance = lanes_metric[np.newaxis, :]*binary_cyto_mask
    
    front = np.max(distance, axis=1)
    
    distance[binary_cyto_mask==0]= front.max()
    
    rear = np.min(distance, axis=1)

    return rear, front, lane_index, two_lanes


def get_single_cells(df):

    #Take only valid cells
    df['single_nucleus'] = np.ones(len(df), dtype='bool')
    #df = df[df.valid==1]
    ids = df.particle.unique()
    #ids = np.arange(uniques.size)

    for i in (ids):

        particle = i
        dfp = df[df.particle==particle]

        dfnp = df[df.particle!=particle]

        #Check positions where the mask is shared between two nuclei
        double_nuclei_frames = pd.merge(dfp, dfnp,  how='inner', on=['frame','cyto_locator']).frame.values
        df.loc[(df.particle==particle) & (df.frame.isin(double_nuclei_frames)), 'single_nucleus']=False
        #df.loc[(df.particle==particle) & (df.frame.isin(double_nuclei_frames)), 'single_nucleus']=0
        
    return df

def remove_close_cells(df, min_distance=10):

    #df['nearest_cell'] = -1*np.ones(len(df), dtype='bool')
    df.loc[:, 'too_close']=np.zeros(len(df))

    #df = df[(df.valid==1) & (df.front!=0)]
    ids = df.particle.unique()
    
    for i in (ids):

        particle = i
        dfp = df[df.particle==particle]

        dfnp = df[df.particle!=particle]

        lane_df = pd.merge(dfp, dfnp, how='inner', on=['frame', 'lane_index'])

        #mybool = ((min_distance>(lane_df.rear_x-lane_df.front_y)) & ((lane_df.rear_x-lane_df.front_y)>=(lane_df.front_y))) | ((min_distance>(lane_df.rear_y-lane_df.front_x)) & ((lane_df.rear_y-lane_df.front_x)>=0))
        mybool = ((lane_df.rear_y< lane_df.front_x+min_distance) & (lane_df.rear_y >lane_df.rear_x)) | ((lane_df.front_y>lane_df.rear_x-10) & (lane_df.front_y<lane_df.front_x)) 
        
        
        close_frames = lane_df.loc[mybool, 'frame']
        
        
        df.loc[((df.particle==particle) & (df.frame.isin(close_frames))), 'too_close'] = 1#(lane_df.rear_x-lane_df.front_y)[mybool]
        if np.sum(np.isnan(lane_df.rear_x-lane_df.front_y))>0:
            print(lane_df.rear_x, lane_df.front_y)
            pass
        #mybool = (min_distance>(lane_df.rear_x-lane_df.front_y)) & ((lane_df.front_x-lane_df.rear_y)>=0)
        #close_frames = lane_df.loc[mybool, 'frame']
        
        #df.loc[((df.particle==particle) & (df.frame.isin(close_frames))), 'too_close'] = (lane_df.rear_x-lane_df.front_y)[mybool]
        

    return df

def get_clean_tracks(df, max_interpolation=5, min_length=6, image_height=1024):
    
    ##filtering and segmenting
    clean_df = get_single_cells(df)
    clean_df = remove_close_cells(clean_df)

    clean_df.loc[:, 'segment']=np.zeros(len(clean_df), dtype=int)
    clean_df['v_nuc'] = np.full(len(clean_df), np.nan, dtype=np.float64)
    clean_df['v_front'] = np.full(len(clean_df), np.nan, dtype=np.float64) 
    clean_df['v_rear'] = np.full(len(clean_df), np.nan, dtype=np.float64)
    clean_df['length'] = np.full(len(clean_df), np.nan, dtype=np.float64)
    clean_df['v_length'] = np.full(len(clean_df), np.nan, dtype=np.float64)

    ids = clean_df.particle.unique()
    for particle in ids:

        dfp = clean_df.loc[clean_df.particle==particle, :]
        invalid = ~((dfp.valid==1) & (dfp.too_close==0) & (dfp.single_nucleus==1) & (dfp.front!=0) & (dfp.rear!=0) & (dfp.nucleus!=0) & (dfp.front<image_height)).values
        
        dfp = tools.segment_dfp(dfp, invalid, min_length=min_length)

        segments = np.unique(dfp.segment.values)
        #print(segments)
        for segment in segments[segments!=0]:
        
            where = (dfp.segment==segment).values
            
            dfps = dfp[where]

            t = dfps.frame.values

            dts = np.diff(t)

            if dts.max()>max_interpolation:
                pass
                #clean_df=clean_df[clean_df.particle!=particle]
                #continue
                
            dfp.loc[where, 'nucleus'] = functions.remove_peaks(dfps.nucleus.values)
            dfp.loc[where, 'rear'] = functions.remove_peaks(dfps.rear.values)
            dfp.loc[where, 'front'] = functions.remove_peaks(dfps.front.values)
            v_nuc = np.gradient(dfps.nucleus.values, dfps.frame.values)
            v_front = np.gradient(dfps.front.values, dfps.frame.values)
            v_rear = np.gradient(dfps.rear.values, dfps.frame.values)
            length = (dfps.front.values - dfps.rear.values)
            v_length = np.gradient(length, dfps.frame.values)
    
            dfp.loc[where, 'v_nuc']= v_nuc
            dfp.loc[where, 'v_front'] = v_front 
            dfp.loc[where, 'v_rear'] = v_rear
            dfp.loc[where, 'length'] = length
            dfp.loc[where, 'v_length'] = v_length
            dfp.loc[where, 'segment'] = segment

        clean_df[clean_df.particle==particle]=dfp

    return clean_df

def classify_tracks(df, tres, coarsen=3, min_episode=1, sm=6, min_length=12, pixelperum=1.31):
    
    df['motion']=''
    df['state']=''
    df['V']=np.nan
    df['O']=np.nan
    
    ids = df.particle.unique()
    
    for particle in ids:
        segments = df.loc[df.particle==particle, 'segment'].unique()
        for segment in segments[segments!=0]:
           
            where = (df.particle==particle) & (df.segment==segment)
     
            df[where], _ = cp.classify_movement(df[where], fps=1/tres, v_min=0.002, min_length=min_length, pixelperum=pixelperum, coarsen=coarsen, Nperm=1000, Lth=0.98, Oth=5, min_episode=min_episode, sm=sm)

    return df