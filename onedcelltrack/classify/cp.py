import pandas as pd
import numpy as np
from skimage.segmentation import find_boundaries

def S(x, axis=0):
    
    S = np.cumsum(x-x.mean(), axis=axis)
    return S

def find_cp(x, N, coarsen=20):
    


    S_0 = S(x)
    
    d_S_0 = S_0.max() - S_0.min()
    #print(d_S_0)
    
    X_boot = np.vstack(
    [x.copy() for i in range(N)])
    
    for i in range(N):
        np.random.shuffle(X_boot[i])
    
    #plt.plot(x)
    #for y in X_boot[:3]:
    #    plt.plot(y)
    
    
    S_boot = S(X_boot, axis=1)
    
    d_S_boot = np.max(S_boot, axis=1) - np.min(S_boot, axis=1)
    
    #print(d_S_boot.shape)
    
    #plt.hist(d_S_boot);
    
    L = np.sum(d_S_boot < d_S_0)/N
    
    cp_index = np.argmax(np.abs(S_0))
    
    return L, cp_index
    
def split(x, x_start, x_end, N, L_th, min_length=100, debug=False):
    
    if debug:
        print('next_loop')
        print(f'x_start:{x_start}, x_end:{x_end}')

    if (x_end-x_start)<min_length:
        return np.array([])
    
    L, cp_index = find_cp(x[x_start:x_end], N)
    cp_index = (cp_index+x_start).copy()
    x_left = x[x_start:cp_index]
    x_right = x[cp_index:x_end]
    if debug:
        print(f'cp_index:{cp_index}')
    if L<L_th:
        if debug:
            print('below threshold')
            print(f'L:{L}')
        return np.array([])
    
    if debug:
        print('above threshold')
        print(print(f'L:{L}'))

    if cp_index==x_start or cp_index==x_end:
        return np.array([])
   
    left_split, right_split = False, False
    
    if x_left.size>=min_length:
        
        x_start_left = x_start
        x_end_left = cp_index
        
        left_split=True
        
    if x_right.size>=min_length:
        
        x_start_right = cp_index
        x_end_right = x_end
        
        right_split=True
    
    if left_split and right_split:
        
        cp_indices = np.concatenate((
            [cp_index],
            split(x, x_start_left, x_end_left, N, L_th, min_length),
            split(x, x_start_right, x_end_right, N, L_th, min_length)))
    
    elif left_split:
        
        cp_indices = np.concatenate((
            [cp_index],
            split(x, x_start_left, x_end_left, N, L_th, min_length)
            ))
        
    
    elif right_split:
        
        cp_indices = np.concatenate((
            [cp_index],
            split(x, x_start_right, x_end_right, N, L_th, min_length)
            ))
    
    else:
        cp_indices = np.array([])
    
    return cp_indices
   
    

def find_cps(x, N, L_th, min_length=100):
    
    x_start, x_end = 0, x.size
    
    return split(x, x_start, x_end, N, L_th, min_length)

        
def classify_movement(dfp, v_min=0.002, min_length=50, pixelperum=1.27, fps=1/30, coarsen=20):
    
    nucleus = dfp.nucleus.values
    rear = dfp.rear.values
    t = dfp.frame.values
    front = dfp.front.values
    #v_nuc = dfp.v_nuc.values
    #v_front=dfp.v_front.values
    #v_rear = dfp.v_rear.values
    
    dfp.loc[:, 'motion']=['' for i in range(len(dfp))]
    
    #Boolean array: True for cells that should be left out
    my_bool = ~((dfp.valid==1) & (dfp.too_close==0) & (dfp.single_nucleus==1) & (dfp.front!=0)).values
    
    #Points where there is a boundary between valid and non valid cells
    boundaries = find_boundaries(my_bool, mode='outer', background=0)
    boundaries = np.argwhere(boundaries)
    
    #If there are no boundaries, no need to check for valid segments
    if boundaries.size<2:
        coarse_frames = np.linspace(0, nucleus.size-1, round(nucleus.size/coarsen)).astype(int)
        nuc_coarse = nucleus[coarse_frames]
        v_nuc_coarse = np.gradient(nuc_coarse, t[coarse_frames])
    
        cp_indices = find_cps(v_nuc_coarse, 10000 , 0.7, min_length)*coarsen
        boundaries = np.array([])

        cp_indices = np.concatenate(([0, t.size], cp_indices))
        
        #Now classify the motion
        sorted_cps = cp_indices.copy().astype(int)
        sorted_cps.sort()
        
        for i in range(sorted_cps.size-1):
            
            cp_0, cp_1 = sorted_cps[i], sorted_cps[i+1]
            segment = (t>=cp_0) & (t<cp_1)    

            t_current = t[cp_0:cp_1]
            if t_current.size<1:
                continue 

            nucleus_current = nucleus[cp_0:cp_1]

            dt = (t_current[-1]-t_current[0])/fps
            
            v_mean = ((nucleus_current[-1]-nucleus_current[0])/pixelperum)/(dt)
            v_mean/=(pixelperum)

            print(v_min, v_mean)

            segment = dfp.frame.isin(t_current)
            if np.abs(v_mean)>=v_min:
                dfp.loc[segment, 'motion']='M'
            else:
                dfp.loc[segment, 'motion']='S'

            valid_boundaries = [0, t.size]

        return dfp, cp_indices, valid_boundaries
    
    #Search valid segments

    boundaries = np.concatenate(([0], boundaries.flatten(), [nucleus.size-1]))
    print(boundaries+t[0])
    valid_boundaries=[]

    cp_indices = np.array([])
    for i in range(boundaries.size-1):

        start, end = boundaries[i], boundaries[i+1]
        
        if (end-start)<min_length:
        
            continue
        if my_bool[start:end].mean()>0.1:
            
            #This is a non valid cell
            continue
        
        print('going from', t[start], t[end])
        #Coarsen the frames for the motion classification
        coarse_frames = np.linspace(start, end-1, round((end-start)/coarsen)).astype(int)
        nuc_coarse = nucleus[coarse_frames]

        t_coarse = t[coarse_frames]
        v_nuc_coarse = np.gradient(nuc_coarse, t_coarse)

        cp_indices_current = find_cps(v_nuc_coarse, 10000 , 0.7, min_length)*coarsen
        
        cp_indices_current+=start
        cp_indices = np.concatenate((cp_indices, cp_indices_current))
        cp_indices_current = np.concatenate(([start, end], cp_indices_current))
        
        #Now classify the motion
        sorted_cps = cp_indices_current.copy().astype(int)
        sorted_cps.sort()
        print(sorted_cps, 'cp_indices')
        
        for i in range(sorted_cps.size-1):
            
            cp_0, cp_1 = sorted_cps[i], sorted_cps[i+1]
            segment = (t>=cp_0) & (t<cp_1)    

            t_current = t[cp_0:cp_1]
            if t_current.size<1:
                continue 

            nucleus_current = nucleus[cp_0:cp_1]

            dt = (t_current[-1]-t_current[0])/fps
            
            v_mean = ((nucleus_current[-1]-nucleus_current[0])/pixelperum)/(dt)
            v_mean/=(pixelperum)

            print(v_min, v_mean)

            segment = dfp.frame.isin(t_current)
            if np.abs(v_mean)>=v_min:
                dfp.loc[segment, 'motion']='M'
            else:
                dfp.loc[segment, 'motion']='S'

            valid_boundaries.append([start, end])

    return dfp, cp_indices, valid_boundaries

        