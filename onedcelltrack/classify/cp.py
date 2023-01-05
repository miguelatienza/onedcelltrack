import pandas as pd
import numpy as np
from skimage.segmentation import find_boundaries
from .. import functions
DEBUG=False

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

def find_cps_2(dfp, TimeRes, Nperm, Lth=0.98):
    
    sm=1
    filterOut = np.round(600/TimeRes).astype('int')
    SamplePoints=np.arange(0, len(dfp), filterOut)
    front= smooth(dfp.nucleus.values[SamplePoints], sm)
    rear = smooth(dfp.rear.values[SamplePoints], sm)
    nucleus = smooth(dfp.nucleus.values[SamplePoints], sm)
    
    nucleus=nucleus-nucleus[0]
    front=front-front[0]
    rear = rear-rear[0]
    
    t = dfp.frame.values[SamplePoints]-dfp.frame.values[0]
    vnuc=np.gradient(nucleus,t)
    vrear=np.gradient(rear,t)
    vfront=np.gradient(front,t);
    initpoint=0
    endpoint=vnuc.size
    CPs=np.array((initpoint, endpoint))
    maxintwithnoPC=0 # intervals that has been checked and contain no more PC
    endloop=0
    repeatfind=0
    while endloop==0:
        print('into the loop')
        print(CPs)
        intstart=CPs[maxintwithnoPC]
        intend=CPs[maxintwithnoPC+1]
        vseg=vnuc[intstart:intend]
        meanvseg=vseg.mean()
        S_0=cp.S(vseg)
        Sdiff0=S_0.max()-S_0.min();
        Sdiff=[]
        
        X_boot = np.vstack(
        [vseg.copy() for i in range(Nperm)])
    
        for i in range(Nperm):
            np.random.shuffle(X_boot[i])    
    
        S_boot = S(X_boot, axis=1)
    
        Sdiff = np.max(S_boot, axis=1) - np.min(S_boot, axis=1)
        
        Lconf=np.mean(Sdiff<Sdiff0)
        print(Lconf)
        if Lconf>Lth:
            indScp=np.argmax(np.abs(S_0[1:-2]))
            IndScp=indScp+1+intstart-1; #index of the max S in the original vC vector(indScp is the index in vseg)
            if not IndScp in CPs:
                CPs=np.concatenate((CPs, [IndScp]))
                CPs.sort()
            
        else:
            maxintwithnoPC=maxintwithnoPC+1
        
        if maxintwithnoPC==CPs.size-1: #if the index of maximum interval with no CP is equal to all current interval.
            endloop=1
        
    if CPs.size>2:
        shortloop=0
        j=1
        while shortloop==0:
            if CPs[j]-CPs[j-1]<(15*2/filterOut):
                CPs=np.delete(CPs, j)
            else:
                j=j+1
            if j==CPs.size:
                shortloop=1
    
    return CPs
        
def classify_movement(dfp, v_min=0.002, min_length=50, pixelperum=1.27, fps=1/30, coarsen=int(20/4), Nperm=1000, Lth=0.98, Oth=5):
    
    nucleus = dfp.nucleus.values
    rear = dfp.rear.values
    t = dfp.frame.values
    front = dfp.front.values


    coarse_frames = np.linspace(0, nucleus.size-1, round(nucleus.size/coarsen)).astype(int)
    nuc_coarse = nucleus[coarse_frames]
    
    v_nuc_coarse = np.gradient(nuc_coarse, t[coarse_frames])

    cp_indices = find_cps(v_nuc_coarse, Nperm , Lth, min_length)*coarsen
    boundaries = np.array([])

    cp_indices = np.concatenate(([0, t.size], cp_indices))
    
    #Now classify the motion
    sorted_cps = cp_indices.copy().astype(int)
    sorted_cps.sort()
   
    for i in range(sorted_cps.size-1):
        
        cp_0, cp_1 = sorted_cps[i], sorted_cps[i+1]
        segment = (t>=cp_0) & (t<cp_1)    
        # print('classifying')
        # print(cp_0, cp_1)
        t_current = t[cp_0:cp_1]
        nucleus_current = nucleus[cp_0:cp_1]
        front_current = front[cp_0:cp_1]
        rear_current = rear[cp_0:cp_1]
        L = front_current-rear_current

        if t_current.size<302/4:
            #print('Too short')
            
            continue 

        V = classify_velocity(nucleus_current, t_current, v_min, 1/fps, pixelperum=pixelperum)
        O = classify_oscillation(L, nucleus_current, t, pixelperum, Oth)
        #print('classified')
        #print(V, O)
        segment = dfp.frame.isin(t_current)
        #print(len(segment==True))

        dfp.loc[segment, 'O']=O
        dfp.loc[segment, 'V']=V

        mov_dir = np.sign(V)*(np.abs(V)>v_min)
        if mov_dir!=0:
            dfp.loc[segment, 'motion']='M'
        else:
            dfp.loc[segment, 'motion']='S'
        
        if mov_dir!=0 and O>=Oth :
            dfp.loc[segment, 'state']='MO'
        elif mov_dir!=0 and O<Oth:
            dfp.loc[segment, 'state']='MS'
        elif mov_dir==0 and O>Oth:
            dfp.loc[segment, 'state']='SO'
        elif mov_dir==0 and O<Oth:
            dfp.loc[segment, 'state']='SS'

    return dfp, cp_indices



def classify_velocity(x, t, v_min, tres, pixelperum, sm=3):

    x_smooth = smooth(x, sm)
    v = np.gradient(x_smooth, t)

    v = v/(pixelperum*tres)
    v_mean = v.mean()

    #return np.sign(v_mean)*(np.abs(v_mean)>v_min)
    return v_mean

def classify_oscillation(L, nucleus, t, pixelperum, Omin=5, min_episode=20, sm=int(300/4)):
    sm = int(300/16)
    min_episode=5
    #print(t.size, L.size)
    #print(L.size)
    L = functions.remove_peaks(L)
    L_filt = (smooth(L, min_episode) - smooth(L, sm))/pixelperum
    nuc_filt = (smooth(nucleus, min_episode) - smooth(nucleus, sm))/pixelperum
    import matplotlib.pyplot as plt
    if DEBUG:
        plt.subplots()
        plt.plot(t, smooth(L, min_episode), color='blue')
        plt.plot(t, smooth(L, sm), color='red')
        plt.show()

    O = np.mean(np.abs(
        L_filt[int(min_episode/2):-int(min_episode/2)])
        + np.abs(nuc_filt[int(min_episode/2):-int(min_episode/2)]))

    #print('O', O)
    return O

def get_cps(dfp, TimeRes, Nperm, Lth=0.98):
    
    sm=1
    filterOut = np.round(600/TimeRes).astype('int')
    SamplePoints=np.arange(0, len(dfp), filterOut)
    front= smooth(dfp.nucleus.values[SamplePoints], sm)
    rear = smooth(dfp.rear.values[SamplePoints], sm)
    nucleus = smooth(dfp.nucleus.values[SamplePoints], sm)
    
    nucleus=nucleus-nucleus[0]
    front=front-front[0]
    rear = rear-rear[0]
    
    t = dfp.frame.values[SamplePoints]-dfp.frame.values[0]
    vnuc=np.gradient(nucleus,t)
    vrear=np.gradient(rear,t)
    vfront=np.gradient(front,t);
    initpoint=0
    endpoint=vnuc.size
    CPs=np.array((initpoint, endpoint))
    maxintwithnoPC=0 # intervals that has been checked and contain no more PC
    endloop=0
    repeatfind=0
    while endloop==0:
        #print('into the loop')
        #print(CPs)
        intstart=CPs[maxintwithnoPC]
        intend=CPs[maxintwithnoPC+1]
        vseg=vnuc[intstart:intend]
        meanvseg=vseg.mean()
        S_0=cp.S(vseg)
        Sdiff0=S_0.max()-S_0.min();
        Sdiff=[]
        
        X_boot = np.vstack(
        [vseg.copy() for i in range(Nperm)])
    
        for i in range(Nperm):
            np.random.shuffle(X_boot[i])    
    
        S_boot = cp.S(X_boot, axis=1)
    
        Sdiff = np.max(S_boot, axis=1) - np.min(S_boot, axis=1)
        
        Lconf=np.mean(Sdiff<Sdiff0)
        #print(Lconf)
        if Lconf>Lth:
            indScp=np.argmax(np.abs(S_0[1:-2]))
            IndScp=indScp+1+intstart-1; #index of the max S in the original vC vector(indScp is the index in vseg)
            if not IndScp in CPs:
                CPs=np.concatenate((CPs, [IndScp]))
                CPs.sort()
            
        else:
            maxintwithnoPC=maxintwithnoPC+1
        
        if maxintwithnoPC==CPs.size-1: #if the index of maximum interval with no CP is equal to all current interval.
            endloop=1
        
    if CPs.size>2:
        shortloop=0
        j=1
        while shortloop==0:
            if CPs[j]-CPs[j-1]<(15*2/filterOut):
                CPs=np.delete(CPs, j)
            else:
                j=j+1
            if j==CPs.size:
                shortloop=1
    
    return CPs

def smooth(a,ws):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    ws += not (ws%2)
    out0 = np.convolve(a,np.ones(ws,dtype=int),'valid')/ws   
    r = np.arange(1,ws-1,2)
    start = np.cumsum(a[:ws-1])[::2]/r
    stop = (np.cumsum(a[:-ws:-1])[::2]/r)[::-1]
    return np.concatenate((start , out0, stop)) 