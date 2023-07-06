import numpy as np
import cupy as cp
from skimage.transform import rescale
from skimage.feature import peak_local_max
from . import functions

def batch_hough(image, delta_y_array, y_0_list, kernel_width):
    """
    Function to get the hough space of an image
    """  
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

def np_batch_hough(image, delta_y_array, y_0_list, kernel_width):
    """
    Function to get the hough space of an image
    """  
    h, w = image.shape
    image = np.array(image)
    y_0_size = np.sum(y_0.size for y_0 in y_0_list)
    y_0_size = y_0_list[0].size
    hough = np.zeros((delta_y_array.size, y_0_size))
    
    i=0
    for delta_y in delta_y_array:
        
        for y_0 in y_0_list:

            x = np.arange(w)
            y = np.arange(h)

            long_enough = ((y_0 + delta_y) > kernel_width/2) & ((y_0 + delta_y) < h-kernel_width/2)
            y_0 = y_0[long_enough]
            Y_0, Y, X = np.meshgrid(y_0, y, x, sparse=True, indexing='ij')

            # p1 = 0, Y_0
            # p2 = w-1, Y_0 + delta_y 

            kernel = Y-(Y_0 + delta_y*X/w)
            kernel = kernel*(np.abs(kernel)<(kernel_width/2))

            current_hough = np.sum(kernel*image[np.newaxis, :, :], axis=(1,2))#/cp.sum(image[cp.newaxis, :,:]*(kernel>0), axis=(1,2))
            hough[i, long_enough] = current_hough

        i+=1
    
    return hough

def gpu_hough(image, kernel_width, scaling=0.25):

    
    scaling = 0.25

    image_rescaled = rescale(image, scaling, anti_aliasing=True)
    h,w = image_rescaled.shape
    delta_y_array = np.arange(int(-h/2), int(h/2) + 1)
    kernel_width=5
    h,w = image_rescaled.shape
    min_y0, max_y0 = int(kernel_width/2), h-int(kernel_width/2)
    y_0_list = [cp.arange(min_y0, max_y0)]
    
    hough = batch_hough(image_rescaled, delta_y_array, y_0_list, kernel_width)
    
    return hough, delta_y_array, y_0_list[0].get()

def np_hough(image, kernel_width, scaling=0.25):

    
    scaling = 0.25

    image_rescaled = rescale(image, scaling, anti_aliasing=True)
    h,w = image_rescaled.shape
    delta_y_array = np.arange(int(-h/2), int(h/2) + 1)
    kernel_width=5
    h,w = image_rescaled.shape
    min_y0, max_y0 = int(kernel_width/2), h-int(kernel_width/2)
    y_0_list = [np.arange(min_y0, max_y0)]
    
    hough = np_batch_hough(image_rescaled, delta_y_array, y_0_list, kernel_width)
    
    return hough, delta_y_array, y_0_list[0]


def get_lane_mask(image, kernel_width=5, line_distance=30, threshold=0.5, scaling=0.25, debug=False, gpu=True):
    """Function that takes in an image of a lines pattern, and returns a mask of the detected lanes. The algorithm assumes that the experimentator has tried to get the lanes to run as close to horizontal as possible.

    Args:
        image (_type_): _description_
        delta_y_max (_type_): _description_
        kernel_width (int, optional): _description_. Defaults to 5.
        line_distance (int, optional): _description_. Defaults to 30. Estimate of the distance between the lines in pixels, where one is still 100% sure that it is below the real distance.
    
    Returns:
        Mask image containing 0s where there is no lane, and a different integer for every separate line.
    """

    
    #print('Detecting lanes...')
    h, w = image.shape
    
    if not gpu:
        hough, delta_y_array, y_0_array = np_hough(image, kernel_width, scaling)
    
    if gpu:
       
        hough, delta_y_array, y_0_array = gpu_hough(image, kernel_width, scaling)
        
    
    min_distance = int(line_distance*scaling*0.8)

    min_coordinates = peak_local_max(hough, min_distance=min_distance, exclude_border=2, threshold_rel=threshold)
    max_coordinates = peak_local_max(-hough, min_distance=min_distance, exclude_border=2, threshold_rel=threshold)
    
    max_coordinates = max_coordinates[tuple([max_coordinates[:, 1].argsort()])].astype(int)
    min_coordinates = min_coordinates[tuple([min_coordinates[:, 1].argsort()])].astype(int)

    # if debug:
    #     #show the hough space
    #     #import matplotlib.pyplot as plt
    #     #plt.subplot(121)
    #     #plt.imshow(myhough)
    #     #plt.scatter(min_coordinates[:,1], min_coordinates[:,0], color='red')
    #     #plt.scatter(max_coordinates[:,1], max_coordinates[:,0], color='white')
    #     #print(min_coordinates[1, :])
    #     #plt.subplot(122)
    #     #plt.imshow(image)
      
    #     #x = [0, image.shape[1]]
    #     #for i in range(min_coordinates.shape[0]):
    #     #    plt.plot(x, [min_coordinates[i,1], sum(min_coordinates[i])])
        
    #     return min_coordinates, max_coordinates
        

    max_coordinates[:,0]=delta_y_array[max_coordinates[:,0]]/scaling
    min_coordinates[:,0]=delta_y_array[min_coordinates[:,0]]/scaling

    max_coordinates[:, 1] = y_0_array[max_coordinates[:, 1]]/scaling
    min_coordinates[:, 1] = y_0_array[min_coordinates[:, 1]]/scaling

    if debug:
        return min_coordinates, max_coordinates

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

    # if debug:
    #     plt.imshow(image, vmin=0, vmax=1000)
    #Loop through the top and bottom lines to create the mask
    for i in range(max_coordinates.shape[0]):
        
        top_coordinates = [0, w, max_coordinates[i,1], np.sum(max_coordinates[i])]
        bottom_coordinates = [0, w, min_coordinates[i,1], np.sum(min_coordinates[i])]
    
        top_x, top_y = functions.get_lines(top_coordinates)
        bottom_x, bottom_y = functions.get_lines(bottom_coordinates)

        y_0_mean = np.round(np.mean((top_y[0], bottom_y[0]))).astype(int)
        y_f_mean = np.round(np.mean((top_y[-1], bottom_y[-1]))).astype(int)

        lane_width = np.mean((top_y[0] - bottom_y[0], top_y[-1]- bottom_y[-1]))

        lane_width = np.round(lane_width).astype(int)
    
        coordinates = 0, w, y_0_mean, y_f_mean
        x_bool, y_bool = functions.get_lanes_for_kymograph_2(coordinates, lane_width, lane_mask.shape)
        
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
        x_bool, y_bool = functions.get_lanes_for_kymograph_2(coordinates, lane_width+10, lane_mask.shape)
        lane_metric[y_bool, x_bool] = functions.distance_to_line(p1, p2, X[y_bool, x_bool], Y[y_bool, x_bool])

    return lane_mask, lane_metric
