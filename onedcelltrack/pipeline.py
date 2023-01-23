# -*- coding: utf-8 -*-
import sys
import time
import logging
import traceback
import pickle
from tqdm import tqdm
#sys.path.append('..')
from . import functions
from .segmentation import segment, segment_looped
#from .omerop import Omero
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


class Track:
    
    def __init__(self, path_out, data_path=None, cyto_file=None, nucleus_file=None, nd2_file=None, lanes_file=None, frame_indices=None, max_memory=None, dataset_id=None, image_id=None, ome_host='omero.physik.uni-muenchen.de', ome_user_name=None, ome_password=None, fov=None, bf_channel=None, nuc_channel=None):
        """
        A class to run full tracking part of the pipeline on one field of view. The class should contain the methods for the cytoplasm segmentation, nucleus tracking, and the rearranging of the nuclei with the cytoplasm contours. The final output should be a set of 3darrays (frame_number, nuclear_position, front, rear) each for one particle.

        Parameters
        ----------
        data_path : string
            Path were data is read. Regular path file or dataset id if from omero
        cyto_file : string
            File containing cyto image. Either tif, nd2 or omero image id.
        nucleus_file : string
            File containing nucleus image.  Either tif, nd2 or omero image id.
        path_out : TYPE
            Path to write all generated data.
        image_indices : list, optional
            Indices of image to be  read  as list [first  index, last index]. The default is None.
        max_memory : TYPE, optional
            The maximum memory that should be used during computations. At the moment just uses the  maximum stack. The default is None.
        nd2_file : string, optional
            ND2file
        fov : int, field of view to read from nd2 file.
            

        Returns
        -------
        None.

        """
        
        self.data_path = data_path
        self.cyto_file = cyto_file
        self.nucleus_file = nucleus_file
        self.lanes_file = lanes_file
        self.path_out = path_out
        self.metadata = {}
        self.df_path = os.path.join(self.path_out, 'tracking_data.csv')
        self.clean_df_path = os.path.join(self.path_out, 'clean_tracking_data.csv')
        self.meta_path = os.path.join(self.path_out, 'metadata.json')
        self.max_memory=max_memory
        self.frame_indices = frame_indices
        self.dataset_id = dataset_id
        self.image_id = image_id
        self.nd2_file = nd2_file
        self.fov = fov

        if dataset_id is not None:

            self.omero=True
            #Assume this is an omero image id
            self.dataset_id = dataset_id
            self.image_id = image_id
            self.conn = Omero(ome_host , ome_user_name)
            shape = self.conn.get_image_shape(image_id)
            self.height, self.width = shape[1:3]

            if self.frame_indices is not None:
                self.n_images = len(frame_indices)
            else: 
                self.n_images = shape[0]
                self.frame_indices = np.arange(shape[0])
        
        else:
            self.omero=False

            if nd2_file is not None:
                #Read from full nd2 file
                f = ND2Reader(os.path.join(data_path, nd2_file))

                self.nfov = f.sizes['v']
                self.n_images = f.sizes['t']
                self.height, self.width = f.sizes['y'], f.sizes['x']
                if frame_indices is None:
                    self.frame_indices = np.arange(0, self.n_images)

                channels = f.metadata['channels']

                if bf_channel is None or nuc_channel is None:
                    ###Infer the channels
                    if 'erry' in channels[0] or 'exas' in channels[0]:
                        self.nucleus_channel=0
                        self.cyto_channel=1
                    elif 'erry' in channels[1] or 'exas' in channels[1]:
                        self.nucleus_channel=1
                        self.cyto_channel=0
                    else:
                        raise ValueError(f"""The channels could not be automatically detected! \n
                        The following channels are available: {channels} . Please specify the indices of bf_channel and nuc_channel as keyword arguments. i.e: bf_channel=0, nuc_channel=1""")
                else:
                    self.cyto_channel=bf_channel
                    self.nucleus_channel=nuc_channel
    
            elif cyto_file.endswith('.mp4'):

                video = io.FFmpegReader(os.path.join(data_path, cyto_file))

                self.n_images, self.height, self.width = video.getShape()[:3]

            elif cyto_file.endswith('.tif'):

                tif = TiffFile(os.path.join(data_path, cyto_file))
                self.height, self.width = tif.pages[0].shape
                if frame_indices is None:
                    self.n_images = len(tif.pages)
                    self.height, self.width = tif.pages[0].shape
                    self.frame_indices = np.arange(0, self.n_images)
                else:
                    self.n_images = len(frame_indices)
                    self.height, self.width = tif.pages[0].shape
                    self.frame_indices=frame_indices

            if max_memory is None:
                self.max_stack = self.n_images
            self.max_stack = max_memory

 
    
    def read_image(self, frames, channel):

        if self.omero:

            if channel==0 or channel=='cyto' or channel=='cytoplasm':
                channel=1
                return self.conn.get_np(self.image_id, frames, channel)
            elif channel==1 or channel=='nucleus':
                channel=0
                return self.conn.get_np(self.image_id, frames, channel)

        if self.nd2_file is not None:

            if channel==0 or channel=='cyto' or channel=='cytoplasm':
                c = self.cyto_channel
                return functions.read_nd2(os.path.join(self.data_path, self.nd2_file), self.fov, frames, c)
            if channel==1 or channel=='nucleus':
                c = self.nucleus_channel
                return functions.read_nd2(os.path.join(self.data_path, self.nd2_file), self.fov, frames, c)

        if self.cyto_file.endswith('.mp4'):

            if channel==0 or channel=='cyto' or channel=='cytoplasm':
                return functions.mp4_to_np(os.path.join(self.data_path, self.cyto_file), frames=frames)
            elif channel==1 or channel=='nucleus':
                return functions.mp4_to_np(os.path.join(self.data_path, self.nucleus_file), frames=frames)
        
        if self.nucleus_file.endswith('.tif'):

            if channel==0 or channel=='cyto' or channel=='cytoplasm':
                    return imread(os.path.join(self.data_path, self.cyto_file), key=frames)
            if channel==1 or channel=='nucleus':
                    return imread(os.path.join(self.data_path, self.nucleus_file), key=frames)



    def preprocess(self, cyto_contrast=1, cyto_brightness=0, nucleus_contrast=1, nucleus_brightness=0, log_nucleus=True, log_cyto=False, cyto_bottom_percentile=0.05, cyto_top_percentile=99.95, nucleus_bottom_percentile=0.05, nucleus_top_percentile=99.95):
        
        self.cyto_path = os.path.join(self.path_out, 'preprocessed_' + self.cyto_file)
        self.nucleus_path = os.path.join(self.path_out, 'preprocessed_' + self.nucleus_file)
        
        if self.cyto_path.endswith('.tif'):
            self.cyto_path = self.cyto_path.split('.tif')[0]+'.mp4'
        if self.nucleus_path.endswith('.tif'):
            self.nucleus_path = self.nucleus_path.split('tif')[0]+'.mp4'

        self.metadata.update(locals())
        self.metadata.pop('self')
        with open(self.meta_path, "w") as outfile:
            json.dump(dict(self.metadata), outfile)

        if os.path.isfile(self.cyto_path) and os.path.isfile(self.nucleus_path):
            self.preprocessed=True
            return

        # if self.cyto_file.endswith('.mp4'):
        #     cytoplasm = functions.mp4_to_np(os.path.join(self.data_path, self.cyto_file), frames = self.image_indices)
        # else:
        #     cytoplasm = imread(os.path.join(self.data_path, self.cyto_file), key=range(self.image_indices[0], self.image_indices[1]))

        cytoplasm = self.read_image(channel=0)
        
        cytoplasm = functions.preprocess(cytoplasm, c=cyto_contrast, b=cyto_brightness, bottom_percentile=cyto_bottom_percentile, top_percentile=cyto_top_percentile, return_type='uint8')
 
        functions.np_to_mp4(cytoplasm, self.cyto_path)
        del cytoplasm
        
        # if self.nucleus_file.endswith('.mp4'):
        #     nucleus = functions.mp4_to_np(os.path.join(self.data_path, self.nucleus_file), frames = self.image_indices)
        # else:
        #     nucleus = imread(os.path.join(self.data_path, self.nucleus_file), key=range(self.image_indices[0], self.image_indices[1]))
        nucleus = self.read_image(channel=1)

        nucleus = functions.preprocess(nucleus, bottom_percentile=nucleus_bottom_percentile, top_percentile=nucleus_top_percentile, c=nucleus_contrast, b=nucleus_brightness, return_type='uint8')
        print(nucleus.dtype)
        print(np.min(nucleus), np.max(nucleus))

        functions.np_to_mp4(nucleus, self.nucleus_path)
        del nucleus
        self.preprocessed=True
        
        return
        

    def get_masks(self, pretrained_model=None, flow_threshold=0.8, mask_threshold=-2, gpu=True,model_type='cyto', cyto_diameter=29):
        
        if not self.preprocessed:
            self.preprocess()
        
        self.metadata.update(locals())
        self.metadata.pop('self')
        with open(self.meta_path, "w") as outfile:
            json.dump(dict(self.metadata), outfile)

        if self.max_memory is None:
            single=True
        elif self.max_memory>=self.n_images:
            single=True
        else:
            single=False

        if single:

            cytoplasm = (functions.mp4_to_np(self.cyto_path, frames=None)/255).astype('float32')
            ##nucleus = (imread(self.nucleus_path, key=range(image_indices[0], image_indices[1]))/255).astype('float32')
            nucleus = (functions.mp4_to_np(self.nucleus_path, frames=None)/255).astype('float32')
            
            cyto_masks = segment(cytoplasm, nucleus, gpu=gpu, model_type=model_type, channels=[1,2], diameter=cyto_diameter, flow_threshold=flow_threshold, mask_threshold=mask_threshold, pretrained_model=pretrained_model, check_preprocessing=False)
            
            functions.np_to_mp4(cyto_masks, os.path.join(self.path_out, 'cyto_masks.mp4'))

            return

        if self.max_memory is None:
            max_stack = self.n_images
        else:
            max_stack = self.max_memory
        
        i_0, i_f = 0, self.n_images
        i_step = max_stack
        index_values = np.append(np.arange(i_0, i_f, i_step), i_f)

        try:
            tmp = os.path.join(self.path_out, 'tmp')
            os.mkdir(tmp)
        except FileExistsError:
            pass
        

        #Segment the cytoplasm
        cyto_paths = []
        for j in range(index_values.size -1):
            
            image_indices = [index_values[j], index_values[j+1]]
            path_out_tmp_cyto = os.path.join(tmp, str(image_indices[0]) + '-' + str(image_indices[1]) + '_cyto_masks.mp4')
                       
            ##cytoplasm = (imread(self.cyto_path, key=range(image_indices[0], image_indices[1]))/255).astype('float32')
            cytoplasm = (functions.mp4_to_np(self.cyto_path, frames=image_indices)/255).astype('float32')
            ##nucleus = (imread(self.nucleus_path, key=range(image_indices[0], image_indices[1]))/255).astype('float32')
            nucleus = (functions.mp4_to_np(self.nucleus_path, frames=image_indices)/255).astype('float32')
            
            tmp_mask = segment(cytoplasm, nucleus, gpu=gpu, model_type=model_type, channels=[1,2], diameter=cyto_diameter, flow_threshold=flow_threshold, mask_threshold=mask_threshold, pretrained_model=pretrained_model, check_preprocessing=False)
            
            functions.np_to_mp4(tmp_mask, path_out_tmp_cyto)
            ##imwrite(path_out_tmp_cyto, (tmp_mask).astype('uint8'))
            cyto_paths.append(path_out_tmp_cyto)
            
        del tmp_mask  

        cyto_masks = np.zeros((self.n_images, self.height, self.width), dtype='uint8')

        last_index = 0
        first_index=0
        for path in cyto_paths:
            x = functions.mp4_to_np(path)
            first_index=last_index*1
            last_index+=x.shape[0]
            cyto_masks[first_index:last_index]=x
        """cyto_masks = imread(cyto_paths[:-1])
        cyto_masks = functions.mp4_to_np(cyto_paths[:-1])
        last_cyto_mask = imread(cyto_paths[-1])
        cyto_masks = cyto_masks.reshape(int(cyto_masks.size/(self.height*self.width)), self.height, self.width) 
        cyto_masks = np.append(cyto_masks, last_cyto_mask, axis=0)
        imwrite(os.path.join(self.path_out, 'cyto_masks.tif'), cyto_masks)
        del cyto_masks 
        del last_cyto_mask#clear memory"""

        functions.np_to_mp4(cyto_masks, os.path.join(self.path_out, 'cyto_masks.mp4'))
        
        for f in os.listdir(tmp):
            os.remove(os.path.join(tmp, f))
        os.rmdir(tmp)

        return
        
    def segment(self, cyto_contrast=1, cyto_brightness=0, nucleus_contrast=1, nucleus_brightness=0, log_nucleus=True, cyto_bottom_percentile=0.05, cyto_top_percentile=99.95, nucleus_bottom_percentile=0.05, nucleus_top_percentile=99.95, pretrained_model=None, flow_threshold=0.8, mask_threshold=-2, gpu=True, model_type='cyto', cyto_diameter=29, verbose=True):
        """Method created for 16bit segmentation. Here the frames of interest are read onto memory and segmented directly, instead of being saved to storage first.

        Args:
        
            cyto_contrast (int, optional): _description_. Defaults to 1.
            cyto_brightness (int, optional): _description_. Defaults to 0.
            nucleus_contrast (int, optional): _description_. Defaults to 1.
            nucleus_brightness (int, optional): _description_. Defaults to 0.
            log_nucleus (bool, optional): _description_. Defaults to True.
            cyto_bottom_percentile (float, optional): _description_. Defaults to 0.05.
            cyto_top_percentile (float, optional): _description_. Defaults to 99.95.
            nucleus_bottom_percentile (float, optional): _description_. Defaults to 0.05.
            nucleus_top_percentile (float, optional): _description_. Defaults to 99.95.
            pretrained_model (_type_, optional): _description_. Defaults to None.
            flow_threshold (float, optional): _description_. Defaults to 0.8.
            mask_threshold (int, optional): _description_. Defaults to -2.
            gpu (bool, optional): _description_. Defaults to True.
            model_type (str, optional): _description_. Defaults to 'cyto'.
            cyto_diameter (int, optional): _description_. Defaults to 29.
        """
        
        # self.cyto_path = os.path.join(self.data_path + self.cyto_file)
        # self.nucleus_path = os.path.join(self.data_path + self.nucleus_file)
        
        self.metadata.update(locals())
        self.metadata.pop('self')
        with open(self.meta_path, "w") as outfile:
            json.dump(dict(self.metadata), outfile)

        if self.max_memory is None:
            single=True
        elif self.max_memory>=self.n_images:
            single=True
        else:
            single=False

        if single:

            cytoplasm = self.read_image(frames=np.arange(self.n_images), channel=0)
            cytoplasm = functions.preprocess(cytoplasm, bottom_percentile=cyto_bottom_percentile, top_percentile=cyto_top_percentile, return_type='float32')
            nucleus = self.read_image(frames=np.arange(self.n_images), channel=1)
            nucleus = functions.preprocess(nucleus, bottom_percentile=nucleus_bottom_percentile, top_percentile=nucleus_top_percentile, log=log_nucleus, return_type='float32')

            cyto_masks = segment(cytoplasm, nucleus, gpu=gpu, model_type=model_type, channels=[1,2], diameter=cyto_diameter, flow_threshold=flow_threshold, mask_threshold=mask_threshold, pretrained_model=pretrained_model, check_preprocessing=False)
            
            functions.np_to_mp4(cyto_masks, os.path.join(self.path_out, 'cyto_masks.mp4'))

        if self.max_memory is None:
            max_stack = self.n_images
        else:
            max_stack = self.max_memory
        
        i_0, i_f = 0, self.n_images
        i_step = max_stack
        index_values = np.append(np.arange(i_0, i_f, i_step), i_f)

        try:
            tmp = os.path.join(self.path_out, 'tmp')
            os.mkdir(tmp)
        except FileExistsError:
            pass
        
        print(self.n_images)
        #Segment the cytoplasm
        cyto_paths = []
        #print(f"Segmenting the {self.n_images} cytoplasm images")
        for j in range(index_values.size -1): #Loop through memory batches to segment each and save result in tmp directory
            
            #print(f'Segmenting batch {j+1}/{index_values.size-1}...')

            image_indices = np.arange(index_values[j], index_values[j+1])
            path_out_tmp_cyto = os.path.join(tmp, str(image_indices[0]) + '-' + str(image_indices[-1]) + '_cyto_masks.mp4')
          
            cytoplasm = self.read_image(image_indices, channel=0)
            cytoplasm = cytoplasm.reshape(
                int(cytoplasm.size/(self.height*self.width)), self.height, self.width)
            # cytoplasm = functions.preprocess(cytoplasm, bottom_percentile=cyto_bottom_percentile, top_percentile=cyto_top_percentile, return_type='uint16')
            cytoplasm = functions.preprocess(cytoplasm, bottom_percentile=cyto_bottom_percentile, top_percentile=cyto_top_percentile, return_type='float32')
            nucleus = self.read_image(image_indices, channel=1).astype('uint16')
            nucleus = nucleus.reshape(
                int(nucleus.size/(self.height*self.width)), self.height, self.width)
            # nucleus = functions.preprocess(nucleus, bottom_percentile=nucleus_bottom_percentile, top_percentile=nucleus_top_percentile, log=False, return_type='uint16')
            nucleus = functions.preprocess(nucleus, bottom_percentile=nucleus_bottom_percentile, top_percentile=nucleus_top_percentile, log=True, return_type='float32')
            
            tmp_mask = segment(cytoplasm, nucleus, gpu=gpu, model_type=model_type, channels=[1,2], diameter=cyto_diameter, flow_threshold=flow_threshold, mask_threshold=mask_threshold, pretrained_model=pretrained_model, check_preprocessing=False, verbose=verbose)
            
            functions.np_to_mp4(tmp_mask, path_out_tmp_cyto)
            ##imwrite(path_out_tmp_cyto, (tmp_mask).astype('uint8'))
            cyto_paths.append(path_out_tmp_cyto)
            
        del tmp_mask  

        #Rearrange segmented masks into one array and save as mp4 file
        cyto_masks = np.zeros((self.n_images, self.height, self.width), dtype='uint8')


        last_index = 0
        first_index=0
        for path in cyto_paths:
            x = functions.mp4_to_np(path)
            first_index=last_index*1
            last_index+=x.shape[0]
            cyto_masks[first_index:last_index]=x
      

        functions.np_to_mp4(cyto_masks, os.path.join(self.path_out, 'cyto_masks.mp4'))
        
        for f in os.listdir(tmp):
            os.remove(os.path.join(tmp, f))
        os.rmdir(tmp)

        return

    def segment_2(self, cyto_contrast=1, cyto_brightness=0, nucleus_contrast=1, nucleus_brightness=0, log_nucleus=True, cyto_bottom_percentile=0.05, cyto_top_percentile=99.95, nucleus_bottom_percentile=0.05, nucleus_top_percentile=99.95, pretrained_model=None, flow_threshold=0.8, mask_threshold=-2, gpu=True, model_type='cyto', cyto_diameter=29, verbose=False, savenumpy=True):
        self.metadata.update(locals())
        self.metadata.pop('self')
        with open(self.meta_path, "w") as outfile:
            json.dump(dict(self.metadata), outfile)

        cytoplasm = self.read_image(frames=self.frame_indices, channel=0)
        #cytoplasm = functions.preprocess(cytoplasm, bottom_percentile=cyto_bottom_percentile, top_percentile=cyto_top_percentile, return_type='float32')
        
        nucleus = self.read_image(frames=self.frame_indices, channel=1)
        #nucleus = functions.preprocess(nucleus, bottom_percentile=nucleus_bottom_percentile, top_percentile=nucleus_top_percentile, log=log_nucleus, return_type='float32')
        
        cyto_masks = segment_looped(cytoplasm, nucleus, gpu=gpu, model_type=model_type, channels=[1,2], diameter=cyto_diameter, flow_threshold=flow_threshold, mask_threshold=mask_threshold, pretrained_model=pretrained_model, check_preprocessing=False)
        if savenumpy:
            np.savez(os.path.join(self.path_out, 'cyto_masks.npz'), cyto_masks)
        else:
            functions.np_to_mp4(cyto_masks, os.path.join(self.path_out, 'cyto_masks.mp4'))

        return

    def get_nucleus_tracks(self, df=None, diameter=19, minmass=None, track_memory=15, max_travel=5):
        
        self.metadata.update(locals())
        self.metadata.pop('self')
        with open(self.meta_path, "w") as outfile:
            json.dump(dict(self.metadata), outfile)

        if df is not None:
            self.df = pd.read_csv(self.df_path)
            return

        #if self.preprocessed:
        if False:
            nuclei = functions.mp4_to_np(self.nucleus_path)
            print(f'nuclei {nuclei.shape}')
            self.df = tracking.track(nuclei, diameter=diameter, minmass=minmass, track_memory=track_memory, max_travel=max_travel)
            self.df.to_csv(self.df_path)

        else:    

            nuclei = self.read_image(frames = self.frame_indices, channel='nucleus')
            print('preprocesing')
            nuclei = functions.preprocess(nuclei, bottom_percentile=0.05, top_percentile=99.95, log=True, return_type='uint16')
            print('done')
            self.df = tracking.track(nuclei, diameter=diameter, minmass=minmass, track_memory=track_memory, max_travel=max_travel)
            self.df.to_csv(self.df_path)
        
        del nuclei
        return

    def detect_lanes(self, lane_distance=30, low_clip=300, high_clip=2000):

        path_to_image = os.path.join(self.data_path, self.lanes_file)

        if path_to_image.endswith('.tif'):

            lanes_image = imread(path_to_image)
        
        elif path_to_image.endswith('.nd2'):

            lanes_image = functions.read_nd2(path_to_image, self.fov)

        lanes_image = np.clip(lanes_image, low_clip, high_clip)

        self.lanes_mask, self.lanes_metric = lane_detection.get_lane_mask(
            lanes_image, kernel_width=5, line_distance=lane_distance
            )
        self.n_lanes = self.lanes_mask.max()

        try:
            lanes_dir = os.path.join(self.path_out, 'lanes')
            os.mkdir(lanes_dir)
        except FileExistsError:
            pass

        imwrite(os.path.join(lanes_dir, 'lanes_mask.tif'), self.lanes_mask)
        imwrite(os.path.join(lanes_dir, 'lanes_metric.tif'), self.lanes_metric)

        return

    def get_locations(self, df=None, cyto_masks_path=None):

        if df is not None:
            self.df = df

        lanes_dir = os.path.join(self.path_out, 'lanes')
        if not os.path.isfile(os.path.join(lanes_dir, 'lanes_mask.tif')):
            self.detect_lanes()
        
        if cyto_masks_path is None:    
            cyto_masks = functions.mp4_to_np(os.path.join(self.path_out, 'cyto_masks.mp4'))
        else:
            #This might also be problematic
            cyto_masks = functions.mp4_to_np(cyto_masks_path, self.image_indices)
        
        try:
            self.lanes_mask
            path_to_patterns = os.path.join(self.data_path, self.lanes_file)
            patterns = functions.read_nd2(path_to_patterns, self.fov)
            self.df = tracking.get_tracking_data(self.df, cyto_masks, self.lanes_mask, self.lanes_metric, patterns)
            
            self.df.index.names = ['Index']

            # self.df = tracking.get_single_cells(self.df)
            # self.df = tracking.remove_close_cells(self.df)
            # self.clean_df = tracking.get_clean_tracks(self.df)

        except AttributeError:
            path_to_patterns = os.path.join(self.data_path, self.lanes_file)
            patterns = functions.read_nd2(path_to_patterns, self.fov)
            lanes_dir = os.path.join(self.path_out, 'lanes')
            self.lanes_mask = imread(os.path.join(lanes_dir, 'lanes_mask.tif'))
            self.lanes_metric = imread(os.path.join(lanes_dir, 'lanes_metric.tif'))
            self.df = tracking.get_tracking_data(self.df, cyto_masks, self.lanes_mask, self.lanes_metric, patterns)

            # self.df = tracking.get_single_cells(self.df)
            # self.df = tracking.remove_close_cells(self.df)

            # self.clean_df = tracking.get_clean_tracks(self.df)

        self.df.to_csv(self.df_path)
        #self.clean_df.to_csv(self.clean_df_path)
        
        return
        
    def get_movie(self, cyto_masks_path=None, crf=20, draw_lanes=True, draw_contours=True, rate=None, from_file=True, draw_nuclei=True):
        
        print(f'Making movie for fov {self.fov}...')
        from skimage.segmentation import find_boundaries
        from skimage import draw
        
        if from_file:

            try:
                original = self.read_image(frames=np.arange(self.n_images), channel='cyto')
                
            except ValueError:
                self.cyto_path = os.path.join(self.data_path, self.cyto_file)
                self.nucleus_path = os.path.join(self.data_path + self.nucleus_file)
                original = self.read_image(self.cyto_path)
                
            original = functions.preprocess(original, bottom_percentile=0.05, top_percentile=99.95, return_type='uint8')

            # original_nuclei = self.read_image(frames=np.arange(self.n_images), channel='nucleus')
            # original_nuclei = self.read_image(frames=np.arange(10), channel='nucleus')
            # original_nuclei = functions.preprocess(original_nuclei, bottom_percentile=0.01, top_percentile=99.95, log=True, return_type='uint8')

        else:
            try:
                original = functions.mp4_to_np(self.cyto_path)
                original_nuclei = functions.mp4_to_np(self.nucleus_path)
            except AttributeError:
                original = functions.mp4_to_np(os.path.join(self.data_path, self.cyto_file), self.image_indices)
                original_nuclei = functions.mp4_to_np(os.path.join(self.data_path, self.nucleus_file), self.image_indices)
        
        if rate is None:
            delta_t = 30
            rate = 15*60/delta_t

        #blue=np.zeros(original.shape, dtype='uint8')
        #original = (original/3).astype('uint8')
        film = np.stack((original, original, original), axis=-1)
        # film = np.stack((original_nuclei, blue, blue), axis=-1)
        # film = np.clip(film[:, :, :, :] + (original/3)[:,:,:,np.newaxis], 0, 255).astype('uint8')

        if draw_contours:

            if cyto_masks_path is None:
                cyto_masks = functions.mp4_to_np(os.path.join(self.path_out, 'cyto_masks.mp4'))

            else:
                cyto_masks = functions.mp4_to_np(os.path.join(self.path_out), 'cyto_masks.mp4')

            cyto_contours = np.zeros(cyto_masks.shape, dtype=np.bool)
            print('finding boundaries from masks')
            #for frame in tqdm(range(cyto_masks.shape[0])):
            for frame in tqdm(range(cyto_masks.shape[0])):
                cyto_contours[frame] = find_boundaries(cyto_masks[frame], background=0, mode='outer')
                film[cyto_contours[:,:,:]!=0, 1] = 255
        
        image_shape = film.shape[1:3]

        if draw_lanes:
            self.lanes_mask = imread(os.path.join(self.path_out, 'lanes/lanes_mask.tif'))
            red = find_boundaries(self.lanes_mask>0)
            red = (100*(red)).astype('uint8')
            film[:, :, :, 0] = np.clip(film[:, :, :, 0] + red[np.newaxis, :,:,], 0, 255).astype('uint8')

            
        if draw_nuclei:
            
            self.df = pd.read_csv(self.df_path)

            p_x = np.round(self.df.x.values).astype(int)
            p_y = np.round(self.df.y.values).astype(int)
            t = self.df.frame.values.astype(int)

            for point in self.df.index:
                rr, cc = draw.disk((p_y[point],p_x[point]), radius=5, shape=image_shape)
                film[t[point], rr, cc] = [0,0,255]
    
        film = functions.label_movie(film)
        
        functions.np_to_mp4(film, os.path.join(self.path_out, f'film_XY{self.fov}.mp4'), crf=crf, rate=rate)
    
    
    def get_kymographs(self):
        
        from skimage.io import imsave
        
        if self.preprocessed:
            cyto = functions.mp4_to_np(self.cyto_path, self.image_indices)
        else:   
            cyto = functions.mp4_to_np(os.path.join(self.data_path, self.cyto_file), self.image_indices)

        for line_index in self.lines_df.index:
            
            df = self.df[self.df.line_index==line_index]
            if len(df)<25:
                continue
            coordinates = [self.lines_df.x_0[line_index], self.lines_df.x_f[line_index], self.lines_df.y_0[line_index], self.lines_df.y_f[line_index]]
     
            kymo = functions.get_kymograph(cyto, coordinates, 20)
            kymo = functions.preprocess(kymo, return_type='uint8', bottom_percentile=0.1, top_percentile=99.9)
            kymo = np.stack((kymo, kymo, kymo), axis=-1)
            front = np.round(df.front.values.astype(int))
            rear = np.round(df.rear.values.astype(int))
            center = np.round(df.center.values.astype(int))
            frame = np.round(df.frame.values.astype(int))
            kymo[front, frame] = [0,255,0]
            kymo[center, frame] = [0,0,255]
            kymo[rear, frame] = [0,255,0]
            imsave(os.path.join(self.path_out, f'segmented_kymo_line_{line_index+1}.png'), kymo)
        
        return

    def get_clean_tracks(self, tres, df=None):

        if df is None:
            df = self.df
        self.clean_df = tracking.get_clean_tracks(df)
        self.clean_df = tracking.classify_tracks(self.clean_df, tres=tres)
        #clean_df_path = self.df_path.split('.')[0]+'clean.csv'
        self.clean_df.to_csv(self.clean_df_path)

def run_pipeline(data_path, nd2_file, lanes_file, path_out, frame_indices=None, manual=False, fovs=None, sql=False, lane_distance=30,
 lane_low_clip=0, lane_high_clip=2000, min_mass=2.65e5, max_travel=15, track_memory=15, diameter=15, min_frames=10, cyto_diameter=29, 
 flow_threshold=1.25, mask_threshold=0, pretrained_model='mdamb231', use_existing_parameters=False, bf_channel=None, nuc_channel=None, tres=120):
    
    args=locals()
    ##Save arguments to file
    args_file = os.path.join(path_out, 'pipeline_args.json')
    if not use_existing_parameters:
        if args['frame_indices'] is not None:
            args['frame_indices'] = list(args['frame_indices'])
    
        with open(args_file, 'wb') as outfile:
            pickle.dump(dict(args), outfile)

    elif use_existing_parameters:        
        try:
            with open(args_file, 'rb') as f:
                args = pickle.load(f)
                args = dict(args)
                #return
        except FileNotFoundError:
            raise RuntimeError('No argument were saved in the last run')
        
        args['use_existing_parameters']=False
        args['fovs']=fovs
        return run_pipeline(*args.values())

    ##prepare logging
    log_filename=path_out+ 'pipeline.log'
    with open(log_filename, 'w') as log_file:
        log_file.write("Logging for Experiment \n")

    
    
    """logger = logging.getLogger('pipeline')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)"""

    if fovs is None:
        f = ND2Reader(os.path.join(data_path, nd2_file))
        fovs = range(f.sizes['t'])


    too_muchtravel=[]
    start_time = time.time()
    for fov in fovs:
        try:
            print(f'Computing field of view {fov}')

            #Create a directory for the output for this field of view
            path_out_fov = os.path.join(path_out, f'XY{fov}/')

            if not os.path.isdir(path_out_fov):
                os.mkdir(path_out_fov)

            #Initiate a Track object
            track = Track(data_path=data_path, path_out=path_out_fov, nd2_file=nd2_file, lanes_file=lanes_file, frame_indices=frame_indices, fov=fov, bf_channel=bf_channel, nuc_channel=nuc_channel)

            #Start by detecting the lanes
            if not os.path.isfile(path_out_fov+'lanes/lanes_mask.tif'):
                try:
                    print('Detecting lanes')
                    track.detect_lanes(lane_distance=lane_distance, low_clip=lane_low_clip, high_clip=lane_high_clip)
                    print('Lane detection succesful')
                except:
                    try:
                        print('Detecting lanes with higer high clip value')
                        track.detect_lanes(lane_distance=lane_distance, low_clip=0, high_clip=5000)
                        print('Lane detection succesful')
                    except:
                        print('Warning! The lane detection failed for field of view {fov}. Skipping to the next field of view.')
                        continue
            else:
                print('Found existing Lanes mask')
                track.lanes_mask, track.lanes_metric = imread(path_out_fov+'lanes/lanes_mask.tif'), imread(path_out_fov+'lanes/lanes_metric.tif')
            # #Run Segmentation of the cytoplasm with Cellpose
            t_0 = time.time()
            track.segment_2(cyto_bottom_percentile=0.0, cyto_top_percentile=100, nucleus_bottom_percentile=0, nucleus_top_percentile=100, flow_threshold=flow_threshold, mask_threshold=mask_threshold, pretrained_model=pretrained_model, cyto_diameter=cyto_diameter, verbose=False)
            t_1 = time.time() - t_0
            print(f'It took {t_1} seconds to run cellpose')

            #Now run the tracking
            try:
                track.get_nucleus_tracks(max_travel=max_travel, diameter=diameter, minmass=min_mass, track_memory=track_memory)
            except:
                try:
                    track.get_nucleus_tracks(max_travel=25, diameter=diameter, minmass=min_mass, track_memory=track_memory)
                except:
                    too_muchtravel.append(fov)
                    continue
            track.df = pd.read_csv(f'{path_out_fov}tracking_data.csv')
            track.get_locations()

            track.get_clean_tracks(tres=tres)
            
            #Now enter the data from the stored data into the tracks table in the database
            if not sql:
                continue

            #First add an entry into the lanes file for each detected lane
            db.add_fov(experiment_id, fov, track.n_lanes)

            try:
                df = track.df
            except AttributeError:
                df = pd.read_csv(track.df_path)

            columns = ['frame', 'y', 'x', 'nucleus_mass', 'nucleus_size', 'particle_id', 'nucleus', 'front', 'rear', 'footprint', 'area', 'valid', 'interpolated', 'Lane_id', 'cyto_locator', 'FN_signal']

            db.add_raw_tracks(df, experiment_id, fov, columns=columns)

            track.get_movie(draw_contours=True)
        except Exception as e:
            print(f'Error at fov {fov}. See {log_filename} for more details')
            print(str(e))
            with open(log_filename, 'a') as log_file:
                log_file.write(str(traceback.format_exc())+'\n')
            continue
        
            
    print(f'It took {(time.time()-start_time)/3600} hours to run pipeline on {len(fovs)} fields of view.')
    print(f'The following fields of view were run with a lower max travel value {too_muchtravel}')