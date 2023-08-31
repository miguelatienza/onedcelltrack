import datetime
from tifffile import TiffFile, imwrite, imread
import os
from . import functions
from .segmentation import Segmentation
from . import tracking 
from . import lane_detection
import pandas as pd
from skvideo import io
import json
from nd2reader import ND2Reader
import glob
import numpy as np
import logging
from tqdm import tqdm

from IPython.display import display

RAW_TRACKING_DATA_FILE_NAME = 'tracking_data.csv'
TRACKING_DATA_FILE_NAME = 'clean_tracking_data.csv'

class Pipeline:
    def __init__(self, pipeline_arguments=None):
        """
        Parameters
        ----------
        pipeline_arguments : str, optional
            Path to the pipeline arguments json file. Default is None.
        """
        param_dict = self.build_param_dict(pipeline_arguments=pipeline_arguments)
        
        self.__dict__.update(param_dict)

    def build_param_dict(self, pipeline_arguments=None):
        """
        Function to build the parameter dictionary from the default_pipeline_arguments.json file

        Returns
        -------
        param_dict : dict
            Dictionary containing all the parameters for the pipeline.
        """
        if pipeline_arguments is None:
            #use defaults
            arguments_file = os.path.join(os.path.dirname(__file__), 'default_pipeline_arguments.json')
        else:
            arguments_file = pipeline_arguments
        
        with open(arguments_file) as f:
            param_dict = json.load(f)
            self.__dict__.update(param_dict)
            self.param_keys = list(param_dict.keys())

        return param_dict

    def update_file_names(self, data_path, lanes_file, path_out, image_file=None, bf_files=None, nuc_files=None):
        """
        Updates the image_file and lanes_file in the param_dict if they are not present in the data_path. 

        The files are checked for the correct format.

        It then builds the Reader which is used to load the images and lanes.

        If an nd2 file is provided, the pixel/micron ratio will be extracted from the metadata and saved to the param_dict. Otherwise it must be specified later.
        
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
        self.check_files(data_path, image_file=image_file, lanes_file=lanes_file, path_out=path_out, bf_files=bf_files, nuc_files=nuc_files)
        
        self.data_path = data_path
        self.image_file = image_file
        self.lanes_file = lanes_file
        self.bf_file = bf_files
        self.bf_file = nuc_files
        self.path_out = path_out

        self.infer_pixel_micron_ratio()
        self.frame_indices=self.infer_frame_indices()
        self.n_frames = len(self.frame_indices)
        self.fovs = self.infer_fovs()
        self.n_fovs = len(self.fovs)
        self.fovs_lanes = self.infer_fovs_lanes()
        self.n_fovs_lanes = len(self.fovs_lanes)

        assert self.n_fovs==self.n_fovs_lanes, 'The number of fovs in the lanes file is not equal to the number of fovs in the nd2 file. Please check if the files are correct. If it is, make sure to specify the fovs and lanes_fovs in the parameters.'

        return self.get_param_dict()

    def check_files(self, data_path, lanes_file, path_out, image_file=None, bf_files=None, nuc_files=None):
        """
        Checks if the files are present in the data_path and if they are of the right type. Accepted types are .nd2 and .tif.
        At the moment it accepts three options for the cell files:
            1. An nd2 file with all fovs, time_points and channels. The user must specify fovs and lanes_fovs as well later.
            2. A tif file with shape [fovs, time_points, height, width, channels] and a lanes file containing all fovs. The tif files should be in the same order as the fovs in the lanes file.
            3. A list of tif files: bf_files, and nuc_files and a lanes file containing all fovs. The tif files should be in the same order as the fovs in the lanes file.
        The lanes images accept two different formats:
            1. An nd2 file with all fovs
            2. A tif file with shape [fovs, height, width]
        
        If the number of fovs in the lanes file is not equal to the number of fovs in the nd2 file, it will raise a Warning

        Parameters
        ----------
        data_path : str
            Path to the data folder.
        image_file : str or list
            Name of the image file or list of names of the image files.
        lanes_file : str
            Name of the lanes file.  
        """
        cell_images_checked=False

        ### Handle the cell images
        ## 1. case: nd2 file with all fovs, time_points and channels and an extra nd2 file for the lanes containing each fov
        if not isinstance(image_file, list):
            if image_file.endswith('.nd2'):
                #make sure the file exists
                assert os.path.isfile(os.path.join(data_path, image_file)), f'Could not find {image_file} in {data_path}'
                f = ND2Reader(os.path.join(data_path, image_file))
                n_fovs = f.sizes['v']
                cell_images_checked=True
                self.nd2_cell_files=True

        ## 2. case: tif file with shape [fovs, time_points, height, width, channels] and a lanes file containing all fovs
        if not isinstance(image_file, list):
            if image_file.endswith('tif') or image_file.endswith('tiff'):
                ## Warn the user that this option is very memory consuming
                print('WARNING: This option is very memory consuming. Please consider using option 3.')
                #make sure the file exists
                assert os.path.isfile(os.path.join(data_path, image_file)), f'Could not find {image_file} in {data_path}'
                image = imread(os.path.join(data_path, image_file))
                assert image.ndim==5, f'{image_file} is not a 3D image stack with 2 channels'
                assert image.shape[-1]==2, 'The tif stack should be of the shape (fovs, time_points, height, width, channels) with 2 channels'
                n_fovs = image.shape[0]
                cell_images_checked=True
                self.single_tif_cell_file=True
        
        ## 3. case: list of tif files: bf_files, and nuc_files and a lanes file containing all fovs
        if bf_files is not None and isinstance(bf_files, list):
            if isinstance(nuc_files, list):
                assert len(bf_files)==len(nuc_files), 'The number of bf_files should be equal to the number of nuc_files'
                n_fovs = len(bf_files)
                for i in range(n_fovs[:1]):# don't go through all files as it should take too long
                    bf_image = imread(os.path.join(data_path, bf_files[i]))
                    nuc_image = imread(os.path.join(data_path, nuc_files[i]))
                    assert bf_image.ndim==3, f'{bf_files[i]} is not a 2D image stack'
                    assert nuc_image.ndim==3, f'{nuc_files[i]} is not a 2D image stack'
                    assert bf_image.shape==nuc_image.shape, f'The shape of {bf_files[i]} is not equal to the shape of {nuc_files[i]}'
                    cell_images_checked=True
                    n_fovs = len(bf_files)
                    self.multiple_tif_cell_files=True
            else:
                raise ValueError('If bf_files is a list, nuc_files should be a list as well')
        elif bf_files is not None and not isinstance(bf_files, list):
            raise ValueError('bf_files should be a list if specified')

        assert cell_images_checked, 'The cell images are not of the right type. Please view Documentation for accepted types'

        ### Handle the lanes images
        if lanes_file.endswith('.nd2'):
            #make sure the file exists
            assert os.path.isfile(os.path.join(data_path, lanes_file)), f'Could not find {lanes_file} in {data_path}'
            f = ND2Reader(os.path.join(data_path, lanes_file))
            n_fovs_lanes = f.sizes['v']
            lanes_images_checked=True
            self.nd2_lanes_file=True

        if lanes_file.endswith('tif'):
            #make sure the file exists
            assert os.path.isfile(os.path.join(data_path, lanes_file)), f'Could not find {lanes_file} in {data_path}'
            lanes_image = imread(os.path.join(data_path, lanes_file))
            assert lanes_image.ndim==3, f'{lanes_file} is not a 2D image stack'
            n_fovs_lanes = lanes_image.shape[0]
            lanes_images_checked=True
            self.single_tif_lanes_file=True
        
        assert lanes_images_checked, 'The lanes images are not of the right type. Please view Documentation for accepted types'

        if n_fovs!=n_fovs_lanes:
             print(f'WARNING! The number of fovs in the lanes file ({n_fovs_lanes}) is not equal to the number of fovs in the nd2 file ({n_fovs}). Please check if the files are correct. If it is, make sure to specify the fovs and lanes_fovs in the parameters.')
        else:
            print('All files are present and of the right type')
        
        ## Save the number of fovs and lanes_fovs   
        self.n_fovs = n_fovs
        self.n_fovs_lanes = n_fovs_lanes


        # Make sure path_out exists
        try:
            os.mkdir(path_out)
        except FileExistsError:
            pass

        return
    
    def read_bf(self, frames=None, fov=0):
        """
        Function to read the brightfield image file
        
        Parameters
        ----------
        frames : slice or list of int, optional
            Frames to be extracted. If None, all frames are extracted.
        fov : int, optional
            Index of the field of view to be extracted. Default is 0.

        Returns
        -------
        image : ndarray

        """
        if self.nd2_cell_files:
            assert self.bf_channel is not None, 'Please specify the brightfield channel'
            image = functions.read_nd2(os.path.join(self.data_path, self.image_file), c=self.bf_channel, frames=frames, v=fov)
            return image
        if self.single_tif_cell_file:
            assert self.bf_channel is not None, 'Please specify the brightfield channel'
            image = imread(os.path.join(self.data_path, self.image_file))
            return image[fov, frames, :, :, self.bf_channel]
        if self.multiple_tif_cell_files:
            bf_image = imread(os.path.join(self.data_path, self.bf_file[fov]))
            return bf_image[frames]
    
    def read_nuc(self, frames=None, fov=0):
        """
        Function to read the nuclear image file
        
        Parameters
        ----------
        frames : slice or list of int, optional
            Frames to be extracted. If None, all frames are extracted.
        fov : int, optional
            Index of the field of view to be extracted. Default is 0.

        Returns
        -------
        image : ndarray

        """
        if self.nd2_cell_files:
            assert self.nuc_channel is not None, 'Please specify the nuclear channel'
            image = functions.read_nd2(os.path.join(self.data_path, self.image_file), c=self.nuc_channel, frames=frames, v=fov)

            return image
        if self.single_tif_cell_file:
            assert self.nuc_channel is not None, 'Please specify the nuclear channel'
            image = imread(os.path.join(self.data_path, self.image_file))
            return image[fov, frames, :, :, self.nuc_channel]
        if self.multiple_tif_cell_files:
            nuc_image = imread(os.path.join(self.data_path, self.nuc_file[fov]))
            return nuc_image[frames]
    
    def read_lanes(self, fov=0):
        """
        Function to read the lanes image file
        
        Parameters
        ----------
        fov : int, optional
            Index of the field of view to be extracted. Default is 0.

        Returns
        -------
        image : ndarray

        """
        if self.nd2_lanes_file:
            image = functions.read_nd2(os.path.join(self.data_path, self.lanes_file), c=0, frames=None, v=fov)
            return image
        if self.single_tif_lanes_file:
            image = imread(os.path.join(self.data_path, self.lanes_file))
            return image[fov]

    def get_param_dict(self):
        """
        Function to get the parameters as a dictionary

        Returns
        -------
        param_dict : dict
            Dictionary containing all the parameters for the pipeline.
        """
        for key in self.param_keys:
            if isinstance(self.__dict__[key], np.ndarray):
                self.__dict__[key] = self.__dict__[key].tolist()
        param_dict = {key: value for key, value in self.__dict__.items() if key in self.param_keys}
        return param_dict

    def save_parameters(self):
        """
        Function to save the parameters to disk
        """
        # Code for saving parameters to disk
        with open(os.path.join(self.path_out, 'pipeline_arguments.json'), 'w') as f:
            json.dump(self.get_param_dict(), f)
      
        return

    def Viewer(self):
        from .viewer.notebook_viewer import Viewer
        self.viewer = Viewer(self)
        self.viewer.show()

    def LaneViewer(self):
        from .viewer.notebook_viewer import LaneViewer
        self.laneviewer = LaneViewer(self)
        self.laneviewer.show()
    
    def TrackingViewer(self):
        from .viewer.notebook_viewer import TrackingViewer
        self.trackingviewer = TrackingViewer(self)
        self.trackingviewer.show()

    def CellposeViewer(self, pretrained_model=None):
        if pretrained_model is not None:
            self.pretrained_model = pretrained_model
        from .viewer.notebook_viewer import CellposeViewer
        self.cellposeviewer = CellposeViewer(self)
        self.cellposeviewer.show()
    
    def infer_pixel_micron_ratio(self):
        """
        Function to infer the pixel/micron ratio from the metadata of the nd2 file. Only works with nd2 files.
        """
        if self.nd2_cell_files:

            f = ND2Reader(os.path.join(self.data_path, self.image_file))
            if 'pixel_microns' in f.metadata.keys():

                self.pixelperum = f.metadata['pixel_microns']
            else:
                self.pixelperum = None
                print(""""WARNING! The pixel/micron ratio couldn't be found in the readme file. Please specify it later.""")
        else:
            self.pixelperum = None
            print(""""WARNING! The pixel/micron ratio can only be automatically inferred with nd2 files. Please specify it later.""")
            return self.pixelperum
    
    def infer_frame_indices(self):
        """
        Function to infer the frame indices to be processed

        Returns
        -------
        frame_indices : list of int
            The frame indices to be processed.
        """
        if self.nd2_cell_files:
            f = ND2Reader(os.path.join(self.data_path, self.image_file))
            frame_indices = list(range(f.sizes['t']))

        elif self.single_tif_cell_file:
            image = imread(os.path.join(self.data_path, self.image_file))
            frame_indices = list(range(image.shape[1]))
        elif self.multiple_tif_cell_files:
            image = imread(os.path.join(self.data_path, self.bf_file[0]))
            frame_indices = list(range(image.shape[0]))
        
        return frame_indices
    
    def infer_fovs(self):
        """
        Function to infer the fovs to be processed

        Returns
        -------
        fovs : list of int
            The fovs to be processed.
        """
        if self.nd2_cell_files:
            f = ND2Reader(os.path.join(self.data_path, self.image_file))
            fovs = list(range(f.sizes['v']))
        elif self.single_tif_cell_file:
            image = imread(os.path.join(self.data_path, self.image_file))
            fovs = list(range(image.shape[0]))
        elif self.multiple_tif_cell_files:
            fovs = list(range(len(self.bf_file)))
        
        return fovs
    
    def infer_fovs_lanes(self):
        """
        Function to infer the fovs to be processed

        Returns
        -------
        fovs : list of int
            The fovs to be processed.
        """
        if self.nd2_lanes_file:
            f = ND2Reader(os.path.join(self.data_path, self.lanes_file))
            fovs = list(range(f.sizes['v']))
        elif self.single_tif_lanes_file:
            image = imread(os.path.join(self.data_path, self.lanes_file))
            fovs = list(range(image.shape[0]))
        
        return fovs

    def detect_lanes(self, fovs, verbose=True):
        """
        Function to detect the lanes in the image file
        """
        # Code for lane detection
        # Read the lanes image
        fovs = [fovs] if isinstance(fovs, int) else fovs
        bad_fovs = []
        for fov in tqdm(fovs, desc='Detecting lanes', disable=not verbose):
            
            try:
                lanes_dir = os.path.join(self.path_out, f'XY{fov}/lanes')
                lanes_image = self.read_lanes()
                
                lanes_mask, lanes_metric = lane_detection.get_lane_mask(
                    lanes_image, kernel_width=3, line_distance=self.lane_distance,
                    threshold=self.lane_threshold, low_clip=self.lane_low_clip, high_clip=self.lane_high_clip)


                imwrite(os.path.join(lanes_dir, 'lanes_mask.tif'), lanes_mask)
                imwrite(os.path.join(lanes_dir, 'lanes_metric.tif'), lanes_metric)
            except Exception as e:
                logging.warning(f'Could not detect lanes for fov {fov} \n {e}')
                bad_fovs.append(fov)
        
        if len(bad_fovs)>0:
            print(f'WARNING! Could not detect lanes for fovs {bad_fovs}')
                

        return lanes_mask, lanes_metric

    def segment(self, fov):
        """
        Function to segment the cells in the image file
        """
        # Code for cell segmentation
        self.segmentation = Segmentation(pretrained_model=self.pretrained_model, gpu=self.use_gpu, omni=self.omni)

        bf = self.read_bf(frames=self.frame_indices, fov=fov)
        nuc = self.read_nuc(frames=self.frame_indices, fov=fov)
        masks = self.segmentation.segment(brightfield=bf, nucleus=nuc, diameter=self.cyto_diameter, flow_threshold=self.flow_threshold, cellprob_threshold=self.cellprob_threshold, verbose=self.verbose)

        ## Now save the segmentation results
        self.save_segmentation_results(masks, fov=fov)

        return
    
    def save_segmentation_results(self, masks, fov=0):
        
        if self.savenumpy_masks:    
            outpath = os.path.join(self.path_out, f'XY{fov}/cyto_masks.npz')
            np.savez(outpath, masks)
        else:
            outpath = os.path.join(self.path_out, f'XY{fov}/cyto_masks.mp4')
            functions.np_to_mp4(masks, outpath)
            

    def track(self, fov):
        """
        Function to track the cells over time
        """
        nuclei = self.read_nuc(frames=self.frame_indices, fov=fov)
        df = tracking.track_nuclei(nuclei, diameter=self.diameter, minmass=self.min_mass, track_memory=self.track_memory, max_travel=self.max_travel, min_frames=self.min_frames, verbose=False, logger=self.logger)
        # Save the df
        dfpath = os.path.join(self.path_out, f'XY{fov}/tracking_data.csv')
        df.to_csv(dfpath)
        
        return
    
    def merge_tracking_data(self, fov, verbose=True):
        """
        Function to merge the tracking data of all fovs
        """
        # Get the tracking data, cyto masks and lanes masks for each fov
        try:
            df = pd.read_csv(os.path.join(self.path_out, f'XY{fov}/tracking_data.csv'))
        except FileNotFoundError:
            print(f'WARNING! Could not find tracking data for fov {fov}')
            self.logger.warning(f'Could not find tracking data for fov {fov}')
            if len(df)==0:
                print(f'WARNING! Tracking data for fov {fov} is empty, skipping this fov')
                self.logger.warning(f'Tracking data for fov {fov} is empty, skipping this fov')
                return  
        try:
            try:
                cyto_masks = np.load(os.path.join(self.path_out, f'XY{fov}/cyto_masks.npz'))['arr_0']
            except:
                cyto_masks = functions.mp4_to_np(os.path.join(self.path_out, f'XY{fov}/cyto_masks.mp4'))
        except FileNotFoundError:
           
            print(f'WARNING! Could not find masks for fov {fov}')
            self.logger.warning(f'Could not find tracking data for fov {fov}')
            return    
        try:
            lanes_image = self.read_lanes(fov=fov)
            lanes_mask = imread(os.path.join(self.path_out, f'XY{fov}/lanes/lanes_mask.tif'))
            lanes_metric = imread(os.path.join(self.path_out, f'XY{fov}/lanes/lanes_metric.tif'))
        except FileNotFoundError:
            print(f'WARNING! Could not find lanes mask for fov {fov}')
            self.logger.warning(f'Could not find tracking data for fov {fov}')
            return

        
        df = tracking.merge_tracking_data(df, cyto_masks, lanes_mask, lanes_metric, patterns=lanes_image, tres=self.tres, min_duration=self.min_duration, verbose=verbose, logger=self.logger)
        #now save the file
        dfpath = os.path.join(self.path_out, f'XY{fov}/tracking_data.csv')
        df.to_csv(dfpath)
        return

    def filter_tracking_data(self, fov):
        """
        Function to filter the tracking data
        """
        # Code for filtering tracking data
        # Read the tracking data
        df = pd.read_csv(os.path.join(self.path_out, f'XY{fov}/tracking_data.csv'))
        # Filter the tracking data
        df = tracking.get_clean_tracks(df, max_interpolation=5, min_length=self.min_track_length, image_height=self.height)
        
        dfpath = os.path.join(self.path_out, f'XY{fov}/clean_tracking_data.csv')

        df.to_csv(dfpath)

        return

    def classify_tracks(self, fov):
        """
        Function to classify the tracks
        """
        # Code for classifying tracks
        # Read the tracking data
        df = pd.read_csv(os.path.join(self.path_out, f'XY{fov}/clean_tracking_data.csv'))
        # Classify the tracks
        # coarsen to 5 minutes
        coarsen = int(5*60/self.tres)
        # smooth to an hour
        sm = int(60*60/(self.tres*coarsen))
        #set minimum length
        min_length= int(2*sm)+1
        df = tracking.classify_tracks(df, tres=self.tres, coarsen=coarsen, sm=sm, pixelperum=self.pixelperum, min_velocity=self.min_velocity, min_o=self.min_o, min_length=min_length)
        # Save the df
        dfpath = os.path.join(self.path_out, f'XY{fov}/clean_tracking_data.csv')
        df.to_csv(dfpath)

        return

    def get_existing_parameters(self):
        """
        Function to get the existing parameters from disk

        Returns
        -------
        parameters : dict
            The existing parameters.
        """
        # Code for getting existing parameters from disk

    def init_logger(self):
        """
        Function to initialise the logger
        """
        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        
        log_filename = f"pipelinelog_{formatted_datetime}.log"
        log_file_path = os.path.join(self.path_out, log_filename)

        # Create a logger instance
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # Create a formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # Create a file handler and set the formatter
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)

        # Add the file handler to the logger
        logger.addHandler(file_handler)

        logger.info('Started pipeline')
        print('Started pipeline')
        print('Logging to {}'.format(log_file_path))
        self.logger=logger
        return
    
    def creat_directories_for_output(self):
        """
        Function to create the directories for the output
        """
        # Code for creating directories for output
        for fov in self.fovs:
            if not os.path.isdir(os.path.join(self.path_out, f'XY{fov}')):
                os.mkdir(os.path.join(self.path_out, f'XY{fov}'))
            if not os.path.isdir(os.path.join(self.path_out, f'XY{fov}/lanes')):    
                os.mkdir(os.path.join(self.path_out, f'XY{fov}/lanes'))
        return

    def convert_pixels_to_um(self, fov):
        """
        Convert clean tracking data and convert the nucleus, rear v_nuc, v_rear, v_front, length, area, x, y into um.
        """
        pathtodf = os.path.join(self.path_out, f'XY{fov}/clean_tracking_data.csv')
        df = pd.read_csv(pathtodf)
        df.nucleus = df.nucleus/self.pixelperum
        df.rear = df.rear/self.pixelperum
        df.front = df.front/self.pixelperum
        df.v_nuc = df.v_nuc*self.pixelperum/self.tres
        df.v_rear = df.v_rear*self.pixelperum/self.tres
        df.v_front = df.v_front*self.pixelperum/self.tres
        df.length = df.length*self.pixelperum
        df.area = df.area*self.pixelperum**2
        df.x = df.x*self.pixelperum
        df.y = df.y*self.pixelperum
        df.to_csv(pathtodf)
        return
    
    def check_for_mergin_data(self, fov):
        """
        Function to check if the data to merge is present already
        """
        assert os.path.isfile(os.path.join(self.path_out, f'XY{fov}/cyto_masks.npz')) or os.path.isfile(os.path.join(self.path_out, f'XY{fov}/cyto_masks.mp4')), f'Could not find cyto masks for fov {fov}'
        assert os.path.isfile(os.path.join(self.path_out, f'XY{fov}/lanes/lanes_mask.tif')), f'Could not find lanes mask for fov {fov}'
        assert os.path.isfile(os.path.join(self.path_out, f'XY{fov}/lanes/lanes_metric.tif')), f'Could not find lanes metric for fov {fov}'
        assert os.path.isfile(os.path.join(self.path_out, f'XY{fov}/tracking_data.csv')), f'Could not find tracking data for fov {fov}'

        return

    def run_pipeline(self, run_segmentation=True, run_tracking=True, run_lane_detection=True, merge_trajectories=True, classify_trajectories=False, verbose=True, convert_to_um=True):
        """
        Function to run the pipeline on a full experiment containing many fields of view
        """
        ## Initialise the logger
        self.init_logger()
        
        # Check if the files are present in the data_path and if they are of the right type
        self.check_files(self.data_path, image_file=self.image_file, lanes_file=self.lanes_file, path_out=self.path_out)

        # Create the directories for the output
        self.creat_directories_for_output()

        ## Now detect all the lanes
        if run_lane_detection:
            self.detect_lanes(self.fovs)
        
        ## Now run the rest of the pipeline by subsequent fovs
        for fov in self.fovs:
            print('Processing fov {}'.format(fov))
            if run_segmentation:
                print('Segmenting cells...')
                self.segment(fov=fov)
            if run_tracking:
                print('Tracking cells with trackpy...')
                self.track(fov=fov)
            
            ## Now get the full dataframe by merging the tracking data and the masks and lanes
            if merge_trajectories:
                # make sure the masks and lanes and tracking_data is present
                self.check_for_mergin_data(fov=fov)

                print('Merging tracking data...')
                self.logger.info('Merging tracking data...')
                self.merge_tracking_data(fov=fov, verbose=verbose)
                
            ## Now some filtering and postprocessing
            print('Filtering and postprocessing...')
            self.logger.info('Filtering and postprocessing...')
            self.filter_tracking_data(fov=fov)
            
            ## Now classify the tracks
            if classify_trajectories:
                ##This is still buggy though
                try:
                    print('Classifying tracks...')
                    self.logger.info('Classifying tracks...')
                    self.classify_tracks(fov=fov)
                except Exception as e:
                    print(f'WARNING! Could not classify tracks for fov {fov} ')
                    self.logger.warning(f'Could not classify tracks for fov {fov} \n {e}')

            ## Now convert to um
            if convert_to_um:
                print('Converting to um...')
                self.logger.info('Converting to um...')
                self.convert_pixels_to_um(fov=fov)

            print(f'Done with fov {fov}!')
            self.logger.info(f'Done! with fov {fov}!')
        print('Done with all fovs!')
        self.logger.info('Done with all fovs!')
        return