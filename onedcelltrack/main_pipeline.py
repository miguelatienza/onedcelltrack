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
        return self.param_dict()

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
        ### Handle the cell images
        ## 1. case: nd2 file with all fovs, time_points and channels and an extra nd2 file for the lanes containing each fov
        cell_images_checked=False
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

    def param_dict(self):
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
            json.dump(self.param_dict(), f)

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

    def segment(self):
        """
        Function to segment the cells in the image file
        """
        # Code for cell segmentation

    def track(self):
        """
        Function to track the cells over time
        """

        # Code for cell tracking

    def detect_lanes(self):
        """
        Function to detect the lanes in the image file
        """
        # Code for lane detection

    def save_results(self):
        """
        Function to save the results to disk
        """
        # Code for saving results

    def load_image(self):
        """
        Function to load the image file
        """
        # Code for loading image

    def load_lanes(self):
        """
        Function to load the lanes file
        """
        # Code for loading lanes

    def get_frame_indices(self):
        """
        Function to get the frame indices to be processed
        """
        # Code for getting frame indices

    def get_fovs(self):
        """
        Function to get the field of views to be processed
        """
        # Code for getting field of views

    def save_to_sql(self):
        """
        Function to save the intermediate results to a SQL database
        """
        # Code for saving to SQL database

    def get_image_shape(self):
        """
        Function to get the shape of the image file
        """
        # Code for getting image shape

    def get_lanes_shape(self):
        """
        Function to get the shape of the lanes file
        """
        # Code for getting lanes shape

    def get_image(self, frame_index, fov):
        """
        Function to get a single image from the image file

        Parameters
        ----------
        frame_index : int
            Index of the frame to be extracted.
        fov : int
            Index of the field of view to be extracted.

        Returns
        -------
        image : ndarray
            The extracted image.
        """
        # Code for getting a single image

    def get_lanes(self, fov):
        """
        Function to get the lanes for a single field of view

        Parameters
        ----------
        fov : int
            Index of the field of view to be extracted.

        Returns
        -------
        lanes : ndarray
            The extracted lanes.
        """
        # Code for getting lanes for a single field of view

    def get_cellpose_model(self):
        """
        Function to get the Cellpose model

        Returns
        -------
        model : tuple
            The Cellpose model.
        """
        # Code for getting the Cellpose model

    def get_existing_parameters(self):
        """
        Function to get the existing parameters from disk

        Returns
        -------
        parameters : dict
            The existing parameters.
        """
        # Code for getting existing parameters from disk

    def save_parameters(self):
        """
        Function to save the parameters to disk

        Parameters
        ----------
        parameters : dict
            The parameters to be saved.
        """
        # Code for saving parameters to disk
        with open(os.path.join(self.path_out, 'parameters.json'), 'w') as f:
            json.dump(self.param_dict(), f)

    def run_pipeline(self):
        """
        Function to run the pipeline on a full experiment containing many fields of view
        """
        # Check if the files are present in the data_path and if they are of the right type
        self.check_files()

        # Run the pipeline
        self.segment()
        self.track()
        self.detect_lanes()
        self.save_results()




def read_image(file_name, fov, frames=None, channel=None):
    """
    Function to read an image file

    Parameters
    ----------
    file_name : str
        Name of the image file.
    fov : int
        Index of the field of view to be extracted.
    frames : slice or list of int, optional
        Frames to be extracted. If None, all frames are extracted.
    channel : int, optional
        Channel to be extracted. If None, all channels are extracted.

    Returns
    -------
    image : ndarray
        The extracted image.
    """
    if file_name.endswith('.nd2'):
        with ND2Reader(file_name) as images:
            images.bundle_axes = 'cyx'
            images.iter_axes = 't'
            if frames is None:
                frames = slice(None)
            if channel is None:
                channel = slice(None)
            return images[frames, channel, fov]

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

def save_to_sql(df, table_name, conn):
    """
    Function to save a DataFrame to a SQL database

    Parameters
    ----------
    df : pandas DataFrame
        The DataFrame to be saved.
    table_name : str
        Name of the table to be created.
    conn : sqlite3.Connection
        Connection to the SQL database.
    """
    df.to_sql(table_name, conn, if_exists='replace', index=False)

def load_from_sql(table_name, conn):
    """
    Function to load a DataFrame from a SQL database

    Parameters
    ----------
    table_name : str
        Name of the table to be loaded.
    conn : sqlite3.Connection
        Connection to the SQL database.

    Returns
    -------
    df : pandas DataFrame
        The loaded DataFrame.
    """
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    return df

def get_image_shape(file_name):
    """
    Function to get the shape of an image file

    Parameters
    ----------
    file_name : str
        Name of the image file.

    Returns
    -------
    shape : tuple
        The shape of the image file.
    """
    if file_name.endswith('.nd2'):
        with ND2Reader(file_name) as images:
            images.bundle_axes = 'cyx'
            return images.sizes['y'], images.sizes['x'], images.sizes['t'], images.sizes['c']

    if file_name.endswith('.mp4'):
        video = io.vreader(file_name)
        frame = next(video)
        return frame.shape

    if file_name.endswith('.tif') or file_name.endswith('.tiff'):
        with TiffFile(file_name) as tif:
            return tif.pages[0].shape

def get_lanes_shape(file_name):
    """
    Function to get the shape of a lanes file

    Parameters
    ----------
    file_name : str
        Name of the lanes file.

    Returns
    -------
    shape : tuple
        The shape of the lanes file.
    """
    if file_name.endswith('.tif') or file_name.endswith('.tiff'):
        with TiffFile(file_name) as tif:
            return tif.pages[0].shape

def get_image(file_name, frame_index, fov):
    """
    Function to get a single image from an image file

    Parameters
    ----------
    file_name : str
        Name of the image file.
    frame_index : int
        Index of the frame to be extracted.
    fov : int
        Index of the field of view to be extracted.

    Returns
    -------
    image : ndarray
        The extracted image.
    """
    if file_name.endswith('.nd2'):
        with ND2Reader(file_name) as images:
            images.bundle_axes = 'cyx'
            images.iter_axes = 't'
            return images[frame_index, :, :, fov]

    if file_name.endswith('.mp4'):
        video = io.vreader(file_name)
        for i, frame in enumerate(video):
            if i == frame_index:
                return frame

    if file_name.endswith('.tif') or file_name.endswith('.tiff'):
        with TiffFile(file_name) as tif:
            return tif.pages[frame_index].asarray()[fov]

def get_lanes(file_name, fov):
    """
    Function to get the lanes for a single field of view

    Parameters
    ----------
    file_name : str
        Name of the lanes file.
    fov : int
        Index of the field of view to be extracted.

    Returns
    -------
    lanes : ndarray
        The extracted lanes.
    """
    if file_name.endswith('.tif') or file_name.endswith('.tiff'):
        with TiffFile(file_name) as tif:
            return tif.pages[fov].asarray()

def get_cellpose_model(pretrained_model='bf'):
    """
    Function to get the Cellpose model

    Parameters
    ----------
    pretrained_model : str, optional
        Name of the pretrained model to be used. Default is 'bf'.

    Returns
    -------
    model : tuple
        The Cellpose model.
    """
    if pretrained_model == 'bf':
        model = models.Cellpose(gpu=False, model_type='bf')
    elif pretrained_model == 'nuclei':
        model = models.Cellpose(gpu=False, model_type='nuclei')
    else:
        raise ValueError(f"Invalid pretrained model: {pretrained_model}")
    return model

def get_existing_parameters(path_out):
    """
    Function to get the existing parameters from disk

    Parameters
    ----------
    path_out : str
        Path to the output directory.

    Returns
    -------
    parameters : dict
        The existing parameters.
    """
    with open(os.path.join(path_out, 'parameters.json'), 'r') as f:
        parameters = json.load(f)
    return parameters

def save_parameters(parameters, path_out):
    """
    Function to save the parameters to disk

    Parameters
    ----------
    parameters : dict
        The parameters to be saved.
    path_out : str
        Path to the output directory.
    """
    with open(os.path.join(path_out, 'parameters.json'), 'w') as f:
        json.dump(parameters, f)

def get_frame_indices(image_file, min_frames=10):
    """
    Function to get the frame indices to be processed

    Parameters
    ----------
    image_file : str
        Name of the image file.
    min_frames : int, optional
        Minimum number of frames to be processed. Default is 10.

    Returns
    -------
    frame_indices : list of int
        The frame indices to be processed.
    """
    if image_file.endswith('.nd2'):
        with ND2Reader(image_file) as images:
            n_frames = images.sizes['t']
    elif image_file.endswith('.mp4'):
        video = io.vreader(image_file)
        n_frames = len(list(video))
    elif image_file.endswith('.tif') or image_file.endswith('.tiff'):
        with TiffFile(image_file) as tif:
            n_frames = len(tif.pages)
    else:
        raise ValueError(f"Invalid image file: {image_file}")

    if n_frames < min_frames:
        raise ValueError(f"Number of frames ({n_frames}) is less than minimum ({min_frames})")

    frame_indices = list(range(0, n_frames, 1))

    return frame_indices

def get_fovs(lanes_file):
    """
    Function to get the field of views to be processed

    Parameters
    ----------
    lanes_file : str
        Name of the lanes file.

    Returns
    -------
    fovs : list of int
        The field of views to be processed.
    """
    if lanes_file.endswith('.tif') or lanes_file.endswith('.tiff'):
        with TiffFile(lanes_file) as tif:
            n_fovs = tif.pages.shape[0]
    else:
        raise ValueError(f"Invalid lanes file: {lanes_file}")
    
    