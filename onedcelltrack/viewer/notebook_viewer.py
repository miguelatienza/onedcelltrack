import ipywidgets as widgets
import matplotlib.pyplot as plt
#import mpl_interactions.ipyplot as iplt
from nd2reader import ND2Reader
import pandas as pd
from IPython.display import display
import sqlite3
import numpy as np
from skimage.segmentation import find_boundaries
import sys
from .. import functions
from .. import tracking
from tqdm import tqdm
from collections.abc import Iterable
import trackpy as tp
from skimage.segmentation import find_boundaries
import os
import json
from cellpose import models
from cellpose.io import logger_setup 
from skimage.morphology import binary_erosion
from skimage.io import imread
from ..classify import cp
import matplotlib.collections as collections
from .. import lane_detection
from ..main_pipeline import Pipeline
import matplotlib.cm as cm
import matplotlib.colors as mcolors

class Viewer(Pipeline):
    """
    This is a general class is to view the images and then chose the bf_channel, nuc_channel and specify the fovs of interest as well as the corresponding fovs corresponding to the lanes_file. Note that if you do not specify the fovs and the lane_fovs, they must have the same length. the viewer contains a button to delete the current fov. The viewer also contains a button to exchange bf_channel and nucleus_channel. The viewer also contains a button to save the fovs and lane_fovs as well as bf_channel and nuc_channel to the param_dict.

    Parameters
    ----------
    pipeline : Pipeline
        Instance of the pipeline class.
    """
    def __init__(self, pipeline, fovs=None, lane_fovs=None, frame_indices=None):
        self.__dict__.update(pipeline.__dict__)
        self.pipeline = pipeline
        if fovs is None:
            fovs = list(range(self.n_fovs))
        if lane_fovs is None:
            lane_fovs = list(range(self.n_fovs_lanes))
        if fovs is None and lane_fovs is None:
            assert len(fovs)==len(lane_fovs), "The length of the fovs was inferred automatically but they are not of the same length. Please specify the fovs and lane_fovs as lists of the same length."
        else:
            assert len(fovs)==len(lane_fovs), "The length of the fovs and lane_fovs must be the same."
        if frame_indices is None:
            self.frame_indices = self.infer_frame_indices()
        else:
            self.frame_indices = frame_indices
        self.n_frames = len(self.frame_indices)
        self.fovs = fovs
        self.lane_fovs = lane_fovs
        self.n_fovs = len(fovs)
        self.n_fovs_lanes = len(lane_fovs)
        ## Initalise the channels
        if self.nuc_channel is None:
            self.nuc_channel = 1
            self.bf_channel = 0
       
        self.channel=0

        ## Initialise the buttons

        #Button to delete a fov
        self.button_delete_fov = widgets.Button(
        description='Delete fov',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click me',
        icon='') # (FontAwesome names without the `fa-` prefix)

        #Button to save everything
        self.button_save = widgets.Button(
        description='Save',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click me',
        icon='') # (FontAwesome names without the `fa-` prefix)
        
        #Dropdown menu to selct bf_channel or nuc_channel
        self.select_channel = widgets.Dropdown(
        options=['Brightfield', 'Nucleus'],
        value='Brightfield',
        description='Channel:',
        disabled=False,
        )

        # Button to exchange the channels
        self.button_exchange_channels = widgets.Button(
        description='Exchange channels',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click me',
        icon='') # (FontAwesome names without the `fa-` prefix)
        
        # Button to cut the frames to the left
        self.button_cut_left = widgets.Button(
        description='Set first time frame',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click me',
        icon='') # (FontAwesome names without the `fa-` prefix)

        # Button to cut the frames to the right
        self.button_cut_right = widgets.Button(
        description='Set last time frame',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click me',
        icon='') # (FontAwesome names without the `fa-` prefix)
        
        ## Add an input box to specify the tres
        self.select_tres = widgets.IntText(
        value=None,
        description='Time resolution:',
        disabled=False,
        )

        ## Add an input box to specify the pixelperum ratio
        self.select_pixelperum = widgets.FloatText(
        value=None,
        description='Pixel per um:',
        disabled=False,
        )

        ## Add slider for the lanes alpha
        self.lanes_alpha = widgets.FloatSlider(
        value=0.1,
        min=0,
        max=1,
        step=0.05,
        description='Lanes alpha:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',
        )

        ## Add a button to show whether the current frame and fov is deleted or not
        self.position_status = widgets.Button(
        description='Not deleted',
        disabled=False,
        button_style='success', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='',
        icon='') # (FontAwesome names without the `fa-` prefix)
        
        #Update functions
        self.button_delete_fov.on_click(self.delete_fov)
        self.button_save.on_click(self.save)
        self.select_channel.observe(self.update_channel, names='value')
        self.button_exchange_channels.on_click(self.exchange_channels)
        self.button_cut_left.on_click(self.cut_left)
        self.button_cut_right.on_click(self.cut_right)
        self.select_tres.observe(self.update_tres, names='value')
        self.lanes_alpha.observe(self.update_lanes_alpha, names='value')
        self.select_pixelperum.observe(self.update_pixelperum, names='value')

        ## Now create the sliders
        
        #slider for the contrast adjustment
        self.clip = widgets.IntRangeSlider(min=0,max=int(2**16 -1), step=1, value=[0,1000], description="clip", continuous_update=True, width='200px')
        #slider for the fovs and make them as wide as te image
        self.fov = widgets.IntSlider(min=0,max=self.n_fovs-1, step=1, description="fov", continuous_update=False)
        self.frame = widgets.IntSlider(min=0,max=self.n_frames-1, step=1, description="frame", continuous_update=False)
        
        ##Initialize the figure
        plt.ioff()
        self.fig = plt.figure()
        # remove the figure title and header
        self.fig.canvas.header_visible = False
        self.fig.canvas.footer_visible = False
        self.fig.canvas.toolbar_visible = True

        self.fig.tight_layout()
        self.ax = self.fig.add_subplot(111)
        #remove x and y ticks
        self.ax.set_xticks([])
        self.ax.set_yticks([])  
        cell_image = self.get_image_cell()
        self.height, self.width = cell_image.shape[:2]
        lanes_image = self.get_image_lanes()
       
        reds = cm.get_cmap('Reds', 5000)
        reds = mcolors.ListedColormap(np.linspace([0,0,0], [1,0,0], 5000))
        self.im_lanes = self.ax.imshow(lanes_image, cmap=reds, vmin=0, vmax=5000, alpha=0.1)
        self.im_cell = self.ax.imshow(cell_image, cmap='gray', vmin=0, vmax=5000, alpha=0.5)
        
        
        #Adjust the slider sizes to the image
        self.fov.layout.width = str(self.fig.get_size_inches()[0]*self.fig.dpi)
        self.clip.layout.width = str(self.fig.get_size_inches()[0]*self.fig.dpi)
        self.frame.layout.width = str(self.fig.get_size_inches()[0]*self.fig.dpi)
        
    def show(self):
        """
        Arranges the widgets and displays them.
        """
        out = widgets.interactive_output(self.update, {'fov': self.fov, 'clip': self.clip, 'frame': self.frame})
        # box for the sliders with same width as the image
        slider_box = widgets.VBox([self.fov, self.clip, self.frame])#,  layout=widgets.Layout(width=str(self.fig.get_size_inches()[0]*self.fig.dpi)))
        # box for the buttons
        buttons = widgets.VBox([self.button_delete_fov, self.button_exchange_channels, self.button_save, self.select_channel, self.button_cut_left, self.button_cut_right, self.select_tres, self.lanes_alpha, self.position_status, self.select_pixelperum])
        
        # box for the sliders and figure
        right_box = widgets.VBox([self.fig.canvas, slider_box])
        # box for the buttons and the sliders and figure
        self.window = widgets.HBox([buttons, right_box])
        self.update(self.fov.value, self.clip.value, self.frame.value)
        display(self.window)

    def get_image_cell(self):

        if self.channel==self.bf_channel:
            image = self.read_bf(frames=self.frame.value, fov=self.fov.value)
        else:
            image = self.read_nuc(frames=self.frame.value, fov=self.fov.value)
        return image
    
    def get_image_lanes(self):
        lanes_image = self.read_lanes(fov=self.fov.value)
        return lanes_image

    def update(self, fov, clip, frame):
        if fov not in self.fovs or not frame in self.frame_indices:
            self.position_status.button_style='danger'
            self.position_status.description='Deleted'
            self.fig.canvas.draw()
            return
        else:
            self.position_status.button_style='success'
            self.position_status.description='Not deleted'

        vmin, vmax = self.clip.value
        cell_image = self.get_image_cell()
        lanes_image = self.get_image_lanes()
   
        if self.channel==self.bf_channel:
            self.im_cell.set_cmap('gray')
        else:
            self.im_cell.set_cmap('Greens')
        self.im_cell.set_data(cell_image)
        self.im_cell.set_clim([vmin, vmax])
        self.im_lanes.set_data(lanes_image)
        self.im_lanes.set_clim([0, 900])

        self.fig.canvas.draw()
    
    def update_channel(self, a):
        """
        Updates the channel value.
        """
        
        if self.select_channel.value=='Brightfield':
            self.channel = self.bf_channel
            self.clip.value=[200, 50_000]
        else:
            self.channel =self.nuc_channel
            self.clip.value=[0, 5_000]
        self.update(self.fov.value, self.clip.value, self.frame.value)

    def delete_fov(self, a):
        """
        Deletes the current fov from the fovs and lane_fovs lists.
        """
        if not self.fov.value in self.fovs:
            return
        self.fovs.remove(self.fov.value)
        self.lane_fovs.remove(self.fov.value)
        self.n_fovs-=1
        self.n_fovs_lanes-=1
        self.fov.max-=1
        self.fov.value+=1
        self.update(self.fov.value, self.clip.value, self.frame.value)
    
    def save(self, a):
        """
        Save the fovs to the param_dict.
        """
        self.update_pipeline()
        self.pipeline.save_parameters()

        if self.tres is None:
            ## Make the save button turn red
            self.button_save.button_style='danger'
            ## add a message
            print('Please specify the time resolution')
        elif self.pixelperum is None:
            ## Make the save button turn red
            self.button_save.button_style='danger'
            ## add a message
            print('Please specify the pixelperum ratio')
        else: #make the button green
            self.button_save.button_style='success'
            return

    def exchange_channels(self, a):
        """
        Exchanges the bf_channel and nuc_channel.
        """
        self.bf_channel, self.nuc_channel = self.nuc_channel, self.bf_channel
        self.update_channel(None)      

        self.update(self.fov.value, self.clip.value, self.frame.value)

    def cut_left(self, a):
        """
        Cuts the frames to the left.
        """
        self.frame_indices = np.arange(self.frame.value, self.frame_indices[-1]+1)
        self.n_frames = len(self.frame_indices)
        #self.frame.max = self.n_frames-1
        #self.frame.value=0
        self.update(self.fov.value, self.clip.value, self.frame.value)

    def cut_right(self, a):
        """
        Cuts the frames to the right.
        """
        self.frame_indices = np.arange(self.frame_indices[0], self.frame.value+1)
        self.n_frames = len(self.frame_indices)
        # self.frame.max = self.n_frames-1
        # self.frame.value=0
        self.update(self.fov.value, self.clip.value, self.frame.value)

    def update_tres(self, a):
        """
        Updates the time resolution.
        """
        if self.select_tres.value is None:
            print('Please specify the time resolution')
            return
        self.tres = self.select_tres.value

    def update_lanes_alpha(self, a):
        """
        Updates the lanes alpha.
        """
        self.im_lanes.set_alpha(self.lanes_alpha.value)
        self.im_cell.set_alpha(1-self.lanes_alpha.value)
        self.update(self.fov.value, self.clip.value, self.frame.value)
    
    def update_pixelperum(self, a):
        """
        Updates the pixelperum ratio.
        """
        if self.select_pixelperum.value is None:
            print('Please specify the pixelperum ratio')
            return
        self.pixelperum = self.select_pixelperum.value

    def update_pipeline(self):
        """
        Updates the pipeline.
        """
        for key in self.param_keys:
            if isinstance(self.__dict__[key], np.ndarray):
                self.__dict__[key] = self.__dict__[key].tolist()
        param_dict = {key: value for key, value in self.__dict__.items() if key in self.param_keys}
        #param_dict = self.pipeline.param_dict()
        self.pipeline.__dict__.update(param_dict)
        self.pipeline.save_parameters()

class LaneViewer(Viewer):
    """
    This class inherits from the general Viewer. It's objective is to view the lanes and detect the lanes for every field of view.
    Parameters
    ----------
    pipeline : Pipeline
        Instance of the pipeline class.
    """

    def __init__(self, pipeline):
        self.__dict__.update(pipeline.__dict__)
    
        ## Initialise the Viewer class
        super().__init__(pipeline)
        
        ## Add new necessary sliders
        self.ld = widgets.IntSlider(min=10,max=60, step=1, description="lane distance", value=self.lane_distance, continuous_update=True)
        self.threshold = widgets.FloatSlider(min=0,max=1, step=0.05, description="threshold", continuous_update=False, value=self.lane_threshold)
        #self.kernel_width= widgets.IntSlider(min=3,max=15, step=2, description="kernel width", continuous_update=False, value=self.kernel_width)
        self.kernel_width=3
        ## override the clip slider for the lanes
        self.clip = widgets.IntRangeSlider(min=0,max=int(20_000), step=1, value=[0,5000], description="clip", continuous_update=True, width='200px')
        
        self.recompute_button = widgets.Button(
        description='Recompute',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click me',
        icon='') # (FontAwesome names without the `fa-` prefix)
        self.recompute_button.on_click(self.recompute)
        
        self.min_coordinates = list(self.fovs.copy())
        self.max_coordinates = list(self.fovs.copy())

        self.ax.plot([], [], color='red')
        
    def get_lanes(self):
        """
        Returns the lanes for all fovs as a list.
        """
        lanes = [self.read_lanes(fov=fov) for fov in self.fovs]
        return lanes
    
    def update(self, fov, clip):

        vmin, vmax = clip
        image = self.lanes[fov]
        self.im_cell.set_alpha(0)
        self.im_lanes.set_alpha(1)
        self.im_lanes.set_cmap('Reds')
        self.im_lanes.set_data(image)
        self.im_lanes.set_clim([vmin, vmax])
        self.im_lanes.set_cmap('gray')
        
        if isinstance(self.min_coordinates[fov], Iterable):
           
            
            [self.ax.axes.lines[0].remove() for j in range(len(self.ax.axes.lines))]
            x = [0, image.shape[1]-1]
            for i in range(self.min_coordinates[fov].shape[0]):
                self.ax.plot(x, [self.min_coordinates[fov][i,1], sum(self.min_coordinates[fov][i])], color='red')
            
            for i in range(self.max_coordinates[fov].shape[0]):
                self.ax.plot(x, [self.max_coordinates[fov][i,1], sum(self.max_coordinates[fov][i])], color='red')
        
        self.fig.canvas.draw()

    def recompute(self, a):
        """
        Recomputes the lanes for the current fov.
        """
        #set the button to working
        self.recompute_button.button_style='warning'
        try:
            vmin, vmax = self.clip.value
            lanes_clipped = np.clip(self.lanes[self.fov.value], vmin, vmax, dtype=self.lanes[self.fov.value].dtype)

            print('recomputing')
            self.min_coordinates[self.fov.value], self.max_coordinates[self.fov.value] = lane_detection.get_lane_mask(lanes_clipped, kernel_width=self.kernel_width, line_distance=self.ld.value, debug=True, gpu=True, threshold=self.threshold.value)
            print('updating')
            self.update(self.fov.value, self.clip.value)
            self.recompute_button.button_style='success'
        except Exception as e:
            print(e)
            self.recompute_button.button_style='danger'

    def show(self):
        """
        Arranges the widgets and displays them.
        """
        lane_detection_sliders = widgets.VBox([self.ld, self.threshold, self.recompute_button, self.button_save])
        self.lanes = self.get_lanes()
        out = widgets.interactive_output(self.update, {'fov': self.fov, 'clip': self.clip})
        # image box
        image_box= widgets.VBox([self.fig.canvas, self.fov, self.clip])#,  layout=widgets.Layout(width=str(self.fig.get_size_inches()[0]*self.fig.dpi)))
        ## Override the window
        self.window = widgets.HBox([widgets.HBox([lane_detection_sliders, image_box])])
        self.recompute(self.fov.value)
        self.update(self.fov.value, self.clip.value)
        display(self.window)

class TrackingViewer(Viewer):
    """
    This class inherits from the general Viewer. It's objective is to preview and calibrate the parameters related to tracking with trackpy.
    Parameters
    ----------
    pipeline : Pipeline
        Instance of the pipeline class.
    """
    def __init__(self, pipeline):
        self.__dict__.update(pipeline.__dict__)
        super().__init__(pipeline)

        self.channel=self.nuc_channel

        ## Initialise the link_dfs
        self.link_dfs = {}
        
        # 
        self.button_save.on_click(self.save)
        ## Create the buttons
        self.button_track = widgets.Button(
        description='Track',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click me',
        icon='')
        self.button_track.on_click(self.track)

        self.min_mass_slider = widgets.FloatSlider(min=1e4, max=1e6, step=0.01e5, description="min_mass", value=self.min_mass, continuous_update=True)
       
        self.diameter_slider = widgets.IntSlider(min=9,max=35, step=2, description="diameter", value=self.diameter, continuous_update=True)
        
        self.min_frames_slider = widgets.FloatSlider(min=0,max=50, step=1, value=self.min_frames, description="min_frames", continuous_update=False)
        
        self.max_travel_slider = widgets.IntSlider(min=3,max=50, step=1, value=self.max_travel, description="max_travel", continuous_update=False)
        
        self.track_memory_slider = widgets.IntSlider(min=0,max=20, step=1, value=self.track_memory, description="track memory", continuous_update=False)
        
        self.track_button = widgets.Button(
        description='Track',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click me',
        icon='')

        self.track_button.on_click(self.link_update)

        self.updated=False
        # ## Add scatter plots for the nuclei
        self.bscat = self.ax.scatter([0,0], [0,0], s=0.2*plt.rcParams['lines.markersize'] ** 2, color='blue', alpha=0.5)
        self.im_cell.set_alpha(1)
        self.im_lanes.set_alpha(0)
        self.im_cell.set_cmap('gray')
        self.lscat = self.ax.scatter([0,0], [0,0], s=0.2*plt.rcParams['lines.markersize'] ** 2, color='red', alpha=0.5)



    def show(self):
        """ 
        Arranges the widgets and displays them.
        """
        #self.update(self.fov.value, self.clip.value, self.frame.value)
        out = widgets.interactive_output(self.update, {'fov': self.fov, 'clip': self.clip, 'frame': self.frame, 'min_mass': self.min_mass_slider, 'diameter': self.diameter_slider})

        # box for sliders for tracking
        tracking_sliders = widgets.VBox([self.min_mass_slider, self.diameter_slider, self.min_frames_slider, self.max_travel_slider, self.track_memory_slider, self.track_button, self.button_save, self.position_status])

        # box for the image
        image_box = widgets.VBox([self.fig.canvas, self.fov, self.clip, self.frame], layout=widgets.Layout(width=str(self.fig.get_size_inches()[0]*self.fig.dpi)))
        
        # box for the sliders and figure
        self.window = widgets.HBox([tracking_sliders, image_box])
       
        #self.update(self.fov.value, self.clip.value, self.frame.value)
        display(self.window)
        self.update(self.fov.value, self.clip.value, self.frame.value, self.min_mass_slider.value, self.diameter_slider.value)

    def link_update(self, a):
        """
        Tracks the whole field of view.
        """
        #set the track button to working
        self.track_button.button_style='warning'
        self.track_button.description='Tracking...'
        # Get the image
        nuclei = self.read_nuc(frames=self.frame_indices, fov=self.fov.value)
        
        self.link_dfs[self.fov.value] = tracking.track_nuclei(nuclei, minmass=self.min_mass_slider.value, diameter=self.diameter_slider.value, min_frames=self.min_frames_slider.value, max_travel=self.max_travel_slider.value, track_memory=self.track_memory_slider.value, verbose=False)
        #update the button again
        self.track_button.button_style='success'
        self.track_button.description='Track'
        self.update(self.fov.value, self.clip.value, self.frame.value, self.min_mass_slider.value, self.diameter_slider.value)

    def batch_update(self, fov, t, min_mass, diameter):
        #set the track button to warning
        self.track_button.button_style='warning'
        image = self.read_nuc(frames=t, fov=fov)
        self.batch_df = tracking.locate_nuclei(raw_image=image, minmass=min_mass, diameter=diameter)
        self.track_button.button_style='success'
        return

    def update(self, fov, clip, frame, min_mass, diameter):
        
        vmin, vmax = clip
        image = self.read_nuc(frames=frame, fov=fov)
        self.im_cell.set_data(image)
        self.im_cell.set_clim([vmin, vmax])

        if not self.updated:
            self.fig.canvas.draw()
            self.updated=True
            return
        
        ## And then plot the tracks
        if (not fov in self.fovs) or (not frame in self.frame_indices):
            self.position_status.button_style='danger'
            self.position_status.description='Deleted position'
            return

        if fov in self.link_dfs.keys():
            df = self.link_dfs[fov]
            df = df[df['frame']==frame]
            self.lscat.set_offsets(df[['x', 'y']].values)
            self.fig.canvas.draw()

        self.batch_update(fov, [frame], min_mass, diameter)
        # self.lscat.set_offsets(np.c_[[], []])
        self.bscat.set_offsets(self.batch_df[['x', 'y']].values)
        self.fig.canvas.draw()
        
    def save(self, a):
       
        ## Make the save button turn orange
        self.button_save.button_style='warning'
        ## add a message
        self.button_save.description='Saving...'
        ## Save the parameters
        self.min_mass = self.min_mass_slider.value
        self.diameter = self.diameter_slider.value
        self.track_memory = self.track_memory_slider.value
        self.max_travel = self.max_travel_slider.value
        self.min_frames = self.min_frames_slider.value

        self.update_pipeline()
        ## Make the save button turn green
        self.button_save.button_style='success'
        self.button_save.description='Saved'

class CellposeViewer(Viewer):
    """
    This class inherits from the general Viewer. It's objective is to preview and calibrate the parameters related to cellpose segmentation."""
    def __init__(self, pipeline):
        self.__dict__.update(pipeline.__dict__)
        super().__init__(pipeline)

        self.channel=self.bf_channel

        ## Initialise the sliders

        self.flow_threshold_slider = widgets.FloatSlider(min=0, max=1.5, step=0.05, description="flow_threshold", value=self.flow_threshold, continuous_update=False)

        self.diameter_slider = widgets.IntSlider(min=9,max=100, step=2, description="diameter", value=self.cyto_diameter, continuous_update=False)

        self.cellprob_threshold_slider = widgets.FloatSlider(min=-10,max=10, step=0.05, value=self.cellprob_threshold, description="cellprob_threshold", continuous_update=False)

        # disable continuous update for the fov and frame sliders and clip
        self.fov.continuous_update=False
        self.frame.continuous_update=False
        self.clip.continuous_update=False

        self.clip.value=[200, 50_000]

    def show(self):
        """
        Arranges the widgets and displays them.
        """
        #Initialise cellpose
        from..segmentation import Segmentation
        self.cellpose = Segmentation(self.pretrained_model, gpu=self.use_gpu)

        out = widgets.interactive_output(self.update, {'fov': self.fov, 'clip': self.clip, 'frame': self.frame, 'flow_threshold': self.flow_threshold_slider, 'diameter': self.diameter_slider, 'cellprob_threshold': self.cellprob_threshold_slider})

        # box for sliders
        cellpose_sliders = widgets.VBox([self.flow_threshold_slider, self.diameter_slider, self.cellprob_threshold_slider, self.button_save, self.position_status])
        self.fig.canvas.draw()
        # box for the image
        image_box = widgets.VBox([self.fig.canvas, self.fov, self.clip, self.frame], layout=widgets.Layout(width=str(self.fig.get_size_inches()[0]*self.fig.dpi)))

        # box for the sliders and figure
        self.window = widgets.HBox([cellpose_sliders, image_box])
        self.update(self.fov.value, self.clip.value, self.frame.value, self.flow_threshold_slider.value, self.diameter_slider.value, self.cellprob_threshold_slider.value)
    
        display(self.window)
    
    def update(self, fov, clip, frame, flow_threshold, diameter, cellprob_threshold):

        vmin, vmax = clip
   
        image = self.read_bf(frames=frame, fov=fov)

        bf = image.copy()
    
        nucleus = self.read_nuc(frames=frame, fov=fov)
        
        # Run segmentation
        if (not fov in self.fovs) or (not frame in self.frame_indices):
            self.position_status.button_style='danger'
            self.position_status.description='Deleted position'
            return
        
        # segment
        self.mask = self.cellpose.segment(bf, nucleus, diameter=diameter, flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold)
        
        image = self.create_rgb_image(fov, frame)
        self.im_cell.set_alpha(1)
        self.im_lanes.set_alpha(0)
        #self.im_lanes.set_cmap('Reds')
        self.im_cell.set_data(image)
        self.im_cell.set_clim([0, 255])
        
        self.fig.canvas.draw()

    def create_rgb_image(self, fov, frame):
        """
        Creates an rgb image from the bf and nucleus channels.
        """
        bf = self.read_bf(frames=frame, fov=fov)
        #clip and convert to uint8
        vmin, vmax = self.clip.value
        bf = (255*(np.clip(bf, vmin, vmax)/vmax)).astype('uint8')
        
        red = find_boundaries(self.mask>0, mode='inner')

        image = np.stack((bf,bf, bf), axis=-1).astype('uint8')

        image[red, :] = [255, 0, 0]

        return image
    
    def save(self, a):
        self.cyto_diameter = self.diameter_slider.value
        self.flow_threshold = self.flow_threshold_slider.value
        self.cellprob_threshold = self.cellprob_threshold_slider.value
        super().save(a)
        
# class CellposeViewer:
    
#     def __init__(self, nd2file, bf_channel=None, nuc_channel=None,pretrained_model='mdamb231', omni=False):
        
#         self.link_dfs = {}
        
#         self.f = ND2Reader(nd2file)
#         self.nfov, self.nframes = self.f.sizes['v'], self.f.sizes['t']
        
#         channels = self.f.metadata['channels']
    
#         if bf_channel is None or nuc_channel is None:
#                     ###Infer the channels
#             if 'erry' in channels[0] or 'exas' in channels[0] and not 'phc' in channels[0]:
#                 self.nucleus_channel=0
#                 self.cyto_channel=1
#             elif 'erry' in channels[1] or 'exas' in channels[1] and not 'phc' in channels[1]:
#                 self.nucleus_channel=1
#                 self.cyto_channel=0
                
#             else:
#                 raise ValueError(f"""The channels could not be automatically detected! \n
#                 The following channels are available: {channels} . Please specify the indices of bf_channel and nuc_channel as keyword arguments. i.e: bf_channel=0, nuc_channel=1""")
#         else:
#             self.cyto_channel=bf_channel
#             self.nucleus_channel=nuc_channel

#         #Widgets
        
#         t_max = self.f.sizes['t']-1
#         self.t = widgets.IntSlider(min=0,max=t_max, step=1, description="t", continuous_update=False)

#         v_max = self.f.sizes['v']-1
#         self.v = widgets.IntSlider(min=0,max=v_max, step=1, description="v", continuous_update=False)
        
#         self.nclip = widgets.FloatRangeSlider(min=0,max=2**16, step=1, value=[0,1000], description="clip nuclei", continuous_update=False, width='200px')
        
#         self.cclip = widgets.FloatRangeSlider(min=0,max=2**16, step=1, value=[50,8000], description="clip cyto", continuous_update=False, width='200px')
        
#         self.flow_threshold = widgets.FloatSlider(min=0, max=1.5, step=0.05, description="flow_threshold", value=1.25, continuous_update=False)
        
#         self.diameter = widgets.IntSlider(min=0,max=1000, step=2, description="diameter", value=29, continuous_update=False)
        
#         self.mask_threshold = widgets.FloatSlider(min=-3,max=3, step=0.1, value=0, description="mask_threshold", continuous_update=False)
        
#         self.max_travel = widgets.IntSlider(min=3,max=50, step=1, value=5, description="max_travel", continuous_update=False)
        
#         self.track_memory = widgets.IntSlider(min=0,max=20, step=1, value=5, description="max_travel", continuous_update=False)

#         self.tp_method = widgets.Button(
#         description='Track',
#         disabled=False,
#         button_style='', # 'success', 'info', 'warning', 'danger' or ''
#         tooltip='Click me',
#         icon='')
        
        
#         vmin, vmax = self.cclip.value
#         cyto = (255*(np.clip(self.f.get_frame_2D(v=0,c=self.cyto_channel,t=0), vmin, vmax)/vmax)).astype('uint8')
#         #cyto = np.clip(self.f.get_frame_2D(v=0,c=self.cyto_channel,t=0), vmin, vmax)
#         #nucleus = self.f.get_frame_2D(v=0,c=self.nucleus_channel,t=0)
#         #nucleus = functions.preprocess(nucleus, log=True, bottom_percentile=0.05, top_percentile=99.95, return_type='uint8')
#         vmin, vmax = self.nclip.value
#         nucleus = (255*(np.clip(self.f.get_frame_2D(v=0,c=self.nucleus_channel,t=0), vmin, vmax)/vmax)).astype('uint8')
#         red = np.zeros_like(nucleus)
       
#         image = np.stack((red, red, nucleus), axis=-1).astype('float32')
#         image+=(cyto[:,:,np.newaxis]/3)
#         image = np.clip(image, 0,255).astype('uint8')
        
#         ##Initialize the figure
#         plt.ioff()
#         self.fig, self.ax = plt.subplots()
#         self.fig.tight_layout()
#         #self.fig.canvas.toolbar_visible = False
#         self.fig.canvas.header_visible = False
#         #self.fig.canvas.footer_visible = False
#         self.im = self.ax.imshow(image)

#         self.init_cellpose(pretrained_model=pretrained_model, omni=omni)
                
#         #Organize layout and display
#         out = widgets.interactive_output(self.update, {'t': self.t, 'v': self.v, 'cclip': self.cclip, 'nclip': self.nclip, 'flow_threshold': self.flow_threshold, 'diameter': self.diameter, 'mask_threshold': self.mask_threshold, 'max_travel': self.max_travel})
        
#         box = widgets.VBox([self.t, self.v, self.cclip, self.nclip, self.flow_threshold, self.diameter, self.mask_threshold, self.tp_method]) #, layout=widgets.Layout(width='400px'))
#         box1 = widgets.VBox([out, box])
#         grid = widgets.widgets.GridspecLayout(3, 3)
        
#         grid[:, :2] = self.fig.canvas
#         grid[1:,2] = box
#         grid[0, 2] = out
        
#         #display(self.fig.canvas)
#         display(grid)
#         plt.ion()
    
#     def init_cellpose(self, pretrained_model='mdamb231', omni=False, model='cyto', gpu=True):
 
#         if omni:
#             from cellpose_omni.models import CellposeModel
#             self.model = CellposeModel(
#             gpu=gpu, omni=True, nclasses=4, nchan=2, pretrained_model=pretrained_model)
#             return

        
#         elif pretrained_model is None:
#             self.model = models.Cellpose(gpu=gpu, model_type='cyto')

#         else:
#             path_to_models = os.path.join(os.path.dirname(__file__), '../models')
#             with open(os.path.join(path_to_models, 'models.json'), 'r') as f:
#                 dic = json.load(f)
#             if pretrained_model in dic.keys():
#                 path_to_model = os.path.join(path_to_models, dic[pretrained_model]['path'])
#                 if os.path.isfile(path_to_model):
#                     pretrained_model = path_to_model
#                 else: 
         
#                     url = dic[pretrained_model]['link']
#                     print('Downloading model from Nextcloud...')
#                     request.urlretrieve(url, os.path.join(path_to_models, path_to_model))
#                     pretrained_model = os.path.join(path_to_models,dic[pretrained_model]['path'])

            
#             if not omni:
#                 self.model = models.CellposeModel(gpu=gpu, pretrained_model=pretrained_model)

#     def update(self, t, v, cclip, nclip, flow_threshold, diameter, mask_threshold, max_travel):      
        
        
#         #contours = get_outlines(image)
#         self.segment(
#         t, v, cclip, nclip, flow_threshold, mask_threshold, diameter)
#         self.im.set_data(self.image)
#         return
#         #lanes = g.get_frame_2D(v=v)
#         #self.im.set_clim([vmin, vmax])
#         #self.fig.canvas.draw()
        

#         self.batch_update(v, t, min_mass, diameter)
#         self.show_tracking(self.batch_df, self.bscat)
#         try:
#             df = self.link_dfs[v][self.link_df.frame==t]
#             self.show_tracking(df, self.lscat)
#         except:
#             self.lscat.set_offsets([]) 
    
#     def show_tracking(self, df, scat):
        
#         t, v= self.t.value, self.v.value
        
#         #[plt.axes.lines[0].remove() for j in range(len(self.ax.axes.lines))]
#         data = np.hstack((df.x.values[:,np.newaxis], df.y.values[:, np.newaxis]))
#         scat.set_offsets(data)
    
#     def segment(self,t, v, cclip, nclip, flow_threshold, mask_threshold, diameter, normalize=True, verbose=False,):
        
#         #nucleus = self.f.get_frame_2D(v=v, t=t, c=self.nucleus_channel)
#         #vmin, vmax = nclip
#         #nucleus = functions.preprocess(nucleus, bottom_percentile=vmin, top_percentile=vmax, log=True, return_type='uint16')
#         #vmin, vmax = cclip
#         #cyto = self.f.get_frame_2D(v=v, t=t, c=self.cyto_channel)
#         #cyto = functions.preprocess(cyto, bottom_percentile=vmin, top_percentile=vmax, return_type='uint16')
       
#         nucleus = self.f.get_frame_2D(v=v, t=t, c=self.nucleus_channel)
#         #nucleus = functions.preprocess(nucleus, bottom_percentile=0, #top_percentile=100, log=True, return_type='uint16')
#         cyto = self.f.get_frame_2D(v=v, t=t, c=self.cyto_channel)
        
    
#         image = np.stack((cyto, nucleus), axis=-1)
#         print(image.shape)
#         print('hjflks1')
#         if diameter == 0:
#             diameter=None
#         mask = self.model.eval(
#             image, diameter=diameter, channels=[1, 0], flow_threshold=flow_threshold, cellprob_threshold=mask_threshold, normalize=normalize, progress=verbose)[0].astype('uint8')
        
#         bin_mask = np.zeros(mask.shape, dtype='bool')
#         cell_ids = np.unique(mask)
#         cell_ids = cell_ids[cell_ids!=0]
        
#         for cell_id in cell_ids:
#             bin_mask+=binary_erosion(mask==cell_id)
        
#         outlines = find_boundaries(bin_mask, mode='outer')
#         try:
#             print(f'{cell_ids.max()} Masks detected')
#         except ValueError:
#             print('No masks detected')
#         self.outlines = outlines
        
#         self.mask=mask
#         self.image = self.get_8bit(outlines, cyto)
        
#         return 
    
#         self.image = np.stack((outlines, cyto.astype('uint8'), nucleus.astype('uint8')), axis=-1)
#         self.outlines = outlines
        
#     def get_8bit(self, outlines, cyto, nuclei=None):
               
#         vmin, vmax = self.cclip.value
#         cyto = np.clip(cyto, vmin, vmax)
#         cyto = (255*(cyto-vmin)/(vmax-vmin)).astype('uint8')
        
#         image = np.stack((cyto, cyto, cyto), axis=-1)
        
#         image[(outlines>0)]=[255,0,0]
        
#         #outlines[:,:,np.newaxis]>0
        
#         return image
        

class ResultsViewer:
    
    def __init__(self, nd2file, outpath, base_path=None, experiment_paths=[], db_path = '/project/ag-moonraedler/MAtienza/database/onedcellmigration.db', path_to_patterns=None):
        
        self.link_dfs = {}
        self.base_path = base_path
        self.cyto_locator=None
        self.path_to_patterns=path_to_patterns
        self.nd2file=nd2file
        self.f = ND2Reader(nd2file)
        self.nfov, self.nframes = self.f.sizes['v'], self.f.sizes['t']
        
        self.outpath=outpath
        self.db_path=db_path
        
        t_max = self.f.sizes['t']-1
        self.t = widgets.IntSlider(min=0,max=t_max, step=1, description="t", continuous_update=True)

        c_max = self.f.sizes['c']-1
        self.c = widgets.IntSlider(min=0,max=c_max, step=1, description="c", value=1, continuous_update=True)

        v_max = self.f.sizes['v']-1
        self.v = widgets.IntSlider(min=0,max=v_max, step=1, description="v", continuous_update=False)
        
        self.clip = widgets.IntRangeSlider(min=0,max=int(2**16 -1), step=1, value=[0,12000], description="clip", continuous_update=True, width='200px')
 
        self.view_nuclei = widgets.Checkbox(
        value=True,
        description='Nuclei',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click me',
        icon='')
        
        self.view_cellpose = widgets.Checkbox(
        value=True,
        description='Contours',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click me',
        icon='')
        
       
        vmin, vmax = self.clip.value
        cyto = self.f.get_frame_2D(v=self.v.value,c=self.c.value,t=self.t.value)
        cyto = np.clip(cyto, vmin, vmax).astype('float32')
        cyto = (255*(cyto-vmin)/(vmax-vmin)).astype('uint8')

        image = np.stack((cyto, cyto, cyto), axis=-1)
        
        
        self.load_masks(outpath, self.v.value)
        self.load_df(self.db_path, self.v.value)
        
        self.oldv=0
        self.update_lanes()
        
        ##Initialize the figure

        fig1=widgets.Output()
        plt.ion()
        with plt.ioff():
            with fig1:
                #fig = plt.figure(tight_layout=True, dpi=200)
                #gs = gridspec.GridSpec(2, 4)
                #fig, axes = plt.subplots(nrows=2, ncols=2, dpi=100, constrained_layout=True)
                self.fig, self.ax = plt.subplots(constrained_layout=True, figsize=(6,6))
                
                display(self.fig.canvas)

        fig2=widgets.Output()
     
        with plt.ioff():
            with fig2:
                #fig = plt.figure(tight_layout=True, dpi=200)
                #gs = gridspec.GridSpec(2, 4)
                #fig, axes = plt.subplots(nrows=2, ncols=2, dpi=100, constrained_layout=True)
                
                self.fig2, self.ax2 = plt.subplots(constrained_layout=True, figsize=(8,6))
                
                #self.cid2 = self.fig2.canvas.mpl_connect('button_press_event', self.onclick_plot)
                
                display(self.fig2.canvas)

        #self.fig.canvas.toolbar_visible = False
        self.fig.canvas.header_visible = False
        self.fig.canvas.footer_visible = True
        self.fig.canvas.toolbar_position='bottom'

        self.fig2.canvas.header_visible = False
        self.fig2.canvas.footer_visible = True
        self.fig2.canvas.toolbar_position='bottom'
        self.im = self.ax.imshow(image, cmap='gray')
        
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        self.bscat = self.ax.scatter([0,0], [0,0], s=0.4*plt.rcParams['lines.markersize'] ** 2, color='blue', alpha=0.5)
        self.lscat = self.ax.scatter([0,0], [0,0], s=0.2*plt.rcParams['lines.markersize'] ** 2, color='red', alpha=0.5)
        
        self.cid2 = self.fig2.canvas.mpl_connect('button_press_event', self.onclick_plot)
        
        self.ax2.plot(np.arange(100), np.ones(100), color='blue')
        self.tmarker=self.ax2.axvline(self.t.value, color='black', lw=1)
        self.ax2.margins(x=0)
        
        
        buttons = [self.t, self.c, self.v, self.clip]
        for button in buttons:
            button.observe(self.update, 'value')
            
        self.file_menu = widgets.Dropdown(options=experiment_paths)
        
        self.file_menu.observe(self.update_experiment, 'value')

        self.buttons_box = widgets.VBox(buttons+ [self.view_nuclei, self.view_cellpose]) #, layout=widgets.]


        self.left_box = widgets.VBox([self.buttons_box, self.file_menu])
        
        self.grid = widgets.HBox([self.left_box, fig1, fig2])
        self.update(None)

    def update(self, change, t=None, c=None, v=None, clip=None):

        vmin, vmax = self.clip.value
        clip=self.clip.value
        t = self.t.value
        c =self.c.value
        v=self.v.value
        #image = self.f.get_frame_2D(v=v,c=c,t=t)
               
        #self.im.set_data(image)
        #lanes = g.get_frame_2D(v=v)
        #self.im.set_clim([vmin, vmax])
        #self.fig.canvas.draw()
        
        if v!=self.oldv:
            self.cyto_locator=None
            self.update_lanes()
        
        if self.view_nuclei.value:
            if v!=self.oldv:
                self.load_df(self.db_path, v)
            self.update_tracks()
            
        if self.view_cellpose.value:
            if v!=self.oldv:
                self.load_masks(self.outpath, v)
    
        self.update_image(t, v, clip)
        
        self.im.set_data(self.image)
        self.tmarker.set_xdata(t)
        
        self.oldv=v
        
    def update_tracks(self):
        
        #Update red and blue nuclei on image

        t, v= self.t.value, self.v.value
        scat=self.bscat
        df = self.clean_df[self.clean_df.frame==self.t.value]
        
        data = np.hstack((df.x.values[:,np.newaxis], df.y.values[:, np.newaxis]))
        scat.set_offsets(data)
        
        scat=self.lscat
        if 'segment' in df.columns:
            df = df[(df.segment>0)]
        
        data = np.hstack((df.x.values[:,np.newaxis], df.y.values[:, np.newaxis]))
        scat.set_offsets(data)
        
        return
    
    def update_lanes(self):
        
        fov = self.v.value
        path_to_lane = os.path.join(self.outpath, f'XY{fov}/lanes/lanes_mask.tif')

        self.lanes = imread(path_to_lane)>0

        return  

    def update_image(self, t, v, clip):
        
        vmin, vmax = clip
        cyto = self.f.get_frame_2D(v=self.v.value,c=self.c.value,t=self.t.value)
        cyto = np.clip(cyto, vmin, vmax).astype('float32')
        cyto = (255*(cyto-vmin)/(vmax-vmin)).astype('uint8')
        image = np.stack((cyto, cyto, cyto), axis=-1)
        image[:,:,0]= np.clip((self.lanes*10).astype('uint16')+image[:,:,0].astype('uint16'), 0, 255).astype('uint8')
        
        if self.view_cellpose.value:       
        
            mask = self.masks[t]

            bin_mask = np.zeros(mask.shape, dtype='bool')
            cell_ids = np.unique(mask)
            cell_ids = cell_ids[cell_ids!=0]

            for cell_id in cell_ids:
                bin_mask+=binary_erosion(mask==cell_id)

            outlines = find_boundaries(bin_mask, mode='outer')
            try:
                pass
                #print(f'{cell_ids.max()} Masks detected')
            except ValueError:
                print('No masks detected')

            self.outlines = outlines
            image[(outlines>0)]=[255,0,0]
            
            if (self.cyto_locator is not None):

                mask_id=self.cyto_locator[t]

                if mask_id!=0:
                    g_outline = find_boundaries(mask==mask_id)
                    image[g_outline]=[0,255,0]

        self.image=image

    def load_df(self, db_path, fov, from_db=False):
        
        if from_db:
            conn = sqlite3.connect(db_path)
            experiment_id=5

            query = f"""
            SELECT * from Raw_tracks
            where Lane_id in
            (SELECT Lane_id From Lanes where lanes.Experiment_id={experiment_id})
            and valid=1
            order by Lane_id"""

            #path_to_df = os.path.join(self.outpath, f'XY{fov}/tracking_data.csv')
            #self.df = pd.read_csv(path_to_df)
            #return

            #df = pd.read_sql(query, conn)
            self.experiment_df = pd.read_sql("""
            Select * from Experiments""", conn)

            self.lanes_df = pd.read_sql(
                f"""Select * from Lanes 
                Where (Experiment_id={experiment_id} and fov={fov})""",
                 conn)

            self.df = pd.read_sql(
                f"""Select * from Raw_tracks 
                Where Lane_id in \
                (Select Lane_id FROM Lanes WHERE
                (Experiment_id={experiment_id} and fov={fov}))""",
                 conn)

            self.metadata = self.experiment_df[self.experiment_df.Experiment_id==experiment_id]
            self.pixelperum = self.metadata['pixels/um'].values
            self.fpm = self.metadata['fpm'].values

            self.lane_ids = np.unique(self.df.Lane_id.values)
            self.lane_ids.sort()

            #self.df['particle']=self.df.particle_id  
            # self.df = tracking.get_single_cells(self.df)
            # self.df = tracking.remove_close_cells(self.df)
            
            # self.clean_df = tracking.get_clean_tracks(self.df)
        
            conn.close()
        
        else:
            
            self.df = pd.read_csv(f'{self.outpath}/XY{fov}/tracking_data.csv')
      
            #self.df['particle_id']=self.df.particle  
            #self.df = tracking.get_single_cells(self.df)
            #self.df = tracking.remove_close_cells(self.df)
            
            self.clean_df = pd.read_csv(f'{self.outpath}/XY{fov}/clean_tracking_data.csv')
            #self.clean_df = tracking.get_clean_tracks(self.df)
            #self.df = self.clean_df
            #self.clean_df = pd.read_csv('/project/ag-moonraedler/MAtienza/UNikon_gradients_27_05_22/extraction/XY0/clean_tracking_data.csv')

    def load_masks(self, outpath, fov):
        
        path_to_mask = os.path.join(outpath, f'XY{fov}/cyto_masks.mp4')
        
        self.masks = functions.mp4_to_np(path_to_mask)
        
        return      
        
    
    def onclick_plot(self, event):
    
        t = event.xdata
        self.t.value=t
        self.update(self.t.value, self.c.value, self.v.value, self.clip.value)
        
        return

    def onclick(self, event):
        
        ix, iy = event.xdata, event.ydata

        self.coords = (ix, iy)
        
        mask_id = self.masks[self.t.value, np.round(iy).astype(int), np.round(ix).astype(int)]
        print(mask_id)
        if mask_id==0:
            #No mask was clicked on
            return
        
        particle_id = self.df.loc[(self.df.frame==self.t.value) & (self.df.cyto_locator==mask_id)].particle.values
        if len(particle_id)<1:
            #No mask was clicked on
            return
            #self.particle_id=self.particle_id[0]
        self.particle_id=particle_id[0]

        self.dfp=self.clean_df[self.clean_df.particle==self.particle_id]

        self.ax2.clear()
        self.ax2.plot(self.dfp.frame, self.dfp.nucleus, color='blue')
        self.ax2.plot(self.dfp.frame, self.dfp.front, color='red')
        self.ax2.plot(self.dfp.frame, self.dfp.rear, color='red')
        self.ax2.margins(x=0)
        
        tres = 30
        def t_to_frame(t):
            return t/(tres/60)
        def frame_to_t(frame):
            return frame*tres/60

        tax = self.ax2.secondary_xaxis('top', functions=(frame_to_t, t_to_frame))
        tax.set_xlabel('Time in minutes')
        self.tmarker=self.ax2.axvline(self.t.value, color='black', lw=1)

        if 'segment' in self.dfp.columns:
            low, high = self.ax2.get_ylim()
            collection = collections.BrokenBarHCollection.span_where(
                self.dfp.frame.values, ymin=low, ymax=high, where=self.dfp.segment==0, facecolor='gray', alpha=0.5)

            self.ax2.add_collection(collection)

        if 'state' in self.dfp.columns:
            #self.dfp, cp_indices, valid_boundaries, segments = cp.classify_movement(self.dfp)

            MO_bool =self.dfp.state=='MO'
            MS_bool =self.dfp.state=='MS'
            SO_bool =self.dfp.state=='SO'
            SS_bool =self.dfp.state=='SS'
        

            x_collection = self.dfp.frame.values

            MO_collection = collections.BrokenBarHCollection.span_where(
                        x_collection, ymin=low, ymax=high, where=MO_bool, facecolor=[1,0,1], alpha=0.2)

            MS_collection = collections.BrokenBarHCollection.span_where(
                        x_collection, ymin=low, ymax=high, where=MS_bool, facecolor=[1,0,0], alpha=0.2)

            SO_collection = collections.BrokenBarHCollection.span_where(
                        x_collection, ymin=low, ymax=high, where=SO_bool, facecolor=[0,0,1], alpha=0.2)

            SS_collection = collections.BrokenBarHCollection.span_where(
                        x_collection, ymin=low, ymax=high, where=SS_bool, facecolor=[0,1,0], alpha=0.2)

            self.ax2.add_collection(MO_collection)
            self.ax2.add_collection(MS_collection) 
            self.ax2.add_collection(SO_collection)
            self.ax2.add_collection(SS_collection)  

            o_change = np.concatenate(([0], np.diff(self.dfp.O.values)!=0))
            v_change = np.concatenate(([0], np.diff(self.dfp.V.values)!=0))
        
            cps = np.argwhere(o_change & v_change & (self.dfp.state.notnull().values))
            #cps = t_[np.clip(cp_indices.astype(int), 0, t_.size-1)]
            for cp_index in cps:
                self.ax2.axvline(cp_index, color='green')
                
        # for boundary in t_[np.clip(np.array(segments).flatten().astype(int), 0, t_.size-1)]:
        #     self.ax2.axvline(boundary, color='green')

        self.cyto_locator = np.zeros(self.masks.shape[0], dtype='uint8')
        
        self.cyto_locator[self.dfp.frame]=self.masks[self.dfp.frame, np.round(self.dfp.y).astype(int), np.round(self.dfp.x).astype(int)]
        
        self.update_image(self.t.value, self.v.value, self.clip.value)
        
        self.im.set_data(self.image)
        return
    
    def update_experiment(self, file_name):

        file_name = file_name.new
        
        base_path = self.base_path
        path_to_meta_data = os.path.join(base_path, file_name, 'Experiment_data.csv')
        experiment_data = pd.read_csv(path_to_meta_data)
        
        self.nd2file = os.path.join(experiment_data.Path.values[0], experiment_data.time_lapse_file.values[0])
        
        self.f = ND2Reader(self.nd2file)
        self.nfov, self.nframes = self.f.sizes['v'], self.f.sizes['t']
        
        self.outpath= os.path.join(base_path, file_name, 'extraction') 
        
        ##update the sliders
        self.t.max = self.f.sizes['t']-1
        self.v.max = self.f.sizes['v']-1 
        
        self.load_masks(self.outpath, self.v.value)
        self.load_df(self.db_path, self.v.value)
        self.update(None)
        #self.update()
        #self.update_image(self.t.value, self.v.value, self.clip.value)
        #self.update_tracks()
        
        return