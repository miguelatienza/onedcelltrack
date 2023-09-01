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
    This is a general class is to view the images and then chose the bf_channel, nuc_channel and specify the fovs of interest as well as the corresponding fovs corresponding to the lanes_file. Note that if you do not specify the fovs and the fovs_lanes, they must have the same length. the viewer contains a button to delete the current fov. The viewer also contains a button to exchange bf_channel and nucleus_channel. The viewer also contains a button to save the fovs and fovs_lanes as well as bf_channel and nuc_channel to the param_dict.

    Parameters
    ----------
    pipeline : Pipeline
        Instance of the pipeline class.
    """
    def __init__(self, pipeline, fovs=None, fovs_lanes=None, frame_indices=None):
        self.__dict__.update(pipeline.__dict__)
        
        self.pipeline = pipeline
        # if (fovs is None) and (self.fovs is None):
        #     print('here')
        #     fovs = list(range(self.n_fovs))
        #     self.fovs=fovs
        #     self.n_fovs=len(fovs)
        # if (fovs_lanes is None) and (self.fovs_lanes) is None:
        #     fovs_lanes = list(range(self.n_fovs_lanes))
        #     self.fovs_lanes=fovs_lanes
        #     self.n_fovs_lanes=len(fovs_lanes)

        # if (fovs is not None) and (fovs_lanes is not None):
        #     assert len(fovs)==len(fovs_lanes), "The length of the fovs was inferred automatically but they are not of the same length. Please specify the fovs and fovs_lanes as lists of the same length."
        # elif (fovs is not None) and (fovs_lanes is None):
        #     assert len(fovs)==len(fovs_lanes), "The length of the fovs and fovs_lanes must be the same."
        # if (frame_indices is None) and (self.frame_indices is None):
        #     self.frame_indices = self.infer_frame_indices()
        #     self.n_frames = len(self.frame_indices)

        # elif frame_indices is not None:
        #     self.frame_indices = frame_indices
        #     self.n_frames = len(self.frame_indices)
        
        ## Initalise the channels
        if self.nuc_channel is None:
            self.nuc_channel = 1
            self.bf_channel = 0

        # Set tchannel to brightfield
        self.channel=self.bf_channel

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
        value=self.pixelperum,
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
        self.clip = widgets.IntRangeSlider(min=0,max=int(2**16 -1), step=1, value=[0,50_000], description="clip", continuous_update=True, width='200px')
        #slider for the fovs and make them as wide as te image
        self.fov = widgets.IntSlider(min=min(self.fovs),max=max(self.fovs), step=1, description="fov", continuous_update=False)
        #slider for the frames and make them as wide as te image
        self.frame = widgets.IntSlider(min=min(self.frame_indices),max=max(self.frame_indices), step=1, description="frame", continuous_update=False)
        
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
        buttons = widgets.VBox([self.button_delete_fov, self.button_exchange_channels, self.button_save, self.select_channel, self.button_cut_left, self.button_cut_right, self.select_tres, self.lanes_alpha, self.position_status, self.select_pixelperum, self.position_status])
        
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

        self.n_fovs = len(self.fovs)
        self.n_fovs_lanes = len(self.fovs_lanes)
        self.n_frames = len(self.frame_indices)

        if fov not in self.fovs or not frame in self.frame_indices:
            self.position_status.button_style='danger'
            self.position_status.description='Deleted position'
            self.fig.canvas.draw()
            return
        else:
            self.position_status.button_style='success'
            self.position_status.description='Not deleted position'

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
        Deletes the current fov from the fovs and fovs_lanes lists.
        """
        if not self.fov.value in self.fovs:
            return
        self.fovs.remove(self.fov.value)
        self.fovs_lanes.remove(self.fov.value)
      
        #self.fov.max-=1
        self.fov.value+=1
        self.update(self.fov.value, self.clip.value, self.frame.value)
    
    def save(self, a):
        """
        Save the fovs to the param_dict.
        """
        ## Make the save button turn orange
        self.button_save.button_style='warning'
        ## add a message
        self.button_save.description='Saving...'
        self.update_pipeline()
        print('udpating pipeline')

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
            self.pipeline.save_parameters()
            self.button_save.button_style='success'
            self.button_save.description='Saved'
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
        # set all attributes in param_dict to the pipeline
        self.pipeline.__dict__.update(param_dict)
        # self.pipeline.__setattr__('param_dict', param_dict)

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
        self.pipeline = pipeline

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
        
        self.min_coordinates = {fov: None for fov in self.fovs}
        self.max_coordinates = {fov: None for fov in self.fovs}

        self.ax.plot([], [], color='red')
        
    def get_lanes(self):
        """
        Returns the lanes for all fovs as a list.
        """
        lanes = {fov: self.read_lanes(fov=fov) for fov in self.fovs}
        return lanes
    
    def update(self, fov, clip):

        self.lane_distance=self.ld.value
        self.lane_threshold=self.threshold.value
        self.lane_low_clip=clip[0]
        self.lane_high_clip=clip[1]
        self.kernel_width=3

        if fov not in self.fovs:
            self.position_status.button_style='danger'
            self.position_status.description='Deleted position'
            self.fig.canvas.draw()
            return
        else:
            self.position_status.button_style='success'
            self.position_status.description='Not deleted position'

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
            #lanes_clipped = np.clip(self.lanes[self.fov.value], vmin, vmax, dtype=self.lanes[self.fov.value].dtype)
            lanes_image = self.get_image_lanes()
            #lanes_clipped = np.clip(lanes_image, vmin, vmax, dtype=lanes_image.dtype)
            print('recomputing')
            self.min_coordinates[self.fov.value], self.max_coordinates[self.fov.value] = lane_detection.get_lane_mask(lanes_image, kernel_width=self.kernel_width, line_distance=self.ld.value, debug=True, gpu=True, threshold=self.threshold.value, low_clip=vmin, high_clip=vmax)
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
        self.pipeline = pipeline
        self.channel=self.nuc_channel

        ## Initialise the link_dfs
        self.link_dfs = {}
        self.clip.value=[0, 5_000]
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

        self.min_mass_slider = widgets.FloatSlider(min=1e3, max=1e5, step=1e2, description="min_mass", value=self.min_mass, continuous_update=True)

        self.diameter_slider = widgets.IntSlider(min=9,max=35, step=2, description="diameter", value=self.diameter, continuous_update=True)
        
        self.min_frames_slider = widgets.FloatSlider(min=0,max=50, step=1, value=self.min_frames, description="min_frames", continuous_update=True)
        
        self.max_travel_slider = widgets.IntSlider(min=3,max=50, step=1, value=self.max_travel, description="max_travel", continuous_update=True)
        
        self.track_memory_slider = widgets.IntSlider(min=0,max=20, step=1, value=self.track_memory, description="track memory", continuous_update=True)
        
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
        out = widgets.interactive_output(self.update, {'fov': self.fov, 'clip': self.clip, 'frame': self.frame, 'min_mass': self.min_mass_slider, 'diameter': self.diameter_slider, 'min_frames': self.min_frames_slider, 'max_travel': self.max_travel_slider, 'track_memory': self.track_memory_slider})

        # box for sliders for tracking
        tracking_sliders = widgets.VBox([self.min_mass_slider, self.diameter_slider, self.min_frames_slider, self.max_travel_slider, self.track_memory_slider, self.track_button, self.button_save, self.position_status])

        # box for the image
        image_box = widgets.VBox([self.fig.canvas, self.fov, self.clip, self.frame], layout=widgets.Layout(width=str(self.fig.get_size_inches()[0]*self.fig.dpi)))
        
        # box for the sliders and figure
        self.window = widgets.HBox([tracking_sliders, image_box])
       
        #self.update(self.fov.value, self.clip.value, self.frame.value)
        display(self.window)
        self.update(self.fov.value, self.clip.value, self.frame.value, self.min_mass_slider.value, self.diameter_slider.value, self.min_frames_slider.value, self.max_travel_slider.value, self.track_memory_slider.value)

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
        self.update(self.fov.value, self.clip.value, self.frame.value, self.min_mass_slider.value, self.diameter_slider.value, self.min_frames_slider.value, self.max_travel_slider.value, self.track_memory_slider.value)

    def batch_update(self, fov, t, min_mass, diameter):
        #set the track button to warning
        self.track_button.button_style='warning'
        image = self.read_nuc(frames=t, fov=fov)
        self.batch_df = tracking.locate_nuclei(raw_image=image, minmass=min_mass, diameter=diameter)
        self.track_button.button_style='success'
        return

    def update(self, fov, clip, frame, min_mass, diameter, min_frames, max_travel, track_memory):
        
        self.min_mass = min_mass
        self.diameter = diameter
        self.max_travel = max_travel
        
        self.track_memory = track_memory
        self.min_frames = min_frames

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
        else:
            self.position_status.button_style='success'
            self.position_status.description='Not deleted position'

        if fov in self.link_dfs.keys():
            df = self.link_dfs[fov]
            df = df[df['frame']==frame]
            self.lscat.set_offsets(df[['x', 'y']].values)
            self.fig.canvas.draw()

        self.batch_update(fov, [frame], min_mass, diameter)
        # self.lscat.set_offsets(np.c_[[], []])
        self.bscat.set_offsets(self.batch_df[['x', 'y']].values)
        self.fig.canvas.draw()

        

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
        else:
            self.position_status.button_style='success'
            self.position_status.description='Not deleted position'
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

class ResultsViewer:
    """
    This class is used to view the results of the tracking and segmentation.
    Parameters
    ----------
    nd2file : str
        Path to the nd2file.
    outpath : str
        Path to the output directory.
    base_path : str
        Path to the base directory.
    experiment_paths : list
        List of paths to the experiments.
    db_path : str
        Path to the database.
    path_to_patterns : str
        Path to the patterns file.
    """
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