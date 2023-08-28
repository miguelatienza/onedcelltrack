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

class Viewer:
    def __init__(self, file, pattern_file, experiment_id=5, df=None):

        self.experiment_id=experiment_id
        self.t, self.v, self.c = [0, 0, 0]
        self.load_df(self.experiment_id, self.v, df=df)
        f = ND2Reader(file)

        axes = f.axes

        if f.sizes['t']>0:
            t_max = f.sizes['t']-1
            t = widgets.IntSlider(min=0,max=t_max, step=1, description="t", continuous_update=True)
        
        if f.sizes['c']>0:
            c_max = f.sizes['c']-1
            c = widgets.IntSlider(min=0,max=c_max, step=1, description="c", continuous_update=True)
        
        if f.sizes['v']>0:
            v_max = f.sizes['v']-1
            v = widgets.IntSlider(min=0,max=v_max, step=1, description="v", continuous_update=False)
        
        clip = widgets.IntRangeSlider(min=0,max=int(2**16 -1), step=1, description="clip", continuous_update=True, width='200px')

        show_nuclei = widgets.Checkbox(
        value=False,
        description='Show trackpy',
        disabled=False,
        indent=False
        )
        
        #live_tracking = widgets.Checkbox(
        #value=False,
        #description='Live tracking',
        #disabled=False,
        #indent=False
        #)

        im_0 = f.get_frame_2D(t=t.value, c=c.value, v=v.value)

        #plt.subplots()
        im = plt.imshow(im_0, cmap='gray')

        scat=plt.scatter([0,0], [0,0])

        def update(t, c, v, clip, show_nuclei):
    
            vmin, vmax = clip
            image = f.get_frame_2D(v=v,c=c,t=t)
            im.set_data(image)
            #lanes = g.get_frame_2D(v=v)
            im.set_clim([vmin, vmax])

            if show_nuclei:
                if v!= self.v:
                    self.load_df(experiment_id=self.experiment_id, fov=v, df=df)
                self.show_tracking(self.df, scat, t)
            else:
                scat.set_offsets([[0,0], [0,0]])
                
            self.t, self.v, self.c = [t, v, c]
        

        out = widgets.interactive_output(update, {'t': t, 'c': c, 'v': v, 'clip': clip, 'show_nuclei': show_nuclei})

        box = widgets.VBox([out, widgets.VBox([t, c, v, clip, show_nuclei],  layout=widgets.Layout(width='400px'))])

        display(box)
        self.t, self.v = [t.value, v.value]

    def load_df(self, experiment_id=None, fov=0, df=None, path_to_db='/project/ag-moonraedler/MAtienza/database/onedcellmigration.db'):

        if df is None:
            conn = sqlite3.connect(path_to_db)

            query = f'''Select * From Raw_tracks where Lane_id in (
                SELECT Lane_id FROM Lanes WHERE (Experiment_id={experiment_id} and fov={fov}))'''
            
            self.df = pd.read_sql(query, conn)
        
        elif df is not None:
            self.df = df

    def show_tracking(self, df, scat, t):

        dft = df[df.frame==t]
        data = np.hstack((dft.x.values[:,np.newaxis], dft.y.values[:, np.newaxis]))
        scat.set_offsets(data)

class LaneViewer:

    def __init__(self, nd2_file, df=None):

        self.v = 0
        f = ND2Reader(nd2_file)
        
        self.lanes = np.array([f.get_frame_2D(v=v) for v in range(f.sizes['v'])])
        axes = f.axes
        
        self.ld = widgets.IntSlider(min=10,max=60, step=1, description="lane_d", value=30, continuous_update=True)
        
        v_max = f.sizes['v']-1
        self.v = widgets.IntSlider(min=0,max=v_max, step=1, description="v", continuous_update=False)

        self.clip = widgets.IntRangeSlider(min=0,max=int(5_000), step=1, description="clip", value=[0,5000], continuous_update=True, width='200px')
        
        self.threshold = widgets.FloatSlider(min=0,max=1, step=0.05, description="threshold", continuous_update=False, value=0.3)
        
        self.kernel_width=5
        
        self.button = widgets.Button(
        description='Recompute',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click me',
        icon='check') # (FontAwesome names without the `fa-` prefix)
        
        self.button.on_click(self.recompute_v)
        
        fig = plt.figure()
        self.ax = fig.add_subplot(111)
        #plt.subplots()
        self.im = self.ax.imshow(self.lanes[0], cmap='gray', vmin=0, vmax=5000)
        
        self.min_coordinates = list(range(f.sizes['v']))
        self.max_coordinates = list(range(f.sizes['v']))

        self.recompute_v(self.v.value)
        #self.min_coordinates=[0,10]
        #self.max_coordinates=[0,20]
        self.ax.plot([], [], color='red')
        
        out = widgets.interactive_output(self.update, {'v': self.v, 'clip': self.clip})

        box = widgets.VBox([out, widgets.VBox([self.v, self.clip, self.ld, self.threshold, self.button],  layout=widgets.Layout(width='400px'))])

        display(box)
        

        
    def update(self, v, clip):
    
        vmin, vmax = self.clip.value
        image = self.lanes[self.v.value]
        self.im.set_data(image)
        self.im.set_clim([vmin, vmax])
        
        if isinstance(self.min_coordinates[v], Iterable):
           
            
            [self.ax.axes.lines[0].remove() for j in range(len(self.ax.axes.lines))]
            x = [0, image.shape[1]-1]
            for i in range(self.min_coordinates[v].shape[0]):
                self.ax.plot(x, [self.min_coordinates[v][i,1], sum(self.min_coordinates[v][i])], color='red')
            
            for i in range(self.max_coordinates[v].shape[0]):
                self.ax.plot(x, [self.max_coordinates[v][i,1], sum(self.max_coordinates[v][i])], color='red')
            
        
    # def recompute(self):

    #     vmin, vmax = self.clip.value
    #     lanes_clipped = np.clip(self.lanes, vmin, vmax, dtype=self.lanes.dtype)

    #     self.min_coordinates = list(range(lanes_clipped.shape[0]))
    #     self.max_coordinates = list(range(lanes_clipped.shape[0]))

    #     for v in tqdm(range(lanes_clipped.shape[0])): 

    #         min_c, max_c = functions.get_lane_mask(lanes_clipped[v], kernel_width=self.kernel_width, line_distance=self.ld.value, debug=True)
            
    #         self.min_coordinates.append(min_c)
    #         self.max_coordinates.append(max_c)

    #     update(v, clip)
    
    def recompute_v(self, v):
        
        v = self.v.value
        vmin, vmax = self.clip.value
        lanes_clipped = np.clip(self.lanes[v], vmin, vmax, dtype=self.lanes.dtype)
        
        print('recomputing')
        self.min_coordinates[v], self.max_coordinates[v] = lane_detection.get_lane_mask(lanes_clipped, kernel_width=self.kernel_width, line_distance=self.ld.value, debug=True, gpu=True, threshold=self.threshold.value)
        print('updating')
        self.update(v, self.clip)
        
class TpViewer:
    
    def __init__(self, nd2file, channel=0):
        
        self.link_dfs = {}
        self.channel=channel
        
        self.f = ND2Reader(nd2file)
        self.nfov, self.nframes = self.f.sizes['v'], self.f.sizes['t']
        
        #Widgets
        
        t_max = self.f.sizes['t']-1
        self.t = widgets.IntSlider(min=0,max=t_max, step=1, description="t", continuous_update=True)

        c_max = self.f.sizes['c']-1
        self.c = widgets.IntSlider(min=0,max=c_max, step=1, description="c", continuous_update=True)

        v_max = self.f.sizes['v']-1
        self.v = widgets.IntSlider(min=0,max=v_max, step=1, description="v", continuous_update=False)
        
        self.clip = widgets.IntRangeSlider(min=0,max=int(2**16 -1), step=1, value=[0,1000], description="clip", continuous_update=True, width='200px')
        
        self.min_mass = widgets.FloatSlider(min=1e5, max=1e6, step=0.01e5, description="min_mass", value=2.65e5, continuous_update=True)
        
        self.diameter = widgets.IntSlider(min=9,max=35, step=2, description="diameter", value=15, continuous_update=True)
        
        self.min_frames = widgets.FloatSlider(min=0,max=50, step=1, value=10, description="min_frames", continuous_update=False)
        
        self.max_travel = widgets.IntSlider(min=3,max=50, step=1, value=15, description="max_travel", continuous_update=False)
        
        self.track_memory = widgets.IntSlider(min=0,max=20, step=1, value=5, description="track memory", continuous_update=False)


        self.tp_method = widgets.Button(
        description='Track',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click me',
        icon='')
        
        self.tp_method.on_click(self.link_update)
        
        vmin, vmax = self.clip.value
        image = self.f.get_frame_2D(v=self.v.value,c=self.c.value,t=self.t.value)
        
        ##Initialize the figure
        plt.ioff()
        self.fig, self.ax = plt.subplots()
        self.fig.tight_layout()
        #self.fig.canvas.toolbar_visible = False
        self.fig.canvas.header_visible = False
        self.fig.canvas.footer_visible = True
        self.im = self.ax.imshow(image, cmap='gray')

        self.bscat = self.ax.scatter([0,0], [0,0], s=0.2*plt.rcParams['lines.markersize'] ** 2, color='blue', alpha=0.5)
        self.lscat = self.ax.scatter([0,0], [0,0], s=0.2*plt.rcParams['lines.markersize'] ** 2, color='red', alpha=0.5)
                
        #Organize layout and display
        out = widgets.interactive_output(self.update, {'t': self.t, 'c': self.c, 'v': self.v, 'clip': self.clip, 'min_mass': self.min_mass, 'diameter': self.diameter, 'min_frames': self.min_frames, 'max_travel': self.max_travel})
        
        box = widgets.VBox([self.t, self.c, self.v, self.clip, self.min_mass, self.diameter, self.min_frames, self.max_travel, self.track_memory, self.tp_method]) #, layout=widgets.Layout(width='400px'))
        box1 = widgets.VBox([out, box])
        grid = widgets.widgets.GridspecLayout(3, 3)
        
        grid[:, :2] = self.fig.canvas
        grid[1:,2] = box
        grid[0, 2] = out
        
        #display(self.fig.canvas)
        display(grid)
        plt.ion()
 
    def update(self, t, c, v, clip, min_mass, diameter, min_frames, max_travel):

        vmin, vmax = clip
        image = self.f.get_frame_2D(v=v,c=c,t=t)
               
        self.im.set_data(image)
        #lanes = g.get_frame_2D(v=v)
        self.im.set_clim([vmin, vmax])
        #self.fig.canvas.draw()
        

        self.batch_update(v, t, min_mass, diameter)
        self.show_tracking(self.batch_df, self.bscat)

        if v in self.link_dfs.keys():
            df = self.link_dfs[v][self.link_df.frame==t]
            self.show_tracking(df, self.lscat)
        else:
            self.lscat.set_offsets([[0,0]]) 
    
    def show_tracking(self, df, scat):
        
        t, v= self.t.value, self.v.value
        
        #[plt.axes.lines[0].remove() for j in range(len(self.ax.axes.lines))]
        data = np.hstack((df.x.values[:,np.newaxis], df.y.values[:, np.newaxis]))
        scat.set_offsets(data)
    
    def batch_update(self, v, t, min_mass, diameter):
        nuclei = self.f.get_frame_2D(v=v, t=t, c=self.channel)
        nuclei = functions.preprocess(nuclei, bottom_percentile=0.05, top_percentile=99.95, log=True, return_type='uint16')
        
        self.batch_df = tp.locate(nuclei, diameter=diameter, minmass=min_mass)
       
        
    def link_update(self, a):
        
        v, min_mass, max_travel, track_memory, diameter, min_frames = self.v.value, self.min_mass.value, self.max_travel.value, self.track_memory.value, self.diameter.value, self.min_frames.value
        if False:#(v in self.link_dfs.keys()):
            self.link_df = self.link_dfs[v]
        else:
            nuclei = np.array([self.f.get_frame_2D(v=v, t=t, c=self.channel) for t in range(self.f.sizes['t'])])
            nuclei = functions.preprocess(nuclei, bottom_percentile=0.05, top_percentile=99.95, log=True, return_type='uint16')
            dft = tp.batch(nuclei, diameter=diameter, minmass=min_mass)
            dftp = tp.link(dft, max_travel, memory=track_memory)

            self.link_df = tp.filter_stubs(dftp, min_frames)
            self.link_dfs[v]=self.link_df
        
        self.update(self.t.value, self.c.value, self.v.value, self.clip.value, self.min_mass.value, self.diameter.value, self.min_frames.value, self.max_travel.value)
        

class CellposeViewer:
    
    def __init__(self, nd2file, bf_channel=None, nuc_channel=None,pretrained_model='mdamb231', omni=False):
        
        self.link_dfs = {}
        
        self.f = ND2Reader(nd2file)
        self.nfov, self.nframes = self.f.sizes['v'], self.f.sizes['t']
        
        channels = self.f.metadata['channels']
    
        if bf_channel is None or nuc_channel is None:
                    ###Infer the channels
            if 'erry' in channels[0] or 'exas' in channels[0] and not 'phc' in channels[0]:
                self.nucleus_channel=0
                self.cyto_channel=1
            elif 'erry' in channels[1] or 'exas' in channels[1] and not 'phc' in channels[1]:
                self.nucleus_channel=1
                self.cyto_channel=0
                
            else:
                raise ValueError(f"""The channels could not be automatically detected! \n
                The following channels are available: {channels} . Please specify the indices of bf_channel and nuc_channel as keyword arguments. i.e: bf_channel=0, nuc_channel=1""")
        else:
            self.cyto_channel=bf_channel
            self.nucleus_channel=nuc_channel

        #Widgets
        
        t_max = self.f.sizes['t']-1
        self.t = widgets.IntSlider(min=0,max=t_max, step=1, description="t", continuous_update=False)

        v_max = self.f.sizes['v']-1
        self.v = widgets.IntSlider(min=0,max=v_max, step=1, description="v", continuous_update=False)
        
        self.nclip = widgets.FloatRangeSlider(min=0,max=2**16, step=1, value=[0,1000], description="clip nuclei", continuous_update=False, width='200px')
        
        self.cclip = widgets.FloatRangeSlider(min=0,max=2**16, step=1, value=[50,8000], description="clip cyto", continuous_update=False, width='200px')
        
        self.flow_threshold = widgets.FloatSlider(min=0, max=1.5, step=0.05, description="flow_threshold", value=1.25, continuous_update=False)
        
        self.diameter = widgets.IntSlider(min=0,max=1000, step=2, description="diameter", value=29, continuous_update=False)
        
        self.mask_threshold = widgets.FloatSlider(min=-3,max=3, step=0.1, value=0, description="mask_threshold", continuous_update=False)
        
        self.max_travel = widgets.IntSlider(min=3,max=50, step=1, value=5, description="max_travel", continuous_update=False)
        
        self.track_memory = widgets.IntSlider(min=0,max=20, step=1, value=5, description="max_travel", continuous_update=False)

        self.tp_method = widgets.Button(
        description='Track',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click me',
        icon='')
        
        #self.tp_method.on_click(self.link_update)
        
        vmin, vmax = self.cclip.value
        cyto = (255*(np.clip(self.f.get_frame_2D(v=0,c=self.cyto_channel,t=0), vmin, vmax)/vmax)).astype('uint8')
        #cyto = np.clip(self.f.get_frame_2D(v=0,c=self.cyto_channel,t=0), vmin, vmax)
        #nucleus = self.f.get_frame_2D(v=0,c=self.nucleus_channel,t=0)
        #nucleus = functions.preprocess(nucleus, log=True, bottom_percentile=0.05, top_percentile=99.95, return_type='uint8')
        vmin, vmax = self.nclip.value
        nucleus = (255*(np.clip(self.f.get_frame_2D(v=0,c=self.nucleus_channel,t=0), vmin, vmax)/vmax)).astype('uint8')
        red = np.zeros_like(nucleus)
       
        image = np.stack((red, red, nucleus), axis=-1).astype('float32')
        image+=(cyto[:,:,np.newaxis]/3)
        image = np.clip(image, 0,255).astype('uint8')
        
        ##Initialize the figure
        plt.ioff()
        self.fig, self.ax = plt.subplots()
        self.fig.tight_layout()
        #self.fig.canvas.toolbar_visible = False
        self.fig.canvas.header_visible = False
        #self.fig.canvas.footer_visible = False
        self.im = self.ax.imshow(image)

        self.init_cellpose(pretrained_model=pretrained_model, omni=omni)
                
        #Organize layout and display
        out = widgets.interactive_output(self.update, {'t': self.t, 'v': self.v, 'cclip': self.cclip, 'nclip': self.nclip, 'flow_threshold': self.flow_threshold, 'diameter': self.diameter, 'mask_threshold': self.mask_threshold, 'max_travel': self.max_travel})
        
        box = widgets.VBox([self.t, self.v, self.cclip, self.nclip, self.flow_threshold, self.diameter, self.mask_threshold, self.tp_method]) #, layout=widgets.Layout(width='400px'))
        box1 = widgets.VBox([out, box])
        grid = widgets.widgets.GridspecLayout(3, 3)
        
        grid[:, :2] = self.fig.canvas
        grid[1:,2] = box
        grid[0, 2] = out
        
        #display(self.fig.canvas)
        display(grid)
        plt.ion()
    
    def init_cellpose(self, pretrained_model='mdamb231', omni=False, model='cyto', gpu=True):
 
        if omni:
            from cellpose_omni.models import CellposeModel
            self.model = CellposeModel(
            gpu=gpu, omni=True, nclasses=4, nchan=2, pretrained_model=pretrained_model)
            return

        
        elif pretrained_model is None:
            self.model = models.Cellpose(gpu=gpu, model_type='cyto')

        else:
            path_to_models = os.path.join(os.path.dirname(__file__), '../models')
            with open(os.path.join(path_to_models, 'models.json'), 'r') as f:
                dic = json.load(f)
            if pretrained_model in dic.keys():
                path_to_model = os.path.join(path_to_models, dic[pretrained_model]['path'])
                if os.path.isfile(path_to_model):
                    pretrained_model = path_to_model
                else: 
         
                    url = dic[pretrained_model]['link']
                    print('Downloading model from Nextcloud...')
                    request.urlretrieve(url, os.path.join(path_to_models, path_to_model))
                    pretrained_model = os.path.join(path_to_models,dic[pretrained_model]['path'])

            
            if not omni:
                
                self.model = models.CellposeModel(gpu=gpu, pretrained_model=pretrained_model)

    def update(self, t, v, cclip, nclip, flow_threshold, diameter, mask_threshold, max_travel):      
        
        
        #contours = get_outlines(image)
        self.segment(
        t, v, cclip, nclip, flow_threshold, mask_threshold, diameter)
        self.im.set_data(self.image)
        return
        #lanes = g.get_frame_2D(v=v)
        #self.im.set_clim([vmin, vmax])
        #self.fig.canvas.draw()
        

        self.batch_update(v, t, min_mass, diameter)
        self.show_tracking(self.batch_df, self.bscat)
        try:
            df = self.link_dfs[v][self.link_df.frame==t]
            self.show_tracking(df, self.lscat)
        except:
            self.lscat.set_offsets([]) 
    
    def show_tracking(self, df, scat):
        
        t, v= self.t.value, self.v.value
        
        #[plt.axes.lines[0].remove() for j in range(len(self.ax.axes.lines))]
        data = np.hstack((df.x.values[:,np.newaxis], df.y.values[:, np.newaxis]))
        scat.set_offsets(data)
    
    def segment(self,t, v, cclip, nclip, flow_threshold, mask_threshold, diameter, normalize=True, verbose=False,):
        
        #nucleus = self.f.get_frame_2D(v=v, t=t, c=self.nucleus_channel)
        #vmin, vmax = nclip
        #nucleus = functions.preprocess(nucleus, bottom_percentile=vmin, top_percentile=vmax, log=True, return_type='uint16')
        #vmin, vmax = cclip
        #cyto = self.f.get_frame_2D(v=v, t=t, c=self.cyto_channel)
        #cyto = functions.preprocess(cyto, bottom_percentile=vmin, top_percentile=vmax, return_type='uint16')
       
        nucleus = self.f.get_frame_2D(v=v, t=t, c=self.nucleus_channel)
        #nucleus = functions.preprocess(nucleus, bottom_percentile=0, #top_percentile=100, log=True, return_type='uint16')
        cyto = self.f.get_frame_2D(v=v, t=t, c=self.cyto_channel)
        
    
        image = np.stack((cyto, nucleus), axis=1)
        if diameter == 0:
            diameter=None
        mask = self.model.eval(
            image, diameter=diameter, channels=[1,0], flow_threshold=flow_threshold, cellprob_threshold=mask_threshold, normalize=normalize, progress=verbose)[0].astype('uint8')
        
        bin_mask = np.zeros(mask.shape, dtype='bool')
        cell_ids = np.unique(mask)
        cell_ids = cell_ids[cell_ids!=0]
        
        for cell_id in cell_ids:
            bin_mask+=binary_erosion(mask==cell_id)
        
        outlines = find_boundaries(bin_mask, mode='outer')
        try:
            print(f'{cell_ids.max()} Masks detected')
        except ValueError:
            print('No masks detected')
        self.outlines = outlines
        
        self.mask=mask
        self.image = self.get_8bit(outlines, cyto)
        
        return 
    
        self.image = np.stack((outlines, cyto.astype('uint8'), nucleus.astype('uint8')), axis=-1)
        self.outlines = outlines
        
    def get_8bit(self, outlines, cyto, nuclei=None):
               
        vmin, vmax = self.cclip.value
        cyto = np.clip(cyto, vmin, vmax)
        cyto = (255*(cyto-vmin)/(vmax-vmin)).astype('uint8')
        
        image = np.stack((cyto, cyto, cyto), axis=-1)
        
        image[(outlines>0)]=[255,0,0]
        
        #outlines[:,:,np.newaxis]>0
        
        return image
        

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