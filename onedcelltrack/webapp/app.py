from flask import Flask, render_template, request, jsonify, Response
from nd2reader import ND2Reader
import base64
import numpy as np
from io import BytesIO
from skimage.io import imsave
from PIL import Image
from matplotlib import pyplot as plt
import os
import pandas as pd
from onedcelltrack import functions
from skimage.segmentation import find_boundaries

from simulations_app import SimulationsApp

class OnedcelltrackApp:
    def __init__(self, experiments, base_path):
        self.app = Flask(__name__)
        simulations_blueprint = SimulationsApp().blueprint
        self.app.register_blueprint(simulations_blueprint)#, url_prefix='/simulations')
        self.nd2_file = '/project/ag-moonraedler/JHeyn/230406_UNikon_FN-lanes_MDAs_2min/230406_UNikon_FN-lanes_MDAs_2min.nd2'
        self.experiments = experiments
        self.experiment='230406_UNikon_FN-lanes_MDAs_2min'
        self.base_path = base_path
        self.f = ND2Reader(self.nd2_file)
        self.img = self.f.get_frame_2D()
        self.fov=0
        self.masks=np.zeros((self.f.sizes['t'], self.f.sizes['y'], self.f.sizes['x']), dtype=np.uint8)
        self.df=pd.DataFrame()
        self.particle_id=0
        self.mask_id=0
        self.img_height, self.img_width=self.img.shape

        @self.app.route("/")
        def index():
            plot_data = [1, 2, 3, 4, 5]  # replace with your plot data
            img = self.get_image_fromnd2(self.f, contrast=(0,30000))

            img_str = self.numpy_to_b64_string(img)
            plot_data = self.numpy_to_b64_string(np.zeros((1000,1000), dtype=np.uint8))
            return render_template("experiment_viewer.html", image=img_str, plot_data=plot_data, max_frame=self.f.sizes['t'], max_channel=self.f.sizes['c'], max_fov=self.f.sizes['v'], experiments=self.experiments)
        
        # @self.app.route('/simulations')
        # def simulations():
        #     # Render the 'simulations.html' template with any dynamic content
        #     return render_template('simulation_viewer.html')
        
        @self.app.route('/update_image', methods=['POST'])
        def update_image():
            # get index value from slider
            
            self.frame = int(request.form['frame'])
            self.channel = int(request.form['channel'])
            self.fov = int(request.form['fov'])
            
            contrast = [int(float(value)) for value in request.form['contrast'].split(',')]
        
            # retrieve image based on index
            img = self.get_image_fromnd2(self.f, t=self.frame, contrast=contrast, c=self.channel, v=self.fov)
            mask = self.masks[self.frame]
            img = self.get_masked_image(img, mask, self.mask_id)
            img_str = self.numpy_to_b64_string(img)
            # send image data to template to update image
            return {'image_data': img_str}

        @self.app.route('/update_fov', methods=['POST'])
        def update_fov():
            print('updating fov')
            # get index value from slider
            fov = int(request.form['fov'])
            self.fov = fov
            print(fov)
            self.load_masks(os.path.join(self.base_path, self.experiment, 'extraction'), fov)
            print('loading dataframe')
            self.load_df(os.path.join(self.base_path, self.experiment, 'extraction'), fov)
            print('done')
            return Response(status=200)

        @self.app.route('/update_experiment', methods=['POST'])
        def update_experiment():
            # get index value from slider
            self.experiment = request.form['experiment']
            print(f'updating experiment to {self.experiment}')
            fov=request.form['fov']
            #experiment = '230322_Ti2_FN-lanes_MDA_2min'
            self.experiment_data = pd.read_csv(os.path.join(self.base_path, self.experiment, 'Experiment_data.csv'))
            nd2_file = os.path.join(self.experiment_data.Path[0], self.experiment_data['time_lapse_file'][0])
            self.f = ND2Reader(nd2_file)
            print('max t', self.f.sizes['t'], 'max fov', self.f.sizes['v'])
            # img = self.get_image_fromnd2(self.f, contrast=(0,30000))
            # img_str = self.numpy_to_b64_string(img)
            self.load_masks(os.path.join(self.base_path, self.experiment, 'extraction'), fov)
            self.load_df(os.path.join(self.base_path, self.experiment, 'extraction'), fov)
            # send image data to template to update image
            return {'max_frame': self.f.sizes['t'], 'max_channel': self.f.sizes['c'], 'max_fov': self.f.sizes['v']}
            
        @self.app.route('/update_plot', methods=['POST'])
        def update_plot():
            # get index value from slider
            #frame = int(request.form['frame'])
            fov = self.fov
            if len(self.df)==0:
                print('no data')
                return Response(status=200)
            print('udpating plot')
            dfp = self.df[self.df.particle==self.particle_id]
            if 'segment' in dfp.columns:
                null = dfp.segment==0
                x0_array, x1_array = self.get_boundaries(null)
                shapes_data = [
                    {"type": "rect","x0": int(x0), "x1": int(x1), "y0": 0, "y1": self.img_height, "fillcolor": "rgb(100,100,100)", "opacity": 0.2, "line": {
              "width": 0
          }} 
                    for x0, x1 in zip(x0_array, x1_array)
                    ]
            else:
                shapes_data = '[]'
           
            data = {
                "time": dfp.frame.values.tolist(),
                "cell_front": dfp.front.values.tolist(),
                "cell_rear": dfp.rear.values.tolist(),
                "cell_nucleus": dfp.nucleus.values.tolist(),
                "shapes_data": shapes_data,
            }
            #print(dfp.front.values)
            return jsonify(data)
        
        @self.app.route('/update_particle_id', methods=['POST'])
        def update_particle_id():
            
            if len(self.df)<1:
                return Response(status=200)
            x=float(request.form['x'])
            y=float(request.form['y'])
            frame=int(request.form['frame'])
            print(f'updating particle id for {x}, {y}, {frame}')
            x, y = x*self.f.sizes['x'], y*self.f.sizes['y']
            
            self.mask_id = self.masks[self.frame, np.round(y).astype(int), np.round(x).astype(int)]
            
            if self.mask_id==0:
                #No mask was clicked on
                return Response(status=200)
            
            particle_id = self.df.loc[(self.df.frame==self.frame) & (self.df.cyto_locator==self.mask_id)].particle.values
            if len(particle_id)<1:
                print('No mask was clicked on, ', particle_id)
                return Response(status=200)
                #self.particle_id=self.particle_id[0]
            self.particle_id=particle_id[0]
            print('selected particle id: ', self.particle_id)
            return Response(status=200)


    def get_image_fromnd2(self, f, t=0,c=0, v=0, contrast=(0, 2**16)):
        # print('debug', type(t), type(c), type(v))
        img = f.get_frame_2D(t=t, c=c, v=v)
        img = (np.clip((img-contrast[0])/(contrast[1]-contrast[0]), 0, 1)*255).astype('uint8')
        return img

    def numpy_to_b64_string(self, image):
        rawBytes = BytesIO()
        im = Image.fromarray(image)
        im.save(rawBytes, format="JPEG")
        rawBytes.seek(0)
        image = base64.b64encode(rawBytes.getvalue())
        img_str = image.decode('utf-8')
        return img_str

    def load_masks(self, outpath, fov):
        try:
            print('Loading masks')
            path_to_mask = os.path.join(outpath, f'XY{fov}/cyto_masks.mp4')
            self.masks = functions.mp4_to_np(path_to_mask)
            print(f'Masks loaded for fov {fov} at {path_to_mask}')
        except FileNotFoundError:
            self.masks = np.zeros((self.f.sizes['t'], self.f.sizes['y'], self.f.sizes['x']), dtype=np.uint8)
            print(f'No masks found for this fov at {path_to_mask}')
        return 
    
    def load_df(self, outpath, fov):
        try:
            path_to_df = os.path.join(outpath, f'XY{fov}/clean_tracking_data.csv')
            self.df = pd.read_csv(path_to_df)
            print(f'Dataframe loaded for fov {fov} at {path_to_df}')
        except FileNotFoundError:
            self.df = pd.DataFrame()
            print(f'No dataframe found for this fov at {path_to_df}')
        return 

    def get_masked_image(self, img, mask, mask_id=0):
        img = np.stack([img, img, img], axis=-1)
        print('this is the mask_id', mask_id)
        outline = find_boundaries(mask>0, mode='outer')
        img[outline==1] = (255,0,0)
        if mask_id>0:
            outline_selected = find_boundaries(mask==mask_id, mode='outer')
            img[outline_selected==1] = (0,255,0)
        return img
    
    def get_boundaries(self, condition):
        diff = np.diff(condition.astype(int))
        if np.all(diff==0):
            return [], []

        left_boundaries = np.where(diff==1)[0]
        right_boundaries = np.where(diff==-1)[0]
        if len(left_boundaries)==0:
            return [0], right_boundaries
        elif len(right_boundaries)==0:
            return left_boundaries, [len(condition)]
        if left_boundaries.min()>right_boundaries.min():
            left_boundaries = np.concatenate([[0], left_boundaries])
        if left_boundaries.max()>right_boundaries.max():
            right_boundaries = np.concatenate([right_boundaries, [len(condition)]])

        return left_boundaries, right_boundaries

if __name__ == "__main__":
    base_path='/project/ag-moonraedler/MAtienza/cell_lines_paper/pipeline/'
    experiments = [directory for directory in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, directory)) and '.' not in directory]
    
    app = OnedcelltrackApp(experiments, base_path)
    app.app.run(debug=True, host='10.153.25.95', port=8899)
