from flask import Flask, render_template, request, jsonify, Response, Blueprint
from matplotlib import pyplot as plt
import os
import numpy as np
import pandas as pd
import json
from julia.api import Julia

JULIA_PATH = "/project/ag-moonraedler/MAtienza/cellsbi/envs/sbi/julia-1.6.7/bin/julia"
PATH_TO_JULIA_SCRIPTS= "/project/ls-raedler/software/onedcellsim/onedcellsim/simulations/"
# jl = Julia(runtime=JULIA_PATH, compiled_modules=False)
# julia_simulate_file = os.path.join(PATH_TO_JULIA_SCRIPTS, "simulate.jl")

# SIMULATE = jl.eval(f"""
#     include("{julia_simulate_file}")""")

PARAMETERS ={
        'E': [7e-5, 5.5e-4, 3e-3],
        'L_0': [10, 30, 50],
        'V_e0': [1e-2, 6.05e-2, 1.1e-1],
        'k_minus': [0,0,0],
        'c_1': [1.5e-4, 1.5e-4, 1.5e-4],
        'c_2': [7.5e-5, 7.5e-5, 7.5e-5],
        'c_3': [7.8e-3, 7.8e-3, 7.8e-3],
        'kappa_max': [35, 35, 35],
        'K_kappa': [20, 20, 20],
        'n_kappa': [7.6, 7.6, 7.6],
        'kappa_0': [1e-2, 1e-2, 1e-2],
        'zeta_max': [1.4, 1.4, 1.4],
        'K_zeta': [32, 32, 32],
        'n_zeta': [5.8, 5.8, 5.8],
        'b': [3, 3, 3],
        'zeta_0': [0, 5e-2, 1e-1],
        'aoverN': [1,1,1],
        'epsilon': [0, 1, 3],
        'B': [5, 37.5, 70],
    }

class SimulationsApp:
    def __init__(self):
        #self.app = Flask(__name__)
        self.blueprint = Blueprint('my_blueprint', __name__)
        self.julia_initialized = False
        
        @self.blueprint.route('/simulations')
        def simulations():
            #self.init_julia()
            # load simulation data
            #data = run_simulation([7e-5, 10, 1e-2, 0, 1.5e-4, 7.5e-5, 7.8e-3, 35, 20, 7.6, 1e-2, 1.4, 32, 5.8, 3, 0, 1, 0, 5])
            #json_data = data.to_json(orient='records')
            # render template with simulation data
            return render_template('simulation_viewer.html', parameters=PARAMETERS)
            
        def register(self, app):
            app.reister_blueprint(self.blueprint)

        @self.blueprint.route('/run_simulation', methods=['POST'])
        def run_simulation():
            params = request.form['params']
            params= params.split(',')
            #params = request.json['params']
            #params = json.loads(request.form['params'])
            #print(params)
            params = [float(param) for param in params]
            params = params[:-3]+[4e-2]+params[-3:]
            full_params = np.array(params)
            #print(full_params)
            DURATION=5
            t_max, t_step = DURATION*60*60,30
            t_step_compute=0.5
            #params, particle_id, verbose, t_max, t_step, t_step_compute = args
            print('heeeey')
            #np.random.seed(int(time.time())
            #variables = self.simulate(parameters=params, t_max=3600*5, t_step=30, t_step_compute=0.5, delta=0, kf0=0)[0]
            variables = self.simulate()
            #print(variables)
            # variables = np.ones((100, 16))
            # variables[:, 1] = np.linspace(0, 100, 100)
            #print(variables.shape)
            #print(variables)
            # df = pd.DataFrame(columns=['t', 'xc', 'xb', 'xf', 'kf', 'kb', 'vrf', 'vrb'])
            #ids = np.ones(length, dtype='int')*particle_id
            data ={'t': (variables[0, :, 1]/3600).tolist(),
            'xf':variables[0, :, 11].tolist(), 
            'xb':list(variables[0, :,12].tolist()),
            'xc':list(variables[0, :,13].tolist()),
            'kf':list(variables[0,:, 5].tolist()),
            'kb':list(variables[0,:,6].tolist()),
            'vrf':list(variables[0,:,9].tolist()),
            'vrb':list(variables[0,:,10].tolist()),
            'vf':list(variables[0,:,14].tolist()),
            'vb':list(variables[0,:,15].tolist()),
            }
            #print(data['t'])
            #df=pd.concat([df, data], ignore_index=True)
            #print(df)
            #print(data)
            return jsonify(data)
    
        @self.blueprint.route('/update_simulation', methods=['POST'])
        def update_simulation():
            print(request.form)
            params = request.form['params']
            params= params.split(',')
            #params = request.json['params']
            #params = json.loads(request.form['params'])
            #print(params)
            params = [float(param) for param in params]
            params = params[:-3]+[4e-2]+params[-3:]
            full_params = np.array(params)
            #print(full_params)
            DURATION=5
            t_max, t_step = DURATION*60*60,30
            t_step_compute=0.5
            #params, particle_id, verbose, t_max, t_step, t_step_compute = args
            print('heeeey')
            #np.random.seed(int(time.time())
            variables = self.simulate()
            print('we have variables')
            #variables = self.data
            #print(variables)
            # variables = np.ones((100, 16))
            # variables[:, 1] = np.linspace(0, 100, 100)
            #print(variables.shape)
            #print(variables)
            # df = pd.DataFrame(columns=['t', 'xc', 'xb', 'xf', 'kf', 'kb', 'vrf', 'vrb'])
            #ids = np.ones(length, dtype='int')*particle_id
            data ={'t': (variables[0, :, 1]/3600).tolist(),
            'xf':variables[0, :, 11].tolist(), 
            'xb':list(variables[0, :,12].tolist()),
            'xc':list(variables[0, :,13].tolist()),
            'kf':list(variables[0,:, 5].tolist()),
            'kb':list(variables[0,:,6].tolist()),
            'vrf':list(variables[0,:,9].tolist()),
            'vrb':list(variables[0,:,10].tolist()),
            'vf':list(variables[0,:,14].tolist()),
            'vb':list(variables[0,:,15].tolist()),
            }
            #print(data['t'])
            #df=pd.concat([df, data], ignore_index=True)
            #print(df)
            #print(data)
            return jsonify(data)
    
    def init_julia(self):
        print('init julia')
        self.data = simulate()
        return
        if self.julia_initialized:
            return

        self.jl = Julia(runtime=JULIA_PATH, compiled_modules=False)
        julia_simulate_file = os.path.join(PATH_TO_JULIA_SCRIPTS, "simulate.jl")

        simulate = self.jl.eval(f"""
            include("{julia_simulate_file}")""")
        self.julia_initialized = True
        self.data=simulate()
        print('simulated')
        #print(data)
        return

    def simulate(self):
        print('here')
        print(JULIA_PATH)
        jl = Julia(runtime=JULIA_PATH, compiled_modules=False)
        julia_simulate_file = os.path.join(PATH_TO_JULIA_SCRIPTS, "simulate.jl")

        simulate = jl.eval(f"""
            include("{julia_simulate_file}")""")
        print('simulating')
        data=simulate()
        return data
        
   