from julia.api import Julia
import os
import numpy as np

def test_julia():
    JULIA_PATH = "/project/ag-moonraedler/MAtienza/cellsbi/envs/sbi/julia-1.6.7/bin/julia"
    PATH_TO_JULIA_SCRIPTS= "/project/ls-raedler/software/onedcellsim/onedcellsim/simulations/"

    jl = Julia(runtime=JULIA_PATH, compiled_modules=False)
    julia_simulate_file = os.path.join(PATH_TO_JULIA_SCRIPTS, "simulate.jl")

    simulate = jl.eval(f"""
        include("{julia_simulate_file}")""")

    data = simulate() 
    assert type(data) == np.ndarray