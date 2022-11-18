# Celltracker

Python project for 1D automated single-cell migration assay. Pipeline which takes in a set of raw microscopy imaging data and creates 1D tracks for each detected cell's nucleus, rear and front position. 

## Installation

`conda create -n onedcelltrack python=3.8`
`pip install git +https://github.com/miguelatienza/onedcelltrack`

You can skip this part and use the pre-installed version instead.

Clone the repository
```
git clone https://gitlab.physik.uni-muenchen.de/Miguel.Atienza/celltracker.git
```
Create a new virtualenv using the provided environment.yml file
```
conda env create -f {path_to_repository}/environment.yml
```

## Pre-installed version on imaging computer
SSH into the workstation either from windows powershell or a linux terminal
```
ssh {username}@lsr-ws-imaging1.roentgen.physik.uni-muenchen.de 
```

Activate my conda environment
```
source /scratch-local/miguel/miniconda/bin/activate
conda activate cellpose
```
When using this for the first time, you need to make the environment visible to jupyter-lab:
```
python -m ipykernel install --user --name celltracker
```
Run a jupyter-lab instance from the base directory
```
cd /
jupyter-lab --no-browser
```
You should find a a link to your jupyterlab server such as:
http://localhost:8888/..., 8888 being the port number in this case
Open a new powershell or terminal and forward the imaging computer server to your own computer
```
ssh -N -f -L localhost:{portnumber}:localhost:{port number} {username}@lsr-ws-imaging1.roentgen.physik.uni-muenchen.de
```
Open the link on any browser

## Usage
Copy the notebook templates/run_full_experiment.ipynb to your working directory of choice.
Open the notebook from jupyterlab. At the top right of the notebook window you should see an option to chose your kernel (to the right of the bug icon). Click on this and select celltracker. 
Fill in the notebook and run it.

Copy the Notebook templates/view_results.ipynb to your working directory and view your results.





