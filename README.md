# Onedcelltrack

<img src="https://github.com/miguelatienza/onedcelltrack/blob/main/pipeline_summary.png" width="800">

Python project for 1D automated single-cell migration assay. Pipeline which takes in a set of raw microscopy imaging data and creates 1D tracks for each detected cell's nucleus, rear and front position. The cells are first segmented using [cellpose](https://github.com/MouseLand/cellpose) and the fluorescently labeled nuclei are tracked using [trackpy](https://github.com/soft-matter/trackpy). The project also provides two different tools for visualising the data. Either in a jupyter notebook or as a website.

## Installation

`conda create -n onedcelltrack python=3.8` <br />
Install onedcelltrack <br />
`python -m pip install git +https://github.com/miguelatienza/onedcelltrack` <br />

Activate the conda environment <br />
`conda activate onedcelltrack` <br />

## Running the pipeline
Copy the notebook `templates/run_full_experiment.ipynb` to your working directory of choice.
Activate the environment and run jupyterlab : <br />
`conda activate onedcelltrack` <br />
`jupyter-lab` <br />
Fill in the notebook and run it.

## Data visualisation in a Jupyter Notebook 
<img src="https://github.com/miguelatienza/onedcelltrack/blob/main/viewer.png" width="800">

Copy the Notebook `templates/view_results.ipynb` to your working directory and view your results.

## Data visualisation in a website
The website provides a more user friendly interface to interact with the microscopy data and resulting cell trajectories. It can be run on a single host server where the (typically large) data is stored and accessed by users from their own computer without the large memory requirements. To run the website:

1. Activate the environment
2. cd into the location of the project
3. run `python onedcelltrack/onedcelltrack/webapp/app.py`








