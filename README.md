# Onedcelltrack

![image](https://github.com/miguelatienza/onedcelltrack/pipeline_summary.png)

Python project for 1D automated single-cell migration assay. Pipeline which takes in a set of raw microscopy imaging data and creates 1D tracks for each detected cell's nucleus, rear and front position. The cells are first segmented using [cellpose](https://github.com/MouseLand/cellpose) and the fluorescently labeled nuclei are tracked using [trackpy](https://github.com/soft-matter/trackpy). 

## Installation

`conda create -n onedcelltrack python=3.8` <br />
Install onedcelltrack <br />
`python -m pip install git +https://github.com/miguelatienza/onedcelltrack` <br />

Activate the conda environment <br />
`conda activate onedcelltrack` <br />

## Data Processing
Copy the notebook `templates/run_full_experiment.ipynb` to your working directory of choice.
Activate the environment and run jupyterlab : <br />
`conda activate onedcelltrack` <br />
`jupyter-lab` <br />
Fill in the notebook and run it.

## Data Visualization
There are two possible ways of viewing the data. Either in a jupyter notebook or as a website.
### Jupyter Notebook 
Copy the Notebook `templates/view_results.ipynb` to your working directory and view your results.





