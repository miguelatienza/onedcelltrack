"""Tools for analysis of cell tracks."""

import numpy as np
import pandas as pd
import os
from tqdm import tqdm


def join_dfs(paths):
    """Join dataframes from different experiments into one dataframe.

    Args:
        paths (list): List of paths to the dataframes to be joined.

    Returns:
        df (pd.DataFrame): Joined dataframe.
    """
    if isinstance(paths, str):
        paths = [paths]
    
    df_list = []
    for path in paths:
        fov_paths = [path for path in os.listdir(path) if 'XY' in path] 

        for fov_path in tqdm(fov_paths):
            fov = float(fov_path.split('XY')[-1])
            # Load the DataFrame
            try:
                df = pd.read_csv(os.path.join(path, fov_path, "clean_tracking_data.csv"), low_memory=False)
            except FileNotFoundError:
                continue
            #df = tracking.get_clean_tracks(df, min_length=3)
            # Add a column to the DataFrame indicating the field of view
            df['fov'] = fov
            df['experiment'] = path.split('/')[-1]
            # Append the DataFrame to the list
            df_list.append(df)

    # Join the DataFrames into a single DataFrame
    df = pd.concat(df_list, ignore_index=True)    
    return df


def create_full_df(paths, fpm=0.5):

    if isinstance(paths, str):
        paths = [paths]
    
    df = join_dfs(paths)
    df = df.sort_values(by=['experiment', 'fov', 'particle', 'segment', 'frame'])
    i=0
    df['unique_id'] = np.zeros(len(df))
    df['duration'] = np.zeros(len(df))
    exps = df['experiment'].unique()

    for exp in exps:
        fovs = df.loc[df['experiment']==exp, 'fov'].unique()
        for fov in tqdm(fovs):
            particles = df.loc[(df['experiment']==exp) & (df['fov']==fov), 'particle'].unique()
            for particle in particles:
                segments = df.loc[(df['experiment']==exp) & (df['fov']==fov) & (df['particle']==particle), 'segment'].unique()
                segments = segments[segments>0]
                for segment in segments:
                    df.loc[(df['experiment']==exp) & (df['fov']==fov) & (df['particle']==particle) & (df['segment']==segment), 'unique_id'] = i
                    frames = df.loc[(df['experiment']==exp) & (df['fov']==fov) & (df['particle']==particle) & (df['segment']==segment), 'frame']
                    duration = (frames.max() - frames.min())/fpm
                    df.loc[(df['experiment']==exp) & (df['fov']==fov) & (df['particle']==particle) & (df['segment']==segment), 'duration'] = duration
                    i+=1
        break            
    return df

class Onedcelldata:
    """Class for analysis of cell tracks."""
    def __init__(self, df):
        self.df = df

    def sample_trajectories(self, n_trajectories):
        """Sample n_trajectories random trajectories."""
        fovs = np.random.choice(self.df['fov'].unique(), n_trajectories)
        df_sample = self.df[self.df['fov'].isin(fovs)]

        ids = np.random.choice(self.df[''].unique(), n_trajectories)
        return self.df[self.df['id'].isin(ids)]
