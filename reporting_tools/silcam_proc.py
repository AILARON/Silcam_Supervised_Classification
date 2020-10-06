#################################################################################################################
# The Pytorch tool module
# Implementation of the dataloader
# Author: Aya Saad
# email: aya.saad@ntnu.no
#
# Date created: 6 April 2020
#
# Project: AILARON
# Contact
# email: annette.stahl@ntnu.no
# funded by RCN IKTPLUSS program (project number 262741) and supported by NTNU AMOS
# Copyright @NTNU 2020
#######################################
import numpy as np
import os
import imageio as imo
import h5py
import pandas as pd
from pathlib import Path
import skimage.io as skio

# -*- coding: utf-8 -*-
local_encoding = 'cp850'  # adapt for other encodings

### load silcam data from the list of exported files
### returns stat
def load_silcam_data(files, path):
    '''
    load silcam data from the list of exported files
    Args:
    @:param files:  filelist
    @:param path:   path
    Returns:
    @:return df: silcam stats file
    '''
    stat = pd.DataFrame([])
    r = 0
    for f in files:
        df = pd.read_csv(path + files[0], parse_dates=['timestamp'])
        df = df[df['export name'] != 'not_exported']
        stat = stat.append(df, sort=True)
        stat = stat.reset_index()
        stat.set_index('timestamp', inplace=True)
        stat = stat.tz_localize('UTC')
        stat = stat.reset_index()
        stat.set_index('timestamp', inplace=True)
        stat.dropna()
    return stat

### create the field of the highest probability
### returns stat
def make_highest_prob(stat):
    #### HIGHEST PROBABILTY
    stat['highest prob'] = stat[['probability_copepod',
                                 'probability_diatom_chain',
                                 'probability_faecal_pellets',
                                 'probability_bubble',
                                 'probability_other']].max(axis=1)  # with the highest prob - 61 copepod
    # ,'probability_oily_gas','probability_oil', , 'probability_bubble' , 'probability_other',
    #                             'probability_faecal_pellets' ,
    #                                  'probability_other'
    return stat

### make the probability assignment to classes
### and create the field of highest probability
### returns the stat updated
def make_prob(stat):
    sum_prob = stat['probability_copepod'] \
               + stat['probability_diatom_chain'] + stat['probability_other'] \
               + stat['probability_faecal_pellets'] + stat['probability_bubble']\
               + stat['probability_oily_gas'] + stat['probability_oil']

    stat['copepod'] = stat['probability_copepod'] / sum_prob
    stat['diatom_chain'] = stat['probability_diatom_chain'] / sum_prob
    stat['other'] = stat['probability_other'] / sum_prob
    stat['faecal_pellets'] = stat['probability_faecal_pellets'] / sum_prob
    stat['bubble'] = stat['probability_bubble'] / sum_prob
    stat['oily_gas'] = stat['probability_oily_gas'] / sum_prob
    stat['oil'] = stat['probability_oil'] / sum_prob
    stat.drop(columns=['probability_copepod',
                       'probability_diatom_chain',
                       'probability_other',
                       'probability_faecal_pellets',
                       'probability_bubble',
                       'probability_oily_gas',
                       'probability_oil'], inplace=True)
    stat.rename(columns={"copepod": "probability_copepod",
                         "diatom_chain": "probability_diatom_chain",
                         "faecal_pellets": "probability_faecal_pellets",
                         "bubble": "probability_bubble",
                         "oily_gas": "probability_oily_gas",
                         "oil": "probability_oil",
                         "other": "probability_other"}, inplace=True)
    stat['highest prob'] = stat[['probability_copepod', 'probability_diatom_chain',
                                 'probability_other', 'probability_faecal_pellets',
                                 'probability_bubble']].max(axis=1)  # with the highest prob - 61 copepod
    #   # 'oily_gas', 'oil'
    print('shape with highest prob ', stat.shape)
    return stat

### group df by timestamp
### returns the dataframe grouped by timestamp
def stat_grouped(df):
    '''
    group df by timestamp
    :param df: dataframe
    :return: dataframe grouped by timestamp
    '''
    #belongs = df.filter(like='probability') >= lim
    comb = df.groupby('timestamp').first() # [#/L]
    #comb = comb.tz_localize('UTC')
    return comb

### delete an image file if it is less that a minimum size
### returns if deleted when small
def del_if_small(file, minsize):
    '''
    delete an image file if it is less that a minimum size
    Args:
    :param file:   name of the file
    :param minsize:    minimum size
    :return: True if deleted / False if not
    '''
    size = os.path.getsize(file)
    if size < minsize:
        os.remove(file)
        return True
    return False

### delete small files in a directory less that a minimum size
def del_small_files(pathname, minsize):
    '''
    delete small files in a directory less that a minimum size
    Args:
    :param pathname:   pathname to files
    :param minsize:    minimum size
    :return:
    '''
    try:
        # print(path)
        path = Path(pathname)
    except TypeError as e:
        print('TypeError: path', e)
        print('path = ', path)
        return None, None
    if not path.exists():
        print('Path does not exists')
        return None, None
    if path.suffix == '.csv':
        files = [path]
    elif path.is_dir():
        files = [f for f in path.glob('*.tiff')]
    else:
        print('Path must be .csv or directory')
        return None, None

    for f in files:
        size = os.path.getsize(pathname + f.stem + '.tiff') #os.stat(f).st_size
        print('file size', f.stem, size)
        if size < minsize:
            os.remove(pathname + f.stem + '.tiff')

### associate speed with raw images
def get_raw_speed(stat_comb, scenes, speed_int):
    '''

    :param stat_comb: stat file contains the raw images and the neptus information
    :param speed_int: interval of speeds we need to report
    :return: new_stat the images associated with the depth, Rpm and water velocity
    '''
    new_stat = stat_comb.loc[sum([stat_comb['export name'].str.contains(f) for f in scenes]) > 0]
    new_stat = new_stat[['file name', 'depth','wv_x', 'Rpm']]
    new_stat = new_stat[['file name', 'depth','wv_x', 'Rpm']]
    new_stat = new_stat[(new_stat['Rpm']>= speed_int[0])]
    new_stat = new_stat[(new_stat['Rpm']<= speed_int[1])]
    return new_stat


### to be continued
### combines the stat of extracted objects and assign corresponding information from the stat file
def stat_populate(stat, stat_comb, flist):
    # populate the depth, density, temperature, etc. to the stat
    for f in flist:
        print(f)
        stat[stat['export name'].str.contains(f)][['AirSaturation', 'Chlorophyll',
                                                   'Conductivity', 'DissolvedOxygen',
                                                   'lat', 'lon', 'height', 'x', 'y', 'z',
                                                   'phi', 'theta', 'psi', 'u', 'v',
                                                   'w', 'vx', 'vy', 'vz',
                                                   'p', 'q', 'r', 'depth', 'alt',
                                                   'Pressure', 'Rpm', 'Salinity', 'Temperature',
                                                   'Turbidity', 'WaterDensity',
                                                   'wv_x', 'wv_y', 'wv_z']] = \
            stat_comb[(stat_comb['file name'] == f)][['AirSaturation', 'Chlorophyll',
                                                      'Conductivity', 'DissolvedOxygen',
                                                      'lat', 'lon', 'height', 'x', 'y', 'z',
                                                      'phi', 'theta', 'psi', 'u', 'v',
                                                      'w', 'vx', 'vy', 'vz',
                                                      'p', 'q', 'r', 'depth', 'alt',
                                                      'Pressure', 'Rpm', 'Salinity', 'Temperature',
                                                      'Turbidity', 'WaterDensity',
                                                      'wv_x', 'wv_y', 'wv_z'
                                                      ]]
    return stat