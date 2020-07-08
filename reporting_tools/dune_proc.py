#################################################################################################################
# The Reporting tool from neptus logs
# Implementation of the dune proc
# Author: Andreas Våge
# email: andreas.vage@ntnu.no
#
# Date created: June 2020
#
# Project: AILARON
# Contact
# email: annette.stahl@ntnu.no
# funded by RCN IKTPLUSS program (project number 262701) and supported by NTNU AMOS
# Copyright @NTNU 2020
#######################################
# neptus logs library (from Andreas Våge repository)
#from dune_proc import load_neptus_csvs
import numpy as np
import pandas as pd
from pathlib import Path

def proc_watervel(df):
    if (df['validity (bitfield)'].dtype == 'object'):
        valid = ' VEL_X|VEL_Y|VEL_Z'
    else:
        valid = 0b0111
    df.loc[df['validity (bitfield)'] != valid, -3:] = np.nan
    df.drop(columns='validity (bitfield)', inplace=True)
    new_names = [(i, 'wv_' + i) for i in df.iloc[:, -3:].columns.values]
    df.rename(columns=dict(new_names), inplace=True)
    return df


def proc_eststate(df):
    df['lon (rad)'] = df['lon (rad)'] * (180 / np.pi)
    df['lat (rad)'] = df['lat (rad)'] * (180 / np.pi)
    df.rename(columns={'lon (rad)': 'lon (°)',
                       'lat (rad)': 'lat (°)'}, inplace=True)
    return df


special_data_proc = {
    'WaterVelocity': proc_watervel,
    'EstimatedState': proc_eststate,
    'Temperature': (lambda df: df[df.entity == ' SmartX']),
    'Pressure': (lambda df: df[df.entity == ' SmartX']),
    #'Acceleration': (lambda df: df.add_prefix('acc_')),
}


def proc_dune_df(name, df):
    df.columns = df.columns.str.strip()
    df.timestamp = pd.to_datetime(df.timestamp, utc=True, unit='s')
    df = df.loc[~df.timestamp.duplicated(keep='first')]
    df.set_index('timestamp', verify_integrity=True, inplace=True)
    if name in special_data_proc.keys():
        df = special_data_proc[name](df)
    if df.entity.unique().size > 1:
        print("Warning: ", name, " has multiple sources:", df.entity.unique())
    df.drop(columns=['system', 'entity'], inplace=True)
    df.columns, units = zip(*df.columns.str.split(' '))
    df.rename(columns={"value": name}, inplace=True)
    return df, units


def load_neptus_csvs(path):
    # -*- coding : utf-8 -*-
    # coding: utf-8
    local_encoding = 'cp850'  # adapt for other encodings
    try:
        #print(path)
        path = Path(path)
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
        files = [f for f in path.glob('*.csv')]
    else:
        print('Path must be .csv or directory')
        return None, None
    if not files:
        print("Using last edited subfolder")
        _, file = max((f.stat().st_ctime, f) for f in path.rglob('*.csv'))
        print("Date: ", file.parents[3].name)
        print("Plan: ", file.parents[2].name)
        files = [f for f in file.parent.glob('*.csv')]
    if not files:
        print("No csv found in directory or subfolders")
        return None, None
    mdf = None
    units = {}
    tol = pd.Timedelta('250ms')
    for f in files:
        name = f.stem
        df = pd.read_csv(f, sep=',', encoding='gbk')  # encoding= 'unicode_escape' 'latin1'
        df, unit = proc_dune_df(name, df)
        units.update(zip(df.columns, unit))
        if mdf is None:
            mdf = df
        else:
            mdf = pd.merge_asof(mdf, df, on='timestamp', direction='nearest', tolerance=tol)
    mdf.set_index('timestamp', verify_integrity=True, inplace=True)
    return mdf, units

# @Aya Saad added on 7 July 2020
# load mission list if more than one mission on the same day
def load_mission_list(mission_list, neptus_dir, csv_path="/mra/csv"):
    '''
    
    :param mission_list: the list of plans created in neptus
    :param neptus_dir: neptus logs directory
    :param csv_path: path to exported csv default = "/mra/csv"
    :return: merged df
    '''
    data = {}
    df = pd.DataFrame([])
    r = 0
    for m in mission_list:
        logdir = Path(neptus_dir + m + csv_path)
        df_dune, units = load_neptus_csvs(logdir)
        df2 = df_dune
        if r == 0:
            df = df2
            r = 1
        else:
            df = df.append(df2, sort=True)

    return df