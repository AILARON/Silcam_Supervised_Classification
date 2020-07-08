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
# funded by RCN IKTPLUSS program (project number 262701) and supported by NTNU AMOS
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

# Interpolating the stats from the export to sensors' data on the timestamp field
# returns df
def df_interpolate(np1, np2, featurename):
    ''' numpy interpolation
    Args:
    @:param np1: numpy from the stat file.
    @:param np2: numpy from the neptus logs.
    @:return df: new numpy combined
    '''
    df = np1
    df[featurename] = np.interp(np.float64(df['timestamp']),
                                np.float64(np2['timestamp']),
                                np2[featurename])

    return df

#### extract middle
### returns stats
def extract_middle(stats, crop_bounds):
    '''
    Temporary cropping solution due to small window in AUV
    Args:
    @:param stats (df)    : silcam stats file
    @:param crop_bounds (tuple) : 4-tuple of lower-left then upper-right coord of crop
    Returns:
    @:return stats (df)    : cropped silcam stats file
    '''
    # print('initial stats length:', len(stats))
    r = np.array(((stats['maxr'] - stats['minr']) / 2) + stats['minr'])
    c = np.array(((stats['maxc'] - stats['minc']) / 2) + stats['minc'])

    points = []
    for i in range(len(c)):
        points.append([(r[i], c[i])])

    pts = np.array(points)
    pts = pts.squeeze()
    ll = np.array(crop_bounds[:2])  # lower-left
    ur = np.array(crop_bounds[2:])  # upper-right
    inidx = np.all(np.logical_and(ll <= pts, pts <= ur), axis=1)
    stats = stats[inidx]
    return stats

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

### Convert a list of silc files to bmp images
def silc_to_bmp(filenames, indir, outdir):
    '''
    Convert a list of silc files to bmp images
    Args:
    :param filenames: list of filenames
    :param indir: input directory
    :param outdir: output directory
    :return:
    '''

    # files = [s for s in os.listdir(directory) if s.endswith('.silc')]
    try:
        os.mkdir(outdir)
    except OSError:
        print("Creation of the directory %s failed" % outdir)
    else:
        print("Successfully created the directory %s " % outdir)


    for f in filenames:
        try:
            with open(os.path.join(indir, f + '.silc'), 'rb') as fh:
                im = np.load(fh, allow_pickle=False)
                #fout = os.path.splitext(f)[0] + '.bmp'
            outname = os.path.join(outdir, f  + '.bmp')
            imo.imwrite(outname, im)
        except:
            print('{0} failed!'.format(f))
            continue

    print('conversion completed!')

### returns an image name from the export name string in the -STATS.csv file
def export_name2im(exportname, path):
    '''
    returns an image from the export name string in the -STATS.csv file
    get the exportname like this: exportname = stats['export name'].values[0]
    Args:
    :param exportname: string containing the name of the exported particle e.g. stats['export name'].values[0]
    :param path: path to exported h5 files
    Returns:
    :return: im : particle ROI image
    '''
    print('inside export_name2im', exportname)
    # the particle number is defined after the time info
    pn = exportname.split('-')[1]
    # the name is the first bit
    name = exportname.split('-')[0] + '.h5'
    # combine the name with the location of the exported HDF5 files
    fullname = os.path.join(path, name)
    # open the H5 file
    fh = h5py.File(fullname ,'r')
    # extract the particle image of interest
    im = fh[pn]

    return im

### extract the roi and save them into the extract directory
### returns df_extract that contains information on the actual saved roi
def extract_to_roi(stat, stat_comb, roidir, extdir, msize=40, minimgsize=1000):
    '''
    extract the roi and save them into the extract directory
    Args:
    :param stat: stat dataframe from the export name
    :param stat_comb: stat dataframe which includes the depth and information from neptus logs
    :param roidir: path to the export directory where the .h5 files are saved
    :param extdir: path to the output directory where the extracted images of the objects are saved
    :param msize: minimum length/width size of the extracted object in pixels
    :param minimgsize: minimum image size acceptable for labeling in bytes
    :return:
    '''
    df_extract = pd.DataFrame([])
    r = 0
    for f in stat_comb['file name'].tolist():
        dfe = stat[stat['export name'].str.contains(f)]
        name = dfe['export name'].tolist()[0]
        particle_image = export_name2im(name, roidir)
        # measure the size of this image
        [height, width] = np.shape(particle_image[:, :, 0])
        # sanity-check on the particle image size
        if (height <= msize) and (width <= msize):
            continue
        copy_to_path = os.path.join(extdir, name)
        file = copy_to_path + '.tiff'
        skio.imsave(file, particle_image)
        if del_if_small(file, minimgsize):
            continue
        if r == 0:
            df_extract = dfe
            r = 1
        else:
            df_extract = df_extract.append(dfe)
    return df_extract

### save the list of objects into a list of image files
def objects_to_file(df, inpath, outpath):
    '''
    save the object into an image file
    Args:
    :param df:   pandas dataframe containing the export name
    :param inpath: path to the export directory where the .h5 files are saved
    :param outpath: path to the output directory where the extracted images of the objects are saved
    :return:
    '''
    for n in df['export name'].tolist():
        im = export_name2im(n, inpath)
        copy_to_path = os.path.join(outpath,n)
        skio.imsave(copy_to_path + '.tiff', im)
    print('export completed!')

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