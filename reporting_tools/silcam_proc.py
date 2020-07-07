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
from pathlib import Path
# -*- coding: utf-8 -*-
local_encoding = 'cp850'  # adapt for other encodings

# Interpolating the stats from the export to sensors' data on the timestamp field
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
def extract_middle(stats, crop_bounds):
    '''
    Temporary cropping solution due to small window in AUV
    Args:
        stats (df)    : silcam stats file
        crop_bounds (tuple) : 4-tuple of lower-left then upper-right coord of crop
    Returns:
        stats (df)    : cropped silcam stats file
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


def stat_combine(df):
    # combine df per timestamp
    #belongs = df.filter(like='probability') >= lim
    comb = df.groupby('timestamp').first() # [#/L]
    comb = comb.tz_localize('UTC')
    return comb


def silc_to_bmp(filenames, indir, outdir):
    '''Convert a list of silc files to bmp images
    Args:
        filenames:  list of filenames
        indir:      input directory
        outdir:     output directory
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