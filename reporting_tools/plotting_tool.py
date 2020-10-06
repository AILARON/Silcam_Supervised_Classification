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
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.io.img_tiles as cimgt

### MAP plotting
def map_plot(gps, request, feature, color):
    ax = plt.gca()
    gl = ax.gridlines(draw_labels=True)
    gl.xlabels_top = gl.ylabels_right = False
    gl.xlocator = mticker.FixedLocator(np.arange(9, 11, 0.05))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)

    ax.set_extent([min(gps['lon'] - 0.004), max(gps['lon'] + 0.004),
                   min(gps['lat'] - 0.003), max(gps['lat'] + 0.003)])

    mp = ax.scatter(gps['lon'], gps['lat'],
                    c=gps[feature], cmap=color, transform=ccrs.Geodetic())
    plt.setp(ax.get_xticklabels(), fontsize=14)
    plt.setp(ax.get_yticklabels(), fontsize=14)

    ax.add_image(request, 12)
    return mp
