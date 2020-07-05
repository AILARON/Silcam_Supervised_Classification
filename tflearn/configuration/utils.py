#################################################################################################################
# A Modularized implementation for
# Image enhancement, segmentation, classification, evaluation and visualization
# Integrating all the algorithms stated above in one framework
# CONFIGURATION - LOAD_DATA - IMAGE_ENHANCEMENTS - IMAGE_SEGMENTATION -
# CLASS_BALANCING - FEATURE_IDENTIFICATION - CLASSIFICATION - EVALUATION_VISUALIZATION
# Author: Aya Saad
# Date created: 29 September 2019
# Project: AILARON
# funded by RCN FRINATEK IKTPLUSS program (project number 262701) and supported by NTNU AMOS
#
#################################################################################################################

import os

def str2bool(v):
    return v.lower() in ('true', '1')

def prepare_dirs(config):
    for path in [config.model_dir, config.output_dir, config.plot_dir]:
        if not os.path.exists(path):
            os.makedirs(path)