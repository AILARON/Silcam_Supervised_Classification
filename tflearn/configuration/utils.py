#################################################################################################################
# The Pytorch supervised classifier module
# Implementation of the utils
# Author: Aya Saad
# email: aya.saad@ntnu.no
#
# Date created: 29 September 2019
#
# Project: AILARON
# Contact
# email: annette.stahl@ntnu.no
# funded by RCN IKTPLUSS program (project number 262741) and supported by NTNU AMOS
# Copyright @NTNU 2020
#######################################


import os

def str2bool(v):
    return v.lower() in ('true', '1')

def prepare_dirs(config):
    for path in [config.model_dir, config.output_dir, config.plot_dir]:
        if not os.path.exists(path):
            os.makedirs(path)
