#################################################################################################################
# The Pytorch neural network architecture module
# Implementation of the configuration
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

import argparse
import os
arg_lists = []
parser = argparse.ArgumentParser(description='Image Classification')

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg



# mode param
mode_arg = add_argument_group('Setup')
mode_arg.add_argument('--num_samples', type=int, default=10000,
                            help='# of samples to compute embeddings on. Becomes slow if very high.')
mode_arg.add_argument('--num_dimensions', type=int, default=2,
                            help='# of tsne dimensions. Can be 2 or 3.')
mode_arg.add_argument('--random_seed', type=int, default=42,
                        help='Seed to ensure reproducibility')


# path params
misc_arg = add_argument_group('Path Params')
misc_arg.add_argument('--data_dir', type=str, default='C:/Users/ayas/Projects/AILARON/Implementation/PySilCam/pysilcam-testdata/unittest-data/silcam_classification_database',
                        help='Directory where data is stored')
misc_arg.add_argument('--output_dir', type=str, default='D:/resnet_test/model',
                        help='Directory where output is stored')
misc_arg.add_argument('--plot_dir', type=str, default='./plots',
                        help='Directory where plots are stored')
misc_arg.add_argument('--model_dir', type=str, default='D:/resnet_test/model',
                        help='Directory where the trained model is stored')


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed

def prepare_dirs(config):
    for path in [config.model_dir, config.output_dir, config.plot_dir]:
        if not os.path.exists(path):
            os.makedirs(path)