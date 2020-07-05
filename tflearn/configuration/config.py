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

import argparse

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
misc_arg.add_argument('--data_dir', type=str, default='./dataset',
                        help='Directory where data is stored')
misc_arg.add_argument('--output_dir', type=str, default='./output',
                        help='Directory where output is stored')
misc_arg.add_argument('--plot_dir', type=str, default='./plots',
                        help='Directory where plots are stored')
misc_arg.add_argument('--model_dir', type=str, default='./model',
                        help='Directory where the trained model is stored')


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed