import os
import numpy as np
import tensorflow as tf

from tflearn.data_utils import image_preloader  # shuffle,
from statistics import mean,stdev
from load_data.load_dataset import DatasetLoader
from DNN.net import *   # Network architectures
import h5py
import tflearn

class TRAIN():
    def __init__(self, network):
        '''
        :network DNN architecture
        '''
        self.network = network
        pass

    def load_data(self, dataset_loader):
        trainX, trainY = dataset_loader.load_hd5_data(type='train', input_width='', round_num='')
        testX, testY = dataset_loader.load_hd5_data(type='test', input_width='', round_num='')
        return trainX, trainY, testX, testY

    def train(self, model_path, trainX, trainY, testX, testY,
              round_num, n_epoch, batch_size):
        pass

    pass

class TRAIN_TFLEARN(TRAIN):

    def train(self, model_path, trainX, trainY, testX, testY,
              round_num, n_epoch, batch_size):
        tf.reset_default_graph()
        model_file = os.path.join(model_path, '/plankton-classifier.tfl')
        model, conv_arr = self.network.build_model(model_file)

        model_name = model_path + '/plankton-classifier'
        print('model_name ', model_file)
        self.network.train(model, trainX, trainY, testX, testY,
                           round_num, n_epoch, batch_size, model_name=model_file)
        return model
        pass

    pass

class TRAIN_TFLEARN_GPU(TRAIN):

    def train(self, model_path, trainX, trainY, testX, testY,
              round_num, n_epoch, batch_size):
        tf.reset_default_graph()

        tflearn.config.init_graph(seed=8888, gpu_memory_fraction=0.4, soft_placement=True)  # num_cores default is All
        # config = tf.ConfigProto(allow_soft_placement=True, allow_growth = True, device_count = {'GPU':2})
        config = tf.ConfigProto(allow_soft_placement=True)

        config.gpu_options.allocator_type = 'BFC'
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        model_file = os.path.join(model_path + round_num, '/plankton-classifier.tfl')

        model, conv_arr = self.network.build_model(model_file)

        tf.get_variable_scope().reuse_variables()

        model_name = model_path + '/plankton-classifier'
        print('model_name ', model_name)
        self.network.train(model, trainX, trainY, testX, testY,
                           round_num, n_epoch, batch_size, model_name=model_name)
        return model
        pass

    pass


class TRAIN_KERAS(TRAIN):

    def train(self, model_path, trainX, trainY, testX, testY,
              round_num, n_epoch, batch_size):
        pass

    pass