#################################################################################################################
# A Modularized implementation for
# Image enhancement, extracting descriptors, clustering, evaluation and visualization
# Integrating all the algorithms stated above in one framework
# CONFIGURATION - LOAD_DATA - IMAGE_ENHANCEMENTS - IMAGE_DESCRIPTORS - CLUSTERING - EVALUATION_VISUALIZATION
# Implementation of the DatasetExporter class
# Author: Aya Saad
# Date created: 30 September 2019
#
#################################################################################################################

import os
import h5py
import numpy as np
from tflearn.data_utils import build_hdf5_image_dataset

class DatasetExporter:
    SPLIT_PERCENT = 0.05  # split the train and test data i.e 0.05 is a 5% for the testing dataset and 95% for the training dataset

    def __init__(self, dataset_loader):
        '''
        DatasetExporter constructor
        :param data_dir:    name of the data directory
        :param header_file: name of the header file
        :param filename:    name of the dataset file
        :param dataset_loader: loader of the dataset
        '''
        self.dataset_loader = dataset_loader

    def export_train_test(self, save_split=True):
        '''
        export the train and test dataset and save them into separate files
        :param save_split: when True save the dataset into the files
        '''
        test_filename = self.dataset_loader.IMSETTEST + self.dataset_loader.WIN + '.dat'
        train_filename = self.dataset_loader.IMSETTRAIN + self.dataset_loader.WIN + '.dat'
        test_file = os.path.join(self.dataset_loader.data_dir, test_filename)
        train_file = os.path.join(self.dataset_loader.data_dir, train_filename)
        print('Split the dataset into 95% training set and 5% test set ...')
        train_x, train_y, test_x, test_y = self.dataset_loader.split_data(split_percent=self.SPLIT_PERCENT)
        Train = np.vstack((train_x,train_y))
        Test = np.vstack((test_x,test_y))
        print('Test set shape ... ', Test.T.shape)
        print('Trainning set shape ... ', Train.T.shape)
        if save_split:
            self.dataset_loader.save_data_to_file(Train.T, train_file)
            self.dataset_loader.save_data_to_file(Test.T, test_file)

    def export_CSV(self, n_splits=10, save_split=True):
        '''
        export the cross validation datasets
        :param set_file:
        :param n_splits: number of chunks/loops the dataset to be splitted into
        :param save_split: when True save the datasets into the files
        '''
        i = 0
        for train_x,train_y,test_x,test_y in self.dataset_loader.split_CSV(n_splits=n_splits):
            i = i + 1
            round_num = str(i)
            if i < 10:
                round_num = '0' + round_num
            test_filename = self.dataset_loader.IMSETTEST + round_num + self.dataset_loader.WIN + '.dat'
            train_filename = self.dataset_loader.IMSETTRAIN + round_num + self.dataset_loader.WIN + '.dat'
            test_file = os.path.join(self.dataset_loader.data_dir, test_filename)
            train_file = os.path.join(self.dataset_loader.data_dir, train_filename)
            Train = np.vstack((train_x, train_y))
            Test = np.vstack((test_x, test_y))
            print('Test set shape ... ', Test.T.shape)
            print('Trainning set shape ... ', Train.T.shape)
            if save_split:
                self.dataset_loader.save_data_to_file(Train.T, train_file)
                self.dataset_loader.save_data_to_file(Test.T, test_file)

    def build_hd5(self, file, input_width, input_height, input_channels=3, round=''):
        '''
        build the hd5 files
        :param file: the file containing the labeled dataset
        :param input_width:
        :param input_height:
        :param input_channels:
        :param round:
        '''
        filename = file.split('.dat')[0] + '.h5'
        print('Building hdf5 for the test set... ', filename)
        out_hd5 = filename # os.path.join(self.dataset_loader.data_dir, filename)
        build_hdf5_image_dataset(file, image_shape=(input_width, input_height, input_channels),
                                 mode='file', output_path=out_hd5, categorical_labels=True, normalize=True)
        h5f = h5py.File(out_hd5, 'r')
        print('Dataset input shape: ', h5f['X'].shape)
        print('Dataset output shape: ', h5f['Y'].shape)

    def build_all_hd5(self, data_dir, input_width=64, input_height=64, input_channels=3, round=''):
        '''
        build hd5 for all .dat files in the directory
        :param input_width:
        :param input_height:
        :param input_channels:
        :param round:
        :return:
        '''
        #filepath = os.path.join(data_dir)  # DATABASE_PATH
        files = [o for o in os.listdir(data_dir) if o.endswith('.dat')]
        for f in files:
            filename = os.path.join(data_dir, f)
            self.build_hd5(file=filename,
                           input_width=input_width, input_height=input_height,input_channels=3, round=round)


