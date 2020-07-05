#################################################################################################################
# A Modularized implementation for
# Image enhancement, extracting descriptors, clustering, evaluation and visualization
# Integrating all the algorithms stated above in one framework
# CONFIGURATION - LOAD_DATA - IMAGE_ENHANCEMENTS - IMAGE_DESCRIPTORS - CLUSTERING - EVALUATION_VISUALIZATION
# Implementation of the DatasetLoader class
# Author: Aya Saad
# Date created: 30 September 2019
#
#################################################################################################################

import os
import csv
import pandas as pd
import numpy as np
import h5py
from sklearn.model_selection import train_test_split, KFold


class DatasetLoader:
    classList = None
    input_data = None       #self.input_data = self.get_data()
    WIN = ''  # '_win' for windows running version
    IMSETTEST = 'image_set_test'  # name of the test set file
    IMSETTRAIN = 'image_set_train'  # name of the train set file

    def __init__(self, data_dir, header_file, filename, WIN=''):
        '''
        Dataset constructor
        :param data_dir:    name of the data directory
        :param header_file: name of the header file
        :param filename:    name of the dataset file
        '''
        print('data_dir ', data_dir, ' header file ', header_file, ' filename ', filename)
        self.header_file = header_file
        self.data_dir = data_dir
        self.filename = filename
        f = filename.split('.dat')[0]
        self.IMSETTEST = f + '_test'  # name of the test set file
        self.IMSETTRAIN = f + '_train'  # name of the train set file
        self.WIN = WIN
        print(filename, f, self.IMSETTEST,self.IMSETTRAIN)


    def get_classes_from_file(self):
        '''
        Get the list of classes from the header file
        '''
        print('Get the list of classes from the header file ', self.header_file)
        cl_file = self.header_file
        with open(cl_file) as f:
            reader = csv.reader(f)
            cl = [r for r in reader]
        self.classList = cl[0]


    def get_classes_from_directory(self):
        '''
        Get the list of classes from the directory
        '''
        print('Get classes from the database directory ', self.data_dir)
        self.classList = [o for o in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, o))]
        print('List of classes from the directory ', self.classList)


    def save_classes_to_file(self):
        '''
        save the list of classes into the header file
        :param classList:  the list of classes
        '''
        print('Save classes to file ', self.header_file)
        df_classes = pd.DataFrame(columns=self.classList)
        df_classes.to_csv(self.header_file, index=False)

    def import_directory_structure(self):
        '''
        import the dataset directory structure and assign the input_data variable
        :param classList:
        '''
        print('Import file list from the directory structure ')
        fileList = []
        for c_ind, c in enumerate(self.classList):
            print('  ', c)
            filepath = os.path.join(self.data_dir, c)   #DATABASE_PATH
            files = [o for o in os.listdir(filepath) if o.endswith('.tiff')]
            for f in files:
                fileList.append([os.path.join(filepath, f), str(c_ind + 1)])
        fileList = np.array(fileList)
        print('Shuffle dataset....')
        np.random.shuffle(fileList)
        self.input_data = fileList

    def get_data_from_file(self):
        '''
        Read the data file and get the list of images along with their labels
        and assign the input_data to the data set
        '''
        print('Get data from file ', self.data_dir, self.filename)
        self.input_data = pd.read_csv(os.path.join(self.data_dir, self.filename), header=None, delimiter=' ')
        print(self.input_data.head())

    def save_data_to_file(self, dataset, filename):
        '''
        Save the labeled data to the data file
        :param dataset: the dataset to be saved in the file
        :param filename: the name of the file
        '''
        print('Save into the data file ....', filename)
        np.savetxt(filename, dataset, delimiter=' ', fmt='%s')


    def split_data(self, split_percent =0.05):
        '''
        Split the dataset into training and testing
        :param split_percent: the percentage of splitting ex: 0.05 -> 95% training 5% testing
        :return train_x, train_y, test_x, test_y
        '''
        print('Split data with a split_percentage ', split_percent)
        print('input data shape ', self.input_data[:,1], self.input_data.shape)
        train_x, test_x, train_y, test_y = train_test_split(self.input_data[:, 0], self.input_data[:, 1],
                         test_size=split_percent,
                         random_state=42, stratify=self.input_data[:,1]
                         )
        print('Size of the training set ', len(train_x))
        print('Size of the output training set ', len(train_y))
        print('Size of the test set ', len(test_x))
        print('Size of the output test set ', len(test_y))
        print(np.unique(train_y))
        print(train_y)
        print(np.unique(test_y))
        print(test_y)

        return train_x.T, train_y.T, test_x.T, test_y.T

    def split_CSV(self, n_splits=10):
        '''
        Generated the cross validation data sets
        :return: X_train input training set, Y_train output training set,
                    X_test input test set, Y_test output test set
        '''
        seed = 7
        print('Split the dataset in KFold for cross validation, number of splits ', n_splits)
        for train_index, test_index in \
                KFold(n_splits=n_splits,shuffle=True,random_state=seed).split(self.input_data[:,0]):
            train_x, test_x = self.input_data[:,0][train_index], self.input_data[:,0][test_index]
            train_y, test_y = self.input_data[:,1][train_index], self.input_data[:,1][test_index]

            yield train_x.T,train_y.T,test_x.T,test_y.T

    def load_hd5_data(self, type='train', write_to_file = 'r', input_width = '', round_num = ''):
        if type == 'train':
            file_name = self.IMSETTRAIN + round_num + self.WIN + ".h5"
        else:
            file_name = self.IMSETTEST + round_num + self.WIN + ".h5"
        #out_hd5 = os.path.join(self.data_dir, file_name)  # str(input_width) + '_db3' +
        print(file_name)
        h5f = h5py.File(file_name, write_to_file)
        X = h5f['X']
        Y = h5f['Y']
        print('X.shape', X.shape, 'Y.shape', Y.shape,'Y=', Y[0])

        return X, Y
