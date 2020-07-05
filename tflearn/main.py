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
import numpy as np
from configuration.utils import *
from configuration.config import get_config
from load_data.export_dataset import DatasetExporter
from load_data.load_dataset import DatasetLoader
from DNN.train_evaluate import *
from DNN.net import *

def build_hd5(data_dir, header_file, filename):
    #### CODE SNIPPET TO LOAD THE DATASET FROM THE DIRECTORY AND CREATE THE HD5 FILES ###########################
    dataset_loader = DatasetLoader(data_dir, header_file, filename)
    dataset_loader.get_classes_from_directory()
    dataset_loader.save_classes_to_file()
    dataset_loader.import_directory_structure()
    dataset_loader.save_data_to_file(dataset_loader.input_data, dataset_loader.filename)
    dataset_exporter = DatasetExporter(dataset_loader)
    dataset_exporter.export_train_test()
    dataset_exporter.export_CSV()
    ## one file creation
    #file = os.path.join(data_dir,
    #                        "image_set_train.dat")  # the file that contains the list of images of the testing dataset along with their classes
    #dataset_exporter.build_hd5(file,input_width=3,input_height=3,input_channels=3,round='')
    ## building all the hd5 from a directory for the cross validation data
    dataset_exporter.build_all_hd5(data_dir)
    ##############################################################################################################
    return

def train_net(data_dir, model_dir, header_file, log_file, filename, round_num=''):
    name = 'SilCamNet'
    n_splits = 1  # 10 for cross_validation, 1 for one time run
    model_file = os.path.join(model_dir, 'plankton-classifier.tfl')
    print('model_dir ', model_dir)
    print('model_file ', model_file)
    myNet = SilCamNet(name, input_width=64, input_height=64, input_channels=3,
                    num_classes=6, learning_rate=0.001,
                    momentum=0.09, keep_prob=0.5, model_file=model_file)

    model, conv_arr = myNet.build_model(model_file=model_file)
    fh = open(log_file, 'w')
    fh.write(name)
    print(name)
    dataset_loader = DatasetLoader(data_dir, header_file, filename)
    traintf = TRAIN_TFLEARN(myNet)
    # loading train data
    trainX, trainY = dataset_loader.load_hd5_data(type='train')
    # loading test data
    testX, testY = dataset_loader.load_hd5_data(type='test', write_to_file='r+')

    model = traintf.train(model_dir, trainX, trainY, testX, testY,
          round_num='', n_epoch=2, batch_size=3)
    print('save into model_file ', model_file)
    model.save(model_file)

    # Evaluate
    model.load(model_file)
    print('load from model_file ', model_file)

    y_pred, y_true, acc, pre, rec, f1sc, conf_matrix, norm_conf_matrix = \
        myNet.evaluate(model, testX, testY)


    ## update summaries ###
    i = 0
    j = ''
    prediction = []
    test = []
    accuracy = []
    precision = []
    recall = []
    f1_score = []
    confusion_matrix = []
    normalised_confusion_matrix = []

    prediction.append(y_pred)
    test.append(y_true)
    accuracy.append(acc)
    precision.append(pre)
    recall.append(rec)
    f1_score.append(f1sc)
    confusion_matrix.append(conf_matrix)
    normalised_confusion_matrix.append(norm_conf_matrix)

    for i in range(0, n_splits):
        fh.write("\nRound ")
        if i < 10:
            j = '0' + str(i)
        fh.write(j)
        print("Round ", j)
        fh.write("\nPredictions: ")
        for el in y_pred:
            fh.write("%s " % el)
        fh.write("\ny_true: ")
        for el in y_true:
            fh.write("%s " % el)
        print("\nAccuracy: {}%".format(100 * accuracy[i]))
        fh.write("\nAccuracy: {}%".format(100 * accuracy[i]))
        print("Precision: {}%".format(100 * precision[i]))
        fh.write("\tPrecision: {}%".format(100 * precision[i]))
        print("Recall: {}%".format(100 * recall[i]))
        fh.write("\tRecall: {}%".format(100 * recall[i]))
        print("F1_Score: {}%".format(100 * f1_score[i]))
        fh.write("\tF1_Score: {}%".format(100 * f1_score[i]))
        print("confusion_matrix: ", confusion_matrix[i])
        print("Normalized_confusion_matrix: ", normalised_confusion_matrix[i])
    fh.close()


    return

def main(config):
    np.random.seed(config.random_seed)
    prepare_dirs(config)
    header_file = config.data_dir + '/header.tfl.txt'
    log_file = os.path.join(config.model_dir, 'SilCamNet.out')
    filename = config.data_dir + '/image_set.dat'

    print(config.data_dir, config.model_dir, header_file, log_file, filename)
    # build_hd5(config.data_dir, header_file, filename)
    train_net(config.data_dir, config.model_dir, header_file, log_file, filename, round_num='')


    return

if __name__ == '__main__':
    config, unparsed = get_config()
    main(config)