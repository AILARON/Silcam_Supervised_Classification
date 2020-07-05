#################################################################################################################
# The Pytorch neural network architecture module
# Implementation of the testing module
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

import torch
from torch_tools.metrics import METRIX
from tqdm import tqdm
import time
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch_tools.trainconfig import *
from torch_tools.net import *
from torch_tools.dataloader import *
from torch import nn
import torchvision.models as models

def adjust_learning_rate(optimizer, epoch, start_lr):
    """Gradually decay learning rate"""
    lr = start_lr
    if epoch > 180:
        lr = lr / 1000000
    elif epoch > 150:
        lr = lr / 100000
    elif epoch > 120:
        lr = lr / 10000
    elif epoch > 90:
        lr = lr / 1000
    elif epoch > 60:
        lr = lr / 100
    elif epoch > 30:
        lr = lr / 10

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def test(model, device, criterion, test_loader, log_file):
    # Testing
    model.eval()
    test_loss = 0
    accuracy = 0
    balanced_accuracy = 0
    fh = open(log_file, 'a+')
    fh.write('*************TEST *************\n')
    for sample in test_loader:
        data, target = sample['image'], sample['label']
        data, target = data.to(device), target.to(device)

        outputs = model(data.float())
        print('outputs ',outputs)
        loss = criterion(outputs, target.squeeze(1).long())
        test_loss += loss.item() * data.size(0)
        acc, bacc, precision, recall, f1_score, rep = \
            METRIX(target.squeeze(1).long(), outputs)
        accuracy += acc
        balanced_accuracy += bacc
        # ----------------------
        fh.write('Test Acc:\t{:.3f}%\tBalanced Acc.:\t{:.3f}%\tPrecision:\t{:.3f}%\t'
              'Recall:\t{:.3f}%\tF1 Score:\t{:.3f}%\t\n'.format(acc * 100, bacc * 100, precision * 100,
                                                          recall * 100, f1_score * 100))
        fh.write('Report: \n{}\n'.format(rep))
        # ----------------------
    fh.close()
    return test_loss, accuracy, balanced_accuracy

def loadData(data_dir, header_file, filename, log_file, batch_size, input_size):
    fh = open(log_file, 'a+')
    fh.write('##### DATASET ######\n')
    composed = transforms.Compose([Resize(input_size), RandomRotation(),
                                   Resize(input_size), ToTensor(), Normalization()])
    dataset = PlanktonDataSet(data_dir=data_dir, header_file=header_file,
                              csv_file=filename, transform=composed)
    class_list = dataset.get_classes_from_file()
    fh.write('class_list\t{}\tlen(class_list)\t{}\n'.format(class_list, len(class_list)))
    fh.write('# split dataset into train, test and validate\n')
    fh.write('len(dataset)\t{}\n'.format(len(dataset)))
    split_size = int(len(dataset) * 0.15)
    train, validate, test = random_split(dataset,
                                         [len(dataset) - 2 * split_size,
                                          split_size,
                                          split_size])
    fh.write('len(train):\t{}\n'.format(len(train)))
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    fh.write('len(validate):\t{}\n'.format(len(validate)))
    validate_loader = DataLoader(validate, batch_size=batch_size, shuffle=True)
    fh.write('len(test):\t{}\n'.format(len(test)))
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)
    fh.close()
    return train_loader, validate_loader, test_loader, class_list


def run(device, net, config, name, log_file,
        number_of_epochs, batch_size, start_lr, weight_decay,
        train_loader, validate_loader, test_loader):
    filename = os.path.join(config.model_dir, name)
    fh = open(log_file, 'a+')

    net = net.float()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=start_lr, weight_decay=weight_decay)

    #fname = str('{}_{}_model.pt'.format(filename, m[0]))
    fname = 'D:/resnet_test/model/norm064/ResNet152/ResNet152_056_model.pt'
    fh.write('Loading best model saved: \t{} \n'.format(fname))
    net.load_state_dict(torch.load(fname))
    fh.close()


    t0 = time.time()
    test_loss, test_acc, test_balacc = test(net, device, criterion, test_loader, log_file)
    time_test = time.time() - t0
    fh = open(log_file, 'a+')
    fh.write('Testing time\t{} seconds\n'.format(time_test))
    fh.write('Accu:\t {:.3f}%\n'.format(test_acc*100/ len(test_loader)))
    fh.write('Test acc:\t{:.3f}%\tbalance acc:\t{:.3f}%\t'
             'loss:\t{:.6f}\t'
             'test batch_size\t{}\n'.format(test_acc*100 / len(test_loader),
                                        test_balacc*100 / len(test_loader),
                                        test_loss, batch_size))
    fh.close()

def load_model(name, num_classes, log_file,
               input_height = 64, input_width = 64, num_of_channels = 3):
    net = models.resnet152(num_of_channels, num_classes) # COAPNet(num_classes=num_classes)

    fh = open(log_file, 'a+')

    fh.write('batch_size:\t{}'.format(batch_size))
    fh.write(str(net))
    params = list(net.parameters())
    fh.write('\nlen(params)\t{}'.format(str(len(params))))
    fh.write('\tparams[0].size()\t{}\n'.format(str(params[0].size())))  # conv1's .weight
    fh.close()
    return net



if __name__ == '__main__':
    # Find the device available to use using torch library
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config, unparsed = get_config()
    np.random.seed(config.random_seed)
    ''''''''''''''
    start_lr = 0.001
    weight_decay = 0.0001
    input_size = 64
    input_height = 64
    input_width = 64
    number_of_channels = 3
    batch_size = [64] # 16, 64, 128, 256
    number_of_epochs = [200] # 200
    norm = 'norm'
    # loading data from data directory
    header_file = config.data_dir + '/header.tfl.txt'
    filename = 'image_set.dat'

    name = ['ResNet152']

    n = 0   # loop over epoch numbers - set to 0 when testing and 1 when on the GPU
    model_dir = config.model_dir
    for i in range(len(name)):  # loop over the network names
        print('#######  THE {} NETWORK ARCHITECTURE ############'.format(name[i]))
        for j in range(len(batch_size)): # loop over bach_size
            print('#######  {} with batch_size {} ############'.format(name[i], batch_size[j]))
            config.model_dir = os.path.join(model_dir, str('{}{:03d}'.format(norm, batch_size[j])), name[i])
            prepare_dirs(config)
            log_file = os.path.join(config.model_dir, name[i] + '_run_test.log')
            fh = open(log_file, 'a+')
            fh.write('#######  THE {} NETWORK ARCHITECTURE with batch_size {} ############\n'.format(name[i],batch_size[j]))
            print('### loading data ### ', name[i])
            fh.write('### loading data ###\n')
            print('number_of_epochs: ', number_of_epochs[n], 'batch_size: ', batch_size[j], '')
            fh.write('number_of_epochs:\t{}\tbatch_size:\t{}\n'.format(number_of_epochs[n], batch_size[j]))
            print('log_file: ', log_file)
            fh.close()
            train_loader, validate_loader, test_loader, class_list = \
                loadData(config.data_dir, header_file, filename, log_file, batch_size[j], input_size=input_size)
            print('len(class_list)', len(class_list))

            net = load_model(name[i], log_file=log_file, num_classes=len(class_list),
                             input_height=input_height, input_width=input_width, num_of_channels=number_of_channels)
            run(device, net, config, name[i], log_file,
                number_of_epochs[n], batch_size[j], start_lr, weight_decay,
                train_loader, validate_loader, test_loader)
