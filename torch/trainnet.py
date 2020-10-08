#################################################################################################################
# The Pytorch neural network architecture module
# Implementation of the training module
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


import torch
from tqdm import tqdm
import time
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
from trainconfig import *
from net import *
from dataloader import *
from torch import nn
import torchvision.models as models
from metrics import METRIX

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
def train(model, name, device, train_loader, optimizer, criterion):
    train_loss = 0
    counter = 0
    model.train()
    for idx_batch, sample in tqdm(enumerate(train_loader),
                                  total=len(train_loader),
                                  leave=False):
        t0 = time.time()
        input, target = sample['image'], sample['label']
        # Move to device
        input, target = input.to(device), target.to(device)
        # Clear optimizers - Clear all accumulated gradients
        optimizer.zero_grad()
        # Predict classes using images from the test set
        outputs = model(input.float())
        ###
        # criterion(outputs.logits, torch.randint(0, 1000, (2,)))
        # Loss
        loss = criterion(outputs, target.squeeze(1).long())
        # Calculate gradients (backpropogation)
        loss.backward()
        # Adjust parameters based on gradients
        optimizer.step()
        # Add the loss to the training set's rnning loss
        train_loss += loss.item() * input.size(0)
        counter += 1
        #print('{}\t{}/ {} seconds'.format(counter, len(train_loader), time.time() - t0))
    return train_loss
def val(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0
    accuracy = 0
    balanced_accuracy = 0
    with torch.no_grad():
        for idx_batch, sample in enumerate(val_loader):
            input, target = sample['image'], sample['label']
            input, target = input.to(device), target.to(device)
            outputs = model(input.float())
            # Calculate Loss
            loss = criterion(outputs, target.squeeze(1).long())
            # Add loss to the validation set's running loss
            val_loss += loss.item() * input.size(0)
            #accuracy, balanced_accuracy, precision, recall, f1_score, rep =
            acc, bacc, _, _, _, _ = \
                METRIX(target.squeeze(1).long(), outputs)
            accuracy+=acc
            balanced_accuracy+=bacc
    return val_loss, accuracy, balanced_accuracy
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
    composed = transforms.Compose([convertToRGB(), Resize(input_size), RandomRotation(),
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

def save_models(epoch, model, name):
    #torch.save(model.state_dict(), "cifar10model_{}.model".format(epoch))
    torch.save(model.state_dict(), name +'_{:03d}_model.pt'.format(epoch))

    print("Chekcpoint saved")

def run(device, net, config, name, log_file,
        number_of_epochs, batch_size, start_lr, weight_decay,
        train_loader, validate_loader, test_loader):
    filename = os.path.join(config.model_dir, name)
    fh = open(log_file, 'a+')
    best_saved_model = []
    net = net.float()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=start_lr, weight_decay=weight_decay)
    best_acc = 0
    acc_list = []
    balacc_list = []
    for epoch_no in range(1, number_of_epochs+1):
        fh.write('\nepoch_no:\t{}\n'.format(epoch_no))
        net.to(device)
        t0 = time.time()
        train_loss = \
            train(net, name, device, train_loader, optimizer, criterion)
        fh.write('Total training time {} seconds\n'.format(time.time() - t0))
        t0 = time.time()
        valid_loss, accuracy, balaccuracy = \
            val(net, device, validate_loader, criterion)
        fh.write('Validation time {} seconds\n'.format(time.time() - t0))

        # Get the average loss for the entire epoch
        train_loss = train_loss / len(train_loader.dataset)
        valid_loss = valid_loss / len(validate_loader.dataset)
        # Print out the information
        #print('Accuracy: ', accuracy / len(validate_loader))
        fh.write('Epoch:\t{}\tTraining Loss:\t{:.6f}\tValidation Loss:\t{:.6f}\n'.format(epoch_no,
                                                                                       train_loss,
                                                                                       valid_loss))
        fh.write('Accu:\t{:.3f}%\tBalanced Accu:\t{:.3f}%\t\n'.format(accuracy*100 / len(validate_loader),
                                                                    balaccuracy*100 / len(validate_loader)))
        acc_list.append(accuracy)
        balacc_list.append(balaccuracy)

        # Validation
        fh.write('*************VALIDATION, val_set_list:*************\n')
        fh.write('Validation accuracy: \t{:.3f}%\t '
                 'Validation_Balanced_Acc: \t{:.3f}% \t'
                 'Validate_loader Length: \t{}\t\n'.format(accuracy*100 / len(validate_loader),
                                                       balaccuracy*100 / len(validate_loader),
                                                       len(validate_loader)))
        acc = balaccuracy*100 / len(validate_loader)
        fh.write('Validation epoch:\t{}\tacc:\t{:.3f}%\t\n'.format(epoch_no, acc))

        # Save model if better than previous
        fh.write('model file path + name {}\n'.format(filename))
        torch.save(net.state_dict(), filename + '_{:03d}_model.pt'.format(epoch_no))
        fh.write('checkpoint\t{}_{:03d}_model.pt\tsaved\n'.format(filename,epoch_no))
        if acc > best_acc:
            best_acc = acc
            best_saved_model.append([str('{:03d}').format(epoch_no),acc])
            fh.write('Best Model\t{}_{:03d}_model.pt\tsaved\n'.format(filename,epoch_no))

        adjust_learning_rate(optimizer, epoch_no, start_lr)
    # Finally test on test-set with the best model
    #'best_coap_{}_model.pt'.format(epoch))
    # return the best saved_model to load for testing
    fh.write('Accuracy List:\t{}\n'.format(*acc_list, sep='\t'))
    fh.write('Balanced accuracy List:\t{}\n'.format(*balacc_list, sep='\t'))

    m = max(best_saved_model, key=lambda x: x[1])
    fname = str('{}_{}_model.pt'.format(filename, m[0]))
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
            log_file = os.path.join(config.model_dir, name[i] + '_run.log')
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

            net = load_model(name[i], log_file=log_file, num_classes=len(class_list),
                             input_height=input_height, input_width=input_width, num_of_channels=number_of_channels)
            run(device, net, config, name[i], log_file,
                number_of_epochs[n], batch_size[j], start_lr, weight_decay,
                train_loader, validate_loader, test_loader)
