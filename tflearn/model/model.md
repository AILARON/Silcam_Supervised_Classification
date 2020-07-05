the trained models are saved under the 
'/mnt/DATA/model'
Each directory is a DNN architecture 
    
    For example:
    modelCoapNet is the COAPNet architecture model trained over the dataset
    the .out files are the reported logs along with the performance during the training
    
    find the file plankton-classifier.tfl.index under each directory. 
    This file has the weights and can be used as the trained model
    
---------------------------------------------------
###Performance Metrics    

    |               | Accuracy        | Precision       | Recall          | F1 Score        | Location                                      |
    | ------------- | --------------- | --------------- | --------------- | --------------- | --------------------------------------------- |
    |               | DBI - DBII      | DBI - DBII      | DBI - DBII      | DBI - DBII      | DBI - DBII                                    |
    | LeNet         | 56.84% - 57.44% | 48.38% - 32.99% | 56.84% - 57.44% | 49.10% - 41.91% | "mnt/DATA/model/modelLENET/LeNetDBI"          | 
    |               |                 |                 |                 |                 | "mnt/DATA/model/modelLENET/LeNetDBI"          |
    | MINST         | 59.94% - 57.44% | 37.53% - 32.99% | 59.94% - 57.44% | 45.92% - 41.91% | "mnt/DATA/model/modelMINST\MINSTDBI"          | 
    |               |                 |                 |                 |                 | "mnt/DATA/model/modelMINST\MINSTDBII"         |
    | SilCamNet     | 93.79% - 83.95% | 93.97% - 83.90% | 93.79% - 83.95% | 93.78% - 83.76% | "mnt/DATA/model/modelOrgNet/OrgNetDBI"        |
    |               |                 |                 |                 |                 | "mnt/DATA/model/modelOrgNet/OrgNetDBII"       |
    | COAPNet       | 95.09% - 82.99% | 95.16% - 83.49% | 95.09% - 82.99% | 95.09% - 83.06% | "mnt/DATA/model/modelCoapNet/CoapNetDBI"      |
    |               |                 |                 |                 |                 | "mnt/DATA/model/modelCoapNet/CoapNetDBII"     |
    | OxfordNet     | 93.28% - 83.51% | 93.50% - 83.51% | 93.28% - 83.51% | 93.31% - 83.44% | "mnt/DATA/model/modelCIFAR10/CIFAR10DBI"      |
    |               |                 |                 |                 |                 | "mnt/DATA/model/modelCIFAR10/CIFAR10DBII"     |
    | AlexNet       | 91.21% - 57.44% | 92.07% - 32.99% | 91.21% - 57.44% | 91.17% - 41.91% | "mnt/DATA/model/modelAlexNet/AlexNetDBI"      |
    |               |                 |                 |                 |                 | "mnt/DATA/model/modelAlexNet/AlexNetDBII"     |
    | VGGNet        | 93.54% -        | 93.76% -        | 93.54% -        | 93.44% -        | "mnt/DATA/model/modelVGGNet/VGGNetDBI"        |
    | PlankNet      | 93.54% -        | 93.57% -        | 93.54% -        | 93.48% -        | "mnt/DATA/model/modelPlankNet/PlankNetDBI"    |
    | GoogleNet     | 82.68% -        | 84.71% -        | 82.68% -        | 81.96% -        | "mnt/DATA/model/modelGoogLeNet/GoogleNetDBI"  |
    | ResNet        | 91.98% - 85.37% | 92.03% - 85.32% | 91.98% - 85.37% | 91.91% - 85.11% | "mnt/DATA/model/modelResNet/ResNetDBI"        |               |
    |               |                 |                 |                 |                 | "mnt/DATA/model/modelResNet/ResNetDBII"       |
    | ResNeXt       | 92.24% - 83.80% | 92.63% - 84.40% | 92.24% - 83.80% | 92.21% - 82.68% | "mnt/DATA/model/modelResNeXt/ResNeXtDBI"      |
    |               |                 |                 |                 |                 | "mnt/DATA/model/modelResNeXt/ResNeXtDBII"     |

---------------------------------------------------    

Performance Metrics of the SilCamNet and COAPNet over DBIII with varying input size and drop-out 

    | Input size    | keep drop out | Accuracy      | Precision     | Recall        | F1 Score      | Location      |
    | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
    |               | probability   | DBIII         | DBIII         | DBIII         | DBIII         |               |
    |               |               |               |               |               |               |               |

