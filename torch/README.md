![NTNU logo](https://qore.no/res/ntnu-logo-100.png)
# Under-Water-Image-Classifier
###### Author: Aya Saad
###### email: aya.saad@ntnu.no
###### Project: AILARON
###### Contact
###### email: annette.stahl@ntnu.no
###### funded by RCN IKTPLUSS program (project number 262701) and supported by NTNU AMOS
###### Copyright @NTNU 2020
---------------------------------------------------
<!-- -->
###### Date created: 3 April 2020
A Modularized implementation for
Network architecture-train-validate-test

---------------------------------------------------
<!-- -->

###Installation
    ## Create a conda environment
    conda create -n <environment name> pip python scikit-image pandas seaborn numpy matplotlib scikit-learn scipy
    ex: conda create -n trainnet pip python scikit-image pandas 
    
    ## Activate the conda environment
    activate <environment name>
    activate trainnet

    ## Install pytorch
    conda install -c pytorch pytorch
    conda install -c conda-forge imageio
    conda install -c pytorch torchvision
    pip install torchsummary

    ## install packages for image enhancements  
    conda install -c conda-forge opencv 
    pip install opencv.contrib-python
    conda install -c anaconda scikit-learn

    ## install tqdm, time
    conda install -c conda-forge tqdm
    conda install -c conda-forge time

---------------------------------------------------

