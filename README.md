# Pytorch-MADF

An Implement of MADF (Image Inpainting by Single-Shot Cascaded Refinement with Mask Awareness) for Pytorch version.

# Requirements
  - Pytorch 1.0.0
  - python 3.*
  - numpy 1.13.1
  - scipy 0.17.0
  
  
# Usages
  ## downlaod repo
  - download this repo by the following instruction:
  
        $ git clone https://github.com/cvpr2020-inpainting-anonymous/Pytorch-MADF.git
        $ cd Pytorch-MADF-master
      
  ## download datasets
  - Firstly, you need to make some directories in the root path
  
        $ mkdir data
        $ cd data
        $ mkdir train
        $ mkdir val
        $ mkdir test   

  ### Places2
  - You can download this datasets [here](http://places2.csail.mit.edu/download.html). We use the High-resolution version in this repo.
  
  - After you have download the datasets, copy ImageNet(here I only use 3137 images) datsets to ***/data/train***, then you have ***/data/train/ImageNet*** path, and training images are stored in ***/data/train/ImageNet***
  
  
  ### Celeba-HQ
  - You can download this datasets [here]() **Set5** dataset is used as **val data**, you can download it [here](http://data.csail.mit.edu/places/places365/val_large.tar).
  
  - After you download **Set5**, please store it in ***/data/val/*** , then you have ***/data/val/Set5*** path, and val images are stored in ***/data/val/Set5***
  
  ### Paris street view
  - **Set14** dataset is used as **test data**, you can download it [here](http://data.csail.mit.edu/places/places365/test_large.tar).
  
  - After you download **Set14**, please store it in ***/data/test/*** , then you have ***/data/test/Set14*** path, and val images are stored in ***/data/test/Set14***
 
  ## training
  
      $ python train.py --train_root train_root --mask_root mask_root --test_root test_root --use_incremental_supervision  
      
  ## testing
  
      $ python test.py --list_file test_list --snapshot model_path
