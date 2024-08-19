
# DB-ATCNet

Dual-Branch Convolution Network with Efficient Channel Attention for EEG-Based Motor Imagery Classification

This paper is an improvement on the paper "Attention temporal convolutional network for EEG-based motor imagery classification". The code is used again in the "https://github.com/Altaheri/EEG-ATCNet". Kudos to the original authors for their open source and contributions.

![1 4](https://github.com/zk-xju/DB-ATCNet/assets/156686159/99f2e790-57f6-43cb-9729-56272b98b027)

# Development environment
Models were trained and tested on Ubuntu 20.04 by a single GPU, Nvidia RTX 3080 10GB (CUDA 11.2), using Python 3.8 with TensorFlow framework. The following packages are required:

TensorFlow 2.9.0

matplotlib 3.5.3

NumPy 1.23.1

scikit-learn 1.3.0

SciPy 1.10.1

mne 0.23.4

# Dataset
The BCI Competition IV-2a dataset needs to be downloaded and the data path placed at 'data_path' variable in BCI_2A_main.py file. The dataset can be downloaded from https://www.bbci.de/competition/iv/#dataset2a.

The Physionet EEG motor movement/imagery dataset needs to be downloaded and the data path placed at 'data_path' variable in Physionet_main.py file. The dataset can be downloaded from https://physionet.org/content/eegmmidb/1.0.0/.


