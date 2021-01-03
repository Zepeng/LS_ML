# Convolutional neural network based vertex reconstruction 
This package is a vertex reconstruction algorithm based on convolutional neural network for JUNO experiment. The default network is a 18-layer ResNet.

## content of the package
1. rootTodataset.py   
   This script reformat the ROOT files produced by JUNO simulation or future JUNO data to the network input. JUNO simulation events stored in ROOT file of TTree are converted into numpy array and saved in a HDF5 file. A hdf5 file is produced after processing one ROOT file, and all the hdf5 files are later merged into one single file with h5merger.py.
2. h5merger.py  
   This script is used to merge hdf5 files produced in the first step. In addition to merge h5 files, a csv file is also generated to store the dataset information inside the h5 file.
3. junodata.py   
   This script is a dataloader for the CNN training/testing.
4. resnet.py   
   This script defines the default ResNet-18 model used in the classifier.
5. vgg.py   
   vgg model is added, but not tested yet.
6. train.py   
   Training/testing of the network.
