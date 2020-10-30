#Convolutional neural network based event classifier
This package is a event classifier based on convolutional neural network designed for event discrimination between atmospheric neutrino background and diffuse supernova neutrino events. The default network is a 18-layer ResNet.

## content of the package
1. DSNBDataset.py - This script reformat the ROOT files produced by JUNO simulation or future JUNO data to the network input.
2. junodata.py - This script is a dataloader for the CNN training/testing.
3. resnet.py - This script defines the default ResNet-18 model used in the classifier.
4. vgg.py - vgg model is added, but not tested yet.
5. train.py - Training/testing of the network.
