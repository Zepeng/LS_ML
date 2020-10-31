import torch
import time
import pandas as pd
import numpy as np
import os

import torch

import argparse
import junodata, vgg, resnet

device = 'cuda'
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
best_acc = 0 # best test accuracy

def cnn_vertex(inputs, epoch):
    net.eval()
    score = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.reshape(inputs.shape[1:]), targets.reshape(targets.shape[1:])
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            for m in range(outputs.size(0)):
                score.append([outputs[m], targets[m]])

        return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CNN vertex reconstruction application.')
    parser.add_argument('--netpath', '-n', type=str, help='location of trained network.')
    args = parser.parse_args()

    print('==> Loading the trained model..')
    net = resnet.resnet18()
    checkpoint = torch.load(args.netpath, map_location=torch.device('cpu') )
    net.load_state_dict(checkpoint['net'])


