import torch
import time
import numpy as np
import os

from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import lr_scheduler
from torch import nn

import argparse
import junodata, vgg, resnet

device = 'cuda'
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
best_acc = 0 # best test accuracy

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(trainloader, epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    train_acc =0
    correct = 0
    total = 0
    fsoftmax = nn.Softmax(dim=1)
    for batch_idx, (inputs, targets, spectators) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        outputs = fsoftmax(outputs)
        loss = criterion(outputs, targets.long())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
        print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total), end='\r')
    return train_loss/len(trainloader), 100.*correct/total

def test(testloader, epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    score = []
    fsoftmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for batch_idx, (inputs, targets, spectators) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            outputs = fsoftmax(outputs)
            loss = criterion(outputs, targets.long())
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            test_loss += loss.item()
            total += targets.size(0)
            for m in range(outputs.size(0)):
                score.append([outputs[m].cpu().numpy(), targets[m].cpu().numpy(), spectators[m].numpy()])
            print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total), end='\r')

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {'net': net.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                    }
            if not os.path.isdir('checkpoint_sens' ):
                os.mkdir('checkpoint_sens' )
            torch.save(state, './checkpoint_sens/ckpt_%d.t7' % epoch)
            torch.save(state, './checkpoint_sens/ckpt.t7' )
            best_acc = acc
        return test_loss/len(testloader), 100.*correct/total, score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch 1d conv net classifier')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--batchsize', '-b', default=400, type=int, help='batch size')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--csvfile', '-c', type=str, help='location of csv file of dataset infor.')
    parser.add_argument('--h5file', '-f', type=str, help='location of hdf5 file.')
    args = parser.parse_args()
    # Data
    print('==> Preparing data..')
    dataset = junodata.H5Dataset(args.h5file, args.csvfile)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    validation_split = .15
    split = int(np.floor(validation_split * dataset_size))
    shuffle_dataset = True
    random_seed= 42
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize, sampler=train_sampler, num_workers=12)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize, sampler=validation_sampler, num_workers=12)

    #initialize the training parameters.
    momentum = 0.9
    weight_decay = 1.0e-3
    start_epoch = 0
    print('==> Building model..')
    net = resnet.resnet18()
    net = net.to(device)
    #use DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print("Let's use ", torch.cuda.device_count(), " GPUs!")
        net = torch.nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count())))
    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()
    # We use SGD
    optimizer = torch.optim.SGD(net.parameters(), args.lr, momentum=momentum, weight_decay=weight_decay)

    if args.resume and os.path.exists('./checkpoint_sens/ckpt.t7'):
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint_sens'), 'Error: no checkpoint directory found!'
        if device == 'cuda':
            checkpoint = torch.load('./checkpoint_sens/ckpt.t7' )
        else:
            checkpoint = torch.load('./checkpoint_sens/ckpt.t7', map_location=torch.device('cpu') )
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
    y_train_loss = np.array([])
    y_train_acc = np.array([])
    y_test_loss = np.array([])
    y_test_acc  = np.array([])
    test_score = []
    start_time = time.time()
    for epoch in range(start_epoch, start_epoch + 20):
        epoch_start = time.time()
        # set the learning rate
        adjust_learning_rate(optimizer, epoch, lr)
        iterout = "Epoch [%d]: "%(epoch)
        for param_group in optimizer.param_groups:
            iterout += "lr=%.3e"%(param_group['lr'])
            print(iterout)
            try:
                train_ave_loss, train_ave_acc = train(train_loader, epoch)
            except Exception as e:
                print("Error in training routine!")
                print(e.message)
                print(e.__class__.__name__)
                traceback.print_exc(e)
                break
            print("Epoch [%d] train aveloss=%.3f aveacc=%.3f"%(epoch,train_ave_loss,train_ave_acc))
            y_train_loss = np.append(y_train_loss, train_ave_loss)
            y_train_acc = np.append(y_train_acc, train_ave_acc)

            # evaluate on validationset
            try:
                test_loss,test_acc, score = test(validation_loader, epoch)
            except Exception as e:
                print("Error in validation routine!")
                print(e.message)
                print(e.__class__.__name__)
                traceback.print_exc(e)
                break
            print("Test[%d]:Result* Prec@1 %.3f\tLoss %.3f"%(epoch,test_acc,test_loss))
            y_test_loss = np.append(y_test_loss, test_loss)
            y_test_acc = np.append(y_test_acc, test_acc)
            test_score.append(score)
        epoch_elapse = time.time() - epoch_start
        print('Epoch %d used %f seconds' % (epoch, epoch_elapse))
        print(y_train_loss, y_train_acc)
        np.save('test_score_%d.npy' % (start_epoch + 1), test_score)
    np.save('loss_acc.npy', np.array([y_train_loss, y_train_acc, y_test_loss, y_test_acc]))
    print('Total time used is %f seconds' % (time.time() - start_time) )
