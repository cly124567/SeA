from __future__ import print_function
import os
from PIL import Image

import logging
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from utils import *

def train(nb_epoch, batch_size, store_name, resume=False, start_epoch=0, model_path=None):
    # setup output
    exp_dir = store_name
    try:
        os.stat(exp_dir)
    except:
        os.makedirs(exp_dir)

    use_cuda = torch.cuda.is_available()
    print(use_cuda)

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.Resize((550, 550)),
        transforms.RandomCrop(448),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    trainset = torchvision.datasets.ImageFolder(root='/data/chenyj/dataset/CUB_200/train', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    device = torch.device("cuda:0")
    # Model
    CELoss = nn.CrossEntropyLoss()

   
    if resume:    
        net = PMG(swin_base_patch4_window7_224_in22k(),512,200).to(device)
        weights_dict = torch.load(model_path,map_location='cpu')
        missing_keys, unexpected_keys=net.load_state_dict(weights_dict,strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)
        else:
            print("all key matched")
        
    else:
        net = load_model(model_name='swin', pretrain=True, require_grad=True)
        
    # GPU
    net.to(device)
    # cudnn.benchmark = True

    optimizer = optim.SGD([
        {'params': net.classifier_concat.parameters(), 'lr': 0.001},
        {'params': net.vit1.parameters(), 'lr': 0.001},
        {'params': net.MSA1.parameters(), 'lr': 0.001},
        {'params': net.classifier1.parameters(), 'lr': 0.001},
        {'params': net.conv_block1.parameters(), 'lr': 0.001},
        {'params': net.vit2.parameters(), 'lr': 0.001},
        {'params': net.MSA2.parameters(), 'lr': 0.001},
        {'params': net.classifier2.parameters(), 'lr': 0.001},
        {'params': net.conv_block2.parameters(), 'lr': 0.001},
        {'params': net.features.parameters(), 'lr': 0.0001}

    ],
        momentum=0.9, weight_decay=5e-4)

    max_val_acc = 0
    lr = [ 0.001, 0.001, 0.001,  0.001,0.001,0.001, 0.001,  0.001,0.001,0.0001]
    for epoch in range(start_epoch, nb_epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0
        correct = 0
        total = 0
        idx = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            idx = batch_idx
            if inputs.shape[0] < batch_size:
                continue
            if use_cuda:
                inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = Variable(inputs), Variable(targets)

            # update learning rate
            for nlr in range(len(optimizer.param_groups)):
                optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, lr[nlr])
           
           
            optimizer.zero_grad()
            output_1, _,_ = net(inputs)
            loss1 = CELoss(output_1, targets) * 1
            loss1.backward()
            optimizer.step()
            
            optimizer.zero_grad()
            _,output_2,_ = net(inputs)
            loss2 = CELoss(output_2, targets) * 1
            loss2.backward()
            optimizer.step()

            optimizer.zero_grad()
            _,_, output_concat = net(inputs)
            concat_loss = CELoss(output_concat, targets) * 2
            concat_loss.backward()
            optimizer.step()

            _, predicted = torch.max(output_concat.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            train_loss += ( loss1.item()+loss2.item() + concat_loss.item())
            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            train_loss3 += concat_loss.item()

            if batch_idx % 50 == 0:
                print(
                    'Step: %d |Loss1: %.5f |Loss2: %.5f | Loss_concat: %.5f | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                    batch_idx,  
                    train_loss1 / (batch_idx + 1),train_loss2 / (batch_idx + 1),train_loss3 / (batch_idx + 1), train_loss / (batch_idx + 1),
                    100. * float(correct) / total, correct, total))

        train_acc = 100. * float(correct) / total
        train_loss = train_loss / (idx + 1)
        with open(exp_dir + '/results_train.txt', 'a') as file:
            file.write(
                'Iteration %d | train_acc = %.5f | train_loss = %.5f  | Loss1: %.5f |Loss2: %.5f | Loss_concat: %.5f |\n' % (
                epoch, train_acc, train_loss, train_loss1 / (idx + 1),train_loss2 / (idx + 1),
                train_loss3 / (idx + 1)))

        val_acc, val_acc_com1,val_acc_com2,val_loss = test(net, CELoss, 4)
           
        net.cpu()
        torch.save(net.state_dict(), './' + store_name + '/model{}.pth'.format(epoch))
        net.to(device)
        with open(exp_dir + '/results_test.txt', 'a') as file:
            file.write('Iteration %d, test_acc = %.5f, test_acc_combined1 = %.5f,test_acc_combined2 = %.5f,test_loss = %.6f\n' % (
            epoch, val_acc, val_acc_com1,val_acc_com2,  val_loss))


train(nb_epoch=200,             # number of epoch
         batch_size=16,         # batch size
         store_name='CUB',     # folder for output
         resume=False,          # resume training from checkpoint
         start_epoch=0,         # the start epoch number when you resume the training
         model_path='')         # the saved model where you want to resume the training
