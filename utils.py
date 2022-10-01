import numpy as np
import random
import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms, models
import torch.nn.functional as F
from model import *
from swin_transformer import *
from PIL import Image

def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1

    return float(lr / 2 * cos_out)


def load_model(model_name, pretrain, require_grad=True):
    print('==> Building model..')
    if model_name == 'swin':
        net = swin_base_patch4_window7_224_in22k(pretrain=pretrain)
        for param in net.parameters():
            param.requires_grad = require_grad
        net = PMG(net, 512, 555)

    return net


def test(net, criterion, batch_size):
    net.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    correct_com1 = 0
    correct_com2 = 0
   
    total = 0
    idx = 0
    device = torch.device("cuda:0")

    transform_test = transforms.Compose([
        transforms.Resize((550, 550)),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    testset = torchvision.datasets.ImageFolder(root='/data/chenyj/dataset/CUB_200/test',
                                               transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4)

    for batch_idx, (inputs, targets) in enumerate(testloader):
        idx = batch_idx
        if use_cuda:
            inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        output_3,output_4,output_concat= net(inputs)
        outputs_com1 =  output_4+output_concat
        outputs_com2 =  output_3+output_4+output_concat

        loss = criterion(output_concat, targets)

        test_loss += loss.item()
        _, predicted = torch.max(output_concat.data, 1)
        _, predicted_com1 = torch.max(outputs_com1.data, 1)
        _, predicted_com2 = torch.max(outputs_com2.data, 1)
       
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        correct_com1 += predicted_com1.eq(targets.data).cpu().sum()
        correct_com2 += predicted_com2.eq(targets.data).cpu().sum()
       
        if batch_idx % 50 == 0:
            print('Step: %d | Loss: %.3f | Acc: %.3f%% (%d/%d) |Combined Acc1: %.3f%% (%d/%d) |Combined Acc2: %.3f%% (%d/%d)' % (
            batch_idx, test_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total, 100. * float(correct_com1) / total, correct_com1,total,100. * float(correct_com2) / total, correct_com2,total))

    test_acc = 100. * float(correct) / total
    test_acc_en1 = 100. * float(correct_com1) / total
    test_acc_en2 = 100. * float(correct_com2) / total
   
    test_loss = test_loss / (idx + 1)

    return test_acc, test_acc_en1, test_acc_en2, test_loss


