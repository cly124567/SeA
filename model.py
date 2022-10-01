import torch.nn as nn
import torch
from math import sqrt
from einops import repeat 
import numpy as np
import time

class PMG(nn.Module):
    def __init__(self, model, feature_size, classes_num,dropout=0.2):
        super(PMG, self).__init__()

        self.features = model
        
        self.max1 = nn.MaxPool2d(kernel_size=28, stride=28)
        self.max2 = nn.MaxPool2d(kernel_size=14, stride=14)
        self.num_ftrs = 1024 * 1 * 1
        self.elu = nn.ELU(inplace=True)

        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(512 *4),
            nn.Linear(512 * 4, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        self.conv_block1 = nn.Sequential(
            BasicConv(self.num_ftrs//2, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs//2, kernel_size=3, stride=1, padding=1, relu=True)
        )
        
        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )
        self.conv_block2 = nn.Sequential(
            BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs//2, kernel_size=3, stride=1, padding=1, relu=True)
        )
        
        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        self.vit1=VisionTransformer(num_patches=28*28,embed_size=512)
        self.vit2=VisionTransformer(num_patches=14*14,embed_size=512)

        self.MSA1=MultiHeadSelfAttention(dim_in=512,dim_k=512,dim_v=512,select_rate=0.5)
        self.MSA2=MultiHeadSelfAttention(dim_in=512,dim_k=512,dim_v=512,select_rate=0.5)
   
    def forward(self, x):
        xf1,xf2, xf3, xf4 = self.features(x)
        
        xl1 = self.conv_block1(xf3)
        xl2 = self.conv_block2(xf4)
  
        xd1=self.max1(xl1)
        xd2=self.max2(xl2)
        xd1 = xd1.view(xd1.size(0), -1)
        xd2 = xd2.view(xd2.size(0), -1)
  
        xl1 = xl1.reshape(xl1.shape[0],xl1.shape[1],-1)
        xl1=xl1.permute(0,2,1)
        xl1 = self.vit1(xl1)
        xl1=self.MSA1(xl1)
        xl1=torch.cat((xl1,xd1),1)
        xc1 = self.classifier1(xl1)

        xl2 = xl2.reshape(xl2.shape[0],xl2.shape[1],-1)
        xl2=xl2.permute(0,2,1)
        xl2 = self.vit2(xl2)
        xl2=self.MSA2(xl2)
        xl2=torch.cat((xl2,xd2),1)
        xc2 = self.classifier2(xl2)
          
        x_concat = torch.cat((xl1,xl2), -1)
        x_concat = self.classifier_concat(x_concat)
        return xc1,xc2, x_concat
    

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class MultiHeadSelfAttention(nn.Module):
    dim_in: int  # input dimension
    dim_k: int   # key and query dimension
    dim_v: int   # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads
    num_select:int
    def __init__(self, dim_in, dim_k, dim_v, select_rate,num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.select_rate=select_rate
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k // num_heads)

    def forward(self, x):
        # x: tensor of shape (batch, n, dim_in)
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head

        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n

        temp = dist[:,:,0,:].reshape(batch, nh, 1, n)
        index = torch.argsort(-temp, dim=-1)
        index=index[:,:,:,int(self.select_rate*n-1)].reshape(batch,nh,1,1)
        index=index.repeat(1,1,1,n)
        max = torch.take_along_dim(temp,index,dim=3)
        zero=torch.zeros(1).cuda()
        rel = torch.where(temp >= max, temp, zero)
        rel=torch.softmax(rel,dim=-1)
        
        att = torch.matmul(rel, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, 1, self.dim_v)  # batch, n, dim_v

        return att[:,0]
        
class VisionTransformer(nn.Module):
    def __init__(self,num_patches,embed_size):
        super(VisionTransformer, self).__init__()
        num_patches = num_patches
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_size))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))
        self.to_latent = nn.Identity()
    def forward(self, img):
        x = img
        b, n, _ = x.shape
        cls=self.cls_token.repeat(b,1,1)
        x = torch.cat((cls, x), dim = 1)
        x += self.pos_embedding
        return self.to_latent(x)


