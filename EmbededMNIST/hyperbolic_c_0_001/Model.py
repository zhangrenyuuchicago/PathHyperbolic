import torch
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.nn.functional as F
import math
import hyptorch.nn as hypnn
import hyptorch.pmath as pmath
import sys

c_limit = 1.0e-20

class LeNet(nn.Module):
    def __init__(self, embed_len):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, embed_len)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class MINet(torch.nn.Module):
    def __init__(self, embed_len, inst_num, c=0.001):
        # for downloading pretrained model
        super(MINet, self).__init__()
        self.inst_num = inst_num
        
        self.lenet = LeNet(embed_len)
        self.c = c
        self.tp = hypnn.ToPoincare(c=self.c, train_x=True, train_c=False, ball_dim=embed_len)
        self.att_weight = hypnn.HypLinear(embed_len, 1, c=self.c)
        self.msi_pred = hypnn.HyperbolicMLR(ball_dim=embed_len, n_classes=2, c=self.c)

    def forward(self, x): 
        #self.tp.c.data = self.tp.c.data.clamp(min=c_limit)
        hyp_result = []
        klein_result = []
        for i in range(self.inst_num):
            med = x[:,i,:,:,:]
            x_size = x.size()
            med_input = med.view(x_size[0], x_size[2], x_size[3], x_size[4])
            med_out = self.lenet(med_input)
            hyp_med_out = self.tp(med_out)
            
            if torch.isnan(hyp_med_out).any():
                print('error: hyp_med_out')
                print(med_out.data.cpu())
                print(hyp_med_out.data.cpu())
                print(self.tp)
                sys.exit()

            hyp_med_out = torch.reshape(hyp_med_out, (hyp_med_out.shape[0], 1, -1))
            hyp_result.append(hyp_med_out)
            
            klein_med_out = pmath.p2k(hyp_med_out, c=self.c)
            klein_med_out = pmath.project(klein_med_out, c=self.c) 
            
            if torch.isnan(klein_med_out).any():
                print('klein_med_out')

            klein_med_out = torch.reshape(klein_med_out, (klein_med_out.shape[0], 1, -1))
            klein_result.append(klein_med_out)
        
        hyp_img_embed = torch.cat(hyp_result, dim=1)
        klein_img_embed = torch.cat(klein_result, dim=1)

        weight_result = []
        lorenz_f_result = []
        for i in range(self.inst_num):
            hyp_img_embed_input_med = hyp_img_embed[:,i,:]
            hyp_img_embed_size = hyp_img_embed.size()
            hyp_img_embed_input = hyp_img_embed_input_med.view(hyp_img_embed_size[0], -1)
            weight = self.att_weight(hyp_img_embed_input)
            
            if torch.isnan(weight).any():
                print('weight')

            klein_img_embed_input_med = klein_img_embed[:,i,:]
            klein_img_embed_size = klein_img_embed.size()
            klein_img_embed_input = klein_img_embed_input_med.view(klein_img_embed_size[0], -1)
            
            lorenz_f = pmath.lorenz_factor(klein_img_embed_input, c=self.c, dim = 1, keepdim=True)
            
            if torch.isnan(lorenz_f).any():
                print('lorenz_f')

            weight_result.append(weight)
            lorenz_f_result.append(lorenz_f)

        embed_weight = torch.cat(weight_result, 1)
        out_weight = embed_weight
        lamb = torch.cat(lorenz_f_result, 1)
        alpha = torch.nn.Softmax(dim=1)(embed_weight)
        
        alpha_lamb = alpha*lamb
        alpha_lamb_sum = torch.sum(alpha_lamb, dim=1)
        alpha_lamb_sum = alpha_lamb_sum.unsqueeze(dim=1)
        alpha_lamb_norm = alpha_lamb / alpha_lamb_sum
        alpha_lamb_norm = torch.reshape(alpha_lamb_norm, (alpha_lamb_norm.shape[0], 1, -1))
        rep = torch.bmm(alpha_lamb_norm, klein_img_embed)
        rep = torch.reshape(rep, (rep.shape[0],-1))
        
        rep = pmath.project(rep, c=self.c)
        
        if torch.isnan(rep).any():
            print('klein_med_out')

        rep = pmath.k2p(rep, c=self.c)
        msi = self.msi_pred(rep)

        if torch.isnan(msi).any():
            print('msi')

        return msi, out_weight

