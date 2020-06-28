import torch
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.nn.functional as F
import math
import hyptorch.nn as hypnn
import hyptorch.pmath as pmath

class MINet(torch.nn.Module):
    def __init__(self, embed_len, inst_num, c=0.05):
        # for downloading pretrained model
        super(MINet, self).__init__()
        self.inst_num = inst_num
        resnet18_pretn = models.resnet18(pretrained=True)
        pretn_state_dict = resnet18_pretn.state_dict()
        self.resnet18 = models.resnet18(num_classes=embed_len)
        model_state_dict = self.resnet18.state_dict()
        update_state = {k:v for k, v in pretn_state_dict.items() if k not in ["fc.weight", "fc.bias"] and k in model_state_dict}
        model_state_dict.update(update_state)
        self.resnet18.load_state_dict(model_state_dict)

        self.c = c
        self.tp = hypnn.ToPoincare(c=self.c, train_x=True, train_c=True, ball_dim=embed_len)
        self.att_weight = hypnn.HypLinear(embed_len, 1, c=self.c)
        self.msi_pred = hypnn.HyperbolicMLR(ball_dim=embed_len, n_classes=3, c=self.c)

    def forward(self, x): 
        hyp_result = []
        klein_result = []
        for i in range(self.inst_num):
            med = x[:,i,:,:,:]
            x_size = x.size()
            med_input = med.view(x_size[0], x_size[2], x_size[3], x_size[4])
            med_out = self.resnet18(med_input)
            hyp_med_out = self.tp(med_out)
            
            hyp_med_out = torch.reshape(hyp_med_out, (hyp_med_out.shape[0], 1, -1))
            hyp_result.append(hyp_med_out)
            
            klein_med_out = pmath.p2k(hyp_med_out, c=self.c)
            klein_med_out = pmath.project(klein_med_out, c=self.c) 
            
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

            klein_img_embed_input_med = klein_img_embed[:,i,:]
            klein_img_embed_size = klein_img_embed.size()
            klein_img_embed_input = klein_img_embed_input_med.view(klein_img_embed_size[0], -1)
            
            lorenz_f = pmath.lorenz_factor(klein_img_embed_input, c=self.c, dim = 1, keepdim=True)
            
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
        
        rep = pmath.k2p(rep, c=self.c)
        msi = self.msi_pred(rep)
            
        return msi, out_weight

