import torch
import torchvision

class MINet(torch.nn.Module):
    def __init__(self, embed_len, inst_num):
        # for downloading pretrained model
        super(MINet, self).__init__()
        self.inst_num = inst_num
        resnet18_pretn = torchvision.models.resnet18(pretrained=True)
        pretn_state_dict = resnet18_pretn.state_dict()
        self.resnet18 = torchvision.models.resnet18(num_classes=embed_len)
        model_state_dict = self.resnet18.state_dict()
        # restore all the weight except the last layer
        update_state = {k:v for k, v in pretn_state_dict.items() if k not in ["fc.weight", "fc.bias"] and k in model_state_dict}
        model_state_dict.update(update_state)
        self.resnet18.load_state_dict(model_state_dict)

        self.att_weight = torch.nn.Sequential(
                        torch.nn.BatchNorm1d(embed_len),
                        torch.nn.ReLU(True),
                        torch.nn.Linear(embed_len, 1)
                        )
        
        self.label_pred = torch.nn.Sequential(
                        torch.nn.BatchNorm1d(embed_len),
                        torch.nn.ReLU(True),
                        torch.nn.Linear(embed_len, 2)
                        )

    def forward(self, x): 
        result = []
        for i in range(self.inst_num):
            #input = torch.squeeze(x[:,i,:,:,:])
            med = x[:,i,:,:,:]
            x_size = x.size()
            input = med.view(x_size[0], x_size[2], x_size[3], x_size[4])
            out = self.resnet18(input)
            out = torch.reshape(out, (out.shape[0], 1, -1))
            result.append(out)
        img_embed = torch.cat(result, dim=1)
        weight_result = []
        for i in range(self.inst_num):
            #img_embed_input = torch.squeeze(img_embed[:,i,:])
            img_embed_input_med = img_embed[:,i,:]
            img_embed_size = img_embed.size()
            img_embed_input = img_embed_input_med.view(img_embed_size[0], -1)
            weight = self.att_weight(img_embed_input)
            weight_result.append(weight)

        embed_weight = torch.cat(weight_result, 1)
        embed_weight = torch.nn.Softmax(dim=1)(embed_weight)
        out_weight = embed_weight
        embed_weight = torch.reshape(embed_weight, (embed_weight.shape[0], 1, -1))
        rep = torch.bmm(embed_weight, img_embed)
        rep = torch.reshape(rep, (rep.shape[0],-1))

        label = self.label_pred(rep)
        
        return label, out_weight

