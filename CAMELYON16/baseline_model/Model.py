import torch
import torchvision

class MINet(torch.nn.Module):
    def __init__(self):
        # for downloading pretrained model
        super(MINet, self).__init__()
        resnet18_pretn = torchvision.models.resnet18(pretrained=True)
        pretn_state_dict = resnet18_pretn.state_dict()
        self.resnet18 = torchvision.models.resnet18(num_classes=2)
        model_state_dict = self.resnet18.state_dict()
        # restore all the weight except the last layer
        update_state = {k:v for k, v in pretn_state_dict.items() if k not in ["fc.weight", "fc.bias"] and k in model_state_dict}
        model_state_dict.update(update_state)
        self.resnet18.load_state_dict(model_state_dict)

    def forward(self, x):
        out = self.resnet18(x)
        return out

 
