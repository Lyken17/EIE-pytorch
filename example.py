import torch
from EIE_module import Module
from alexnet import alexnet

if __name__ == "__main__":
    net = alexnet(pretrained=True)
    print(type(net))
    seq = None
    for name, c in net.named_param_list():
        print(name, c.nelement())
        data = c.view(-1)
        if seq is None:
            seq = data
        else:
            seq = torch.cat((seq, data), 0)


