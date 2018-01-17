from collections import defaultdict, OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.modules.module import _addindent

EIE_target_module = (
    nn.modules.conv._ConvNd,
    nn.Conv2d,
    nn.Linear,
)


def To_EIE_model(model):
    for module in model.children():
        if module not in EIE_target_module:
            pass
        module.train()

import numpy as np


class Module(nn.Module):
    def __init__(self, *args, **kwargs):
        if isinstance(args[0], nn.Module):
            model = args[0]
            self.__dict__ = model.__dict__

            # dirty fix, any better method?
            self.forward = model.forward
        else:
            super(Module, self).__init__(*args, **kwargs)
        self.quantized = False
        self.hist = dict()

    def load_state_dict(self, state_dict, strict=True):
        super(Module, self).load_state_dict(state_dict, strict)

    def parameters(self):
        for name, param in self.named_parameters():
            yield param

    def named_param_list(self):
        for name, param in self.named_parameters():
            yield name, param.data

    def param_list(self):
        for name, data in self.named_param_list():
            yield data

    def param_list(self):
        pass


    def quantization(self):
        pass

    def flatted_parameter_list(self):
        x = None
        for _ in self.parameters():
            n = _.view(-1).data
            if x is None:
                x = n
            x = torch.cat((x, n), 0)
        return x


# make sure to put Module at the first position
class Conv2d(Module, nn.Conv2d):
    pass


class Linear(Module, nn.Linear):
    pass

from pprint import pprint
from torch.in
if __name__ == "__main__":
    d = Variable(torch.zeros([1, 20]).float())
    c = Linear(10, 20)
    c = nn.Sequential(
        nn.Linear(20, 10),
        nn.Linear(10, 10)
    )
    pprint(c.__dict__)
    pprint(c.forward(d))

    b = Module(c)
    pprint(b.__dict__)
    pprint(b.forward(d))
    # res = c.parameter_list
    # print(res, type(res))
    #
    # for name, c in c.named_param_list():
    #     print(name, c.size())
