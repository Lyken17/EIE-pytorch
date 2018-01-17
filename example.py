from pprint import pprint

import torch
import torch.nn as nn
from torch.autograd import Variable

from sklearn.cluster import KMeans

from EIE_module import Module
from alexnet import alexnet

EIE_target_list = (nn.modules.conv._ConvNd, nn.Conv2d, nn.Linear,)


def is_EIE_module(module):
    for each in EIE_target_list:
        if isinstance(module, each):
            return True
    return False


import numpy as np


def enumrate_params(model):
    # print(name, model._parameters)
    if is_EIE_module(model):
        for key, value in model._parameters.items():
            yield model, key, value

    for child in model.children():
        if is_EIE_module(child):
            for name, key, value in enumrate_params(child):
                yield name, key, value


def select_kmeans_center(params, centers=8):
    kmeans = KMeans(n_clusters=8).fit(params)
    return kmeans.labels_, kmeans.centers_


def convert_index_to_binary(ind):
    pass


def quantize_state_dict(table, state_dict):
    pass


def dequantize_state_dict(table, state_dict):
    pass


if __name__ == "__main__":
    d = Variable(torch.zeros([1, 20]).float())
    c = nn.Sequential(nn.Linear(20, 10), nn.Linear(10, 10), nn.Linear(10, 5))
    # print(is_EIE_module(nn.Linear(20, 10)))
    plist = []
    for module, key, params in enumrate_params(c):
        print(module, key, params.size())
        plist.append(params.data.view(-1))

    torch.save({'table': table, "load_dict": data, "interest": meaning, })
