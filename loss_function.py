# encoding: utf-8

import torch
import torch.nn.functional as F


def cross_entropy(y_hat, y):

    epsilon = 0.00001
    y_diff  = y_hat.gather(1, y.view(-1, 1)) + epsilon

    # print(y_diff.view(1,-1))

    return - torch.log(y_diff).sum()
    # return -y_diff.sum()



def cross_entropy_weighted(y_hat, y, weight={}):
    return sum([F.cross_entropy(y_hat[y==c], y[y==c])*weight.get(c.item(),1) for c in y.unique()]).sum()
