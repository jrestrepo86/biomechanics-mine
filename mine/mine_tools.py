#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mine Tools
"""

import numpy as np
import ray
import scipy.io as spio
import torch
import torch.nn as nn
from ray.experimental.tqdm_ray import tqdm


def toColVector(x):
    """
    Change vectors to column vectors
    """
    x = x.reshape(x.shape[0], -1)
    if x.shape[0] < x.shape[1]:
        x = x.T
    x.reshape((-1, 1))
    return x


def get_activation_fn(afn):
    """
    Get activation functions
    """
    activation_functions = {
        "linear": lambda: lambda x: x,
        "relu": nn.ReLU,
        "relu6": nn.ReLU6,
        "elu": nn.ELU,
        "prelu": nn.PReLU,
        "leaky_relu": nn.LeakyReLU,
        "threshold": nn.Threshold,
        "hardtanh": nn.Hardtanh,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "log_sigmoid": nn.LogSigmoid,
        "softplus": nn.Softplus,
        "softshrink": nn.Softshrink,
        "softsign": nn.Softsign,
        "tanhshrink": nn.Tanhshrink,
        "softmax": nn.Softmax,
    }

    if afn not in activation_functions:
        raise ValueError(
            f"'{afn}' is not included in activation_functions. Use below one \n {activation_functions.keys()}"
        )

    return activation_functions[afn]


class MovingAverageSmooth:
    def __init__(self):
        self.ca = None
        self.n = 0

    def __call__(self, loss):
        self.n += 1
        if self.ca is None:
            self.ca = loss
        else:
            self.ca = self.ca + (loss - self.ca) / self.n
        return self.ca


class ExpMovingAverageSmooth:
    def __init__(self, alpha=0.01):
        self.ema = None
        self.alpha = alpha

    def __call__(self, loss):
        if self.ema is None:
            self.ema = loss
        else:
            self.ema = self.alpha * loss + (1.0 - self.alpha) * self.ema
        return self.ema


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = np.abs(delta)
        self.counter = 0
        self.early_stop = False
        self.min_loss = np.inf

    def __call__(self, loss):
        if loss < self.min_loss - self.delta:
            self.min_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def mine_data_loader(X, Y, val_size=0.2, device="cuda"):
    n = X.shape[0]
    # send data top device
    X = torch.from_numpy(X)
    Y = torch.from_numpy(Y)

    # mix samples
    inds = np.random.permutation(n)
    X = X[inds, :].to(device)
    Y = Y[inds, :].to(device)
    # split data in training and validation sets
    val_size = int(val_size * n)
    inds = torch.randperm(n)
    (val_idx, train_idx) = (inds[:val_size], inds[val_size:])

    Xtrain = X[train_idx, :]
    Ytrain = Y[train_idx, :]
    Xval = X[val_idx, :]
    Yval = Y[val_idx, :]
    Xtrain = Xtrain.to(device)
    Ytrain = Ytrain.to(device)
    Xval = Xval.to(device)
    Yval = Yval.to(device)
    return Xtrain, Ytrain, Xval, Yval, X, Y


@ray.remote
class Progress:
    def __init__(self, max_it=1, pbar=True):
        self.pbar_flag = pbar
        self.pbar = tqdm(total=max_it)
        self.count = -1
        self.max_it = max_it
        self.update()

    def update(self):
        if self.pbar_flag:
            self.pbar.update()
        else:
            self.count += 1
            p = 100 * self.count / self.max_it
            print(f"Progress: {p:.2f}%  {self.count}/{self.max_it}")


def loadmat(filename):
    """
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    """
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    """
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict
