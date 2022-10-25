import torch
import numpy as np
import torch.nn.functional as F


def dl_to_sampler(dl):
    dl_iter = iter(dl)
    def sample():
        nonlocal dl_iter
        try:
            return next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl)
            return next(dl_iter)
    return sample


@torch.no_grad()
def get_acc(model, dl):
    assert model.training == False
    acc = []
    for X, y in dl:
        acc.append(torch.argmax(model(X), dim=1) == y)
    acc = torch.cat(acc)
    acc = torch.sum(acc)/len(acc)
    return acc.item()


@torch.no_grad()
def get_acc_ensemble(ensemble, dl):
    assert all(model.training == False for model in ensemble)
    acc = []
    for X, y in dl:
        outs = [torch.softmax(model(X), dim=1) for model in ensemble]
        outs = torch.stack(outs, dim=0).mean(dim=0)
        acc.append(torch.argmax(outs, dim=1) == y)
    acc = torch.cat(acc)
    acc = torch.sum(acc)/len(acc)
    return acc.item()

