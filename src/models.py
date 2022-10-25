import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as model_zoo
import numpy as np 
    
    
def get_model_func(args):
    if args.dataset == 'camelyon17':
        if args.model == 'resnet50':
            def m_f():
                m = model_zoo.resnet50(pretrained=True)
                d = m.fc.in_features
                m.fc = nn.Linear(d, 2)
                return m.to(args.device)
            return m_f
        else:
            raise NotImplementedError(f"Missing implemntation for model '{args.model}'.")
    elif args.dataset == 'waterbird':
        if args.model == 'resnet50':
            def m_f():
                m = model_zoo.resnet50(pretrained=True)
                d = m.fc.in_features
                m.fc = nn.Linear(d, 2)
                return m.to(args.device)
            return m_f
        else:
            raise NotImplementedError(f"Missing implemntation for model '{args.model}'.")
    elif args.dataset == 'oh-65cls':
        if args.model == 'resnet18':
            def m_f():
                m = model_zoo.resnet18(pretrained=True)
                d = m.fc.in_features
                m.fc = nn.Linear(d, 65)
                return m.to(args.device)
            return m_f
        if args.model == 'resnet50':
            def m_f():
                m = model_zoo.resnet50(pretrained=True)
                d = m.fc.in_features
                m.fc = nn.Linear(d, 65)
                return m.to(args.device)
            return m_f
        else:
            raise NotImplementedError(f"Missing implemntation for model '{args.model}'.")
    else:
        raise KeyError(f"Unknown dataseet '{args.dataset}'.")
