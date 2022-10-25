import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision
import torchvision.utils as vision_utils
from wilds import get_dataset as get_wild_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func
        self.dataset = self.dl.dataset

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))


def process_dataset_65_classes_office_home(datasets, device=None, shuffle=True):
    final_X, final_Y = [], []
    for dataset in datasets:
        for i, (x, y) in enumerate(dataset):
            final_X.append(x)
            final_Y.append(y)
    X = torch.stack(final_X)
    Y = torch.tensor(final_Y).long()
    if shuffle:
        perm = torch.randperm(len(X))
        X = X[perm]
        Y = Y[perm]
    if device is not None:
        X = X.to(device)
        Y = Y.to(device)
    return TensorDataset(X, Y)


def get_OH_65classes_v1(args):
    data_transform = transforms.Compose([
        transforms.Resize((90, 90)),
        transforms.ToTensor()
    ])

    data_train_1 = ImageFolder("./datasets/OfficeHomeDataset_10072016/Product", transform=data_transform)
    data_train_2 = ImageFolder("./datasets/OfficeHomeDataset_10072016/Clipart", transform=data_transform)
    data_train = process_dataset_65_classes_office_home([data_train_1, data_train_2], device=args.device, shuffle=False)

    data_test_1 = ImageFolder("./datasets/OfficeHomeDataset_10072016/Real-world", transform=data_transform)
    data_test = process_dataset_65_classes_office_home([data_test_1], device=args.device, shuffle=False)
    data_test, data_valid, data_perturb = torch.utils.data.random_split(data_test, [1900, 500, 1957], 
                                                                        generator=torch.Generator().manual_seed(42))
    
    train_dl = DataLoader(data_train, batch_size=args.batch_size_train, shuffle=True)
    valid_dl = DataLoader(data_valid, batch_size=args.batch_size_eval, shuffle=False)
    test_dl = DataLoader(data_test, batch_size=args.batch_size_eval, shuffle=False)
    perturb_dl = DataLoader(data_perturb, batch_size=args.batch_size_train, shuffle=True)
    
    return train_dl, valid_dl, test_dl, perturb_dl


def get_OH_65classes_v2(args):
    data_transform = transforms.Compose([
        transforms.Resize((90, 90)), # spatial size of vgg-f input
        transforms.ToTensor()
    ])

    data_train_1 = ImageFolder("./datasets/OfficeHomeDataset_10072016/Product", transform=data_transform)
    data_train_2 = ImageFolder("./datasets/OfficeHomeDataset_10072016/Clipart", transform=data_transform)
    data_train = process_dataset_65_classes_office_home([data_train_1, data_train_2], device=args.device, shuffle=True)

    data_test = ImageFolder("./datasets/OfficeHomeDataset_10072016/Real-world", transform=data_transform)
    data_test = process_dataset_65_classes_office_home([data_test], device=args.device, shuffle=True)

    data_perturb = ImageFolder("./datasets/OfficeHomeDataset_10072016/Art", transform=data_transform)
    data_perturb = process_dataset_65_classes_office_home([data_perturb], device=args.device, shuffle=True)

    data_test, data_valid = torch.utils.data.random_split(data_test, [len(data_test)-800, 800], 
                                                          generator=torch.Generator().manual_seed(42))
    
    train_dl = DataLoader(data_train, batch_size=args.batch_size_train, shuffle=True)
    valid_dl = DataLoader(data_valid, batch_size=args.batch_size_eval, shuffle=False)
    test_dl = DataLoader(data_test, batch_size=args.batch_size_eval, shuffle=False)
    perturb_dl = DataLoader(data_perturb, batch_size=args.batch_size_train, shuffle=True)
    
    return train_dl, valid_dl, test_dl, perturb_dl
    

def get_waterbird_v1(args): # confounder_strength = 0.95
    scale = 256.0/224.0
    transform_test = transforms.Compose([
        transforms.Resize((int(224*scale), int(224*scale))),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    try:
        dataset = get_wild_dataset(dataset="waterbirds", download=False, root_dir="./datasets")
    except:
        dataset = get_wild_dataset(dataset="waterbirds", download=True, root_dir="./datasets")
    
    data_train = dataset.get_subset("train", transform=transform_test)
    data_test  = dataset.get_subset("test", transform=transform_test)
    data_valid = dataset.get_subset("val", transform=transform_test)
    data_perturb = data_valid
    
    train_dl = get_train_loader("standard", data_train, batch_size=args.batch_size_train, num_workers=8, pin_memory=True)
    perturb_dl = get_train_loader("standard", data_perturb, batch_size=args.batch_size_train, num_workers=8, pin_memory=True)
    test_dl = get_eval_loader("standard", data_test, batch_size=args.batch_size_eval, num_workers=5, pin_memory=True)
    valid_dl = get_eval_loader("standard", data_valid, batch_size=args.batch_size_eval, num_workers=5, pin_memory=True)
    
    train_dl = WrappedDataLoader(train_dl, lambda x, y, meta: (x.to(args.device), y.to(args.device)))
    valid_dl = WrappedDataLoader(valid_dl, lambda x, y, meta: (x.to(args.device), y.to(args.device)))
    test_dl = WrappedDataLoader(test_dl, lambda x, y, meta: (x.to(args.device), y.to(args.device)))
    perturb_dl = WrappedDataLoader(perturb_dl, lambda x, y, meta: (x.to(args.device), y.to(args.device)))
    
    return train_dl, valid_dl, test_dl, perturb_dl


def get_camelyon17_v2(args): # ood = val_unlabeled
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    try:
        dataset = get_wild_dataset(dataset="camelyon17", download=False, root_dir="./datasets/")
    except:
        dataset = get_wild_dataset(dataset="camelyon17", download=True, root_dir="./datasets/")
    
    data_train = dataset.get_subset("train", transform=transform)
    data_test  = dataset.get_subset("test", transform=transform)
    data_valid = dataset.get_subset("val", transform=transform)
    data_perturb = get_wild_dataset(dataset="camelyon17", download=False, unlabeled=True, root_dir="./datasets/")
    data_perturb = data_perturb.get_subset("val_unlabeled", transform=transform) # train_unlabeled, val_unlabeled, test_unlabeled
    
    train_dl = get_train_loader("standard", data_train, batch_size=args.batch_size_train, num_workers=8, pin_memory=True)
    perturb_dl = get_train_loader("standard", data_perturb, batch_size=args.batch_size_train, num_workers=8, pin_memory=True)
    test_dl = get_eval_loader("standard", data_test, batch_size=args.batch_size_eval, num_workers=5, pin_memory=True)
    valid_dl = get_eval_loader("standard", data_valid, batch_size=args.batch_size_eval, num_workers=5, pin_memory=True)
    
    train_dl = WrappedDataLoader(train_dl, lambda x, y, meta: (x.to(args.device), y.to(args.device)))
    valid_dl = WrappedDataLoader(valid_dl, lambda x, y, meta: (x.to(args.device), y.to(args.device)))
    test_dl = WrappedDataLoader(test_dl, lambda x, y, meta: (x.to(args.device), y.to(args.device)))
    perturb_dl = WrappedDataLoader(perturb_dl, lambda x, y: (x.to(args.device), y.to(args.device)))
    
    return train_dl, valid_dl, test_dl, perturb_dl


def get_camelyon17_v1(args): # ood = test_unlabeled
    transform = transforms.Compose([
        transforms.ToTensor()
    ])  
    try:
        dataset = get_wild_dataset(dataset="camelyon17", download=False, root_dir="./datasets/")
    except:
        dataset = get_wild_dataset(dataset="camelyon17", download=True, root_dir="./datasets/")
    
    data_train = dataset.get_subset("train", transform=transform)
    data_test  = dataset.get_subset("test", transform=transform)
    data_valid = dataset.get_subset("val", transform=transform)
    data_perturb = get_wild_dataset(dataset="camelyon17", download=False, unlabeled=True, root_dir="./datasets/")
    data_perturb = data_perturb.get_subset("test_unlabeled", transform=transform) # train_unlabeled, val_unlabeled, test_unlabeled
    
    train_dl = get_train_loader("standard", data_train, batch_size=args.batch_size_train, num_workers=8, pin_memory=True)
    perturb_dl = get_train_loader("standard", data_perturb, batch_size=args.batch_size_train, num_workers=8, pin_memory=True)
    test_dl = get_eval_loader("standard", data_test, batch_size=args.batch_size_eval, num_workers=5, pin_memory=True)
    valid_dl = get_eval_loader("standard", data_valid, batch_size=args.batch_size_eval, num_workers=5, pin_memory=True)
    
    train_dl = WrappedDataLoader(train_dl, lambda x, y, meta: (x.to(args.device), y.to(args.device)))
    valid_dl = WrappedDataLoader(valid_dl, lambda x, y, meta: (x.to(args.device), y.to(args.device)))
    test_dl = WrappedDataLoader(test_dl, lambda x, y, meta: (x.to(args.device), y.to(args.device)))
    perturb_dl = WrappedDataLoader(perturb_dl, lambda x, y: (x.to(args.device), y.to(args.device)))
    
    return train_dl, valid_dl, test_dl, perturb_dl


def get_dataset(args):
    if args.dataset == 'kaggle-bird':
        if args.perturb_type == 'ood_is_test':
            return get_kaggle_bird_dl_v1(args)
        else:
            NotImplementedError(f"Version of perturbations '{args.perturb_type}' not implemented for dataset '{args.dataset}'.")
    elif args.dataset == 'camelyon17':
        if args.perturb_type == 'ood_is_not_test':
            return get_camelyon17_v2(args)
        if args.perturb_type == 'ood_is_test':
            return get_camelyon17_v1(args)
        else:
            NotImplementedError(f"Version of perturbations '{args.perturb_type}' not implemented for dataset '{args.dataset}'.")
    elif args.dataset == 'waterbird':
        if args.perturb_type == 'ood_is_test':
            return get_waterbird_v1(args)
        else:
            NotImplementedError(f"Version of perturbations '{args.perturb_type}' not implemented for dataset '{args.dataset}'.")
    elif args.dataset == 'oh-65cls':
        if args.perturb_type == 'ood_is_test':
            return get_OH_65classes_v1(args)
        if args.perturb_type == 'ood_is_not_test':
            return get_OH_65classes_v2(args)
        else:
            NotImplementedError(f"Version of perturbations '{args.perturb_type}' not implemented for dataset '{args.dataset}'.")
    else:
        raise KeyError(f"Unknown dataset '{args.dataset}'.")
