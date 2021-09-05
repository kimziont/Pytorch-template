import os, time, gc, argparse, json, random, wandb
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from omegaconf import OmegaConf
from importlib import import_module

from torch.optim import lr_scheduler
from optimizer import get_optimizer_args
from scheduler import get_scheduler_args





def get_config(config):
    parser = argparse.ArgumentParser(description="Mask Classification")

    parser.add_argument("--PROJECT_PATH", type=str, default=config["PROJECT_PATH"])
    parser.add_argument("--BASE_DATA_PATH", type=str, default=config["BASE_DATA_PATH"])
    parser.add_argument("--csv_path", type=str, default=config["csv_path"])

    # hyper parameters
    parser.add_argument("--lr", type=float, default=config["learning_rate"])
    parser.add_argument("--batch_size", type=int, default=config["batch_size"])
    parser.add_argument("--num_workers", type=int, default=config["num_workers"])
    parser.add_argument("--epochs", type=int, default=config["epochs"], help=f'number of epochs to train (default: {config["epochs"]})')
    parser.add_argument("--seed", type=int, default=config["seed"], help=f'random seed (default: {config["seed"]})')
    parser.add_argument("--patience", type=int, default=config["patience"], help=f'early stopping patience (default: {config["patience"]})')
    parser.add_argument("--resize_width", type=int, default=config["resize_width"], help='resize_width size for image when training')
    parser.add_argument("--resize_height", type=int, default=config["resize_height"], help='resize_height size for image when training')

    # model environment
    parser.add_argument('--model', type=str, default=config["model"], help=f'model type (default: {config["model"]})')
    parser.add_argument('--kfold', type=int, default=config["kfold"], help=f'K-Fold (default: {config["kfold"]})')
    parser.add_argument('--optimizer', type=str, default=config["optimizer"], help=f'optimizer type (default: {config["optimizer"]})')
    parser.add_argument('--criterion', type=str, default=config["criterion"], help=f'criterion type (default: {config["criterion"]})')
    parser.add_argument('--lr_scheduler', type=str, default=config["lr_scheduler"], help=f'scheduler type (default: {config["lr_scheduler"]})')
    parser.add_argument('--description', type=str, default=config["description"], help='model description')

    # CLI에서 입력 또는 default로 받아온 값을 사용하도록 해준다
    args = parser.parse_args()

    with open("config.json", "r") as jsonfile:
        data = json.load(jsonfile)
        
    data["PROJECT_PATH"] = args.PROJECT_PATH
    data["BASE_DATA_PATH"] = args.BASE_DATA_PATH
    data["csv_path"] = args.csv_path

    data["learning_rate"] = args.lr
    data["batch_size"] = args.batch_size
    data["num_workers"] = args.num_workers
    data["epochs"] = args.epochs
    data["seed"] = args.seed
    data["patience"] = args.patience
    data["resize_width"] = args.resize_width
    data["resize_height"] = args.resize_height      

    data["model"] = args.model
    data["kfold"] = args.kfold
    data["optimizer"] = args.optimizer
    data["criterion"] = args.criterion
    data["lr_scheduler"] = args.lr_scheduler
    data["description"] = args.description

    data["img_dir"] = os.path.join(data["BASE_DATA_PATH"], data["img_dir"])
    data["csv_path"] = os.path.join(data["BASE_DATA_PATH"], data["csv_path"])
    data["docs_path"] = os.path.join(data["PROJECT_PATH"], data["docs_path"])
    data["model_path"] = os.path.join(data["PROJECT_PATH"], data["model_path"])
    
    with open("config.json", "w") as jsonfile:
        json.dump(data, jsonfile, indent=4)
    return OmegaConf.load("config.json")



def set_random_seed(seed):
    """
    for Reproducible Model experiment, set random seed
    """
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU


def get_model(config):
    model_module = getattr(import_module("models"), config["model"])
    model = model_module()
    model.cuda()
    return model


def get_optimizer(config, model):
    # optimizer.py 에서 가져오자
    optimizer_module = getattr(optim, config["optimizer"])
    optimizer_args = get_optimizer_args(config, config["optimizer"])
    optimizer = optimizer_module(params=model.parameters(), **optimizer_args)
    return optimizer

def get_criterion(config):
    criterion_module = getattr(nn, config["criterion"])
    criterion = criterion_module()
    return criterion

def get_lr_scheduler(config, optimizer):
    lr_scheduler_module = getattr(optim.lr_scheduler, config["lr_scheduler"])
    scheduler_args = get_scheduler_args(config, config["lr_scheduler"])
    lr_scheduler = lr_scheduler_module(optimizer, **scheduler_args)
    return lr_scheduler

def get_dataset(config):
    dataset_module = getattr(import_module("dataloaders.dataset"), config["dataset"])
    dataset = dataset_module()
    return dataset


def get_trainer(config, device, model, optimizer, criterion, dataset, lr_scheduler):
    trainer_module = trainer = getattr(import_module("trainer"), config["trainer"])
    trainer = trainer_module(config, device, model, optimizer, criterion, dataset, lr_scheduler)
    return trainer


def main():
    config = OmegaConf.load("config.json")
    config = get_config(config)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    set_random_seed(seed=config["seed"])
    print(f"device: {device}")
    print(f"torch_version: {torch.__version__}")
    print(f"dataset: {config['dataset']}")
    print(f"model: {config['model']}")
    print(f"optimizer: {config['optimizer']}")
    print(f"batch_size: {config['batch_size']}")
    print(f"kfold: {config['kfold']}")

    model = get_model(config)
    optimizer = get_optimizer(config, model)
    criterion = get_criterion(config)
    lr_scheduler = get_lr_scheduler(config, optimizer)
    wandb.watch(model)
    dataset = get_dataset(config)

    trainer = get_trainer(config, device, model, optimizer, criterion, dataset, lr_scheduler)
    trainer.train()



if __name__ == '__main__':
    wandb.init(project="Mask-classification")
    main()