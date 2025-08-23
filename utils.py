import yaml
import os
import torch
from torch.backends import cudnn
import random
import importlib
from models.BaseModel import BaseModel
import numpy as np
import data_process
import torchvision
import time
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


def read_config(yaml_path):
    with open(yaml_path, "r") as imf:
        config = yaml.load(imf.read())
    return config


def save_checkpoint(state, config):
    expr_dir = os.path.join(config["checkpoint_dir"], config["dataset"], config["experiment_name"])

    if not os.path.exists(expr_dir):
        os.makedirs(expr_dir)
    epoch = state["epoch"]
    save_dir = os.path.join(expr_dir, str(epoch) + ".pth")
    torch.save(state, save_dir)


def set_seed(seed, cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def init(config, local_rank, use_ddp):
    cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    cudnn.deterministic = True
    torch.manual_seed(config["seed"])  # Set random seed for CPU
    torch.cuda.manual_seed(config["seed"])  # Set random seed for the current GPU
    torch.cuda.manual_seed_all(config["seed"])  # Set random seed for all GPUs
    random.seed(config["seed"])
    config["local_rank"] = local_rank
    config["use_ddp"] = use_ddp
    set_seed(config["seed"])

    os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu_ids"]

    if config["use_ddp"]:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")

    print_options(config)


def print_options(config):
    expr_dir = os.path.join(config["checkpoint_dir"], config["dataset"], config["experiment_name"])

    if not os.path.exists(expr_dir) and ((config["local_rank"] == 0 and config["use_ddp"]) or not config["use_ddp"]):
        os.makedirs(expr_dir)

    message = "--------------------Options----------------------\n"
    for k in list(config.keys()):
        val = config[k]
        comment = str(k) + ":\t"
        if val == None:
            comment += "None\n"
        else:
            comment += str(val) + "\n"
        message += comment
    message += "--------------------End----------------------\n"

    # phase = 'train' if not config['eval'] else 'val'
    phase = "train"
    file_name = "{}_{}_opt.txt".format(phase, get_time())
    path = os.path.join(config["checkpoint_dir"], config["dataset"], config["experiment_name"], file_name)
    with open(path, "w") as imf:
        imf.write(message)


def get_time():
    return str(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))


def create_model(config):
    model_name = "models." + config["model_name"]  # e.g. 'Models.nd'
    modellib = importlib.import_module(model_name)  # e.g. 'Models.nd'
    model = None
    target_model_name = config["model_name"]  # e.g. 'nd'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() and issubclass(cls, BaseModel):
            model = cls

    if model == None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_name, target_model_name))
        exit(0)

    instance = model(config)
    return instance
