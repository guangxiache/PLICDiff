import logging
import os
import random
import time

import numpy as np
import torch
import yaml
from easydict import EasyDict
from utils.runtime import project_path


def load_config(path):
    with open(project_path(path), 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    for section, keys in [('data', ('path', 'split')), ('model', ('checkpoint',))]:
        if section in config:
            for key in keys:
                if config[section].get(key):
                    config[section][key] = project_path(config[section][key])
    return config

def get_logger(name, log_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def get_new_log_dir(root='./logs', prefix='', tag='', custom_name=''):
    fn = time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime())
    if prefix != '':
        fn = prefix + '_' + fn
    if tag != '':
        fn = fn + '_' + tag
    log_dir = os.path.join(root, fn)
    basename = os.path.basename(log_dir)
    new_basename = basename + '_' + custom_name
    log_dir = os.path.join(os.path.dirname(log_dir), new_basename)
    if os.path.exists(log_dir):
        for i in range(1, 1000):
            new_log_dir = f'{log_dir}_{i}'
            if not os.path.exists(new_log_dir):
                log_dir = new_log_dir
                break
    os.makedirs(log_dir)
    return log_dir

def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
