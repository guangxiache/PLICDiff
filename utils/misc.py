import logging  # 用于记录日志
import os  # 用于文件和目录操作
import random  # 用于生成随机数
import time  # 用于时间处理

import numpy as np  # 处理数组和数学计算
import torch  # PyTorch深度学习库
import yaml  # 处理YAML格式的文件
from easydict import EasyDict  # 允许使用点号访问字典的属性

# 定义一个空的黑洞类，用于占位或替代某些属性
class BlackHole(object):
    def __setattr__(self, name, value):
        pass  # 设置属性时无操作

    def __call__(self, *args, **kwargs):
        return self  # 将对象设为可调用，调用时返回自身

    def __getattr__(self, name):
        return self  # 获取属性时返回自身

# 从指定路径加载配置文件
def load_config(path):
    with open(path, 'r') as f:
        return EasyDict(yaml.safe_load(f))  # 返回EasyDict以便于使用点号访问

# 创建和配置日志记录器
def get_logger(name, log_dir=None):
    logger = logging.getLogger(name)  # 根据名称获取日志记录器
    logger.setLevel(logging.DEBUG)  # 设置日志级别为DEBUG
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')  # 设置日志格式
    '''这里是自己的修改部分'''
    # 移除之前的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    stream_handler = logging.StreamHandler()  # 创建控制台输出处理器
    stream_handler.setLevel(logging.DEBUG)  # 设置处理器的日志级别
    stream_handler.setFormatter(formatter)  # 设置处理器的日志格式
    logger.addHandler(stream_handler)  # 添加处理器到日志记录器

    if log_dir is not None:  # 如果指定了日志目录
        file_handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))  # 创建文件输出处理器
        file_handler.setLevel(logging.DEBUG)  # 设置处理器的日志级别
        file_handler.setFormatter(formatter)  # 设置处理器的日志格式
        logger.addHandler(file_handler)  # 添加文件处理器到日志记录器

    return logger  # 返回配置好的日志记录器

# 创建新的日志目录，包含时间戳
def get_new_log_dir(root='./logs', prefix='', tag='', custom_name=''):
    fn = time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime())  # 生成时间戳
    if prefix != '':
        fn = prefix + '_' + fn  # 如果有前缀，添加前缀
    if tag != '':
        fn = fn + '_' + tag  # 如果有标签，添加标签
    log_dir = os.path.join(root, fn)  # 创建完整的日志目录路径
    '''下面的三行是额外加上的，后续需要删去'''
    basename = os.path.basename(log_dir)
    new_basename = basename + '_' + custom_name
    log_dir = os.path.join(os.path.dirname(log_dir), new_basename)
    if os.path.exists(log_dir):
        # 如果目录已经存在，添加一个唯一的后缀
        for i in range(1, 1000):
            new_log_dir = f'{log_dir}_{i}'
            if not os.path.exists(new_log_dir):
                log_dir = new_log_dir
                break    
    os.makedirs(log_dir)  # 创建日志目录
    return log_dir  # 返回日志目录路径

# 设置随机种子以确保可重复性
def seed_all(seed):
    torch.manual_seed(seed)  # 设置PyTorch的随机种子
    np.random.seed(seed)  # 设置NumPy的随机种子
    random.seed(seed)  # 设置Python内置random模块的随机种子

# 记录超参数到TensorBoard
def log_hyperparams(writer, args):
    from torch.utils.tensorboard.summary import hparams  # 导入超参数记录功能
    vars_args = {k: v if isinstance(v, str) else repr(v) for k, v in vars(args).items()}    # 获取参数字典。在字典推导式中，k: v 的意思是创建一个字典的键值对，repr(v) 将值转换为其字符串表示形式。
                                                                                            # vars(args): 这个函数返回一个字典，包含了 args 对象的所有属性以及对应的值。
    exp, ssi, sei = hparams(vars_args, {})  # 记录超参数
    writer.file_writer.add_summary(exp)  # 添加实验记录
    writer.file_writer.add_summary(ssi)  # 添加超参数记录
    writer.file_writer.add_summary(sei)  # 添加超参数记录

# 将逗号分隔的字符串转换为整数元组
def int_tuple(argstr):
    return tuple(map(int, argstr.split(',')))  # 将字符串分割并转换为整数元组

# 将逗号分隔的字符串转换为字符串元组
def str_tuple(argstr):
    return tuple(argstr.split(','))  # 将字符串分割并转换为字符串元组

# 计算模型中可训练参数的数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  # 返回可训练参数的总数
