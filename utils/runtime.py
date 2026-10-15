"""Paths and device validation shared by the command-line entry points."""
from pathlib import Path
import os
import argparse

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value):
    """Resolve configuration paths relative to the checkout, independent of cwd."""
    path = Path(os.path.expandvars(str(value))).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


def validate_device(value):
    device = torch.device(value)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable. Use --device cpu or install a matching CUDA build.')
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f'CUDA device {device.index} does not exist.')
    return device


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'true', '1', 'yes'}:
        return True
    if value.lower() in {'false', '0', 'no'}:
        return False
    raise argparse.ArgumentTypeError('Use true or false.')


def load_checkpoint(path, device='cpu'):
    path = project_path(path)
    if not Path(path).is_file():
        raise FileNotFoundError(
            f'Checkpoint not found: {path}. Set model.checkpoint in the sampling configuration '
            'or pass --checkpoint.'
        )
    # Project checkpoints include the training EasyDict and, for sample files, PyG objects.
    return torch.load(path, map_location=device, weights_only=False)
