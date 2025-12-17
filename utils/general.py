"""General utility functions."""

import os
import sys
import yaml
import logging
import random
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, Any
import platform

# Set up logging
LOGGER = logging.getLogger(__name__)

def set_logging(name='', verbose=True):
    """Set up logging configuration."""
    rank = int(os.getenv('RANK', -1))
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=level
    )

def init_seeds(seed=0, deterministic=False):
    """Initialize random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def colorstr(*input):
    """Colorize string output."""
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])
    colors = {
        'black': '\033[30m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',
        'bold': '\033[1m',
        'underline': '\033[4m',
    }
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

def increment_path(path, exist_ok=False, mkdir=False):
    """Increment path if it exists."""
    path = Path(path)
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        for n in range(2, 9999):
            p = f'{path}{n}{suffix}'
            if not Path(p).exists():
                break
        path = Path(p)
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    return path

def check_file(file: str, suffix=''):
    """Check if file exists and has correct suffix."""
    file = Path(str(file).strip().replace("'", '').replace('"', ''))
    if not file.exists():
        raise FileNotFoundError(f"File not found: {file}")
    if suffix and file.suffix != suffix:
        raise ValueError(f"File {file} does not have suffix {suffix}")
    return str(file)

def check_yaml(file: str, suffix=('.yaml', '.yml')):
    """Check if YAML file exists and is valid."""
    file = Path(file)
    if not file.exists():
        raise FileNotFoundError(f"YAML file not found: {file}")
    if file.suffix not in suffix:
        raise ValueError(f"File {file} is not a YAML file")
    try:
        with open(file, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"Error loading YAML file {file}: {e}")

def check_dataset(data: str):
    """Check and load dataset configuration."""
    data = check_file(data, suffix='.yaml')
    data_dict = check_yaml(data)
    
    # Validate dataset structure
    required_keys = ['path', 'train', 'val', 'nc', 'names']
    for key in required_keys:
        if key not in data_dict:
            raise ValueError(f"Dataset YAML missing required key: {key}")
    
    # Resolve paths
    path = Path(data_dict['path'])
    if not path.is_absolute():
        path = Path(data).parent / path
    data_dict['path'] = str(path.resolve())
    
    return data_dict

def check_img_size(img_size, s=32):
    """Check image size is multiple of stride s."""
    new_size = make_divisible(img_size, int(s))
    if new_size != img_size:
        LOGGER.warning(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
    return new_size

def make_divisible(x, divisor):
    """Make x divisible by divisor."""
    return int(np.ceil(x / divisor)) * divisor

def copy_attr(a, b, include=(), exclude=()):
    """Copy attributes from b to a, optionally filtering."""
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)

def intersect_dicts(da, db, exclude=()):
    """Return intersection of two dictionaries."""
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}

def print_args(args=None, show_file=True, show_func=False):
    """Print function arguments."""
    x = sys._getframe(1).f_locals
    s = ', '.join(f'{k}={v!r}' for k, v in sorted(x.items()) if k not in ('self', 'args'))
    if show_file:
        s = f'{Path(sys._getframe(1).f_code.co_filename).name}: {s}'
    if show_func:
        s = f'{sys._getframe(1).f_code.co_name}: {s}'
    LOGGER.info(s)

def yaml_save(file='data.yaml', data=None):
    """Save data to YAML file."""
    file = Path(file)
    if data is None:
        data = {}
    file.parent.mkdir(parents=True, exist_ok=True)
    with open(file, 'w') as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True)

def yaml_load(file='data.yaml', append_filename=False):
    """Load YAML file."""
    with open(file, errors='ignore', encoding='utf-8') as f:
        s = yaml.safe_load(f) or {}
        if append_filename:
            s['yaml_file'] = str(file)
        return s

def one_cycle(y1=0.0, y2=1.0, steps=100):
    """Generate one cycle learning rate schedule."""
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1

import math

