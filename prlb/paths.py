"""Filesystem layout shared by every script.

All locations derive from two roots that can be overridden by environment variables:

    PRLB_DATA     datasets and embeddings     (default: <repo>/data)
    PRLB_RESULTS  pickles, logs and figures   (default: <repo>/results)
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('PRLB_DATA', ROOT / 'data'))
RESULTS = Path(os.environ.get('PRLB_RESULTS', ROOT / 'results'))

CIFAR_RAW = DATA / 'cifar10'
CIFAR_C = CIFAR_RAW / 'CIFAR-10-C'
CIFAR_EMB = DATA / 'cifar10_emb'

DNET = DATA / 'domainnet'
DNET_TXT = DNET / 'txt'
DNET_ZIPS = DNET / 'zips'
DNET_EMB = DNET / 'emb'

SYN_OUT = RESULTS / 'synthetic'
CIFAR_OUT = RESULTS / 'cifar10c'
DNET_OUT = RESULTS / 'domainnet'
FIG_OUT = RESULTS / 'figures'

CORRUPTIONS = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
               'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
               'brightness', 'contrast', 'elastic_transform', 'pixelate',
               'jpeg_compression', 'speckle_noise', 'gaussian_blur', 'spatter',
               'saturate']

DOMAINS = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']


def ensure(*dirs):
    """Create each directory if needed and return the first one."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    return Path(dirs[0])
