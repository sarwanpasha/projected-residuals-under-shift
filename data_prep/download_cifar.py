"""Download CIFAR-10 (train and test) and CIFAR-10-C into PRLB_DATA/cifar10.

CIFAR-10 is read from the Hugging Face dataset uoft-cs/cifar10 and stored as uint8
arrays.  CIFAR-10-C is the archive released by Hendrycks and Dietterich (2019) on
Zenodo.  The script finishes by checking that the label order of CIFAR-10-C matches
the CIFAR-10 test split, which every corruption experiment relies on.
"""
import io
import tarfile
import urllib.request

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

from prlb.paths import CIFAR_RAW, CIFAR_C, ensure

CIFAR_C_URL = 'https://zenodo.org/records/2535967/files/CIFAR-10-C.tar'


def fetch_cifar10():
    for split, fn in [('train', 'plain_text/train-00000-of-00001.parquet'),
                      ('test', 'plain_text/test-00000-of-00001.parquet')]:
        p = hf_hub_download('uoft-cs/cifar10', fn, repo_type='dataset')
        t = pq.read_table(p).to_pydict()
        keys = list(t.keys())
        ik = 'img' if 'img' in keys else 'image'
        lk = 'label' if 'label' in keys else 'fine_label'
        X = np.stack([np.array(Image.open(io.BytesIO(r['bytes'])).convert('RGB')) for r in t[ik]])
        y = np.array(t[lk], dtype=np.int64)
        np.save(CIFAR_RAW / ('c10_%s_x.npy' % split), X)
        np.save(CIFAR_RAW / ('c10_%s_y.npy' % split), y)
        print(split, X.shape, y.shape, X.dtype, flush=True)


def fetch_cifar10c():
    if (CIFAR_C / 'labels.npy').exists():
        print('CIFAR-10-C already present', flush=True)
        return
    tar_path = CIFAR_RAW / 'CIFAR-10-C.tar'
    if not tar_path.exists():
        print('downloading', CIFAR_C_URL, flush=True)
        urllib.request.urlretrieve(CIFAR_C_URL, tar_path)
    with tarfile.open(tar_path) as tf:
        tf.extractall(CIFAR_RAW)
    tar_path.unlink()


def main():
    ensure(CIFAR_RAW)
    fetch_cifar10()
    fetch_cifar10c()
    lc = np.load(CIFAR_C / 'labels.npy')
    yt = np.load(CIFAR_RAW / 'c10_test_y.npy')
    print('LABEL_ORDER_MATCH', bool((lc[:10000] == yt).all()), flush=True)


if __name__ == '__main__':
    main()
