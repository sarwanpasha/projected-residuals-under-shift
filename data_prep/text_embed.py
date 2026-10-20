"""CLIP ViT-B/32 text prototypes for the zero-shot head.

Usage:  python data_prep/text_embed.py

Each class prototype is the normalized mean of the normalized text embeddings of a
small prompt ensemble.  Writes data/cifar10_emb/clip_text.npy (10 x 512) and
data/domainnet/emb/clip_text.npy (345 x 512, in the order of classes.txt).
"""
import numpy as np
import torch
from transformers import CLIPModel, CLIPTokenizer

from prlb.paths import CIFAR_EMB, DNET_EMB

CIFAR_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
                 'horse', 'ship', 'truck']
TEMPLATES = ['a photo of a {}.', 'a blurry photo of a {}.', 'a low resolution photo of a {}.',
             'a drawing of a {}.', 'a painting of a {}.', 'a sketch of a {}.',
             'art of the {}.']

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
m = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(dev).eval()
tok = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')


@torch.no_grad()
def protos(names):
    out = []
    for nm in names:
        t = tok([tp.format(nm) for tp in TEMPLATES], padding=True, return_tensors='pt').to(dev)
        e = m.get_text_features(**t).float()
        e = e / e.norm(dim=1, keepdim=True)
        e = e.mean(0)
        out.append((e / e.norm()).cpu().numpy())
    return np.stack(out)


np.save(CIFAR_EMB / 'clip_text.npy', protos(CIFAR_CLASSES))
names = [c.replace('_', ' ') for c in (DNET_EMB / 'classes.txt').read_text().split()]
np.save(DNET_EMB / 'clip_text.npy', protos(names))
print('text prototypes written', flush=True)
