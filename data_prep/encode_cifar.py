"""Encode CIFAR-10 and the 19 x 5 CIFAR-10-C shift sets with a frozen encoder.

Usage:  python data_prep/encode_cifar.py {clip|dinov2}

clip    CLIP ViT-B/32 image embedding (openai/clip-vit-base-patch32), d = 512
dinov2  DINOv2 ViT-B/14 class token (facebook/dinov2-base), d = 768

Images are resized to 224 x 224 with bicubic interpolation and normalized with the
statistics of each encoder.  Outputs, in PRLB_DATA/cifar10_emb:
    <enc>_train.npy, <enc>_clean.npy, <enc>_<corruption>.npz (keys s1..s5),
    labels_train.npy, labels_test.npy
"""
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from prlb.paths import CIFAR_RAW, CIFAR_C, CIFAR_EMB, CORRUPTIONS, ensure

dev = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_cifar():
    return (np.load(CIFAR_RAW / 'c10_train_x.npy'), np.load(CIFAR_RAW / 'c10_train_y.npy'),
            np.load(CIFAR_RAW / 'c10_test_x.npy'), np.load(CIFAR_RAW / 'c10_test_y.npy'))


def build(name):
    if name == 'clip':
        from transformers import CLIPModel
        m = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(dev).eval()
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        fn = lambda x: m.get_image_features(pixel_values=x)
    else:
        from transformers import Dinov2Model
        m = Dinov2Model.from_pretrained('facebook/dinov2-base').to(dev).eval()
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        fn = lambda x: m(pixel_values=x).last_hidden_state[:, 0]
    mu = torch.tensor(mean, device=dev).view(1, 3, 1, 1)
    sd = torch.tensor(std, device=dev).view(1, 3, 1, 1)
    return fn, mu, sd


@torch.no_grad()
def encode(arr, fn, mu, sd, bs=384, res=224):
    out = []
    for i in range(0, len(arr), bs):
        x = torch.from_numpy(arr[i:i + bs]).to(dev).permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(x, size=(res, res), mode='bicubic', align_corners=False)
        x = (x - mu) / sd
        with torch.autocast('cuda', dtype=torch.float16, enabled=(dev == 'cuda')):
            z = fn(x)
        out.append(z.float().cpu().numpy().astype(np.float32))
    return np.concatenate(out)


def main():
    enc = sys.argv[1]
    if enc not in ('clip', 'dinov2'):
        raise SystemExit('encoder must be clip or dinov2')
    ensure(CIFAR_EMB)
    fn, mu, sd = build(enc)
    Xtr, ytr, Xte, yte = load_cifar()
    np.save(CIFAR_EMB / 'labels_train.npy', ytr)
    np.save(CIFAR_EMB / 'labels_test.npy', yte)
    t0 = time.time()
    np.save(CIFAR_EMB / ('%s_train.npy' % enc), encode(Xtr, fn, mu, sd))
    print('train done %.1fs' % (time.time() - t0), flush=True)
    np.save(CIFAR_EMB / ('%s_clean.npy' % enc), encode(Xte, fn, mu, sd))
    print('clean done %.1fs' % (time.time() - t0), flush=True)
    for cn in CORRUPTIONS:
        p = CIFAR_C / ('%s.npy' % cn)
        if not p.exists():
            print('MISSING', cn, flush=True)
            continue
        A = np.load(p)
        Z = {}
        for s in range(5):
            Z['s%d' % (s + 1)] = encode(A[s * 10000:(s + 1) * 10000], fn, mu, sd)
        np.savez(CIFAR_EMB / ('%s_%s.npz' % (enc, cn)), **Z)
        print('%s done %.1fs' % (cn, time.time() - t0), flush=True)
    print('ENCODE_COMPLETE', enc, flush=True)


if __name__ == '__main__':
    main()
