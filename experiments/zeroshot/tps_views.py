"""Test-time prototype shifting with augmented views on CIFAR-10-C (zero-shot CLIP).

Usage:  python experiments/zeroshot/tps_views.py

Follows the test-time prototype shifting protocol of Sui et al. (2025) on the zero-shot
CLIP ViT-B/32 head: for every test image, 63 random resized crops with random
horizontal flips are embedded together with the original view, the 10 percent of
views with the lowest prediction entropy are kept, and one Adam step from zero shifts
minimizes the entropy of their averaged prediction.  The shifted prototypes then
classify the original view, and the shifts are reset for the next image.  The shift
is one vector per class or one vector shared by all classes.  Alongside, on the same
images:

  * the zero-shot head with no correction, with the common-mean feature correction,
    and with the per-class oracle translation of the paper;
  * the confident-view ensemble alone, which averages the kept views with the
    unshifted prototypes, to separate the effect of the views from that of the shift;
  * prototype shifting applied after the common-mean correction.

1000 test images per corruption and severity (severities 1, 3 and 5) are used, the
same images for every method.  Every learning rate of LRS is stored.

Output: results/zeroshot/tps_views.pkl
"""
import pickle

import numpy as np
import torch
from torchvision.ops import roi_align
from transformers import CLIPModel

from prlb import core as R
from prlb.paths import CIFAR_C, CIFAR_EMB, CORRUPTIONS, RESULTS, ensure

OUT = RESULTS / 'zeroshot'
NV = 63
KEEP = 6
NIMG = 1000
SEVS = (1, 3, 5)
LRS = (1e-3, 5e-3, 1e-2)
SCALE = 100.0
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
MU = torch.tensor((0.48145466, 0.4578275, 0.40821073), device=dev).view(1, 3, 1, 1)
SD = torch.tensor((0.26862954, 0.26130258, 0.27577711), device=dev).view(1, 3, 1, 1)


def views(imgs, g):
    """NV random resized crops (area 0.3 to 1, aspect 3/4 to 4/3) with random flips,
    resampled to 224 x 224, for a batch of uint8 images of shape (n, 32, 32, 3)."""
    x = torch.from_numpy(imgs).to(dev).permute(0, 3, 1, 2).float() / 255.0
    n, _, H, W = x.shape
    area = torch.empty(n * NV, device=dev).uniform_(0.3, 1.0, generator=g)
    logr = torch.empty(n * NV, device=dev).uniform_(np.log(3 / 4), np.log(4 / 3), generator=g)
    r = torch.exp(logr)
    w = torch.clamp(torch.sqrt(area * r) * W, 4, W)
    h = torch.clamp(torch.sqrt(area / r) * H, 4, H)
    x0 = torch.rand(n * NV, device=dev, generator=g) * (W - w)
    y0 = torch.rand(n * NV, device=dev, generator=g) * (H - h)
    idx = torch.arange(n, device=dev).repeat_interleave(NV).float()
    boxes = torch.stack([idx, x0, y0, x0 + w, y0 + h], 1)
    v = roi_align(x, boxes, output_size=(224, 224), spatial_scale=1.0, sampling_ratio=2, aligned=True)
    flip = torch.rand(n * NV, device=dev, generator=g) < 0.5
    v[flip] = v[flip].flip(3)
    return (v - MU) / SD


@torch.no_grad()
def embed_views(model, imgs, seed, bs=16):
    g = torch.Generator(device=dev); g.manual_seed(seed)
    out = []
    for i in range(0, len(imgs), bs):
        v = views(imgs[i:i + bs], g)
        with torch.autocast('cuda', dtype=torch.float16, enabled=(dev == 'cuda')):
            z = model.get_image_features(pixel_values=v)
        out.append(z.float().view(-1, NV, z.shape[-1]))
    return torch.cat(out)


def nrm(t):
    return t / t.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def probs(V, P):
    """V (n, m, d) normalized views, P (n, K, d) or (K, d) prototypes."""
    Pn = nrm(P)
    S = SCALE * (torch.einsum('nmd,kd->nmk', V, Pn) if Pn.dim() == 2 else torch.einsum('nmd,nkd->nmk', V, Pn))
    return torch.softmax(S, -1)


def confident(V, T):
    p = probs(V, T)
    ent = -(p * torch.log(p + 1e-12)).sum(-1)
    idx = ent.argsort(1)[:, :KEEP]
    return torch.gather(V, 1, idx[..., None].expand(-1, -1, V.shape[-1]))


def tps_predict(Vk, X0, T, lr, shared):
    """One Adam step from zero on the entropy of the averaged prediction of the kept
    views; the first Adam step equals lr times the sign of the gradient."""
    n, _, d = Vk.shape
    K = T.shape[0]
    s = torch.zeros((n, 1 if shared else K, d), device=dev, requires_grad=True)
    pbar = probs(Vk, T[None] + s).mean(1)
    L = -(pbar * torch.log(pbar + 1e-12)).sum()
    g, = torch.autograd.grad(L, s)
    with torch.no_grad():
        P = T[None] - lr * g / (g.abs() + 1e-8)
        return probs(X0[:, None], P)[:, 0].argmax(1)


def main():
    ensure(OUT)
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(dev).eval()
    T = torch.tensor(np.load(CIFAR_EMB / 'clip_text.npy'), dtype=torch.float32, device=dev)
    Xtr = np.load(CIFAR_EMB / 'clip_train.npy').astype(np.float64)
    ytr = np.load(CIFAR_EMB / 'labels_train.npy')
    yte = np.load(CIFAR_EMB / 'labels_test.npy')
    K = 10
    MS = R.class_means(Xtr, ytr, K)
    sel = np.sort(np.random.default_rng(0).permutation(len(yte))[:NIMG])
    y = torch.tensor(yte[sel], device=dev)
    rows = []
    for ci, cn in enumerate(CORRUPTIONS):
        A = np.load(CIFAR_C / ('%s.npy' % cn), mmap_mode='r')
        Z = np.load(CIFAR_EMB / ('clip_%s.npz' % cn))
        for sv in SEVS:
            Xf = Z['s%d' % sv].astype(np.float64)
            D = R.class_means(Xf, yte, K) - MS
            dbar = Xf.mean(0) - Xtr.mean(0)
            X = Xf[sel]
            imgs = np.ascontiguousarray(A[(sv - 1) * 10000 + sel])
            Vr = embed_views(model, imgs, seed=100 * ci + sv)
            r = dict(corr=cn, sev=sv, group=cn)
            for tag, shift in (('', np.zeros_like(dbar)), ('_mean', dbar)):
                X0 = nrm(torch.tensor(X - shift, dtype=torch.float32, device=dev))
                V = nrm(Vr - torch.tensor(shift, dtype=torch.float32, device=dev))
                Vall = torch.cat([X0[:, None], V], 1)
                if tag == '':
                    r['zs'] = float((probs(X0[:, None], T)[:, 0].argmax(1) == y).float().mean())
                else:
                    r['mean'] = float((probs(X0[:, None], T)[:, 0].argmax(1) == y).float().mean())
                Vk = confident(Vall, T)
                r['views' + tag] = float((probs(Vk, T).mean(1).argmax(1) == y).float().mean())
                for shared in (False, True):
                    key = 'tps' + ('_shared' if shared else '_class') + tag
                    r[key] = {}
                    for lr in LRS:
                        pr = torch.cat([tps_predict(Vk[i:i + 250], X0[i:i + 250], T, lr, shared)
                                        for i in range(0, NIMG, 250)])
                        r[key][lr] = float((pr == y).float().mean())
            Xo = nrm(torch.tensor(X - D[yte[sel]], dtype=torch.float32, device=dev))
            r['oracle'] = float((probs(Xo[:, None], T)[:, 0].argmax(1) == y).float().mean())
            rows.append(r)
            print('%s s%d zs %.4f mean %.4f orc %.4f views %.4f views_mean %.4f tps_class %s tps_shared %s tps_class_mean %s'
                  % (cn, sv, r['zs'], r['mean'], r['oracle'], r['views'], r['views_mean'],
                     r['tps_class'], r['tps_shared'], r['tps_class_mean']), flush=True)
    with open(OUT / 'tps_views.pkl', 'wb') as f:
        pickle.dump(rows, f)
    print('\n=== SUMMARY over %d cells, learning rate selected on the other corruptions ===' % len(rows))
    for k in ('zs', 'mean', 'views', 'views_mean', 'oracle'):
        print('  %-16s %.4f' % (k, np.mean([r[k] for r in rows])))
    for k in ('tps_class', 'tps_shared', 'tps_class_mean', 'tps_shared_mean'):
        sel_acc = []
        for r in rows:
            oth = [q for q in rows if q['group'] != r['group']]
            lr = max(LRS, key=lambda l: np.mean([q[k][l] for q in oth]))
            sel_acc.append(r[k][lr])
        print('  %-16s %.4f   per lr %s' % (k, np.mean(sel_acc),
              {l: round(float(np.mean([r[k][l] for r in rows])), 4) for l in LRS}))
    for sv in SEVS:
        s = [r for r in rows if r['sev'] == sv]
        print('  sev %d: zs %.4f mean %.4f views %.4f tps_class@5e-3 %.4f tps_shared@5e-3 %.4f oracle %.4f'
              % (sv, np.mean([r['zs'] for r in s]), np.mean([r['mean'] for r in s]),
                 np.mean([r['views'] for r in s]), np.mean([r['tps_class'][5e-3] for r in s]),
                 np.mean([r['tps_shared'][5e-3] for r in s]), np.mean([r['oracle'] for r in s])))


if __name__ == '__main__':
    main()
