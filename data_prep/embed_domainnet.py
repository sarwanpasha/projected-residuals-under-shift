"""Embed one DomainNet domain with both frozen encoders.

Usage:  python data_prep/embed_domainnet.py <domain>

<domain> is one of clipart, infograph, painting, quickdraw, real, sketch.  The train
and test split lists are downloaded into PRLB_DATA/domainnet/txt when missing.  If
PRLB_DATA/domainnet/zips/<domain>.zip exists it is read locally; otherwise the
archive is read directly from the DomainNet server with HTTP range requests, so the
multi-gigabyte zip never has to be stored.  For clipart and painting the cleaned
archives in the groundtruth/ folder are used, which match the official split lists.

Each image is resized to 224 x 224 (bicubic) and encoded with CLIP ViT-B/32 and DINOv2
ViT-B/14.  Outputs, in PRLB_DATA/domainnet/emb:
    <domain>_clip.npy, <domain>_dino.npy   float16 embeddings
    <domain>_y.npy                         int16 global class index
    <domain>_ok.npy                        int8, 0 marks an unreadable image
    classes.txt                            the 345 class names in index order
"""
import io
import sys
import urllib.request as UR
import zipfile

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from prlb.paths import DNET_TXT, DNET_ZIPS, DNET_EMB, DOMAINS, ensure

SERVER = 'http://csr.bu.edu/ftp/visda/2019/multi-source'
SUBDIR = {'clipart': 'groundtruth/', 'painting': 'groundtruth/'}
RES = 224
BUF = 8 * 1024 * 1024
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


class HttpFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests, buffered in BUF-byte blocks."""

    def __init__(self, url, buf=BUF):
        self.url = url; self.buf = buf; self.pos = 0; self._c0 = 0; self._c = b''
        rq = UR.Request(url, method='HEAD')
        self.size = int(UR.urlopen(rq, timeout=60).headers['Content-Length'])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, o, w=0):
        self.pos = o if w == 0 else (self.pos + o if w == 1 else self.size + o)
        return self.pos

    def _fetch(self, a, n):
        b = min(self.size - 1, a + max(n, self.buf) - 1)
        for att in range(6):
            try:
                rq = UR.Request(self.url, headers={'Range': 'bytes=%d-%d' % (a, b)})
                return UR.urlopen(rq, timeout=120).read()
            except Exception:
                if att == 5:
                    raise
        return b''

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b''
        if not (self._c0 <= self.pos and self.pos + n <= self._c0 + len(self._c)):
            self._c = self._fetch(self.pos, n); self._c0 = self.pos
        o = self.pos - self._c0
        d = self._c[o:o + n]; self.pos += len(d)
        return d

    def readinto(self, bb):
        d = self.read(len(bb)); bb[:len(d)] = d
        return len(d)


def fetch_split_lists():
    ensure(DNET_TXT)
    for d in DOMAINS:
        for s in ['train', 'test']:
            f = DNET_TXT / ('%s_%s.txt' % (d, s))
            if not f.exists() or f.stat().st_size == 0:
                UR.urlretrieve('%s/domainnet/txt/%s_%s.txt' % (SERVER, d, s), f)


def classes_all():
    names = set()
    for d in DOMAINS:
        for s in ['train', 'test']:
            for ln in open(DNET_TXT / ('%s_%s.txt' % (d, s))):
                ln = ln.strip()
                if ln:
                    names.add(ln.split()[0].split('/')[1])
    return sorted(names)


def open_archive(dom):
    local = DNET_ZIPS / ('%s.zip' % dom)
    if local.exists():
        return str(local), True
    return '%s/%s%s.zip' % (SERVER, SUBDIR.get(dom, ''), dom), False


class ZD(Dataset):
    def __init__(self, items, src, is_local):
        self.it = items; self.src = src; self.is_local = is_local; self.z = None

    def __len__(self):
        return len(self.it)

    def __getitem__(self, i):
        if self.z is None:
            self.z = zipfile.ZipFile(self.src if self.is_local else HttpFile(self.src))
        p, y = self.it[i]
        try:
            im = Image.open(io.BytesIO(self.z.read(p))).convert('RGB').resize((RES, RES), Image.BICUBIC)
            a = np.asarray(im, dtype=np.uint8).copy(); ok = 1
        except Exception:
            a = np.zeros((RES, RES, 3), np.uint8); ok = 0
        return torch.from_numpy(a), y, ok


def build(name):
    if name == 'clip':
        from transformers import CLIPModel
        m = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(dev).eval()
        return ((lambda x: m.get_image_features(pixel_values=x)),
                (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    from transformers import Dinov2Model
    m = Dinov2Model.from_pretrained('facebook/dinov2-base').to(dev).eval()
    return ((lambda x: m(pixel_values=x).last_hidden_state[:, 0]),
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def main():
    dom = sys.argv[1]
    if dom not in DOMAINS:
        raise SystemExit('domain must be one of %s' % DOMAINS)
    ensure(DNET_EMB)
    fetch_split_lists()
    CLS = classes_all(); CIX = {c: i for i, c in enumerate(CLS)}
    (DNET_EMB / 'classes.txt').write_text('\n'.join(CLS))
    items = []
    for s in ['train', 'test']:
        for ln in open(DNET_TXT / ('%s_%s.txt' % (dom, s))):
            ln = ln.strip()
            if ln:
                p = ln.split()[0]; items.append((p, CIX[p.split('/')[1]]))
    src, is_local = open_archive(dom)
    z = zipfile.ZipFile(src if is_local else HttpFile(src))
    ordr = {n: i for i, n in enumerate(z.namelist())}
    z.close()
    miss = [p for p, _ in items if p not in ordr]
    print('DOM', dom, 'SRC', src, 'N', len(items), 'K_global', len(CLS),
          'missing_in_zip', len(miss), flush=True)
    if miss:
        raise SystemExit('%d of %d split-list paths are absent from the archive; first: %s'
                         % (len(miss), len(items), miss[0]))
    if not is_local:
        items.sort(key=lambda t: ordr[t[0]])

    FC, mc, sc = build('clip'); FD, md, sd = build('dino')
    MC = torch.tensor(mc, device=dev).view(1, 3, 1, 1); SC = torch.tensor(sc, device=dev).view(1, 3, 1, 1)
    MD = torch.tensor(md, device=dev).view(1, 3, 1, 1); SD = torch.tensor(sd, device=dev).view(1, 3, 1, 1)
    dl = DataLoader(ZD(items, src, is_local), batch_size=256, num_workers=6,
                    shuffle=False, pin_memory=True)
    OC = []; OD = []; YY = []; OK = []
    with torch.no_grad():
        for bi, (a, y, ok) in enumerate(dl):
            x = a.to(dev, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
            with torch.autocast('cuda', dtype=torch.float16, enabled=(dev == 'cuda')):
                zc = FC((x - MC) / SC); zd = FD((x - MD) / SD)
            OC.append(zc.float().cpu().numpy().astype(np.float16))
            OD.append(zd.float().cpu().numpy().astype(np.float16))
            YY.append(y.numpy().astype(np.int16)); OK.append(ok.numpy().astype(np.int8))
            if bi % 50 == 0:
                print('batch', bi, flush=True)
    np.save(DNET_EMB / ('%s_clip.npy' % dom), np.concatenate(OC))
    np.save(DNET_EMB / ('%s_dino.npy' % dom), np.concatenate(OD))
    np.save(DNET_EMB / ('%s_y.npy' % dom), np.concatenate(YY))
    np.save(DNET_EMB / ('%s_ok.npy' % dom), np.concatenate(OK))
    print('SAVED', dom, 'badreads', int((np.concatenate(OK) == 0).sum()), flush=True)


if __name__ == '__main__':
    main()
