"""DomainNet remaining label-free baselines on the same 4000-point subsamples as tta.py.

Usage:  python experiments/domainnet/tta_aux.py {clip|dino} <K list> <tag>

BN alignment, CORAL, T3A, scale-only alignment and the per-class correction with
predicted assignments.

Output: results/domainnet/dnet_tta2_<enc>_<tag>.pkl
"""
import os,sys,pickle,numpy as np
from prlb import core as R
from prlb import extra as R2
from prlb.paths import DNET_EMB, DNET_OUT, ensure
EMB=str(DNET_EMB); ODIR=str(ensure(DNET_OUT))
DOMS=['clipart','infograph','painting','quickdraw','real','sketch']
ENC=sys.argv[1]
KS=[int(x) for x in sys.argv[2].split(',')]
TAG=sys.argv[3]
NSUB=4000
SEED=0
D={}
for d in DOMS:
    X=np.load(EMB+'/%s_%s.npy'%(d,ENC))
    y=np.load(EMB+'/%s_y.npy'%d).astype(np.int64)
    ok=np.load(EMB+'/%s_ok.npy'%d).astype(bool)
    D[d]=(X[ok],y[ok])
bad=[d for d in D if D[d][0].shape[0]==0]
if bad: raise SystemExit('ABORT zero rows: %s'%bad)
KG=len(open(EMB+'/classes.txt').read().split())
cnt=np.zeros(KG,np.int64)
for d in DOMS: cnt+=np.bincount(D[d][1],minlength=KG)
RANK=np.argsort(-cnt)
def prep(dom,keep,cmap):
    X,y=D[dom]
    m=np.isin(y,keep)
    return X[m].astype(np.float32),cmap[y[m]]
rows=[]
for K in KS:
    keep=RANK[:K]
    cmap=-np.ones(KG,np.int64); cmap[keep]=np.arange(K)
    PR={}
    for S in DOMS:
        Xs,ys=prep(S,keep,cmap)
        idx=np.random.default_rng(SEED).permutation(len(ys))
        ntr=int(0.6*len(ys))
        Xtr=Xs[idx[:ntr]]; ytr=ys[idx[:ntr]]
        W,b=R.ridge_probe(Xtr,ytr,K)
        PR[S]=(Xtr,ytr,W,b)
    for si,S in enumerate(DOMS):
        Xtr,ytr,W,b=PR[S]
        MS=R.class_means(Xtr,ytr,K)
        for ti,Tg in enumerate(DOMS):
            if Tg==S: continue
            Xf,yf=prep(Tg,keep,cmap)
            g=np.random.default_rng(1000*K+10*si+ti)
            sel=g.permutation(len(yf))[:min(NSUB,len(yf))]
            Xt=Xf[sel]; yt=yf[sel]
            r=dict(K=K,enc=ENC,src=S,tgt=Tg,n=int(len(yt)))
            r['a_bn']=R.acc(R.bn_align(Xt,Xtr),yt,W,b)
            try: r['a_coral']=R.acc(R.coral_align(Xt,Xtr),yt,W,b)
            except Exception: r['a_coral']=float('nan')
            r['a_t3a']=float(np.mean(R.t3a(Xt,W,b,K)==yt))
            r['a_scale']=R.acc(R2.scale_align(Xt,Xtr),yt,W,b)
            MT=R.class_means(Xt,yt,K)
            deltas=MT-MS; dbar=Xt.mean(0)-Xtr.mean(0)
            plg=R.predict(Xt-dbar,W,b)
            r['a_pcp']=R.acc(Xt-deltas[plg],yt,W,b)
            rows.append(r)
            print('%d %s %s->%s bn %.5f coral %.5f t3a %.5f scale %.5f pcp %.5f'%(K,ENC,S,Tg,r['a_bn'],r['a_coral'],r['a_t3a'],r['a_scale'],r['a_pcp']),flush=True)
    pickle.dump(rows,open(ODIR+'/dnet_tta2_%s_%s.pkl'%(ENC,TAG),'wb'))
print('DNET_TTA2_DONE',ENC,TAG,len(rows),flush=True)
