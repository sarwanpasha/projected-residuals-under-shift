"""DomainNet transductive baselines on a 4000-point target subsample per pair.

Usage:  python experiments/domainnet/tta.py {clip|dino} <K list> <tag>

TENT and SHOT (best of head and affine adaptation, selected by their own label-free
objective), LAME with default and oracle-tuned graph strength, and the common-mean
correction followed by LAME.

Output: results/domainnet/dnet_tta_<enc>_<tag>.pkl
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
ALPHAS=[0.25,0.5,1.0,2.0,4.0,8.0]
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
        print('probe',K,S,len(ytr),flush=True)
    for si,S in enumerate(DOMS):
        Xtr,ytr,W,b=PR[S]
        MS=R.class_means(Xtr,ytr,K)
        for ti,Tg in enumerate(DOMS):
            if Tg==S: continue
            Xf,yf=prep(Tg,keep,cmap)
            g=np.random.default_rng(1000*K+10*si+ti)
            sel=g.permutation(len(yf))[:min(NSUB,len(yf))]
            Xt=Xf[sel]; yt=yf[sel]
            MT=R.class_means(Xt,yt,K)
            deltas=MT-MS
            dbar=Xt.mean(0)-Xtr.mean(0)
            Xg=Xt-dbar; Xo=Xt-deltas[yt]
            r=dict(K=K,enc=ENC,src=S,tgt=Tg,n=int(len(yt)),nfull=int(len(yf)),
                   a_naive=R.acc(Xt,yt,W,b),a_mean=R.acc(Xg,yt,W,b),a_orc=R.acc(Xo,yt,W,b))
            for nm,md in (('tent','tent'),('shot','shot')):
                r[nm]=float('nan')
                try:
                    Xa,Wa,ba=R.tta_baseline(Xt,W,b,mode=md)
                    r[nm]=R.acc(Xa,yt,Wa,ba)
                except Exception as e:
                    print('   %s unavailable: %s'%(nm,type(e).__name__),flush=True)
            idx1=R2.knn_index(Xt.astype(np.float32),5)
            r['lame_def']=float((R2.lame(Xt,W,b,K,alpha=1.0,idx=idx1)==yt).mean())
            best=-1.0; bal=1.0
            for al in ALPHAS:
                v=float((R2.lame(Xt,W,b,K,alpha=al,idx=idx1)==yt).mean())
                if v>best: best,bal=v,al
            r['lame_orc']=best; r['lame_alpha']=bal
            idxm=R2.knn_index(Xg.astype(np.float32),5)
            r['mean_lame']=float((R2.lame(Xg,W,b,K,alpha=1.0,idx=idxm)==yt).mean())
            rows.append(r)
            print('%d %s %s->%s n %d naive %.5f mean %.5f orc %.5f tent %.5f shot %.5f lame %.5f lameorc %.5f meanlame %.5f'%(
                K,ENC,S,Tg,r['n'],r['a_naive'],r['a_mean'],r['a_orc'],r['tent'],r['shot'],
                r['lame_def'],r['lame_orc'],r['mean_lame']),flush=True)
    pickle.dump(rows,open(ODIR+'/dnet_tta_%s_%s.pkl'%(ENC,TAG),'wb'))
print('DNET_TTA_DONE',ENC,TAG,len(rows),flush=True)
