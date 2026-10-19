"""DomainNet selective corrector under a label budget.

Usage:  python experiments/domainnet/selective.py {clip|dino} <K list> <tag>

On a 4000-point target subsample per pair, the certificate-driven selective corrector
and the cross-validated refit are run at B in {50, ..., 1600}.  The tag names the
output file so that several K lists can run in parallel.

Output: results/domainnet/dnet_sel_<enc>_<tag>.pkl
"""
import os,sys,pickle,numpy as np
from prlb import core as R
from prlb.core import ridge_cv
from prlb.paths import DNET_EMB, DNET_OUT, ensure
EMB=str(DNET_EMB); ODIR=str(ensure(DNET_OUT))
DOMS=['clipart','infograph','painting','quickdraw','real','sketch']
ENC=sys.argv[1]
KS=[int(x) for x in sys.argv[2].split(',')]
TAG=sys.argv[3]
NSUB=4000
SEED=0
BUDG=[50,100,200,400,800,1600]
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
    for si,Sd in enumerate(DOMS):
        Xtr,ytr,W,b=PR[Sd]
        for ti,Tg in enumerate(DOMS):
            if Tg==Sd: continue
            Xf,yf=prep(Tg,keep,cmap)
            g=np.random.default_rng(1000*K+10*si+ti)
            sel=g.permutation(len(yf))[:min(NSUB,len(yf))]
            Xt=Xf[sel]; yt=yf[sel]
            dbar=Xt.mean(0)-Xtr.mean(0)
            MT=R.class_means(Xt,yt,K); MS=R.class_means(Xtr,ytr,K)
            deltas=MT-MS
            r=dict(K=K,enc=ENC,src=Sd,tgt=Tg,n=int(len(yt)),
                   a_mean=R.acc(Xt-dbar,yt,W,b), a_orc=R.acc(Xt-deltas[yt],yt,W,b))
            Ghat,_=R.plugin_certificate(Xt,Xtr,ytr,W,b,K,dbar)
            r['nposhat']=int((Ghat>0).sum())
            rng=np.random.default_rng(7000+1000*K+10*si+ti)
            for B in BUDG:
                if B>=len(yt)//2:
                    r['sel_%d'%B]=float('nan'); r['nsel_%d'%B]=-1; r['rcv_%d'%B]=float('nan'); continue
                Xc,Ssel=R.selective_corrector(Xt,yt,Xtr,ytr,W,b,K,dbar,B,rng,n_min=8,score=Ghat)
                r['sel_%d'%B]=R.acc(Xc,yt,W,b); r['nsel_%d'%B]=len(Ssel)
                q=np.random.default_rng(9000+B+1000*K+10*si+ti).choice(len(yt),size=B,replace=False)
                Wc,bc,_l=ridge_cv(Xt[q],yt[q],K)
                r['rcv_%d'%B]=R.acc(Xt,yt,Wc,bc)
            rows.append(r)
            print('%d %s %s->%s mean %.5f orc %.5f | %s'%(K,ENC,Sd,Tg,r['a_mean'],r['a_orc'],
                  ' '.join('B%d sel %.5f (%d cls) rcv %.5f'%(B,r['sel_%d'%B],r['nsel_%d'%B],r['rcv_%d'%B]) for B in BUDG)),flush=True)
    pickle.dump(rows,open(ODIR+'/dnet_sel_%s_%s.pkl'%(ENC,TAG),'wb'))
print('DNET_SEL_DONE',ENC,TAG,len(rows),flush=True)
