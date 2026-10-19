"""DomainNet decision rule and cross-validated Route A.

Usage:  python experiments/domainnet/decision_rule.py {clip|dino} <K list>

For every pair the script stores the plug-in certificate, the q = 0.02 margin-quantile
threshold computed on pseudo-labeled target margins, the true per-class loss of the
common-mean correction and the mean-confidence null, together with Route A refits
using a fixed and a cross-validated ridge penalty at budgets 5K, 10K and 25K.

Output: results/domainnet/dnet_extra_<enc>.pkl
"""
import os,sys,pickle,numpy as np
from prlb import core as R
from prlb.paths import DNET_EMB, DNET_OUT, ensure
EMB=str(DNET_EMB); ODIR=str(ensure(DNET_OUT))
DOMS=['clipart','infograph','painting','quickdraw','real','sketch']
ENC=sys.argv[1]
KS=[int(x) for x in sys.argv[2].split(',')]
Q=0.02
SEED=0
LAMS=(1e-3,1e-2,1e-1,1.0,10.0,100.0)

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

def ridge_cv(XB,yB,K,rng,folds=3):
    """Ridge probe whose penalty is chosen by folds-fold cross-validation on the
    labeled budget, with the fold assignment drawn from rng."""
    n,d=XB.shape
    Y=np.zeros((n,K)); Y[np.arange(n),yB]=1.0
    Xa=np.c_[XB,np.ones(n,dtype=XB.dtype)].astype(np.float64)
    I=np.eye(d+1)
    idx=rng.permutation(n)
    sc={l:[0,0] for l in LAMS}
    for f in range(folds):
        te=idx[f::folds]
        mask=np.zeros(n,bool); mask[te]=True
        tr=~mask
        if tr.sum()<2 or mask.sum()<1: continue
        A0=Xa[tr].T@Xa[tr]; B0=Xa[tr].T@Y[tr]
        for l in LAMS:
            try: Wa=np.linalg.solve(A0+l*I,B0)
            except Exception: continue
            W=Wa[:d].T; b=Wa[d]
            pr=(XB[mask]@W.T+b).argmax(1)
            sc[l][0]+=int((pr==yB[mask]).sum()); sc[l][1]+=int(mask.sum())
    cand=[l for l in LAMS if sc[l][1]>0]
    best=max(cand,key=lambda l: sc[l][0]/sc[l][1]) if cand else 1e-2
    A=Xa.T@Xa; Bm=Xa.T@Y
    Wa=np.linalg.solve(A+best*I,Bm)
    return Wa[:d].T, Wa[d], float(best)

rows=[]
rng=np.random.default_rng(SEED)
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
        print('probe',K,S,flush=True)
    for S in DOMS:
        Xtr,ytr,W,b=PR[S]
        MS=R.class_means(Xtr,ytr,K)
        for Tg in DOMS:
            if Tg==S: continue
            Xt,yt=prep(Tg,keep,cmap)
            MT=R.class_means(Xt,yt,K)
            deltas=MT-MS
            dbar=Xt.mean(0)-Xtr.mean(0)
            Xg=Xt-dbar; Xo=Xt-deltas[yt]
            plg=R.predict(Xg,W,b)
            Ghat,_=R.plugin_certificate(Xt,Xtr,ytr,W,b,K,dbar)
            thr=np.zeros(K)
            for c in range(K):
                m=(plg==c)
                thr[c]=R.pos_quantile(R.margins(Xg[m],plg[m],W,b),Q) if m.sum()>1 else 0.0
            po=R.predict(Xo,W,b); pg=plg
            loss=np.zeros(K); npc=np.zeros(K,np.int64)
            for c in range(K):
                m=(yt==c); npc[c]=int(m.sum())
                if npc[c]>0:
                    loss[c]=float((po[m]==c).mean()-(pg[m]==c).mean())
            conf=R.mean_conf_score(Xt,W,b,K,dbar)
            routes={}
            for Bud in [5*K,10*K,25*K]:
                if Bud>=len(yt)//2: continue
                qq=rng.permutation(len(yt))[:Bud]
                rest=np.setdiff1d(np.arange(len(yt)),qq)
                W2,b2,lam=ridge_cv(Xt[qq],yt[qq],K,np.random.default_rng(SEED))
                aA=R.acc(Xt[rest],yt[rest],W2,b2)
                W3,b3=R.ridge_probe(Xt[qq],yt[qq],K)
                aAfix=R.acc(Xt[rest],yt[rest],W3,b3)
                routes[Bud]={'A_cv':aA,'A_fixed':aAfix,'lam':lam,
                             'mean':R.acc(Xt[rest]-dbar,yt[rest],W,b)}
            rows.append(dict(K=K,enc=ENC,src=S,tgt=Tg,Ghat=Ghat,thr=thr,loss=loss,
                             conf=conf,npc=npc,routes=routes))
            print('%d %s %s->%s firing %d/%d truesafe %d/%d lam %s'%(
                K,ENC,S,Tg,int((Ghat>thr).sum()),K,int((loss<=Q).sum()),K,
                [routes[t]['lam'] for t in sorted(routes)]),flush=True)
    pickle.dump(rows,open(ODIR+'/dnet_extra_%s.pkl'%ENC,'wb'))
print('DNET_EXTRA_DONE',ENC,len(rows),flush=True)
