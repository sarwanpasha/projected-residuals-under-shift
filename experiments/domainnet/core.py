"""DomainNet core analysis.

Usage:  python experiments/domainnet/core.py {clip|dino} [K list, default 10,25,50,100,345]

The K most frequent classes over the six domains are kept.  For each source domain a
ridge probe is trained on a 60 percent split and the remaining 40 percent is a holdout.
For each of the 30 ordered source-target pairs the script records the label drift,
the common-mean, oracle and label-free baseline accuracies, the gap, the gap
bound with oracle-corrected, source and pseudo-labeled margins, the number of classes
with Gamma_c > 0 and with a positive plug-in estimate, label-shift estimates (BBSE,
RLLS, MLLS with and without bias-corrected temperature scaling), prior-corrected
accuracies on calibrated posteriors, and Routes A, B and C at budgets 5K, 10K and 25K.

Output: results/domainnet/dnet_<enc>.pkl
"""
import os,sys,pickle,numpy as np
from prlb import core as R
from prlb import extra as R2
from prlb.paths import DNET_EMB, DNET_OUT, ensure
EMB=str(DNET_EMB); ODIR=str(ensure(DNET_OUT))
DOMS=['clipart','infograph','painting','quickdraw','real','sketch']
ENC=sys.argv[1] if len(sys.argv)>1 else 'clip'
KS=[int(x) for x in (sys.argv[2].split(',') if len(sys.argv)>2 else ['10','25','50','100','345'])]
SEED=0

def load():
    D={}
    for d in DOMS:
        f=EMB+'/%s_%s.npy'%(d,ENC)
        if not os.path.exists(f): print('MISSING',f); continue
        X=np.load(f); y=np.load(EMB+'/%s_y.npy'%d).astype(np.int64)
        ok=np.load(EMB+'/%s_ok.npy'%d).astype(bool)
        D[d]=(X[ok],y[ok])
        print('loaded',d,D[d][0].shape,'dropped',int((~ok).sum()),flush=True)
    return D

D=load()
bad=[d for d in D if D[d][0].shape[0]==0]
if bad: raise SystemExit('ABORT: zero usable rows for %s -- re-embed those domains before analysing'%bad)
DOMS=[d for d in DOMS if d in D]
print('DOMAINS',DOMS,'ENC',ENC,flush=True)
KG=len(open(EMB+'/classes.txt').read().split())
cnt=np.zeros(KG,np.int64)
for d in DOMS: cnt+=np.bincount(D[d][1],minlength=KG)
RANK=np.argsort(-cnt)

def prep(dom,keep,cmap):
    X,y=D[dom]
    m=np.isin(y,keep)
    return X[m].astype(np.float32),cmap[y[m]]

def pi_of(y,K):
    c=np.bincount(y,minlength=K).astype(float)
    return c/max(c.sum(),1)

def conf_q(Xho,yho,Xt,W,b,K):
    ps=R.predict(Xho,W,b)
    C=np.zeros((K,K))
    np.add.at(C,(ps,yho),1.0)
    C/=float(len(yho))
    q=np.bincount(R.predict(Xt,W,b),minlength=K).astype(float)/float(len(Xt))
    return C,q

def bbse_fast(C,q,K,pis):
    w=np.linalg.lstsq(C,q,rcond=None)[0]
    pit=np.clip(w*pis,1e-8,None)
    return pit/pit.sum()

def rlls_fast(C,q,K,pis,n_t,lam=None):
    if lam is None: lam=1.0/np.sqrt(n_t)
    b0=C@pis
    A=C.T@C+lam*np.eye(K)
    th=np.linalg.solve(A,C.T@(q-b0))
    pit=np.clip(pis+th,1e-8,None)
    return pit/pit.sum()

def bcts_fit(S,y,K,iters=300,lr=0.05):
    import torch
    Z=torch.tensor(S,dtype=torch.float64); Y=torch.tensor(y)
    lt=torch.zeros(1,dtype=torch.float64,requires_grad=True)
    bb=torch.zeros(K,dtype=torch.float64,requires_grad=True)
    opt=torch.optim.Adam([lt,bb],lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        L=torch.nn.functional.cross_entropy(Z*torch.exp(lt)+bb,Y)
        L.backward(); opt.step()
    T=float(torch.exp(lt).item()); B=bb.detach().numpy()
    nll0=float(torch.nn.functional.cross_entropy(Z,Y).item())
    nll1=float(L.item())
    return T,B,nll0,nll1

def mlls_pi(P,pis,K,iters=200):
    pit=pis.copy()
    for _ in range(iters):
        w=pit/np.maximum(pis,1e-12)
        Q=P*w
        Q=Q/np.maximum(Q.sum(1,keepdims=True),1e-30)
        new=Q.mean(0)
        if np.abs(new-pit).sum()<1e-10: pit=new; break
        pit=new
    return np.clip(pit,1e-8,None)/np.clip(pit,1e-8,None).sum()

def cal_post(S,T_,B_):
    Z=S.astype(np.float64)*T_+B_
    Z=Z-Z.max(1,keepdims=True)
    P=np.exp(Z)
    return P/P.sum(1,keepdims=True)

def prior_acc(P,y,pihat,pis):
    """Accuracy after the label-shift correction Q ~ P * (pihat / pis) applied to
    calibrated posteriors P.  Ridge scores are not log-probabilities, so the prior
    ratio is applied only after bias-corrected temperature scaling."""
    w=np.maximum(pihat,1e-12)/np.maximum(pis,1e-12)
    return float(np.mean((P*w).argmax(1)==y))

rows=[]
rng=np.random.default_rng(SEED)
for K in KS:
    keep=RANK[:K]
    cmap=-np.ones(KG,np.int64); cmap[keep]=np.arange(K)
    PR={}
    for S in DOMS:
        Xs,ys=prep(S,keep,cmap)
        idx=np.random.default_rng(SEED).permutation(len(ys))
        ntr=int(0.6*len(ys)); tr=idx[:ntr]; ho=idx[ntr:]
        Xtr,ytr,Xho,yho=Xs[tr],ys[tr],Xs[ho],ys[ho]
        W,b=R.ridge_probe(Xtr,ytr,K)
        PR[S]=(Xtr,ytr,Xho,yho,W,b)
        print('probe',K,S,'ntr',len(ytr),'srcacc',round(R.acc(Xho,yho,W,b),4),flush=True)
    for S in DOMS:
        Xtr,ytr,Xho,yho,W,b=PR[S]
        pis=pi_of(ytr,K)
        MS=R.class_means(Xtr,ytr,K)
        Sho=R.scores(Xho,W,b)
        sub=np.random.default_rng(1).permutation(len(yho))[:10000]
        T_,B_,n0,n1=bcts_fit(Sho[sub].astype(np.float64),yho[sub],K)
        for Tg in DOMS:
            if Tg==S: continue
            Xt,yt=prep(Tg,keep,cmap)
            pit=pi_of(yt,K); drift=float(np.abs(pit-pis).sum())
            MT=R.class_means(Xt,yt,K)
            deltas=MT-MS
            dbar=Xt.mean(0)-Xtr.mean(0)
            rhos=deltas-dbar
            G=R.all_gamma_caps(rhos,W)
            Xg=Xt-dbar; Xo=Xt-deltas[yt]
            a_nv=R.acc(Xt,yt,W,b); a_mn=R.acc(Xg,yt,W,b); a_or=R.acc(Xo,yt,W,b)
            gap=a_or-a_mn
            b_orc=R.theory_bound(Xo,yt,W,b,rhos,pit)[0]
            b_src=R.theory_bound(Xtr,ytr,W,b,rhos,pit)[0]
            Ghat,rhohat=R.plugin_certificate(Xt,Xtr,ytr,W,b,K,dbar)
            plg=R.predict(Xg,W,b)
            b_ps=R.theory_bound(Xg,plg,W,b,rhohat,pit)[0]
            pchg=float(np.mean(plg!=R.predict(Xo,W,b)))
            a_bn=R.acc(R.bn_align(Xt,Xtr),yt,W,b)
            try: a_cor=R.acc(R.coral_align(Xt,Xtr),yt,W,b)
            except Exception: a_cor=float('nan')
            a_t3a=float(np.mean(R.t3a(Xt,W,b,K)==yt))
            Cm,qv=conf_q(Xho,yho,Xt,W,b,K)
            pi_bb=bbse_fast(Cm,qv,K,pis)
            pi_rl=rlls_fast(Cm,qv,K,pis,len(yt))
            St=R.scores(Xt,W,b).astype(np.float32)
            sub2=np.random.default_rng(2).permutation(len(yt))[:20000]
            Z=(St[sub2]*T_+B_); Z=Z-Z.max(1,keepdims=True)
            P=np.exp(Z); P=P/P.sum(1,keepdims=True)
            pi_ml=mlls_pi(P.astype(np.float64),pis,K)
            Zu=St[sub2]-St[sub2].max(1,keepdims=True)
            Pu=np.exp(Zu); Pu=Pu/Pu.sum(1,keepdims=True)
            pi_mlu=mlls_pi(Pu.astype(np.float64),pis,K)
            l1={'naive':float(np.abs(pis-pit).sum()),'bbse':float(np.abs(pi_bb-pit).sum()),
                'rlls':float(np.abs(pi_rl-pit).sum()),'mlls_bcts':float(np.abs(pi_ml-pit).sum()),
                'mlls_raw':float(np.abs(pi_mlu-pit).sum())}
            Pg=cal_post(R.scores(Xg,W,b),T_,B_)
            pa={'uncal':a_mn,'naive':prior_acc(Pg,yt,pis,pis),'bbse':prior_acc(Pg,yt,pi_bb,pis),
                'rlls':prior_acc(Pg,yt,pi_rl,pis),'mlls_bcts':prior_acc(Pg,yt,pi_ml,pis),
                'orc':prior_acc(Pg,yt,pit,pis)}
            routes={}
            for Bud in [5*K,10*K,25*K]:
                if Bud>=len(yt)//2: continue
                q=rng.permutation(len(yt))[:Bud]; rest=np.setdiff1d(np.arange(len(yt)),q,assume_unique=False)
                XB,yB=Xt[q],yt[q]
                try:
                    W2,b2=R.ridge_probe(XB,yB,K); aA=R.acc(Xt[rest],yt[rest],W2,b2)
                except Exception: aA=float('nan')
                dh=dbar[None,:].repeat(K,0).copy()
                for c in range(K):
                    mm=(yB==c)
                    if mm.sum()>=2: dh[c]=XB[mm].mean(0)-MS[c]
                plr=R.predict(Xt[rest]-dbar,W,b)
                aB=float(np.mean(R.predict(Xt[rest]-dh[plr],W,b)==yt[rest]))
                pihat=(np.bincount(yB,minlength=K)+1.0); pihat=pihat/pihat.sum()
                Pr=cal_post(R.scores(Xt[rest]-dbar,W,b),T_,B_)
                aC=prior_acc(Pr,yt[rest],pihat,pis)
                aCo=prior_acc(Pr,yt[rest],pit,pis)
                aC0=prior_acc(Pr,yt[rest],pis,pis)
                routes[Bud]={'A':aA,'B':aB,'C':aC,'Corc':aCo,'C0':aC0,
                             'mean':R.acc(Xt[rest]-dbar,yt[rest],W,b)}
            rows.append(dict(K=K,enc=ENC,src=S,tgt=Tg,nS=len(ytr),nT=len(yt),drift=drift,
                a_naive=a_nv,a_mean=a_mn,a_orc=a_or,gap=gap,b_orc=b_orc,b_src=b_src,b_ps=b_ps,
                npos=int((G>0).sum()),nposhat=int((Ghat>0).sum()),pchg=pchg,
                a_bn=a_bn,a_coral=a_cor,a_t3a=a_t3a,l1=l1,pa=pa,routes=routes,
                bcts=(T_,n0,n1)))
            print('%d %s %s->%s drift %.4f gap %.5f b_orc %.5f b_src %.5f b_ps %.5f npos %d/%d pchg %.4f'%(
                K,ENC,S,Tg,drift,gap,b_orc,b_src,b_ps,int((G>0).sum()),K,pchg),flush=True)
    pickle.dump(rows,open(ODIR+'/dnet_%s.pkl'%ENC,'wb'))
print('DNET_CORE_DONE',ENC,len(rows),flush=True)
