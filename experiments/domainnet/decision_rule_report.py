"""Summary of the DomainNet decision rule and of the Route A penalty check, with
Route C from the core analysis for comparison.  Writes results/domainnet/dnet_extra_report.txt.
"""
import sys,pickle,numpy as np
from collections import Counter
from prlb import extra as R2
from prlb.paths import DNET_OUT
ODIR=str(DNET_OUT)
Q=0.02
OUT=[]
def P(*a):
    s=' '.join(str(x) for x in a); OUT.append(s); print(s,flush=True)
def med(v):
    v=np.asarray([x for x in v if np.isfinite(x)])
    return float(np.median(v)) if v.size else float('nan')
def perm(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    m=np.isfinite(a)&np.isfinite(b)
    if m.sum()<3: return (float('nan'),)*4
    return R2.paired_perm(a[m],b[m],n=20000,rng=np.random.default_rng(0))

for enc in ['clip','dino']:
    try: rows=pickle.load(open(ODIR+'/dnet_extra_%s.pkl'%enc,'rb'))
    except Exception as e: P('NO DATA',enc,e); continue
    try: core={(r['K'],r['src'],r['tgt']):r for r in pickle.load(open(ODIR+'/dnet_%s.pkl'%enc,'rb'))}
    except Exception: core={}
    P('='*78); P('ENCODER',enc,'cells',len(rows))
    for K in sorted(set(r['K'] for r in rows)):
        rr=[r for r in rows if r['K']==K]
        P('-'*78); P('K=%d pairs=%d'%(K,len(rr)))
        ps=np.concatenate([r['Ghat']<=r['thr'] for r in rr])
        ts=np.concatenate([r['loss']<=Q for r in rr])
        cf=np.concatenate([r['conf'] for r in rr])
        n=len(ps)
        tp=int(np.sum(ps&ts)); fp=int(np.sum(ps&~ts))
        fn=int(np.sum(~ps&ts)); tn=int(np.sum(~ps&~ts))
        P('  decision rule at q=%.2f over %d class-instances:'%(Q,n))
        P('    TP=%d FP=%d FN=%d TN=%d  acc=%.5f prec=%.5f rec=%.5f npv=%.5f'%(
            tp,fp,fn,tn,(tp+tn)/n,tp/max(tp+fp,1),tp/max(tp+fn,1),tn/max(tn+fn,1)))
        P('    true_safe base rate %.5f   rule fires (asks for labels) on %.5f'%(
            ts.mean(),1.0-ps.mean()))
        best=(-1,None)
        for t in np.quantile(cf,np.linspace(0.01,0.99,99)):
            a=float(np.mean((cf<=t)==ts))
            if a>best[0]: best=(a,float(t))
        P('    mean-confidence null, oracle-tuned threshold: acc=%.5f'%best[0])
        for Bud in sorted(set(b for r in rr for b in r['routes'])):
            sel=[r for r in rr if Bud in r['routes']]
            if not sel: continue
            acv=[r['routes'][Bud]['A_cv'] for r in sel]
            afx=[r['routes'][Bud]['A_fixed'] for r in sel]
            mn=[r['routes'][Bud]['mean'] for r in sel]
            lm=Counter(r['routes'][Bud]['lam'] for r in sel).most_common(2)
            P('  B=%d (B/K=%.0f) routeA_cv %.5f  routeA_fixed %.5f  mean %.5f  modal lambda %s'%(
                Bud,Bud/K,med(acv),med(afx),med(mn),lm))
            o,p,lo,hi=perm(acv,afx)
            P('     A_cv-vs-A_fixed %+.5f [%+.5f,%+.5f] p=%.5g'%(o,lo,hi,p))
            cs=[core.get((K,r['src'],r['tgt']),{}).get('routes',{}).get(Bud,{}).get('C',float('nan')) for r in sel]
            if np.isfinite(np.asarray(cs,float)).sum()>=3:
                o,p,lo,hi=perm(cs,acv)
                P('     routeC %.5f   C-vs-A_cv %+.5f [%+.5f,%+.5f] p=%.5g  (d/K=%.2f)'%(
                    med(cs),o,lo,hi,p,(512.0 if enc=='clip' else 768.0)/K))
open(ODIR+'/dnet_extra_report.txt','w').write('\n'.join(OUT)+'\n')
print('WROTE',ODIR+'/dnet_extra_report.txt')
