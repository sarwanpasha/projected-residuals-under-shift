"""Summary of the DomainNet core analysis, per encoder and per K, with paired
permutation tests.  Writes results/domainnet/dnet_report.txt.
"""
import sys,pickle,numpy as np
from prlb import extra as R2
from prlb.paths import DNET_OUT
ODIR=str(DNET_OUT)
OUT=[]
def P(*a):
    s=' '.join(str(x) for x in a); OUT.append(s); print(s,flush=True)

def med(v):
    v=np.asarray([x for x in v if np.isfinite(x)]);
    return float(np.median(v)) if v.size else float('nan')

def perm(a,bb):
    a=np.asarray(a,float); bb=np.asarray(bb,float)
    m=np.isfinite(a)&np.isfinite(bb)
    if m.sum()<3: return (float('nan'),)*4
    o,p,lo,hi=R2.paired_perm(a[m],bb[m],n=20000,rng=np.random.default_rng(0))
    return o,p,lo,hi

for enc in ['clip','dino']:
    try: rows=pickle.load(open(ODIR+'/dnet_%s.pkl'%enc,'rb'))
    except Exception as e: P('NO DATA',enc,e); continue
    P('='*78); P('ENCODER',enc,'cells',len(rows))
    for K in sorted(set(r['K'] for r in rows)):
        rr=[r for r in rows if r['K']==K]
        n=len(rr)
        dr=[r['drift'] for r in rr]
        P('-'*78)
        P('K=%d pairs=%d  drift l1: min %.5f med %.5f max %.5f'%(K,n,min(dr),med(dr),max(dr)))
        P('  nT med %d  source-domain n med %d'%(int(med([r['nT'] for r in rr])),int(med([r['nS'] for r in rr]))))
        P('  acc naive %.5f mean %.5f bn %.5f coral %.5f t3a %.5f orc %.5f'%(
            med([r['a_naive'] for r in rr]),med([r['a_mean'] for r in rr]),med([r['a_bn'] for r in rr]),
            med([r['a_coral'] for r in rr]),med([r['a_t3a'] for r in rr]),med([r['a_orc'] for r in rr])))
        o,p,lo,hi=perm([r['a_mean'] for r in rr],[r['a_naive'] for r in rr])
        P('  mean-vs-naive  %+.5f [%+.5f,%+.5f] p=%.5g'%(o,lo,hi,p))
        o,p,lo,hi=perm([r['a_orc'] for r in rr],[r['a_mean'] for r in rr])
        P('  orc-vs-mean    %+.5f [%+.5f,%+.5f] p=%.5g   (this is the gap)'%(o,lo,hi,p))
        for law in ['b_orc','b_src','b_ps']:
            hold=sum(1 for r in rr if r[law]>=r['gap']-1e-12)
            rat=[r[law]/r['gap'] for r in rr if r['gap']>1e-6]
            P('  %-6s holds %d/%d   median ratio %.5f   p90 %.5f'%(law,hold,n,med(rat),
                float(np.percentile([x for x in rat if np.isfinite(x)],90)) if rat else float('nan')))
        P('  band: b_ps==0 on %d/%d pairs; classes with Gamma_c>0 med %.1f/%d; Ghat>0 med %.1f'%(
            sum(1 for r in rr if r['b_ps']<=1e-12),n,med([r['npos'] for r in rr]),K,med([r['nposhat'] for r in rr])))
        P('  pred-change global-vs-oracle med %.5f'%med([r['pchg'] for r in rr]))
        ks=['naive','bbse','rlls','mlls_bcts','mlls_raw']
        P('  label-shift l1 err: '+'  '.join('%s %.5f'%(k,med([r['l1'][k] for r in rr])) for k in ks))
        ks2=['uncal','naive','bbse','rlls','mlls_bcts','orc']
        P('  prior-corrected acc: '+'  '.join('%s %.5f'%(k,med([r['pa'][k] for r in rr])) for k in ks2))
        for k in ['bbse','rlls','mlls_bcts']:
            o,p,lo,hi=perm([r['pa'][k] for r in rr],[r['pa']['naive'] for r in rr])
            P('    %s-vs-naive-prior %+.5f [%+.5f,%+.5f] p=%.5g'%(k,o,lo,hi,p))
        buds=sorted(set(b for r in rr for b in r['routes']))
        for Bud in buds:
            sel=[r for r in rr if Bud in r['routes']]
            if not sel: continue
            A=[r['routes'][Bud]['A'] for r in sel]; Bl=[r['routes'][Bud]['B'] for r in sel]
            C=[r['routes'][Bud]['C'] for r in sel]; M=[r['routes'][Bud]['mean'] for r in sel]
            Co=[r['routes'][Bud].get('Corc',float('nan')) for r in sel]
            C0=[r['routes'][Bud].get('C0',float('nan')) for r in sel]
            P('  B=%d (B/K=%.0f) routeA %.5f routeB %.5f routeC %.5f  [C-oracle-prior %.5f  C-no-reweight %.5f]  mean %.5f'%(
                Bud,Bud/K,med(A),med(Bl),med(C),med(Co),med(C0),med(M)))
            o,p,lo,hi=perm(C,A)
            P('     C-vs-A %+.5f [%+.5f,%+.5f] p=%.5g  (d/K=%s)'%(o,lo,hi,p,'512/%d=%.2f'%(K,512.0/K) if enc=='clip' else '768/%d=%.2f'%(K,768.0/K)))
        P('  bcts T med %.5f  src NLL %.5f -> %.5f'%(med([r['bcts'][0] for r in rr]),
            med([r['bcts'][1] for r in rr]),med([r['bcts'][2] for r in rr])))
open(ODIR+'/dnet_report.txt','w').write('\n'.join(OUT)+'\n')
print('WROTE',ODIR+'/dnet_report.txt')
