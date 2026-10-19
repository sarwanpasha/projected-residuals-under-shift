"""Departure of the selective corrector from the common-mean correction on DomainNet,
per K, encoder and budget.
"""
import pickle,glob,numpy as np
from prlb.paths import DNET_OUT
ODIR=str(DNET_OUT)
ENCS=['clip','dino']
BUDG=[50,100,200,400,800,1600]
def ci(v):
    v=np.asarray([x for x in v if x==x],float)
    if v.size<2: return float('nan'),float('nan'),int(v.size)
    return float(v.mean()), float(1.96*v.std(ddof=1)/np.sqrt(v.size)), int(v.size)
DAT={}
for e in ENCS:
    rows=[]
    for f in sorted(glob.glob(ODIR+'/dnet_sel_%s_*.pkl'%e)):
        rows+=pickle.load(open(f,'rb'))
    DAT[e]=rows
    print('LOADED',e,len(rows),'cells')
KS=sorted({r['K'] for r in DAT['clip']})
worst=0.0; worstwhere=None
print()
for K in KS:
    print('=== K=%d ==='%K)
    for e in ENCS:
        rs=[r for r in DAT[e] if r['K']==K]
        am=np.array([r['a_mean'] for r in rs]); ao=np.array([r['a_orc'] for r in rs])
        print('  %-4s common-mean %.5f  per-class oracle %.5f  headroom %+.5f  plug-in positive %.1f/%d'%(
            e,am.mean(),ao.mean(),(ao-am).mean(),float(np.mean([r['nposhat'] for r in rs])),K))
        for B in BUDG:
            d=np.array([r['sel_%d'%B]-r['a_mean'] for r in rs],float)
            d=d[~np.isnan(d)]
            if d.size<2: continue
            m,h,n=ci(d)
            rv=np.array([r['rcv_%d'%B]-r['a_mean'] for r in rs],float); rv=rv[~np.isnan(rv)]
            ns=np.mean([r['nsel_%d'%B] for r in rs if r['nsel_%d'%B]>=0])
            if abs(m)>worst: worst=abs(m); worstwhere=(K,e,B)
            print('     B=%-5d departure %+0.5f +- %0.5f   worst cell %+0.5f   classes corrected %.1f   refit-cv %+0.5f'%(
                B,m,h,d[np.argmax(np.abs(d))],ns,rv.mean() if rv.size else float('nan')))
    print()
print('LARGEST MEAN DEPARTURE over all cells: %+0.5f at K=%d %s B=%d'%(worst if worstwhere else 0,
      worstwhere[0],worstwhere[1],worstwhere[2]))
allv=[r['sel_%d'%B]-r['a_mean'] for e in ENCS for r in DAT[e] for B in BUDG if r['sel_%d'%B]==r['sel_%d'%B]]
allv=np.array(allv,float)
print('OVER ALL %d cell-budget pairs: min %+0.5f  max %+0.5f  mean %+0.5f  frac below -0.01: %.4f'%(
      allv.size,allv.min(),allv.max(),allv.mean(),float((allv<-0.01).mean())))
