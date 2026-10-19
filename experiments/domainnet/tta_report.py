"""DomainNet label-free leaderboard: merges tta.py and tta_aux.py, prints mean accuracy
with 95 percent intervals per K and encoder, the LaTeX table body, and paired contrasts
against the common-mean correction.
"""
import sys,pickle,glob,numpy as np
from prlb.paths import DNET_OUT
ODIR=str(DNET_OUT)
ENCS=['clip','dino']
ARMS=[('No correction','a_naive'),('TENT','tent'),('SHOT','shot'),('T3A','a_t3a'),
      ('LAME, default','lame_def'),('LAME, oracle-tuned','lame_orc'),('CORAL','a_coral'),
      ('Scale alignment','a_scale'),('Common-mean $+$ LAME','mean_lame'),
      ('Common-mean correction','a_mean'),('BN alignment','a_bn'),
      ('Per-class, predicted label','a_pcp'),('Per-class oracle label','a_orc')]
def ci95(v):
    v=np.asarray([x for x in v if x==x],float)
    if v.size<2: return float('nan'),float('nan'),int(v.size)
    return float(v.mean()), float(1.96*v.std(ddof=1)/np.sqrt(v.size)), int(v.size)
DAT={}
for e in ENCS:
    rows=[]
    for f in sorted(glob.glob(ODIR+'/dnet_tta_%s_*.pkl'%e)):
        rows+=pickle.load(open(f,'rb'))
    ex={}
    for f in sorted(glob.glob(ODIR+'/dnet_tta2_%s_*.pkl'%e)):
        for r in pickle.load(open(f,'rb')):
            ex[(r['K'],r['src'],r['tgt'])]=r
    for r in rows:
        k=(r['K'],r['src'],r['tgt'])
        if k in ex:
            for f2 in ('a_bn','a_coral','a_t3a','a_scale','a_pcp'): r[f2]=ex[k][f2]
    DAT[e]=rows
    print('LOADED',e,len(rows),'cells,',len(ex),'aux cells',flush=True)
KS=sorted({r['K'] for r in DAT['clip']})
print()
for K in KS:
    print('=== K=%d ==='%K)
    for lab,key in ARMS:
        out=[]
        for e in ENCS:
            v=[r.get(key,float('nan')) for r in DAT[e] if r['K']==K]
            m,h,n=ci95(v)
            out.append('%s %.5f+-%.5f (n=%d)'%(e,m,h,n))
        print('  %-24s %s'%(lab,' | '.join(out)))
    print()
for K in KS:
    print('%% --- LaTeX body, DomainNet K=%d ---'%K)
    for lab,key in ARMS:
        cells=[]
        for e in ENCS:
            v=[r.get(key,float('nan')) for r in DAT[e] if r['K']==K]
            m,h,n=ci95(v)
            cells.append('$%.4f\\pm%.4f$'%(m,h))
        print('%s & %s \\\\'%(lab,' & '.join(cells)))
    print()
v=[r.get('lame_alpha') for e in ENCS for r in DAT[e]]
print('LAME alpha choices:',{a:v.count(a) for a in sorted(set(v))})
ns=sorted({r['n'] for e in ENCS for r in DAT[e]})
nf=sorted({r['nfull'] for e in ENCS for r in DAT[e]})
print('subsample sizes:',ns[:5],'... full target sizes min/max:',min(nf),max(nf))
print()
print('=== paired contrasts vs common-mean (mean diff +- 1.96 se over 30 pairs) ===')
for K in KS:
    for e in ENCS:
        rs=[r for r in DAT[e] if r['K']==K]
        base=np.array([r['a_mean'] for r in rs],float)
        for lab,key in [('T3A','a_t3a'),('per-class predicted','a_pcp'),('LAME orc','lame_orc'),('oracle','a_orc')]:
            v=np.array([r.get(key,np.nan) for r in rs],float)
            d=v-base; d=d[~np.isnan(d)]
            m=d.mean(); h=1.96*d.std(ddof=1)/np.sqrt(d.size)
            print('  K=%-3d %-4s %-20s %+0.5f +- %0.5f  %s'%(K,e,lab,m,h,'sig' if abs(m)>h else 'ns'))
