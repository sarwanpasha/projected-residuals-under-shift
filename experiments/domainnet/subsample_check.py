"""Effect of the 4000-point target subsample: compares the accuracies and the gap on
the subsample (tta.py) with the full-target values (core.py) on matched pairs.
"""
import pickle,glob,numpy as np
from prlb.paths import DNET_OUT
ODIR=str(DNET_OUT)
for e in ('clip','dino'):
    full={}
    for r in pickle.load(open(ODIR+'/dnet_%s.pkl'%e,'rb')):
        full[(r['K'],r['src'],r['tgt'])]=r
    sub=[]
    for f in sorted(glob.glob(ODIR+'/dnet_tta_%s_*.pkl'%e)):
        sub+=pickle.load(open(f,'rb'))
    D={'a_naive':[], 'a_mean':[], 'a_orc':[], 'gap':[]}
    nsub=[]
    for r in sub:
        k=(r['K'],r['src'],r['tgt'])
        if k not in full: continue
        F=full[k]
        nsub.append(r['n'])
        for key in ('a_naive','a_mean','a_orc'):
            D[key].append(r[key]-F[key])
        D['gap'].append((r['a_orc']-r['a_mean'])-(F['a_orc']-F['a_mean']))
    print('ENC',e,'matched cells',len(D['a_naive']),'subsample n min',min(nsub),'max',max(nsub))
    for key in ('a_naive','a_mean','a_orc','gap'):
        v=np.array(D[key])
        print('   %-8s mean %+0.5f  max|.| %0.5f  rms %0.5f'%(key,v.mean(),np.abs(v).max(),np.sqrt((v**2).mean())))
