#!/usr/bin/env python3
"""Zero-retraining test on larger quantum registers, disjoint circuits/devices."""
import os,sys
import numpy as np,pandas as pd,torch
from .paths import CHECKPOINTS_DIR, ensure_output
ROOT=str(ensure_output())
from .neurosym import make_case
from .physnet import PhysicsGNN,predict
from .eval_physics import batched_policy

def edge_index(n): return np.array([(i,i+1) for i in range(n-1)]+[(i,i+2) for i in range(n-2)],dtype=np.int32)
def init_noise(n,rng,regime):
    ei=edge_index(n);E=len(ei)
    p=rng.uniform(.012,.09,n).astype('float32')
    if regime=='correlated':
        c=rng.uniform(.035,.20,E).astype('float32')*(rng.random(E)<.74);c[n-1:]*=.48
    elif regime=='sparse':c=rng.uniform(.025,.23,E).astype('float32')*(rng.random(E)<.28)
    else:c=np.zeros(E,dtype='float32')
    return p,c,ei

def errors(rng,p,c,ei,shots):
    n=len(p);err=(rng.random((shots,n))<p[None,:]).astype('uint8');bur=(rng.random((shots,len(ei)))<c[None,:]).astype('uint8')
    for j,(a,b) in enumerate(ei):err[:,a]^=bur[:,j];err[:,b]^=bur[:,j]
    return err

def qtrue(p,c,ei,masks):
    border=(masks[:,ei[:,0]]^masks[:,ei[:,1]]).astype('float32')
    l=(masks*np.log1p(-2*p)).sum(1)+(border*np.log1p(-2*c)).sum(1)
    return np.clip(-.5*np.expm1(l),.001,.499)

def q_indep(err,masks):
    p=np.clip(err.mean(0),.001,.49)
    return np.clip(-.5*np.expm1(masks@np.log1p(-2*p)),.001,.499)

def run(trials=120, sizes=(8,10,12), circuits=(59,73,101,131), devices=(503,607,701)):
    model=PhysicsGNN(); ck=torch.load(os.path.join(CHECKPOINTS_DIR,'physics_gnn.pt'),weights_only=True);model.load_state_dict(ck['state']);model.eval()
    rows=[];cal=[]
    for n in sizes:
      for regime in ['correlated','sparse','independent']:
        for circ in circuits:
            faults,_,measured,S,_,_,_=make_case(n,circ,.06)
            masks=(measured[:,:n]|measured[:,n:]).astype('uint8')
            for ds in devices:
                rng=np.random.default_rng(90000+n*10000+circ*31+ds*37+{'correlated':1,'sparse':2,'independent':3}[regime]);p,c,ei=init_noise(n,rng,regime)
                err=errors(rng,p,c,ei,48);qt=qtrue(p,c,ei,masks)
                pred=predict(model,err,masks,ei)
                emp=np.clip(((err.astype('int16')@masks.T.astype('int16'))%2).mean(0),.01,.49)
                ind=q_indep(err,masks)
                f=rng.integers(len(faults),size=trials);ys=rng.random((trials,20));loc=rng.integers(S.shape[1],size=(trials,20))
                for m,q in [('PhysicsGNN',pred),('Empirical',emp),('Independent',ind),('Oracle',qt)]:
                    acc,ent=batched_policy(S,qt,q,ys,f,loc)
                    for b,x in zip([4,8,12,16,20],acc):
                        rows.append(dict(n=n,regime=regime,circuit=circ,device=ds,method=m,budget=b,acc=x,episodes=trials,classes=len(faults),tests=len(qt)))
                    cal.append(dict(n=n,regime=regime,circuit=circ,device=ds,method=m,mse=np.mean((q-qt)**2)))
            print('transfer',n,regime,circ,'K',len(faults),'C',S.shape[1],flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(ROOT,'transfer_results.csv'),index=False)
    pd.DataFrame(cal).to_csv(os.path.join(ROOT,'transfer_calibration.csv'),index=False)
    print(pd.DataFrame(rows).query('budget==20').groupby(['n','regime','method']).acc.mean().to_string())

if __name__ == '__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=120)
    ap.add_argument('--sizes', type=int, nargs='+', default=[8,10,12])
    args=ap.parse_args()
    run(trials=args.episodes, sizes=args.sizes)
