#!/usr/bin/env python3
"""Paired, batched held-out evaluation (same episode random numbers as neurosym.py)."""
import os,csv,sys,time
import numpy as np,pandas as pd,torch
from .paths import CHECKPOINTS_DIR, ensure_output
ROOT=str(ensure_output())
from .neurosym import N,device,sample_errors,true_q,neural_q,NoiseNet,independent_q,parity_cal_q,moment_q,features,make_case,entropy
from .physnet import PhysicsGNN,predict
import pickle

def batched_policy(S,qt,qpred,ys,faults,randomloc,policy='mi'):
    S=np.asarray(S,dtype=np.float64);qt=np.asarray(qt,dtype=np.float64);qpred=np.asarray(qpred,dtype=np.float64)
    B=len(faults);K,C=S.shape;post=np.full((B,K),1.0/K)
    selacc=[];selent=[]
    pen=entropy(qpred)
    for step in range(ys.shape[1]):
        if policy=='random':choices=randomloc[:,step]
        else:
            prob=qpred[None,:]+(1-2*qpred[None,:])*(post@S)
            mi=entropy(prob)-pen[None,:]
            choices=mi.argmax(axis=1)
        noise=ys[:,step] < qt[choices]
        y=np.bitwise_xor(S[faults,choices].astype(np.uint8),noise.astype(np.uint8))
        st=S[:,choices].T
        like=np.where(st==y[:,None],1-qpred[choices,None],qpred[choices,None]);post*=like
        post/=post.sum(1,keepdims=True)
        if step+1 in (4,8,12,16,20):
            selacc.append((post.argmax(1)==faults).mean())
            selent.append((-post*np.log(np.maximum(post,1e-20))).sum(1).mean())
    return selacc,selent

def run(episodes=100):
    ck=torch.load(os.path.join(CHECKPOINTS_DIR,'physics_gnn.pt'),map_location='cpu',weights_only=True)
    ph=PhysicsGNN();ph.load_state_dict(ck['state']);ph.eval()
    ck0=torch.load(os.path.join(CHECKPOINTS_DIR,'neural_channel.pt'),map_location='cpu',weights_only=True)
    old=NoiseNet(ck0['dim']);old.load_state_dict(ck0['state']);old.eval()
    with open(os.path.join(CHECKPOINTS_DIR,'ridge_channel.pkl'),'rb') as f:ridge=pickle.load(f)
    modes=['correlated','independent','sparse'];circs=[11,29,47,83];devs=[201,301,401];rows=[];cal=[]
    t0=time.time()
    for reg in modes:
      for circ in circs:
        faults,_,measured,S,w,_,_=make_case(N,circ,.06)
        masks=(measured[:,:N]|measured[:,N:]).astype('uint8')
        for ds in devs:
            rng=np.random.default_rng(100000+circ*13+ds*19+{'correlated':1,'independent':2,'sparse':3}[reg]);p,c=device(rng,reg)
            err=sample_errors(rng,p,c,48);qt=true_q(p,c,masks)
            qs={'Oracle':qt,'PhysicsGNN':predict(ph,err,masks),'NeuralResidual':neural_q(old,err,masks),
                'Ridge':np.clip(ridge.predict(features(err,masks)),.005,.495),
                'Empirical':np.clip(parity_cal_q(err,masks),.01,.49),
                'Independent':independent_q(err,masks), 'Moment':moment_q(err,masks)}
            for a,q in qs.items():
                cal.append(dict(regime=reg,circuit=circ,device=ds,model=a,mse=np.mean((q-qt)**2)))
            truefaults=rng.integers(len(faults),size=episodes)
            shots=rng.random((episodes,20));randomloc=rng.integers(S.shape[1],size=(episodes,20))
            for a,q in qs.items():
                ac,en=batched_policy(S,qt,q,shots,truefaults,randomloc)
                for k,t in enumerate((4,8,12,16,20)):
                    rows.append(dict(regime=reg,circuit=circ,device=ds,method=a,budget=t,acc=float(ac[k]),entropy=float(en[k]),episodes=episodes))
            print('finished',reg,circ,ds,'s',round(time.time()-t0,1),flush=True)
    for name,data in [('physics_results.csv',rows),('physics_calibration.csv',cal)]:
        pd.DataFrame(data).to_csv(os.path.join(ROOT,name),index=False)
    a=pd.DataFrame(rows);print('\nRESULTS budget 20\n',a[a.budget==20].groupby(['regime','method']).acc.agg(['mean','std']).to_string())
    b=pd.DataFrame(cal);print('\nMSE\n',b.groupby(['regime','model']).mse.mean().to_string())
    return a,b
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--episodes',type=int,default=100)
    run(ap.parse_args().episodes)
