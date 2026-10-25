#!/usr/bin/env python3
"""Deterministic reproducible Pauli-fingerprint fault diagnosis simulation.
Model: ideal Clifford circuits, one propagated Pauli fault per run,
independent symmetric per-qubit readout bit-flip probability p.
Each distinct selected test is run on a fresh input copy; no hardware claims.
"""
import os, csv, json, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .paths import ensure_output
ROOT=str(ensure_output())
OUT=os.path.join(ROOT,'figures')

def propagate(v,gates,n):
    v=v.copy()
    for g in gates:
        if g[0]=='H':
            j=g[1]; v[j],v[n+j]=v[n+j].copy(),v[j].copy()
        elif g[0]=='S':
            j=g[1]; v[n+j]^=v[j]
        else:
            c,t=g[1],g[2];v[t]^=v[c];v[n+c]^=v[n+t]
    return v

def base_circuit(n,seed):
    rng=np.random.default_rng(seed)
    gs=[]
    for layer in range(5):
        order=rng.permutation(n)
        for j in order:
            if rng.random()<0.54:gs.append(('H',int(j)))
            if rng.random()<0.42:gs.append(('S',int(j)))
        cxs=list(range(layer%2,n-1,2))
        for j in cxs: gs.append(('CX',j,j+1))
    return gs

def candidate_pool(n,seed):
    rng=np.random.default_rng(seed)
    cand=[];seen=set()
    def add(v):
        b=bytes(v)
        if b not in seen and np.any(v):seen.add(b);cand.append(v)
    # every weight-1, weight-2 input Pauli test
    for j in range(n):
        for p in ((1,0),(0,1),(1,1)):
            v=np.zeros(2*n,np.uint8);v[j],v[n+j]=p;add(v)
    for j in range(n):
        for k in range(j+1,n):
            for a in ((1,0),(0,1),(1,1)):
                for b in ((1,0),(0,1),(1,1)):
                    v=np.zeros(2*n,np.uint8);v[j],v[n+j]=a;v[k],v[n+k]=b;add(v)
    # deterministic fixed-seed diverse weight-3/4 Paulis
    for _ in range(160):
        w=int(rng.choice([3,4])) if n>=4 else n
        ids=rng.choice(n,size=w,replace=False)
        v=np.zeros(2*n,np.uint8)
        for j in ids:
            p=((1,0),(0,1),(1,1))[int(rng.integers(3))]
            v[j],v[n+j]=p
        add(v)
    return np.stack(cand)

def fault_library(n,gates):
    snapshots=np.unique(np.linspace(0,len(gates),6,dtype=int))
    unique={}
    zero=np.zeros(2*n,np.uint8);unique[bytes(zero)]=zero
    for at in snapshots:
        suffix=gates[int(at):]
        for j in range(n):
            for p in ((1,0),(0,1),(1,1)):
                v=zero.copy();v[j],v[n+j]=p
                u=propagate(v,suffix,n);unique[bytes(u)]=u
        for j in range(n-1):
            for p1,p2 in (((1,0),(1,0)),((0,1),(0,1)),((1,1),(1,1))):
                v=zero.copy();v[j],v[n+j]=p1;v[j+1],v[n+j+1]=p2
                u=propagate(v,suffix,n);unique[bytes(u)]=u
    return np.stack(list(unique.values()))

def symp(A,B,n):
    return ((A[:,:n]@B[:,n:].T + A[:,n:]@B[:,:n].T)%2).astype(np.uint8)

def make_case(n,seed,p):
    gates=base_circuit(n,seed)
    candidates=candidate_pool(n,seed+19)
    measured=np.stack([propagate(v,gates,n) for v in candidates])
    faults=fault_library(n,gates)
    synd=symp(faults,measured,n) # K x C
    # per-test weight of propagated Pauli observable
    w=np.sum((measured[:,:n] | measured[:,n:]),axis=1)
    q=(1-(1-2*p)**w)/2
    b=-np.log(2*np.sqrt(q*(1-q)))
    return faults,candidates,measured,synd,w,q,b

def greedy(synd,b,budget):
    k,c=synd.shape
    ia,ib=np.triu_indices(k,1)
    dif=(synd[ia]!=synd[ib]).T.astype(np.float32)  # C x pairs
    remaining=np.ones(c,dtype=bool)
    residual=np.ones(len(ia),dtype=np.float32)
    selected=[]
    # maximize sum_{pairs} (1-exp(-distance))
    gains_scale=(1-np.exp(-b)).astype(np.float32)
    for _ in range(min(budget,c)):
        gains=(dif @ residual)*gains_scale
        gains[~remaining]=-np.inf
        j=int(np.argmax(gains)); selected.append(j)
        remaining[j]=False
        residual*=np.where(dif[j]>0,np.exp(-b[j]),1.0)
    return selected

def rollout_score(synd,q,inds,seed,trials=1400,retain=None):
    rng=np.random.default_rng(seed)
    kk=synd.shape[0]
    labels=rng.integers(kk,size=trials)
    cols=np.array(inds,dtype=int);S=synd[:,cols]
    qq=q[cols]
    targets=S[labels]
    reads=np.bitwise_xor(targets, (rng.random(targets.shape)<qq).astype(np.uint8))
    # independent symmetric readout channel log-likelihood for all hypothesized signatures
    ll=np.zeros((trials,kk),dtype=np.float32)
    for t in range(len(cols)):
        lmatch=np.log1p(-qq[t]);lmismatch=np.log(qq[t])
        ll += np.where(reads[:,t,None]==S[None,:,t],lmatch,lmismatch).astype(np.float32)
    # deterministic random tie breaks only where multiple candidates with exactly the same LL
    eps=rng.random(ll.shape)*1e-7
    pred=(ll+eps).argmax(axis=1)
    acc=float(np.mean(pred==labels))
    sep=(S[:,None,:]!=S[None,:,:]).any(axis=-1)
    identifiable=float(np.mean(sep[np.triu_indices(kk,1)]))
    min_dist=int((S[:,None,:]!=S[None,:,:]).sum(axis=-1)[np.triu_indices(kk,1)].min())
    return acc,identifiable,min_dist

def experiment(seeds=[11,29,47,83],ns=[4,6,8],ps=[0.025,0.06,0.10],budgets=[8,12,16,20,24],trials=1400):
    rows=[];details=[]
    for n in ns:
        for circuit_seed in seeds:
            for p in ps:
                faults,cands,measured,S,w,q,b=make_case(n,circuit_seed,p)
                maxb=max(budgets)
                chosen={
                    'Noise-aware':greedy(S,b,maxb),
                    'Unweighted':greedy(S,np.ones_like(b),maxb),
                    'Random':np.random.default_rng(circuit_seed+109).permutation(S.shape[1])[:maxb].tolist(),
                    'Local basis':sorted(range(S.shape[1]),key=lambda i: (i>=3*n,i))[:maxb],
                }
                for method, inds in chosen.items():
                    for m in budgets:
                        scores=rollout_score(S,q,inds[:m],int(1000*n + circuit_seed*11 + p*10000 + m*15),trials)
                        rows.append(dict(n=n,seed=circuit_seed,p=p,budget=m,method=method,acc=scores[0],pair_coverage=scores[1],minimum_distance=scores[2],fault_classes=len(faults),candidate_tests=len(cands),mean_weight=float(w[inds[:m]].mean()),mean_parity_flip=float(q[inds[:m]].mean())))
                details.append({'n':n,'seed':circuit_seed,'p':p,'hypotheses':len(faults),'tests':len(cands)})
                print(n,circuit_seed,p,len(faults),len(cands),flush=True)
    with open(os.path.join(ROOT,'results.csv'),'w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with open(os.path.join(ROOT,'manifest.json'),'w') as f:
        json.dump({'model':'exact Pauli/Clifford commutation + independent classical readout-bit errors', 'seeds':seeds,'n':ns,'p':ps,'budgets':budgets,'trials_each':trials,'detail':details},f,indent=2)
    return rows

def plot(rows):
    os.makedirs(OUT,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    methods=['Noise-aware','Unweighted','Local basis','Random']
    colors={'Noise-aware':'#186E86','Unweighted':'#D3812B','Local basis':'#6B63A3','Random':'#87939B'}
    markers={'Noise-aware':'o','Unweighted':'s','Local basis':'^','Random':'D'}
    n=8;p=0.06
    fig,ax=plt.subplots(figsize=(5.8,3.15))
    for met in methods:
        sub=[r for r in rows if r['n']==n and r['p']==p and r['method']==met]
        bs=sorted(set(r['budget'] for r in sub));avg=[np.mean([r['acc'] for r in sub if r['budget']==m]) for m in bs]
        ax.plot(bs,avg,marker=markers[met],color=colors[met],linewidth=1.9,markersize=5,label=met)
    ax.set(xlabel='Circuit executions / test budget',ylabel='Top-1 fault-class diagnosis accuracy',ylim=(0,1.03));ax.set_xticks(bs)
    ax.grid(alpha=.18,linestyle='--'); ax.legend(fontsize=8,frameon=False,loc='lower right')
    fig.tight_layout(pad=.6)
    fig.savefig(os.path.join(OUT,'accuracy_budget.pdf'),bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(5.8,3.15))
    for met in methods:
        sub=[r for r in rows if r['p']==0.06 and r['budget']==20 and r['method']==met]
        xs=sorted(set(r['n'] for r in sub));v=[np.mean([r['acc'] for r in sub if r['n']==x]) for x in xs]
        ax.plot(xs,v,marker=markers[met],color=colors[met],linewidth=1.9,markersize=5,label=met)
    ax.set(xlabel='Number of qubits',ylabel='Top-1 diagnosis accuracy',ylim=(0,1.03));ax.set_xticks([4,6,8]);ax.grid(alpha=.18,linestyle='--');ax.legend(fontsize=8,frameon=False)
    fig.tight_layout(pad=.6);fig.savefig(os.path.join(OUT,'scaling.pdf'),bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(5.8,3.15))
    for met in methods:
        sub=[r for r in rows if r['n']==8 and r['budget']==20 and r['method']==met]
        xs=sorted(set(r['p'] for r in sub));v=[np.mean([r['acc'] for r in sub if r['p']==x]) for x in xs]
        ax.plot([x*100 for x in xs],v,marker=markers[met],color=colors[met],linewidth=1.9,markersize=5,label=met)
    ax.set(xlabel='Per-qubit readout flip probability (%)',ylabel='Top-1 diagnosis accuracy',ylim=(0,1.03));ax.set_xticks([2.5,6,10]);ax.grid(alpha=.18,linestyle='--');ax.legend(fontsize=8,frameon=False)
    fig.tight_layout(pad=.6);fig.savefig(os.path.join(OUT,'noise.pdf'),bbox_inches='tight');plt.close(fig)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--trials',type=int,default=1400);args=ap.parse_args()
    rows=experiment(trials=args.trials);plot(rows)
    print('finished',len(rows),'results')
