#!/usr/bin/env python3
"""NSAD: neural-calibrated symplectic active fault diagnosis.
Explicit simulator: ideal Clifford U, one persistent Pauli fault, correlated
classical readout noise sampled independently per circuit execution.
No real-device measurements; independent circuit/device train/test seeds.
"""
import os, csv, json, sys, argparse, time, pickle
import numpy as np
import torch
from torch import nn
from sklearn.linear_model import Ridge
from .paths import OUTPUT_DIR, CHECKPOINTS_DIR, ensure_output
from .run_experiments import make_case, greedy
ROOT=str(ensure_output())
torch.set_num_threads(2)
N=8
EDGES=np.array([(i,i+1) for i in range(N-1)]+[(i,i+2) for i in range(N-2)],dtype=np.int32)
E=len(EDGES)

def device(rng, regime='correlated'):
    p=rng.uniform(.012,.09,N).astype('float32')
    if regime=='correlated':
        cs=rng.uniform(.035,.20,E).astype('float32') * (rng.random(E)<.74)
        cs[N-1:]*=.48
    elif regime=='independent': cs=np.zeros(E,dtype='float32')
    elif regime=='sparse':
        cs=rng.uniform(.025,.23,E).astype('float32') * (rng.random(E)<.28)
    else:raise ValueError(regime)
    return p,cs

def sample_errors(rng,p,cs,shots):
    errors=(rng.random((shots,N))<p[None,:]).astype('uint8')
    bursts=(rng.random((shots,E))<cs[None,:]).astype('uint8')
    for e,(i,j) in enumerate(EDGES):
        errors[:,i]^=bursts[:,e];errors[:,j]^=bursts[:,e]
    return errors

def true_q(p,cs,masks):
    masks=np.asarray(masks,dtype=np.uint8)
    cross=(masks[:,EDGES[:,0]]^masks[:,EDGES[:,1]]).astype('float32')
    ln=(masks*np.log1p(-2*p)[None,:]).sum(axis=1)+(cross*np.log1p(-2*cs)[None,:]).sum(axis=1)
    return np.clip(.5*(-np.expm1(ln)),.001,.499).astype('float32')

def parity_cal_q(errors,masks):
    return ((errors.astype('int16')@masks.T.astype('int16'))%2).mean(axis=0).astype('float32')

def independent_q(errors,masks):
    pp=np.clip(errors.mean(axis=0),.001,.49)
    logprod=masks@np.log1p(-2*pp)
    return np.clip(.5*(-np.expm1(logprod)),.001,.499).astype('float32')

def moment_q(errors,masks,shrink=.72):
    """Structured method-of-moments estimator from calibrated Z parities.
    For the modeled XOR-edge readout channel, moment_i moment_j / moment_ij
    equals (1-2c_ij)^2 exactly at population level.
    """
    z=1-2*errors.astype('float32')
    mi=np.clip(z.mean(0),.02,.999)
    ze=(z[:,EDGES[:,0]]*z[:,EDGES[:,1]]).mean(0)
    ce=(1-np.sqrt(np.clip(mi[EDGES[:,0]]*mi[EDGES[:,1]]/np.clip(ze,.02,.999),.3,1)))/2
    ce=np.clip(ce,0,.27)*shrink
    ae=1-2*ce
    prod=np.ones(N,dtype='float32')
    for j,(a,b) in enumerate(EDGES):
        prod[a]*=ae[j];prod[b]*=ae[j]
    pp=np.clip(.5*(1-mi/np.maximum(prod,.01)),.001,.35)
    return true_q(pp,ce,masks)

def features(errors,masks):
    """Permutation-compatible local/node/edge signals pooled for all tests.
    Inputs from a finite calibration bitstring collection, not oracle p,c.
    Each X has 15 physics-guided summaries and a direct empirical parity.
    """
    masks=masks.astype('float32'); shots=len(errors)
    mu=errors.mean(0).astype('float32')
    cov=np.array([np.mean(errors[:,a]*errors[:,b])-mu[a]*mu[b] for a,b in EDGES],dtype='float32')
    bnd=np.abs(masks[:,EDGES[:,0]]-masks[:,EDGES[:,1]])
    within=masks[:,EDGES[:,0]]*masks[:,EDGES[:,1]]
    qemp=parity_cal_q(errors,masks)
    qind=independent_q(errors,masks)
    # preserve device-level calibration; expose all node and edge observations
    # and pair-border interactions to a trainable neural residual network.
    X=np.concatenate([masks, mu[None,:].repeat(len(masks),axis=0),
       cov[None,:].repeat(len(masks),axis=0), bnd, within,
       (masks*mu[None,:]),(bnd*cov[None,:]),
       qemp[:,None],qind[:,None],masks.mean(axis=1)[:,None]],axis=1)
    return X.astype('float32')

class NoiseNet(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d,128),nn.GELU(),nn.Linear(128,128),nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,1))
    def forward(self,x):
        q0=x[:,-2:-1].clamp(.004,.496)
        # physics residual logit: independent readout as anchor, learned crosstalk correction
        raw=torch.logit(2*q0)+self.net(x)
        return .5*torch.sigmoid(raw)

def train(seed=2027, devices=580, ncal=48, samples=110):
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    rng=np.random.default_rng(seed);torch.manual_seed(seed)
    X=[]; Y=[]
    for idx in range(devices):
        regime='independent' if idx%5==0 else ('sparse' if idx%7==0 else 'correlated')
        p,c=device(rng,regime);err=sample_errors(rng,p,c,ncal)
        m=(rng.random((samples,N))<rng.uniform(.18,.85,(samples,1))).astype('uint8')
        m[m.sum(1)==0,0]=1
        X.append(features(err,m));Y.append(true_q(p,c,m))
    X=np.concatenate(X);Y=np.concatenate(Y)
    order=rng.permutation(devices)
    xdev=X.reshape(devices,samples,-1);ydev=Y.reshape(devices,samples)
    tr=order[:int(devices*.85)];va=order[int(devices*.85):]
    Xt=torch.tensor(xdev[tr].reshape(-1,X.shape[1]));Yt=torch.tensor(ydev[tr].reshape(-1,1));Xv=torch.tensor(xdev[va].reshape(-1,X.shape[1]));Yv=torch.tensor(ydev[va].reshape(-1,1))
    model=NoiseNet(X.shape[1]);opt=torch.optim.AdamW(model.parameters(),lr=.0018,weight_decay=.0002)
    best=1e10; hist=[]; state=None
    for epoch in range(40):
        model.train();ids=torch.randperm(len(Yt)); losses=[]
        for batch in ids.split(1024):
            pred=model(Xt[batch]);loss=((pred-Yt[batch])**2).mean()
            opt.zero_grad();loss.backward();opt.step();losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            v=((model(Xv)-Yv)**2).mean().item()
        hist.append(dict(epoch=epoch+1,train_mse=float(np.mean(losses)),val_mse=v))
        if v<best: best=v;state={k:v.clone() for k,v in model.state_dict().items()}
        if epoch%10==9:print('epoch',epoch+1,'val_MSE',round(v,6),flush=True)
    model.load_state_dict(state)
    ridge=Ridge(alpha=10.0).fit(xdev[tr].reshape(-1,X.shape[1]),ydev[tr].reshape(-1))
    with open(os.path.join(CHECKPOINTS_DIR,'ridge_channel.pkl'),'wb') as f:pickle.dump(ridge,f)
    torch.save(dict(state=state,dim=X.shape[1],calibration=ncal,devices=devices,seed=seed),os.path.join(CHECKPOINTS_DIR,'neural_channel.pt'))
    with open(os.path.join(ROOT,'train_history.csv'),'w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(hist[0]));writer.writeheader();writer.writerows(hist)
    return model

@torch.no_grad()
def neural_q(model,errors,masks):
    model.eval();X=features(errors,masks)
    return np.clip(model(torch.from_numpy(X)).numpy().ravel(),.005,.495)

def entropy(x):
    a=np.clip(x,1e-7,1-1e-7)
    return -a*np.log(a)-(1-a)*np.log1p(-a)

def policy_step(posterior,S,q,method):
    if method.endswith('risk'):
        # one-step Bayes classification reward = sum_y max_f p(f)P(y|f)
        positive=q[None,:]+(1-2*q[None,:])*S
        a=np.max(posterior[:,None]*positive,axis=0)
        b=np.max(posterior[:,None]*(1-positive),axis=0)
        return int(np.argmax(a+b))
    # exact mutual information under assumed memoryless Bernoulli channel
    py=q+(1-2*q)*(posterior@S)
    mi=entropy(py)-entropy(q)
    return int(np.argmax(mi))

def simulate_episode(S,trueq,qass,method,ys,fs,randomlist,staticinds=None):
    K,C=S.shape;posterior=np.ones(K,dtype='float64')/K
    predictions=[]; entropy_v=[]; choices=[]
    for step,u in enumerate(ys):
        if staticinds is not None: t=int(staticinds[step])
        elif method=='random':t=int(randomlist[step])
        else:t=policy_step(posterior,S,qass,method)
        choices.append(t)
        y=int(S[fs,t])^int(u<float(trueq[t]))
        qq=float(qass[t]);like=np.where(S[:,t]==y,1-qq,qq)
        posterior*=like
        posterior=posterior/max(posterior.sum(),1e-200)
        predictions.append(int(np.argmax(posterior)==fs))
        entropy_v.append(float(-np.dot(posterior,np.log(np.maximum(posterior,1e-20)))))
    return np.array(predictions),np.array(entropy_v),choices

def evaluation(model, ncal=48, episodes=34, circuit_seeds=(11,29,47,83), dev_seeds=(201,301,401),regimes=('correlated','independent','sparse')):
    rows=[];calrows=[]
    with open(os.path.join(CHECKPOINTS_DIR,'ridge_channel.pkl'),'rb') as f: ridge=pickle.load(f)
    for regime in regimes:
      for circ in circuit_seeds:
        faults,_,measured,S,w,_,_=make_case(8,circ,.06)
        masks=(measured[:,:8] | measured[:,8:]).astype('uint8');Sf=S.astype('float64')
        for ds in dev_seeds:
            rng=np.random.default_rng(100000+circ*13+ds*19+{'correlated':1,'independent':2,'sparse':3}[regime])
            p,c=device(rng,regime);e=sample_errors(rng,p,c,ncal)
            qt=true_q(p,c,masks)
            qi=independent_q(e,masks)
            qe=np.clip(parity_cal_q(e,masks),.01,.49)
            qn=neural_q(model,e,masks)
            qr=np.clip(ridge.predict(features(e,masks)),.005,.495)
            qm=moment_q(e,masks)
            for name,q in [('Neural',qn),('Ridge',qr),('Moment',qm),('Independent',qi),('Empirical',qe)]:
                calrows.append(dict(regime=regime,circuit=circ,device_seed=ds,method=name,mse=float(np.mean((q-qt)**2)),mae=float(np.mean(np.abs(q-qt))),ncal=ncal))
            # matched budget static (including repeated trials of same physical circuits if specified)
            ws=-np.log(2*np.sqrt(qt*(1-qt)))
            wn=-np.log(2*np.sqrt(qn*(1-qn)))
            static_sift=greedy(S,wn,20)
            static_oracle=greedy(S,ws,20)
            static_unweighted=greedy(S,np.ones_like(qt),20)
            local=sorted(range(len(qt)),key=lambda x:(x>=24,x))[:20]
            methods={'NSAD (neural MI)':(qn,None),'Neural Bayes-risk':(qn,None),
                  'Oracle MI':(qt,None),'Oracle Bayes-risk':(qt,None),
                  'Empirical MI':(qe,None),'Independent MI':(qi,None),
                  'Ridge MI':(qr,None),'Moment MI':(qm,None),'Constant MI':(np.full_like(qt, .18),None),
                  'SIFT neural static':(qn,static_sift),'SIFT oracle static':(qt,static_oracle),
                  'Unweighted static':(qi,static_unweighted),'Local Pauli':(qi,local),
                  'Random':(qi,None)}
            truefaults=rng.integers(len(faults),size=episodes)
            shot_noise=rng.random((episodes,20));random_locs=rng.integers(S.shape[1],size=(episodes,20))
            for name,(qx,statics) in methods.items():
                pred=[];ents=[]
                for i,fl in enumerate(truefaults):
                    method=('random' if name=='Random' else ('risk' if 'risk' in name else 'mi'))
                    yy,ee,_=simulate_episode(Sf,qt,qx,method,shot_noise[i],fl,random_locs[i],statics if name!='Random' else None)
                    pred.append(yy);ents.append(ee)
                pp=np.stack(pred);hh=np.stack(ents)
                for budget in [4,8,12,16,20]:
                    rows.append(dict(regime=regime,circuit=circ,device_seed=ds,method=name,budget=budget,
                                     acc=float(pp[:,budget-1].mean()),entropy=float(hh[:,budget-1].mean()),episodes=episodes,
                                     fault_classes=len(faults),candidate_tests=len(qt)))
            print('evaluated',regime,circ,ds,'K',len(faults),flush=True)
    for fname,r in [('results_adaptive.csv',rows),('channel_calibration.csv',calrows)]:
        with open(os.path.join(ROOT,fname),'w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(r[0]));writer.writeheader();writer.writerows(r)
    json.dump(dict(n_qubits=8,calibration_shots=ncal,episodes_each=episodes,train_device_seed=2027,
       circuit_seeds=list(circuit_seeds),evaluation_device_seeds=list(dev_seeds),
       noise_regimes=list(regimes),calibration_included_in_diagnosis_budget=False,
       model='independent qubit flips XOR pair-burst flips on known 1st/2nd-neighbor edges',
       consistency='fresh persistent Pauli mutation fault per diagnostic episode; independent noise across measurements'),
       open(os.path.join(ROOT,'protocol.json'),'w'),indent=2)
    return rows,calrows

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--episodes',type=int,default=34);parser.add_argument('--train-devices',type=int,default=580);parser.add_argument('--eval-only',action='store_true');args=parser.parse_args()
    st=time.time()
    if args.eval_only:
        ck=torch.load(os.path.join(CHECKPOINTS_DIR,'neural_channel.pt'),map_location='cpu',weights_only=True)
        net=NoiseNet(ck['dim']);net.load_state_dict(ck['state'])
    else: net=train(devices=args.train_devices)
    rr,cr=evaluation(net,episodes=args.episodes)
    print('total seconds',round(time.time()-st,2));
    for reg in ['correlated','independent','sparse']:
        for name in ['NSAD (neural MI)','Oracle MI','Neural Bayes-risk','Independent MI','SIFT neural static','Random']:
            scores=[r['acc'] for r in rr if r['regime']==reg and r['method']==name and r['budget']==20]
            print(reg,name,round(np.mean(scores),4))
        for name in ['Neural','Ridge','Moment','Independent','Empirical']:
            print('calibration',reg,name,round(np.mean([r['mse'] for r in cr if r['regime']==reg and r['method']==name]),5))
