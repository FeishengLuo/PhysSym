#!/usr/bin/env python3
"""Physics-constrained graph channel learner for quantum Pauli parity tests.
Learning outputs physical factor rates; all test parity likelihoods use exact
symplectic/characteristic-function propagation. Device-disjoint synthetic train/test.
"""
import os,sys,time,csv
import numpy as np
import torch
from torch import nn
from .paths import OUTPUT_DIR, CHECKPOINTS_DIR, ensure_output
from .neurosym import EDGES,N,E,device,sample_errors,true_q
from .run_experiments import make_case
ROOT=str(ensure_output())

L=torch.tensor(EDGES[:,0],dtype=torch.long);R=torch.tensor(EDGES[:,1],dtype=torch.long)
torch.set_num_threads(2)

def cal_stats(errors, edge_index=None):
    edge_index=EDGES if edge_index is None else np.asarray(edge_index,dtype=np.int32)
    # one row per device: node means; edge co-flip and parity means
    err=errors.astype(np.float32); mu=err.mean(0)
    co=(err[:,edge_index[:,0]]*err[:,edge_index[:,1]]).mean(0)
    parity=(err[:,edge_index[:,0]]!=err[:,edge_index[:,1]]).mean(0)
    nfeatures=np.stack([mu,np.sqrt(mu*(1-mu)/len(err))],axis=-1)
    efeat=np.stack([co,parity,mu[edge_index[:,0]],mu[edge_index[:,1]],co-mu[edge_index[:,0]]*mu[edge_index[:,1]]],axis=-1)
    return nfeatures.astype('float32'),efeat.astype('float32')

class PhysicsGNN(nn.Module):
  def __init__(self,h=48,depth=3):
    super().__init__();self.node=nn.Sequential(nn.Linear(2,h),nn.SiLU(),nn.Linear(h,h));self.edge=nn.Sequential(nn.Linear(5,h),nn.SiLU(),nn.Linear(h,h))
    self.en=nn.ModuleList([nn.Sequential(nn.Linear(h*3,h),nn.SiLU(),nn.Linear(h,h)) for _ in range(depth)])
    self.nn=nn.ModuleList([nn.Sequential(nn.Linear(h*2,h),nn.SiLU(),nn.Linear(h,h)) for _ in range(depth)])
    self.phead=nn.Sequential(nn.Linear(h+2,32),nn.SiLU(),nn.Linear(32,1))
    self.chead=nn.Sequential(nn.Linear(h+5,32),nn.SiLU(),nn.Linear(32,1))
  def forward(self,node,edge,edge_index=None):
    li,ri=(L,R) if edge_index is None else (torch.as_tensor(edge_index[:,0],dtype=torch.long),torch.as_tensor(edge_index[:,1],dtype=torch.long))
    # [batch, node, features], [batch, edge, features]
    n=self.node(node);e=self.edge(edge)
    B=n.shape[0]
    for ef,nf in zip(self.en,self.nn):
      u=ef(torch.cat([e,n[:,li],n[:,ri]],axis=-1))
      e=e+.35*u
      agg=torch.zeros_like(n)
      agg.index_add_(1,li,u);agg.index_add_(1,ri,u)
      n=n+.35*nf(torch.cat([n,agg/3.0],axis=-1))
    # physical bounds p in [0,.25], c in [0,.30] via sigmoid with prior biases
    p=.25*torch.sigmoid(self.phead(torch.cat([n,node],-1)).squeeze(-1)-1.0)
    c=.30*torch.sigmoid(self.chead(torch.cat([e,edge],-1)).squeeze(-1)-2.0)
    return p,c

def q_from_pc(p,c,masks,edge_index=None):
    masks=masks.to(p.device,dtype=p.dtype)
    if masks.ndim==2: masks=masks[None,:,:].expand(p.shape[0],-1,-1)
    li,ri=(L,R) if edge_index is None else (torch.as_tensor(edge_index[:,0],dtype=torch.long),torch.as_tensor(edge_index[:,1],dtype=torch.long))
    cross=torch.abs(masks[:,:,li]-masks[:,:,ri])
    lp=torch.log1p(-2*p).unsqueeze(1)
    lc=torch.log1p(-2*c).unsqueeze(1)
    logrel=(masks*lp).sum(-1)+(cross*lc).sum(-1)
    return -.5*torch.expm1(logrel)

def make_training(devices=1600,ncal=48,seed=8433,masks_per=100):
    rng=np.random.default_rng(seed)
    # mixed device family, only physical probability latent labels, train supervision parity q
    nodes=[];edges=[];pall=[];call=[]
    for k in range(devices):
        mode=('correlated','independent','sparse')[k%3]
        p,c=device(rng,mode);err=sample_errors(rng,p,c,ncal)
        a,b=cal_stats(err);nodes.append(a);edges.append(b);pall.append(p);call.append(c)
    # fixed random variable-weight queries paired with device independent of eval circuit pool
    masks=(rng.random((masks_per,N))<rng.uniform(.18,.8,(masks_per,1))).astype('float32')
    masks[masks.sum(1)==0,0]=1
    return (torch.from_numpy(np.stack(nodes)),torch.from_numpy(np.stack(edges)),
           torch.from_numpy(np.stack(pall)),torch.from_numpy(np.stack(call)),torch.from_numpy(masks))

def fit(out=None,devs=2000,ncal=48,epochs=35,seed=643):
    if out is None:out=os.path.join(CHECKPOINTS_DIR,'physics_gnn.pt')
    os.makedirs(os.path.dirname(os.path.abspath(out)),exist_ok=True)
    t0=time.time();torch.manual_seed(seed)
    a,b,p,c,m=make_training(devs,ncal,seed=8433+seed-643)
    rng=np.random.default_rng(71);perm=rng.permutation(devs);train=perm[:int(devs*.83)];val=perm[int(devs*.83):]
    net=PhysicsGNN();opt=torch.optim.AdamW(net.parameters(),lr=.0015,weight_decay=.0004)
    best=1e5;best_state=None;hist=[]
    for epoch in range(epochs):
        net.train();losses=[]
        for inds in torch.from_numpy(np.random.default_rng(epoch+33).permutation(train)).split(64):
            pp,cc=net(a[inds],b[inds])
            tq=q_from_pc(p[inds],c[inds],m)
            pq=q_from_pc(pp,cc,m)
            # prediction-focused objective: channel accuracy and latent parameter recovery
            qloss=((pq-tq)**2).mean()
            theta=((pp-p[inds])**2).mean()+((cc-c[inds])**2).mean()
            loss=qloss+0.15*theta
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step()
            losses.append(loss.item())
        net.eval()
        with torch.no_grad():
            vv=[]
            for ids in torch.from_numpy(val).split(80):
                pp,cc=net(a[ids],b[ids]);vv.append(((q_from_pc(pp,cc,m)-q_from_pc(p[ids],c[ids],m))**2).mean().item())
            v=float(np.mean(vv))
        hist.append([epoch+1,float(np.mean(losses)),v]);
        if v<best:
            best=v;best_state={k:t.clone() for k,t in net.state_dict().items()}
        if epoch%10==9:print('physics_epoch',epoch+1,'validation_mse',round(v,6),'elapsed',round(time.time()-t0,1),flush=True)
    net.load_state_dict(best_state);torch.save({'state':best_state,'devs':devs,'cal_shots':ncal,'architecture':'graph physical factor estimation'},out)
    with open(os.path.join(ROOT,'physics_training.csv'),'w',newline='') as f:
        w=csv.writer(f);w.writerow(['epoch','train_loss','val_parity_mse']);w.writerows(hist)
    print('physics_best',best,'elapsed',time.time()-t0,flush=True)
    return net

@torch.no_grad()
def predict(net, errors,masks,edge_index=None):
    net.eval(); a,b=cal_stats(errors,edge_index)
    p,c=net(torch.from_numpy(a)[None],torch.from_numpy(b)[None],edge_index)
    return q_from_pc(p,c,torch.from_numpy(np.asarray(masks,dtype='float32')),edge_index).cpu().numpy()[0].clip(.002,.499)

if __name__=='__main__':
    import argparse
    par=argparse.ArgumentParser();par.add_argument('--train-devs',type=int,default=2000);par.add_argument('--epochs',type=int,default=35)
    args=par.parse_args();fit(devs=args.train_devs,epochs=args.epochs)
