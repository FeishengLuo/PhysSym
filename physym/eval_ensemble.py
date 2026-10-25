import os,sys
import numpy as np,pandas as pd,torch
from .paths import CHECKPOINTS_DIR, ensure_output
ROOT=str(ensure_output())
from .neurosym import make_case,device,sample_errors,true_q,independent_q,parity_cal_q,neural_q,NoiseNet,N
from .physnet import PhysicsGNN,predict
from .eval_physics import batched_policy
def run(episodes=100):
    models=[]
    for f in ['physics_gnn.pt','physics_gnn_101.pt','physics_gnn_203.pt']:
      path=os.path.join(CHECKPOINTS_DIR,f)
      net=PhysicsGNN();net.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['state']);net.eval();models.append(net)
    rows=[];qresults=[]
    for regime in ['correlated','independent','sparse']:
     for circ in [11,29,47,83]:
      faults,_,measured,S,_,_,_=make_case(N,circ,.06)
      masks=(measured[:,:N]|measured[:,N:]).astype('uint8')
      for ds in [201,301,401]:
       rng=np.random.default_rng(100000+circ*13+ds*19+{'correlated':1,'independent':2,'sparse':3}[regime])
       p,c=device(rng,regime);err=sample_errors(rng,p,c,48);qt=true_q(p,c,masks)
       q=[predict(model,err,masks) for model in models];qe=np.mean(q,axis=0);qp=q[0]
       qarr={'PhysicsGNN':qp,'PhysicsEnsemble':qe,'Ensemble+empirical 25%':.75*qe+.25*np.clip(parity_cal_q(err,masks),.01,.49),'Oracle':qt}
       f=rng.integers(len(faults),size=episodes);ys=rng.random((episodes,20));rr=rng.integers(S.shape[1],size=(episodes,20))
       for name,pred in qarr.items():
        ac,en=batched_policy(S,qt,pred,ys,f,rr)
        for i,b in enumerate([4,8,12,16,20]):
         rows.append(dict(regime=regime,circuit=circ,device=ds,method=name,budget=b,acc=ac[i],entropy=en[i]))
        qresults.append(dict(regime=regime,circuit=circ,device=ds,method=name,mse=np.mean((pred-qt)**2)))
    pd.DataFrame(rows).to_csv(os.path.join(ROOT,'ensemble_results.csv'),index=False)
    pd.DataFrame(qresults).to_csv(os.path.join(ROOT,'ensemble_calibration.csv'),index=False)
    print(pd.DataFrame(rows).query('budget==20').groupby(['regime','method']).acc.mean().to_string())
    print(pd.DataFrame(qresults).groupby(['regime','method']).mse.mean().to_string())

if __name__ == '__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=100)
    run(ap.parse_args().episodes)
