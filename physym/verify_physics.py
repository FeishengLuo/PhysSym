#!/usr/bin/env python3
"""Monte Carlo validation of analytic correlated parity law + held-out paired stats."""
import os,json
import numpy as np,pandas as pd
from scipy.stats import t
from .paths import REFERENCE_DIR, ensure_output
from .eval_transfer import init_noise, errors, qtrue
ROOT=str(REFERENCE_DIR)
def run(samples=240000):
    rng=np.random.default_rng(31337);n=8
    p,c,ei=init_noise(n,rng,'correlated')
    m=(rng.random((36,n))<.5).astype('uint8')
    bit=errors(rng,p,c,ei,samples)
    emp=((bit.astype('int16')@m.T.astype('int16'))%2).mean(0)
    exact=qtrue(p,c,ei,m)
    err=float(np.max(np.abs(emp-exact)))
    tolerance=max(.006, 6.0*np.sqrt(.25/samples))
    assert err<tolerance,(err, tolerance, exact, emp)
    checks=[]
    for filename,method,base,reg,nn in [('physics_results.csv','PhysicsGNN','NeuralResidual','sparse',None),('transfer_results.csv','PhysicsGNN','Empirical','correlated',12),('ensemble_results.csv','PhysicsEnsemble','PhysicsGNN','correlated',None)]:
     d=pd.read_csv(os.path.join(ROOT,filename)).query('budget==20 and regime==@reg')
     if nn is not None:d=d[d.n==nn]
     dif=(d.pivot_table(index=['circuit','device'],columns='method',values='acc')[method]-d.pivot_table(index=['circuit','device'],columns='method',values='acc')[base]).dropna().values
     low,high=t.interval(.95,len(dif)-1,loc=dif.mean(),scale=dif.std(ddof=1)/np.sqrt(len(dif)))
     checks.append(dict(experiment=filename,method=method,baseline=base,regime=reg,n_qubits=nn,delta=float(dif.mean()),ci95=[float(low),float(high)],clusters=len(dif)))
    report={'analytic_correlated_parity_vs_monte_carlo':{'qubits':n,'samples':samples,'masks':len(m),'max_absolute_error':err,'pass':True},'paired_results':checks}
    with open(os.path.join(ensure_output(),'verification.json'),'w') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))

    return report

if __name__ == '__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--samples', type=int, default=240000)
    run(ap.parse_args().samples)
