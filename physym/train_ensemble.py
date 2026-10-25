"""Train ensemble members; the main seed-643 model is trained separately."""
from .paths import CHECKPOINTS_DIR
from .physnet import fit

def run(devices=2000, epochs=35, seeds=(101,203)):
    for seed in seeds:
        print('ensemble seed',seed,flush=True)
        fit(out=str(CHECKPOINTS_DIR / f'physics_gnn_{seed}.pt'),devs=devices,ncal=48,epochs=epochs,seed=seed)

if __name__ == '__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--train-devices',type=int,default=2000)
    ap.add_argument('--epochs',type=int,default=35)
    ap.add_argument('--seeds',nargs='+',type=int,default=[101,203])
    a=ap.parse_args()
    run(devices=a.train_devices,epochs=a.epochs,seeds=a.seeds)
