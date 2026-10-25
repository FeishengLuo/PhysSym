"""Public command-line entry point: python -m physym ..."""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(
        prog="python -m physym",
        description="PhysSym: reproduce physics-consistent quantum diagnosis experiments",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("demo", help="Run a tiny, deterministic active-diagnosis example")
    p.add_argument("--episodes", type=int, default=8)
    p = sub.add_parser("verify", help="Verify correlated parity formula and reference CIs")
    p.add_argument("--samples", type=int, default=240000)
    p = sub.add_parser("train", help="Train the main physical graph learner")
    p.add_argument("--train-devices", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--seed", type=int, default=643)
    p = sub.add_parser("train-ensemble", help="Train seeds 101 and 203")
    p.add_argument("--train-devices", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=35)
    p = sub.add_parser("eval-main", help="Evaluate calibrated baselines on 8-qubit test devices")
    p.add_argument("--episodes", type=int, default=100)
    p = sub.add_parser("eval-ensemble", help="Evaluate the three-model physics ensemble")
    p.add_argument("--episodes", type=int, default=100)
    p = sub.add_parser("eval-transfer", help="Evaluate frozen 8-qubit weights at larger sizes")
    p.add_argument("--episodes", type=int, default=120)
    p.add_argument("--sizes", type=int, nargs="+", default=[8, 10, 12])
    p = sub.add_parser("train-baselines", help="Train unconstrained neural and ridge channel baselines")
    p.add_argument("--train-devices", type=int, default=580)
    p = sub.add_parser("eval-adaptive", help="Evaluate original static and adaptive controller suite")
    p.add_argument("--episodes", type=int, default=34)
    sub.add_parser("figures", help="Rebuild all 6 vector figures from frozen reference CSV files")
    sub.add_parser("protocol", help="Print original benchmark protocol JSON")
    a = parser.parse_args()
    if a.command == "demo":
        import numpy as np
        import torch
        from .neurosym import N, device, sample_errors, true_q, make_case
        from .physnet import PhysicsGNN, predict
        from .eval_physics import batched_policy
        from .paths import CHECKPOINTS_DIR
        rng = np.random.default_rng(2027)
        _, _, measured, syndrome, _, _, _ = make_case(N, 11, .06)
        mask = (measured[:, :N] | measured[:, N:]).astype("uint8")
        p, c = device(rng, "correlated")
        cal = sample_errors(rng, p, c, 48)
        checkpoint = torch.load(CHECKPOINTS_DIR / "physics_gnn.pt", map_location="cpu", weights_only=True)
        net = PhysicsGNN()
        net.load_state_dict(checkpoint["state"])
        q_pred = predict(net, cal, mask)
        q_true = true_q(p, c, mask)
        labels = rng.integers(syndrome.shape[0], size=a.episodes)
        u = rng.random((a.episodes, 20))
        r = rng.integers(syndrome.shape[1], size=(a.episodes, 20))
        acc, entropy = batched_policy(syndrome, q_true, q_pred, u, labels, r)
        print(json.dumps({
            "mode": "synthetic_clifford_pauli_demo",
            "qubits": N,
            "fault_equivalence_classes": int(syndrome.shape[0]),
            "candidate_tests": int(syndrome.shape[1]),
            "calibration_shots_per_device": 48,
            "diagnosis_shots_per_episode": 20,
            "episodes": a.episodes,
            "top1_accuracy_at_20": round(float(acc[-1]), 5),
            "posterior_entropy_at_20": round(float(entropy[-1]), 5),
            "note": "Toy demonstration only: not a published benchmark estimate",
        }, indent=2))
    elif a.command == "verify":
        from .verify_physics import run
        run(samples=a.samples)
    elif a.command == "train":
        from .physnet import fit
        fit(devs=a.train_devices, epochs=a.epochs, seed=a.seed)
    elif a.command == "train-ensemble":
        from .train_ensemble import run
        run(devices=a.train_devices, epochs=a.epochs)
    elif a.command == "eval-main":
        from .eval_physics import run
        run(episodes=a.episodes)
    elif a.command == "eval-ensemble":
        from .eval_ensemble import run
        run(episodes=a.episodes)
    elif a.command == "eval-transfer":
        from .eval_transfer import run
        run(trials=a.episodes, sizes=a.sizes)
    elif a.command == "train-baselines":
        from .neurosym import train
        train(devices=a.train_devices)
    elif a.command == "eval-adaptive":
        import torch
        from .neurosym import NoiseNet, evaluation
        from .paths import CHECKPOINTS_DIR
        ck = torch.load(CHECKPOINTS_DIR / 'neural_channel.pt', map_location='cpu', weights_only=True)
        net = NoiseNet(ck['dim'])
        net.load_state_dict(ck['state'])
        evaluation(net, episodes=a.episodes)
    elif a.command == "figures":
        from . import figures_camera   # intentionally generates figures on invocation
    elif a.command == "protocol":
        from .paths import REFERENCE_DIR
        print((REFERENCE_DIR / "protocol.json").read_text())


if __name__ == "__main__":
    main()
