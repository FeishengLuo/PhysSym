<div align="center">

# ⚛️ PhysSym

### Physics-Consistent Graph Learning and Active Symplectic Testing for Quantum Circuit Fault Diagnosis

**A neural–symbolic AI4QC agent that learns *how* a quantum device misreads — then chooses the next Pauli test by Bayesian information gain.**

[![Python ≥3.10](https://img.shields.io/badge/Python-%E2%89%A53.10-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.2-EE4C2C?logo=pytorch&logoColor=white)](requirements.txt)
[![License: MIT](https://img.shields.io/badge/code%20license-MIT-green)](LICENSE)
[![Paper PDF](https://img.shields.io/badge/paper-PDF%20in%20%2Fpaper-orange)](paper/PhysSym.pdf)
[![Springer CCIS](https://img.shields.io/badge/proceedings-Springer%20CCIS%20(to%20appear)-blue)](paper/)

</div>

---

> **What PhysSym does.** Given a synthetic Clifford circuit, a dictionary of distinct *propagated Pauli fault classes*, and a small batch of **48 device-calibration readouts**, PhysSym learns a physically consistent correlated-readout model and **adaptively selects the most informative Pauli tests** to identify the fault class — within a budget of **20 diagnosis executions**.

> **Scope & honesty.** These are **simulated Clifford/Pauli mutations and classical readout channels**, not real-QPU experiments. Reported accuracies measure *fault-equivalence-class identification* on the specified factorized XOR readout model — not arbitrary faulty-gate location, and not general-purpose quantum error correction. The shipped pretrained checkpoints reproduce the synthetic benchmark of the manuscript.

## ✨ Highlights

| 🏆 | Result |
|:--|:--|
| 🧠 **Physics-structured GNN** | Learns *local flip rates* `p_i` and *pair-burst rates* `c_ij` — not independent errors per observable — and emits **coherent closed-form probabilities for arbitrary Pauli supports** |
| 🎯 **3-model ensemble** | **45.3% / 69.4% / 93.7%** fault-class accuracy on correlated / sparse-burst / independent 8-qubit readout — **leading all non-oracle baselines** in both correlated regimes |
| 📈 **Zero-retraining transfer** | Frozen 8-qubit GNN generalizes to **10- and 12-qubit** devices: **+5.97 pp** over empirical parity calibration on held-out 12-qubit correlated devices |
| 🧪 **Sample-efficient** | Only **48 calibration shots + 20 diagnosis executions** per episode |
| 📐 **Theory-backed** | Universal `2n`-test noiseless identifiability theorem · exact correlated-channel mapping · information-gain regret bound under estimation error |
| 🔁 **Fully reproducible** | One-command demo, pretrained checkpoints, frozen reference records, vector figures, GitHub Actions smoke tests |

## 🔬 Method at a glance

```text
 calibration readouts                candidate Pauli tests
 (48 shots / device)                       │
        │                                  ▼
        ▼                    ┌────────────────────────────┐
  ┌───────────┐  p_i, c_ij   │  Bayesian active selector  │◄──── expected
  │ Physics   │─────────────►│  (mutual information gain) │      mutual
  │ GNN       │              └────────────┬───────────────┘      information
  │ (3-model  │                           │ chosen test m_t
  │ ensemble) │                           ▼
  └───────────┘              ┌────────────────────────────┐
        │                    │  Exact symplectic oracle    │ syndrome
        ▼                    │  (Clifford propagation of   │
  ┌───────────┐  q_t(m)      │   Pauli fault classes)      │
  │ Analytic  │─────────────►└────────────┬───────────────┘
  │ parity    │                           ▼
  │ channel   │              posterior update ──► fault-class verdict
  └───────────┘
```

For a binary test support $m_t\in\{0,1\}^n$, the channel is **exact and closed-form**:

$$
q_t = \frac{1}{2}\left[1 - \prod_i (1-2p_i)^{m_{ti}}
  \prod_{(i,j)\in E}(1-2c_{ij})^{m_{ti}\oplus m_{tj}}\right].
$$

Independent local flips and independent pair-burst sources, composed by XOR on each execution — a strict extension of the independent-noise SIFT model (set $c_{ij}=0$ to recover it). It is **not** a universal description of quantum-device noise.

## 📊 Results shipped with this repository

Episodes use **20 diagnosis executions** and **48 separate calibration bitstrings per device**. Values are the frozen records of the manuscript benchmark (regenerate the full tables with the commands below).

**8-qubit fault-class identification:**

| Noise family | Physical GNN (single) | **PhysSym ensemble (3 GNNs)** | Unconstrained NN | Oracle channel |
|:--|--:|--:|--:|--:|
| Dense correlated | 43.42% | **45.33%** 🥇 | 44.00% | 50.67% |
| Sparse bursts | 68.58% | **69.42%** 🥇 | 62.08% | 72.08% |
| Independent | **93.92%** 🥇 | 93.67% | 93.50% | 94.17% |

**Zero-retraining transfer (frozen single model):**

| Held-out setting | Physical GNN | Empirical parity | Perfect calibration |
|:--|--:|--:|--:|
| 12 qubits, correlated | **27.85%** | 21.88% | 32.92% |
| 12 qubits, sparse bursts | **53.96%** | 52.85% | 61.04% |

> **Reading the table.** The ensemble is not uniformly best in every setting, and the oracle has privileged true-channel parameters. Results support improvements on this controlled benchmark only — no real-hardware or universal SOTA claim. Cluster-paired intervals ship with the reference records.

## 🚀 Quick start

Python **≥3.10** (developed on 3.11–3.13, CPU PyTorch):

```bash
git clone https://github.com/FeishengLuo/PhysSym.git
cd PhysSym
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
python -m pip install -U pip && python -m pip install -e '.[test]'

python -m physym demo        # fast end-to-end inference, bundled GNN checkpoint
python -m pytest -q          # deterministic physics + model checks
python -m physym protocol    # exact reference experiment settings
```

`demo` runs a small new synthetic episode to verify the inference path — it **does not** reproduce the paper's accuracy table and needs no datasets, API keys, or quantum backend.

### 🔁 Reproduce the reported evaluations

```bash
python -m physym verify        # analytic parity law + paired intervals (240k MC shots)
python -m physym eval-main --episodes 100            # 8-qubit clusters + calibration baselines
python -m physym eval-ensemble --episodes 100        # 3-member ensemble, same 48+20 protocol
python -m physym eval-adaptive --episodes 34         # sequential MI controller
python -m physym eval-transfer --episodes 120 --sizes 8 10 12   # zero-retraining transfer
python -m physym figures                             # regenerate all six paper figures
```

New tabular outputs go to `results/generated/` (gitignored); archived reference records in `results/reference/` stay untouched. Redirect with `PHYSYM_OUTPUT_DIR=/path/to/dir` (`$env:PHYSYM_OUTPUT_DIR=...` on PowerShell). Reference schemas include `regime`, `circuit`, `device`, `method`, `budget`, `acc`. **Never compare a low-episode smoke run with the reported table.**

### 🏋️ Retrain from scratch

```bash
python -m physym train --train-devices 2000 --epochs 35 --seed 643            # main model
python -m physym train-ensemble --train-devices 2000 --epochs 35              # seeds 101 & 203
python -m physym train-baselines --train-devices 580                          # NN + ridge baselines
```

Set `PHYSYM_CHECKPOINT_DIR=/path/to/new_checkpoints` first to protect the distributed checkpoints — training overwrites `checkpoints/` as written. Training data is generated in memory with deterministic NumPy seeds; no private or proprietary datasets are used. CPU-only by default; training is substantially slower than the demo. See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## 🧪 Tests & reproducibility

```bash
python -m pytest -q                       # symplectic commutation, exact parity law, MC agreement,
                                          # GNN inference at larger graph size, sequential controller
python -m physym verify --samples 240000  # full statistical verification
python scripts/verify_manifest.py         # package integrity
```

GitHub Actions re-runs the small checks on every push and PR. On the current numerical stack, full reruns land near the archived averages but are **not perfectly cell-by-cell identical**; details and the uncompleted full-size transfer rerun are documented in [docs/VALIDATION.md](docs/VALIDATION.md).

## 📁 Repository layout

```text
PhysSym/
├── README.md                  # this file
├── pyproject.toml             # pip-installable package (physym-research)
├── requirements.txt
├── LICENSE                    # original Python code: MIT
├── THIRD_PARTY.md             # Springer template + model pickle notices
├── physym/                    # importable core algorithms + CLI
│   ├── __main__.py            # python -m physym {demo,verify,train,eval-*,figures,protocol}
│   ├── run_experiments.py     # Clifford/Pauli oracle; static comparators
│   ├── neurosym.py            # noise simulator, baselines, MI controller
│   ├── physnet.py             # PhysicsGNN, analytic channel, training
│   ├── eval_physics.py        # paired main evaluation
│   ├── eval_ensemble.py       # 3-model evaluation
│   ├── eval_transfer.py       # cross-size frozen-model evaluation
│   ├── verify_physics.py      # analytic law + CI consistency
│   ├── train_ensemble.py
│   ├── figures_camera.py      # regenerates the vector research figures
│   └── paths.py               # checkpoint / reference / output directories
├── checkpoints/               # pretrained physics GNNs + neural channel + ridge baseline
├── paper/
│   └── PhysSym.pdf            # 📖 the paper, open-sourced here
├── docs/
│   ├── REPRODUCIBILITY.md     # command provenance and caveats
│   └── VALIDATION.md          # packaging validation status
├── tests/                     # quick physics / model / API tests
└── .github/workflows/         # CI smoke tests
```

## 📖 Citation

The paper is **open-sourced in this repository**: [`paper/PhysSym.pdf`](paper/PhysSym.pdf) (camera-ready Springer CCIS layout). If PhysSym helps your research, please cite:

```bibtex
@inproceedings{chen2027physym,
  title     = {PhysSym: Physics-Consistent Graph Learning and Active
               Symplectic Testing for Quantum Circuit Fault Diagnosis},
  author    = {Chen, Luodi and Yang, Ziyu},
  booktitle = {TBD},
  publisher = {TBD},
  series    = {TBD},
  year      = {2027},
  note      = {To appear. Volume/DOI metadata will be updated upon publication.
               Open-source PDF: \url{https://github.com/FeishengLuo/PhysSym/blob/main/paper/PhysSym.pdf}},
}
```

> Venue, volume, and DOI details are pending official publication and will be filled in — please do not cite placeholder metadata as final.

## ⚠️ Responsible research use

- This repository is a **research implementation for simulated Clifford/Pauli benchmarks**; it is not certified diagnostic software for real quantum hardware.
- The MIT license covers the original research Python software; Springer styles and final paper publication rights are treated separately in [THIRD_PARTY.md](THIRD_PARTY.md).
- All training data is synthetic and deterministically seeded — no private clinical data or proprietary quantum telemetry is used.
