# Reproducibility, provenance and limitations

## Original material preserved

The initial package shipped with the manuscript is the source of this repository's
pretrained `.pt` models, the `ridge_channel.pkl` baseline, all six vector figures,
and reported metric CSVs. The research algorithms were retained and their
module imports/output locations were reorganized to remove the dependency on
a nearby `../qcai2027_symfit` directory. There is no data download step.

The files under `results/reference/` are **frozen reference records**. Evaluation
scripts generate new results separately under `results/generated/` rather than
changing the previously archived numbers. Plot regeneration intentionally reads
`results/reference/`, so figures do not change as a side effect of a quick run.

## Exact protocol

- 8-qubit main experiment: circuit seeds `11,29,47,83`; independently generated
  device seeds `201,301,401`; regimes `correlated, independent, sparse`.
- Device calibration: **48** error bitstrings per device. These are not counted
  inside the separate **20** diagnosis executions per episode.
- Main held-out evaluation: **100** fault episodes per `(regime,circuit,device)`;
  12 circuit-device clusters per noise family. Reported interim budgets:
  `4,8,12,16,20` shots.
- Larger-register transfer: circuit seeds `59,73,101,131`, device seeds
  `503,607,701`, evaluation qubit sizes `8,10,12`, **120** fault episodes per
  circuit-device cluster. The GNN weights are frozen and were trained only on
  the 8-qubit device population.
- Graph: known 1st/2nd-neighbor readout edges. Physical error factors are
  independent per shot *before* XOR composition; readout bits after composition
  can be correlated.
- Labels: the *equivalence classes of propagated Pauli mutations* produced
  by `run_experiments.fault_library`, including no-fault. The model cannot
  identify insertion locations that produce identical final Paulis.
- All source models use a uniform prior over these classes and independent
  shot-level measurement noise in the assumed simulated channel.

## Data and code map

| File or module | Definition |
|---|---|
| `physym/run_experiments.py` | H, S, CX Clifford propagation, candidate tests, fault-equivalence classes, symplectic syndromes, original static selectors |
| `physym/neurosym.py` | Pauli parity + latent XOR readout simulator; neural residual, empirical, ridge, method-of-moments baselines, sequential MI policy |
| `physym/physnet.py` | Message-passing physics GNN, `q_from_pc` analytic parity channel, supervised synthetic noise parameter training |
| `physym/eval_physics.py` | Matched random-number trajectories for main channel-model comparisons |
| `physym/eval_ensemble.py` | Mean of three independently trained physical channel predictions |
| `physym/eval_transfer.py` | Variable-size graph readout simulation and zero-retraining evaluation |
| `physym/verify_physics.py` | Large-shot Monte Carlo parity validation + cluster-paired Student intervals from frozen CSVs |
| `physym/figures_camera.py` | Vector plots generated from frozen evaluation tables |

## Commands

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python -m physym verify --samples 240000
python -m physym eval-main --episodes 100
python -m physym eval-ensemble --episodes 100
python -m physym eval-adaptive --episodes 34
python -m physym eval-transfer --episodes 120 --sizes 8 10 12
python -m physym figures
```

For quick pipeline checks, use `--episodes 2` or `--samples 12000` but do not
report those as headline results. Run published settings when comparing against
the frozen results. Some output differences can arise across floating-point
libraries or PyTorch versions; the archived CSVs are immutable provenance.

## Train all main physical GNNs

```bash
# Do this in a scratch location to protect the official checkpoints:
export PHYSYM_CHECKPOINT_DIR=/tmp/physym-fresh-weights
python -m physym train --train-devices 2000 --epochs 35 --seed 643
python -m physym train-ensemble --train-devices 2000 --epochs 35
```

The model uses two node calibration statistics and five edge statistics,
48-dimensional node/edge embeddings, three residual message-passing rounds,
and bounded sigmoid heads. The training objective combines held-out support
parity mean-square-error and latent local/edge noise factor errors. Random
synthetic devices generated for train/validation do not share latent draws
with evaluation devices. The training scripts emit epoch diagnostics under
`results/generated/`.

## What the benchmark does not establish

- No fault-location identification beyond equivalent propagated Pauli strings.
- No direct noise model for coherent quantum errors, quantum gate noise,
  asymmetric SPAM, temporal drift or non-Clifford faults.
- No wall-clock quantum processor speedups or deployed QPU runtime results.
- Results are specific to the documented simulator and sampled candidate
  Pauli tests. A separate hardware benchmark is needed for hardware claims.

## File integrity

The archive includes `MANIFEST_SHA256.txt` for all files except the manifest
itself. A program can validate it using `sha256sum -c MANIFEST_SHA256.txt`
from the repository root (Linux/macOS) or Python's `hashlib` (all platforms).
