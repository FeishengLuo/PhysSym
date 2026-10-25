# Validation report for repository packaging

The checks below were performed while converting the original PhysSym research
bundle into an importable GitHub repository.

## Passed checks

1. `python -m pytest -q`: five tests passed.
2. `python -m physym demo --episodes 8`: ran from repository root with the
   original `checkpoints/physics_gnn.pt` model, without any other project
   checked out or connected quantum backend.
3. `python -m physym verify --samples 240000`: passed the correlated-XOR parity
   Monte Carlo check and recomputed paired confidence intervals from the
   published `results/reference/` records.
4. `python -m physym eval-main --episodes 100` and
   `python -m physym eval-ensemble --episodes 100`: completed under the current
   Python/PyTorch environment. See the note below on exact reproducibility.
5. `python -m physym eval-adaptive --episodes 2`: original 14-policy adaptive
   comparison script completed its smoke run.
6. `python -m physym train --train-devices 9 --epochs 1 --seed 643`, and
   `python -m physym train-baselines --train-devices 8`: small scratch-training
   runs completed. Scratch GNN weights were written outside the repo.
7. `python -m physym eval-transfer --episodes 2 --sizes 8` completed, and a
   reduced 10/12-qubit transfer run using one circuit/device cluster and two
   episodes per regime completed. The **full 10/12-qubit transfer** was not
   completed during packaging because it exceeded the execution time limit;
   original full-scale CSVs remain available under `results/reference/`.
8. `python -m physym figures`: produced all six vector figures from archived
   CSVs. Their publication copies are included in `paper/figures/`.
9. Springer `pdflatex -halt-on-error main.tex`, using the supplied original
   bibliography `.bbl`, compiled the camera-ready-style source into a **15-page**
   PDF with exactly two placeholder authors and the first marked corresponding.
   The test PDF is deliberately not committed; the repository user generates
   their own final `paper/PhysSym.pdf` after editing the author metadata.

## Important distinction: faithful protocol vs byte-for-byte identical results

The reorganized repository preserves the benchmark seeds, algorithms and
pretrained artifacts, but the **full main and ensemble reruns did not match
all saved per-cluster CSV cells exactly** on the current numerical stack.
Some accuracies differed by up to **0.04 (four percentage points) for an
individual 100-episode circuit/device cell**. Mean headline results were
close (e.g. dense-correlated physical ensemble `0.4533` archived vs `0.4550`
rerun; dense-correlated single GNN `0.4342` archived vs `0.4375` rerun).
Calibration MSEs matched up to tiny floating-point differences.

The cause of all discrepancies was **not conclusively isolated** during
packaging. This is not an exact byte-reproducibility claim. The immutable
archived result files remain the values referred to by the original paper;
new runs use the isolated `results/generated/` directory. For a camera-ready
artifact, disclose this if readers expect bitwise reproducibility, and capture
Python/PyTorch/NumPy versions when publishing new runs.

## Publication readiness

The LaTeX source is formatted for Springer CCIS/LNCS camera-ready production,
but named-person metadata remains deliberately unfilled. The manuscript is
**not** claimed to be accepted, published or tested on physical hardware.
Confirm the eventual workshop-specific author, permissions, page-limit and
Springer licence instructions before submitting a final file.
