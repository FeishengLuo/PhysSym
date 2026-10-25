# Third-party components and publication rights

- `paper/llncs.cls` and `paper/splncs04.bst` originate from the Springer
  LNCS/CCIS LaTeX proceedings package (TeX Live distribution). They are **not**
  relicensed by this repository's MIT software license. Refer to the upstream
  Springer source package and its licence for permitted distribution and use.
- Python dependencies (PyTorch, NumPy, pandas, SciPy, scikit-learn, Matplotlib)
  retain their respective upstream licences.
- The MIT licence in the root is offered for the original Python software,
  not as a grant to republish the final scholarly article under MIT. Authors
  should check Springer licence-to-publish requirements before public release
  of any accepted version of the manuscript.
- `checkpoints/ridge_channel.pkl` is a Python pickle: load only after checking
  the provenance of this repository. Loading an arbitrary pickle can execute code.
