# CoRP

## LIBERO Evaluation

### Environment setup

Linux, Python 3.12, Git, and an NVIDIA GPU with EGL support are required.
Run from the repository root:

```bash
bash scripts/install.sh
source .venv/bin/activate
```

The script installs CUDA dependencies and the pinned LIBERO version.
Git LFS is required if the checkpoint has not been downloaded.

### Run

Full 2,000-episode evaluation:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python scripts/evaluate_libero.py --suite all --seed 7 --output-dir outputs/seed7
```

Single-suite evaluation (500 episodes), for example Spatial:

```bash
python scripts/evaluate_libero.py --suite libero_spatial --seed 7 --output-dir outputs/seed7_spatial
```

Choose `--suite` from `libero_spatial`, `libero_object`, `libero_goal`, or
`libero_10` (Long).


## RoboTwin2.0

TODO