# Kernel Conformal Prediction

Reproducibility code for the paper **"[Paper Title]"**.

This repository contains all code to reproduce the synthetic and real-data experiments.

## Repository Structure

```
mvcp/
├── scorers.py              # Nonconformity scorer classes (KernelScorer, MahalanobisScorer, DensityScorer, BonferroniScorer)
├── volume_estimator.py     # Prediction region volume estimation (Monte Carlo & importance sampling)
├── metrics.py              # Evaluation metrics (coverage, worst-slab coverage)
├── pipeline.py             # Data split and model fitting pipeline
├── experiment.py           # Synthetic experiment orchestration (multi-seed runner)
├── table_viz.py            # Styled notebook tables for result inspection
│
├── synthetic_experiments/  # Synthetic data experiments
│   ├── data_generators.py              # Synthetic DGP
│   ├── run_one_seed.py                 # SLURM runner: one seed, fixed lengthscale
│   ├── run_one_seed_alpha_sweep.py     # SLURM runner: alpha sweep
│   ├── run_one_seed_lengthscale.py     # SLURM runner: lengthscale ablation
│   ├── latex_tables.py                 # LaTeX table generation for synthetic results
│   ├── scatter_plotter.py              # Contour/scatter plot utilities
│   ├── synthetic_results.ipynb         # Analysis notebook
│   └── results_*/                      # Saved JSON results per seed
│
├── realdata_experiments/   # Real data experiments (Bio, House, Blog datasets)
│   ├── run_experiment.py               # Core pipeline: one seed × one model
│   ├── run_bio.py / run_bio_3d.py / run_bio_4d.py
│   ├── run_house.py / run_house_3d.py / run_house_4d.py
│   ├── run_blog_2d.py / run_blog_3d.py / run_blog_4d.py
│   ├── make_table.py                   # DataFrame builder and styled tables
│   ├── latex_tables.py                 # LaTeX table generation for real data results
│   ├── real_data_results.ipynb         # Analysis notebook
│   └── experiment_results/             # Saved JSON results per seed
│
└── data/feldman/           # Datasets: bio.csv, blog_data.csv, house.csv
```

## Reproducing Results

### Synthetic Experiments

Run all seeds (designed for SLURM array jobs):
```bash
cd synthetic_experiments
python run_one_seed.py <seed>           # e.g. for seed in {1..50}
python run_one_seed_lengthscale.py <seed>
python run_one_seed_alpha_sweep.py <seed>
```

Analyze results in `synthetic_experiments/synthetic_results.ipynb`.

### Real Data Experiments

Run all seeds and models (designed for SLURM array jobs):
```bash
cd realdata_experiments
python run_bio.py <seed> <model>        # model ∈ {Ridge, MLP, GBR}
python run_bio_3d.py <seed> <model>
python run_bio_4d.py <seed> <model>
# similarly for run_house_*.py and run_blog_*.py
```

Analyze results in `realdata_experiments/real_data_results.ipynb`.

## Core Modules

| Module | Purpose |
|--------|---------|
| `scorers.py` | Nonconformity scores: `KernelScorer` (novel), `MahalanobisScorer`, `DensityScorer`, `BonferroniScorer` |
| `volume_estimator.py` | Volume of prediction regions via bounding-box MC (d≤3) or importance sampling (d≥4) |
| `metrics.py` | `compute_coverage`, `compute_wsc` (worst-slab conditional coverage) |
| `pipeline.py` | `generate_and_fit`: splits data, fits models, computes residuals |
| `experiment.py` | `run_single_seed`, `run_experiment`: full synthetic experiment loop |

## Requirements

- Python ≥ 3.9
- numpy, scipy, scikit-learn, pandas, matplotlib
