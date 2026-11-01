# Reliability-aware source-term inversion under degraded atmospheric radiation monitoring

Reproducibility package for the manuscript *"Reliability-aware source-term inversion for nuclear accidents under degraded atmospheric radiation monitoring"*
(submitted to *Journal of Environmental Radioactivity*).

The study couples a CNN–LSTM–Attention (CLA) estimator with split-conformal calibration and a hybrid
learned + protocol gating stage, and audits its behaviour when the monitoring network degrades
(station loss, temporal truncation, multiplicative bias, limit-of-detection censoring).

This repository contains the **code**, the **audit scripts** and the **source data behind every figure and
every number quoted in the paper**. The large CALPUFF simulated datasets and the trained model checkpoints
are archived separately on Zenodo (see [Data availability](#data-availability)).

---

## Repository map

```
.
├── code/                     Model and experiment code
│   ├── models.py             CLA / CNN-LSTM / LSTM-Attention / 2D-CNN-Attention definitions
│   ├── data_pipeline.py      Dataset construction, splitting, transforms
│   ├── metrics.py            Error, coverage and interval-width metrics
│   ├── run_experiments.py    Capacity, robustness, generalization and ablation runs
│   ├── ensemble_uncertainty.py   Deep-ensemble split-conformal intervals
│   ├── robustness.py         Noise / station-loss / truncation / bias / LOD protocols
│   ├── tune_optuna.py        Historical TPE hyper-parameter search (candidates only)
│   ├── validate_dispersion.py    CALPUFF forward-model sanity checks
│   ├── aggregate_*.py        Result aggregation across seeds
│   └── ...
├── analysis/                 Audit, re-computation and figure-generation scripts
│   ├── audit_receptor_geometry.py          Receptor order locality + INP/CSV mapping
│   ├── audit_source_receptor_geometry.py   Source–receptor geometry heterogeneity
│   ├── recompute_predicted_group_coverage.py  Truth- vs prediction-grouped conformal coverage
│   ├── run_structured_station_outage.py    Radial-band and azimuth-sector structured outages
│   ├── replot_structured_outage.py         Structured-outage figure
│   ├── plot_reliability_workflow.py        Reliability workflow overview figure
│   ├── fit_validate_gate.py / apply_gate.py    Hybrid gate fitting and application
│   ├── run_patch_conformal_ood.py          Conditional (Mondrian) conformal patch experiments
│   └── analyze_tadrg_outputs.py            Gate discrimination and calibration audits
├── data/
│   ├── figure_source/        One CSV per manuscript figure
│   └── audits/               Configuration, geometry, gate and coverage audits
├── figures/                  Rendered audit figures (PDF / PNG / SVG)
│   ├── Fig_gate_audit.*              Reliability-gate audit, panels a-d
│   └── Fig_coverage_observability.*  Truth- vs prediction-grouped coverage
├── docs/
│   └── figure_source_map.md  Figure → source CSV → script correspondence
├── config.json               Experiment configuration
├── frozen_hyperparameters_manuscript.json   Frozen hyper-parameters used for all reported runs
└── requirements.txt
```

---

## Reproducing every reported number

| Result in the manuscript | Source file | Generating script |
|---|---|---|
| Conformal calibration, marginal coverage | `data/figure_source/Fig01_conformal_calibration_source.csv` | `code/ensemble_uncertainty.py` |
| Worst-subgroup coverage (global vs Mondrian) | `data/figure_source/Fig02a_worst_subgroup_coverage_source.csv` | `code/ensemble_uncertainty.py` |
| Risk-stratified Mondrian intervals | `data/figure_source/Fig02b_risk_mondrian_source.csv` | `analysis/run_patch_conformal_ood.py` |
| Station-deactivation scaling | `data/figure_source/Fig03_station_deactivation_source.csv` | `code/robustness.py` |
| Hybrid gate acceptance / rejection | `data/figure_source/Fig04_reliability_gate_source.csv` | `analysis/fit_validate_gate.py` |
| Per-condition residual failure rate + Wilson bound | `data/audits/condition_audit.csv` | `analysis/analyze_tadrg_outputs.py` |
| Gate discrimination (AUROC / AUPRC) by scope | `data/audits/discrimination_scope_audit.csv` | `analysis/analyze_tadrg_outputs.py` |
| Gate threshold feasibility | `data/audits/gate_calibration_audit.csv` | `analysis/analyze_tadrg_outputs.py` |
| Risk-indicator redundancy | `data/audits/feature_redundancy_audit.csv` | `analysis/analyze_tadrg_outputs.py` |
| **Truth- vs prediction-grouped subgroup coverage** | `data/audits/predicted_group_coverage_audit.csv` | `analysis/recompute_predicted_group_coverage.py` |
| Inference latency | `data/audits/runtime_summary.csv` | `code/run_experiments.py` |
| Structured (radial / azimuth) station outage | `data/figure_source/structured_outage_summary.csv`, `Fig_structured_outage_performance_JER_source.csv` | `analysis/run_structured_station_outage.py` |
| Mask sizes (46/45/45 radial; 34 per sector) | `data/figure_source/structured_outage_mask_manifest.csv` | `analysis/run_structured_station_outage.py` |
| Task-dependent negative transfer | `data/figure_source/Fig06_task_decomposition_source.csv` | `code/aggregate_results.py` |
| CALPUFF configuration (MCHEM/MWET/MDRY = 0, Z0IN = 0.25 m, NSECDT = 1200 s) | `data/audits/calpuff_configuration_audit.csv` | `analysis/audit_receptor_geometry.py` |
| Receptor coordinates and order locality | `data/audits/receptor_coordinates_exact.csv`, `station_order_locality_summary.csv`, `station_order_permutation_distribution.csv` | `analysis/audit_receptor_geometry.py` |
| Source–receptor geometry heterogeneity | `data/audits/source_receptor_geometry_audit.csv` | `analysis/audit_source_receptor_geometry.py` |

### Re-running the audit scripts

The three audit scripts added for the revision are pure re-computations: they read already-archived
prediction tables and re-group samples. They do **not** retrain models or re-run CALPUFF.

```bash
python analysis/recompute_predicted_group_coverage.py   # -> data/audits/predicted_group_coverage_audit.csv
python analysis/audit_source_receptor_geometry.py       # -> data/audits/source_receptor_geometry_audit.csv
```

Both scripts resolve paths relative to their own location; adjust the `BASE`/`SRC` constants if you
relocate the prediction tables.

---

## Key audit results

**1. Conditional-coverage subgroups are recoverable from model outputs.**
Holding the numerical group boundaries fixed and assigning the same 4 000 test samples by *predicted*
instead of *true* values reproduces the subgroup coverage to within 1.3 percentage points
(90 % global split-conformal, Scenario 3, CLA):

| Subgroup | Grouped by truth | Grouped by prediction | Difference |
|---|---|---|---|
| Release intensity, upper tercile | 79.40 % | 79.04 % | −0.36 pp |
| Release height 1–40 m | 70.36 % | 71.62 % | +1.25 pp |
| Spatial boundary (X) | 84.87 % | 84.73 % | −0.14 pp |
| Spatial boundary (Y) | 84.87 % | 84.65 % | −0.22 pp |

**2. The learned gate detects station and temporal degradation, not failure in general.**
Failure-detection AUROC is 0.949 / 0.940 / 0.869 (release rate / horizontal location / release height)
over the union of all conditions, and falls to 0.747 / 0.749 / 0.834 once extreme conditions are excluded.
Horizontal location admits no threshold satisfying the 10 % tolerance on the calibration set
(best available Wilson upper bound 10.53 %), so that output is handled by the protocol gate only.

**3. Source–receptor geometry is heterogeneous across scenarios.**
Scenario 1 has a fixed source with receptors at 1.77–6.20 km (median 3.66 km) spanning bearings
38.2°–152.9°. Scenarios 2–3 resample the source inside a 5 km × 5 km box
(UTM X 340.08–345.08 km, Y 6177.11–6182.11 km); the nearest-receptor distance varies from
0.004 km to 4.07 km (median 1.42 km).

The two audit figures above are rendered in `figures/` (`Fig_gate_audit` and
`Fig_coverage_observability`, PDF / PNG / SVG at 300 dpi) and can be regenerated with
`python analysis/plot_gate_and_coverage_audit.py`.

---

## Data availability

| Item | Location |
|---|---|
| Code, figure source data, audits, receptor coordinates, CALPUFF input-parameter audit | this repository |
| ~900 MB CALPUFF simulated datasets (`data s1/s2/s3.csv`) | Zenodo |
| Six trained model checkpoints (CLA, CNN-LSTM, LSTM-Attention, 2D-CNN-Attention; Scenario 3) | Zenodo |

Zenodo DOI: *to be inserted at proof stage.*

---

## Environment

```
python >= 3.9
numpy>=1.24,<3   pandas>=2.0,<3   scipy>=1.10,<2   tensorflow>=2.15,<2.19
```

Install with `pip install -r requirements.txt`. Figure scripts additionally use
`matplotlib` and `seaborn`.

---

## Hyper-parameters

All reported results come from a **frozen** hyper-parameter set
(`frozen_hyperparameters_manuscript.json`), revalidated from scratch with identical data splits,
epoch budget, early-stopping rule and three training seeds. `tune_optuna.py` is retained only to
document how the candidate configurations were generated; its search-time validation loss is not
used as evidence anywhere in the paper.

---

## License

Code: MIT (see `LICENSE`).
Data and figure source tables: CC BY 4.0 (see `DATA_LICENSE.md`).

## Citation

*To be completed once the manuscript reference is available.*
