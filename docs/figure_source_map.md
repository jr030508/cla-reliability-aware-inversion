# Figure → source data → script

Every figure and table in the manuscript maps to exactly one archived CSV and, where applicable,
one script. No number in the paper is produced by hand.

## Main-text figures

| Figure | Content | Source CSV | Script |
|---|---|---|---|
| Fig. 1 | Reliability-aware inversion workflow (estimator → conformal calibration → hybrid gate → accept/flag/reject) | — (schematic) | `analysis/plot_reliability_workflow.py` |
| Fig. 2a | Worst pre-specified subgroup coverage, global vs Mondrian | `data/figure_source/Fig02a_worst_subgroup_coverage_source.csv` | `code/ensemble_uncertainty.py` |
| Fig. 2b | Risk-stratified Mondrian intervals under clean / LOD conditions | `data/figure_source/Fig02b_risk_mondrian_source.csv` | `analysis/run_patch_conformal_ood.py` |
| Fig. 3 | Station-deactivation error scaling (136 → 96 → 64 → 32 → 16) | `data/figure_source/Fig03_station_deactivation_source.csv` | `code/robustness.py` |
| Fig. 4 | Hybrid gate acceptance and rejection rates by condition | `data/figure_source/Fig04_reliability_gate_source.csv` | `analysis/fit_validate_gate.py` |
| Fig. 5 | Inference latency vs batch size | `data/figure_source/Fig05_runtime_source.csv`, `data/audits/runtime_summary.csv` | `code/run_experiments.py` |
| Fig. 6 | Task decomposition / negative transfer | `data/figure_source/Fig06_task_decomposition_source.csv` | `code/aggregate_results.py` |
| Fig. 7 | Structured station outage (radial bands, azimuth sectors) vs matched random | `data/figure_source/Fig_structured_outage_performance_JER_source.csv`, `structured_outage_summary.csv` | `analysis/run_structured_station_outage.py`, `analysis/replot_structured_outage.py` |
| Fig. S1 | Structured outage mask geometry | `data/figure_source/structured_outage_mask_manifest.csv` | `analysis/run_structured_station_outage.py` |

## Audit tables referenced in the text

| Statement | Source CSV | Script |
|---|---|---|
| Truth- vs prediction-grouped subgroup coverage (max deviation 1.25 pp) | `data/audits/predicted_group_coverage_audit.csv` | `analysis/recompute_predicted_group_coverage.py` |
| Per-condition accepted-sample failure rate and Wilson 95 % upper bound | `data/audits/condition_audit.csv` | `analysis/analyze_tadrg_outputs.py` |
| Gate discrimination AUROC / AUPRC by evaluation scope | `data/audits/discrimination_scope_audit.csv` | `analysis/analyze_tadrg_outputs.py` |
| Gate threshold feasibility per output | `data/audits/gate_calibration_audit.csv` | `analysis/analyze_tadrg_outputs.py` |
| Risk-indicator redundancy (ensemble uncertainty vs interval half-width, Pearson = 1.000) | `data/audits/feature_redundancy_audit.csv` | `analysis/analyze_tadrg_outputs.py` |
| CALPUFF configuration audit | `data/audits/calpuff_configuration_audit.csv` | `analysis/audit_receptor_geometry.py` |
| Exact receptor coordinates, distances and bearings | `data/audits/receptor_coordinates_exact.csv` | `analysis/audit_receptor_geometry.py` |
| Receptor-order locality permutation test | `data/audits/station_order_locality_summary.csv`, `station_order_permutation_distribution.csv` | `analysis/audit_receptor_geometry.py` |
| Source–receptor geometry heterogeneity | `data/audits/source_receptor_geometry_audit.csv` | `analysis/audit_source_receptor_geometry.py` |

## Column dictionary for the audit tables

### `predicted_group_coverage_audit.csv`
`parameter`, `group_type`, `group`, `boundary_source`, `boundary_value`,
`n_truth_grouped`, `coverage_truth_grouped`, `n_predicted_grouped`,
`coverage_predicted_grouped`, `delta_percentage_points`.
Group boundaries are taken from the truth distribution and then applied unchanged to the predicted
values, so the two columns differ only in the assignment variable.

### `condition_audit.csv`
`task`, `condition`, `samples`, `accepted`, `acceptance_rate`, `failure_rate_all`,
`failure_rate_accepted`, `wilson_upper_accepted`, `mean_error_all`, `mean_error_accepted`,
`failure_detection_auroc`, `failure_detection_auprc`, `tolerance`,
`gate_feasible_on_calibration`, `conformal_coverage`, `conformal_mean_width`,
`median_predicted_risk`, `accepted_failures`, `passes_10pct_wilson`.

### `discrimination_scope_audit.csv`
`task`, `scope` (`all_conditions` / `non_extreme` / `clean_and_bias`), `samples`,
`failure_prevalence`, `auroc`, `auprc`.

### `gate_calibration_audit.csv`
`task`, `tolerance`, `tolerance_source`, `gate_feasible`, `risk_threshold`,
`calibration_accepted`, `calibration_failures`, `calibration_failure_rate`,
`calibration_wilson_upper`, `best_available_threshold`, `best_available_accepted`,
`best_available_wilson_upper`.

### `source_receptor_geometry_audit.csv`
Metric/value pairs covering Scenario 1 source-to-receptor distances and bearings, the Scenario 2–3
source sampling box, nearest-receptor distances, and the fraction of receptors outside the nominal
plume sector (defined as the bearing span of the receptor arc seen from the Scenario 1 source).
