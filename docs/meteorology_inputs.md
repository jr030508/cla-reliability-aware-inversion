# Meteorological and terrain inputs (GBHG / Copenhagen 1978-10-19)

## The key fact: one meteorological field for all three scenarios

All three CALPUFF scenarios share a **single** CALMET output file:
`data/calpuff_inputs/CALMET01_shared.DAT`. It is the file that actually generated the three
`data s1/s2/s3.csv` datasets.

| Property | Value |
|---|---|
| File | `CALMET01_shared.DAT` (2.4 MB) |
| Domain | 20 × 20 cells, 0.5 km spacing |
| Origin | X = 341.580 km, Y = 6174.610 km |
| UTM zone | 33 |
| Vertical layers | NZ = 3 |
| Period | 1978-10-19, IBHR = 0 → **IEHR = 23** (24 hourly steps) |
| Output step | NSECDT = 1200 s (20-min fields) |
| Version | CALMET 6.334, level 110421 |

Because the CALPUFF simulations reported in the paper run from **12:00 to 16:00**, this field
(0–23 h) covers them completely. An earlier archived copy of the same domain only extended to
14 h and was therefore **not** the one used; it has been replaced by this file.

The three CALPUFF control files then sample that field on different receptor grids:

| Scenario | CALPUFF grid | Origin (km) | Receptors |
|---|---|---|---|
| 1 | 20 × 20 | 341.58 / 6174.61 | 136 |
| 2 | 25 × 25 | 339.08 / 6174.61 | 136 |
| 3 | 25 × 25 | 339.08 / 6174.61 | 136 |

The 25 × 25 grids of Scenarios 2–3 extend further west (and slightly further north) than the
meteorological domain, so CALPUFF interpolates within the shared field; this is the configuration
used for the reported runs and is reproduced faithfully by the archived `.INP` files.

## File inventory

| File | Role |
|---|---|
| `CALMET_shared.INP` | CALMET control file, extracted verbatim from the "embedded control file" block of `CALMET01_shared.DAT` (1 113 records). It is the control file that produced the archived field. |
| `CALMET01_shared.DAT` | the CALMET wind/turbulence field itself (binary) |
| `CALMET_shared.LST` | CALMET listing; independent record of every parameter used |
| `surf.dat`, `up2.dat` | surface and upper-air station data (1978-10-19) |
| `geo.dat`, `lu.dat` | CTG-processed geophysical (terrain/elevation) and land-use fields |
| `ctgproc.inp` | CTGPROC control file used to build `geo.dat` / `lu.dat` |
| `GBHG_terrain.tif` | terrain raster for the GBHG area (open with Surfer or any GIS) |
| `CALPUFF-s1/s2/s3.INP` | the three CALPUFF control files of the manuscript |

## Reproducing the three datasets

1. Build the geophysical fields (if not using the archived outputs):
   `CTGCOMP` → `CTGPROC ctgproc.inp` → `geo.dat`, `lu.dat`
2. Run CALMET with `CALMET_shared.INP` plus `surf.dat`, `up2.dat`, `geo.dat`, `lu.dat`
   → `CALMET01.DAT` (should match `CALMET01_shared.DAT`)
3. Run CALPUFF three times, each with `CALPUFF-s1.INP` / `-s2.INP` / `-s3.INP` and the shared
   meteorological field, then CALPOST for the receptor time series
   → `data s1.csv`, `data s2.csv`, `data s3.csv`
4. Continue with `code/data_pipeline.py` and the training runs described in the main README.

## Not archived

`euogel20.tif` (169 MB) is the large-scale European terrain/land-cover raster that feeds CTGCOMP.
It is not part of this repository because of its size; it is produced by the standard CTG
terrain-processing chain and is not required once `geo.dat` / `lu.dat` are available.
