# Ground-Motion Assimilation Enables Adaptive Shaking Prediction for Earthquake Early Warning

Yi-Sheng Huang and Hsin-Hua Huang
Institute of Earth Sciences, Academia Sinica, Taipei, Taiwan

Paper DOI: `10.xxxx/xxxxx` (to be added on publication)

This repository contains the code and processed inputs used in the paper. A
Kalman filter (KF) updates the coefficients of a ground-motion model in real
time as peak ground acceleration (PGA) observations arrive, and predicts PGA at
all stations after each update.

## Study events

| ID | Event | Event time (`earthquake_index.csv`) | Mw | Stations | GMPE |
| --- | --- | --- | --- | --- | --- |
| 1 | 2024 Hualien | 20240402T2358 | 7.3 | 439 | Chao20 |
| 2 | 2016 Meinong | 20160205T1957 | 6.39 | 496 | Chao20 |
| 3 | 2022 Chihshang | 20220918T0644 | 7.0 | 344 | Chao20 |
| 4 | 2026 Kumamoto | 20260728T1627 | 6.8 | 249 | Zhao06 |
| 5 | 2016 Kumamoto | 20160416T0125 | 7.0 | 266 | Zhao06 |

Hypocenters and magnitudes are listed in `earthquake_index.csv`. Station
KMMH07 is included in the 2026 Kumamoto dataset.

## Files

- `seismic_data_preprocess/Run_KalmanFilter.ipynb`: main workflow for one
  selected event: GMPE prediction, observed final PGA distribution,
  least-square initialization and KF updates. Stored outputs are for event 1.
- `analysis/point_line_benchmark.py`: point- and line-source Chao20
  benchmarks and 25-gal accuracy, missed- and false-alert rates for events 1–3.
- `analysis/lead_time.py`: warning lead times for the 2024 Hualien earthquake.
- `RuptureParaTime_new/`: processed station-by-time PGA tables for the five events.
- `earthquake_index.csv`: event IDs, event times, hypocenters and Mw.
- `GMPE code/`: Chao et al. (2020) implementation and coefficients
  (`GMPE_Chao19.py`, `GMPE_Chao20_model_coefficients.csv`) and a wrapper for
  the OpenQuake Zhao et al. (2006) model (`GMPE_Zhao06.py`).
- `vdist.py`: geodesic distance and azimuth on the WGS84 ellipsoid.
- `PGA_colorbar.cpt`, `intensity_colorbar.cpt`, `TW_city_boundary.txt`: map
  palettes and Taiwan administrative boundaries.
- `environment.yml`: software environment.
- `FILE_MANIFEST.csv`: file sizes and SHA-256 checksums.
- `LICENSE`: MIT license.

## Installation

```bash
conda env create -f environment.yml
conda activate eew-data-assimilation
```

## Usage

### Main workflow

```bash
jupyter lab
```

Open `seismic_data_preprocess/Run_KalmanFilter.ipynb`, set `EVENT_ID` (1–5)
and run all cells. The notebook can be started from the repository root or
from its own directory, and processes one event per run.

- `EVALUATION_SECOND` sets the evaluation time of each event (Hualien 35 s,
  Meinong 25 s, Chihshang 25 s, 2026 Kumamoto 27 s, 2016 Kumamoto 25 s).
- `HISTORY_END_SECOND` sets the end of the KF time history (40 s).
- The notebook prints the median (μ) and standard deviation (σ) of
  ln(predicted PGA) − ln(observed final PGA) for the initial point-source
  GMPE and at the evaluation time, and the azimuth of the directional term.
- `coefficient_history` and `pga_history` hold the per-update coefficients
  and predicted PGA in memory.

The filter is initialized by a least-square fit to the **GMPE predictions,
not to the observed PGA**; the two directional coefficients start at zero and
are learned from the data. Chao20 is evaluated with crustal strike-slip flags,
hypocentral depth as Ztor and hypocentral distance as Rrup (point-source
approximation), and converted from g to gal with `980 × 1.22`; Zhao06 returns
gal and is scaled by `1.2`. Taiwan events use the covariance update of
manuscript Eq. 9, `(I − KH)P`, and Japan events use the equivalent Joseph form.

### Additional analyses

```bash
python analysis/point_line_benchmark.py [output_dir]
python analysis/lead_time.py [output_dir]
```

Both scripts write figures and CSV tables to `output_dir` (default
`analysis/output/`) and print the summary values.

- Benchmarks use the AutoBATS Mw (Hualien 7.3, Meinong 6.4, Chihshang 7.0)
  and the final CWA hypocenter. The line source starts at the epicenter and
  runs along strike 21°, 299° and 26°, with Wells & Coppersmith (1994) lengths
  of 82, 14 and 45 km. Rates are percentages of all stations.
- The lead time is the time the observed PGA first reaches 25 gal minus the
  time the KF-predicted PGA first reaches 25 gal.

## Input data format

Each file in `RuptureParaTime_new/` has the columns: time index (seconds after
origin time), `Longitude`, `Latitude`, `RealPGA` (PGA observed up to that
time), `EndRealPGA` (final PGA), `Station` and `Vs30`. PGA is in gal and Vs30
in m/s.

## Data sources

- Taiwan: Taiwan Strong Motion Instrumentation Program (TSMIP) records of the
  Central Weather Administration (CWA), obtained from the Geophysical Data
  Management System (GDMS, https://gdms.cwa.gov.tw).
- Japan: K-NET and KiK-net records of the National Research Institute for
  Earth Science and Disaster Resilience (NIED),
  https://doi.org/10.17598/nied.0004.
- Vs30: National Center for Research on Earthquake Engineering (NCREE) station
  Vs30 for Taiwan, and J-SHIS V4 AVS30 for Japan.

The processed PGA tables in `RuptureParaTime_new/` are derived from these
datasets. Users of the original waveform or site data should obtain them from
the providers and follow their terms of use.

## Citation

If you use this code, please cite:

> Huang, Y.-S. & Huang, H.-H. (2026). Ground-Motion Assimilation Enables
> Adaptive Shaking Prediction for Earthquake Early Warning. *Nature
> Communications*. https://doi.org/10.xxxx/xxxxx

## License

The code is released under the MIT License; see `LICENSE`. The license does not
cover the third-party data listed under Data sources.

## Change log

- 2026-10-30: Focal mechanisms in the final-PGA map are now keyed by the event
  IDs above (event 2 previously failed and event 3 showed another event's
  mechanism; map only, no numerical change). Added the initial point-source
  μ/σ print, the benchmark and lead-time scripts and the license; translated
  the remaining non-English comments in `vdist.py`.
