# Earthquake ground-motion data assimilation

Code and processed inputs for five study events.

## Study events

The selection follows the source notebook's explicit manuscript column order.
Times and magnitudes retain the source event index values.

| ID | Source event time | Label |
| --- | --- | --- |
| 1 | 20240402T2358 | Hualien |
| 2 | 20160205T1957 | Meinong |
| 3 | 20220918T0644 | Chihshang |
| 4 | 20260728T1627 | 2026 Kumamoto |
| 5 | 20160416T0125 | 2016 Kumamoto |

## Files

- `seismic_data_preprocess/Run_KalmanFilter.ipynb`: one event-selectable core workflow: GMPE prediction, observed final PGA distribution, least-square initialization, and KF updates.
- `RuptureParaTime_new/`: processed station-by-time PGA tables for the five events.
- `earthquake_index.csv`: event metadata with IDs 1–5.
- `GMPE code/`: Chao and Zhao implementations and Chao coefficients.
- `vdist.py`: distance and azimuth helper.
- `PGA_colorbar.cpt`, `intensity_colorbar.cpt`, `TW_city_boundary.txt`: original map palettes and Taiwan boundary input.
- `environment.yml`: dependencies; `FILE_MANIFEST.csv`: provenance and checksums.

## Run

```bash
conda env create -f environment.yml
conda activate eew-data-assimilation
jupyter lab
```

Open `seismic_data_preprocess/Run_KalmanFilter.ipynb`, set `EVENT_ID` (1–5),
and run all cells. The notebook works from the release root or notebook directory.
Only one event is processed per run. `HISTORY_END_SECOND` controls the end of the full time series. `EVALUATION_SECOND` sets each event's manuscript evaluation time (Hualien 35 s; Meinong 25 s; Chihshang 25 s; 2026 Kumamoto 27 s; 2016 Kumamoto 25 s).
`coefficient_history` and `pga_history` contain the per-update outputs in memory.
GMPE prediction and final PGA distribution use the original PyGMT map style
(surface interpolation, coastlines, PGA colors and station markers). Least-square
and KF comparisons use Matplotlib. GMT, GSHHG coastlines and DCW data are
provided by the environment. No Japan administrative-boundary file or aftershock
catalog is needed. The notebook does not create output directories automatically.

Input headers retain the source layout: time index, Longitude, Latitude,
RealPGA, EndRealPGA, Station and Vs30. PGA is in gal and Vs30 in m/s.

## Reproducibility and attribution

The notebook retains the source five-event calculation settings, including
region-specific zero-PGA handling and covariance updates. KMMH07 is retained; the
2026 Kumamoto dataset contains 249 stations. The initial least-square fit uses GMPE predictions, not final observed
PGA. Auxiliary figure experiments, batch comparisons, magnitude-report schedules
and directional/residual diagnostics have been removed.

All five events were executed through all four notebook steps in the existing
`xdas310` environment. Four figures per event were produced. After restoring PyGMT, the Taiwan and
Japan GMPE/final-PGA maps were rendered and visually checked. Per-update KF
residual medians and standard deviations matched the original core workflows
with maximum absolute difference 3.2e-15. The environment specification has not yet been tested by creating a fresh environment.

Paper title, authors, DOI, code license and third-party data attribution/terms
remain to be supplied by the author. No license is assigned by this preparation copy.
