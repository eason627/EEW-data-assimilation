# Figure 2 animation provenance

The playback executes scientific calculations from
`seismic_data_preprocess/Run_KalmanFilter.ipynb`, event 1 (2024 Hualien).
It executes the existing parameter-loading, Chao20 prediction, least-square
initialization, KF loop and evaluation statements from that notebook using
Python's AST. Plot-only statements are skipped; the KF equations are not
reimplemented. The sole numerical configuration override is
`HISTORY_END_SECOND = 45`, applied in memory. The source notebook is not edited.

The first successful update is 13 s, with six arrived stations. The existing
minimum is five stations; 5–12 s are skipped. The source-to-site azimuth,
epicentral and hypocentral distances, Vs30 terms, wave-front selection,
zero-PGA handling, process and observation covariances, covariance-update
form and coefficients all come directly from the existing workflow.
The 439-station observation order is checked at every second.
All 33 updates from 13 to 45 s must exist; otherwise generation fails rather
than filling a missing update.

## Original Figure 2 plotting cells

The research source inspected on 2026-10-30 was:

`/home/eason/Work/Data-assimilation_project/seismic_data_preprocess/03_Run_KalmanFilter.ipynb`

The companion `figure2_plotting_reference.json` preserves verbatim:

- Cell 8: `plot_pred_vs_obs`, including residual definitions and PGA/distance plots.
- Cell 10: `Plot_shaking_map`, including the original GMT surface interpolation,
  region, CPT, triangular station symbols, 25-gal highlighting, white epicenter,
  black amplification-direction vector and red time/direction annotations.
- Cell 15: the Figure 2e/2f plotting code, unwrapped azimuth display, median μ,
  original colors, reference times (14 and 25 s), and independently computed
  reported-magnitude GMPE benchmarks.

Cell numbers above are zero-based notebook cell indices.
The complete research-notebook SHA-256 is stored in the companion JSON.
The source notebook is not needed when reproducing from this repository.

At execution, two absolute map-input paths are redirected to repository
files, and the map function returns its figure instead of saving into the
research working tree. The original diagnostic function draws its three-panel
figure; only the azimuth and distance panels are cropped for composition.
The original two time-series panels are rendered directly, with additional
red current-time cursors and points. Legends clarify the initial-GMPE/final-
observation residual definition. The composition contains only the plots, without an added header, panel titles
or footer. Time and direction remain in the original map annotations.
Raster resizing adapts font sizes to the combined animation.

The original map `surface` warnings about coincident grid points are retained;
no new station aggregation, interpolation parameters or smoothing is applied.
This is spatial interpolation used by the original plotting function, not
temporal interpolation of KF results.

## Definitions and verification

Gray residual points: ln(initial point-source GMPE PGA) − ln(final observed PGA).
Blue residual points: ln(current KF predicted PGA) − ln(final observed PGA).
μ is the median residual; σ uses NumPy's population standard deviation.
ME = √(μ² + σ²). These statements are executed from the notebook.
Final PGA is `EndRealPGA`; real-time PGA is `RealPGA` at the arrived stations.
The horizontal distance-panel threshold is 25 gal and its red vertical line is
the existing PGA wave-front radius.

The direction is the original atan2 of the learned sine/cosine coefficients.
There is no separate direction-confidence flag in these sources; the playback
does not invent one. Every displayed direction comes from an actual update.

The time curves retain original plotting conventions and are extended to 45 s
with the same calculations. The 2e azimuth curve remains unwrapped and shifted
by whole turns for display, while tick labels and direction annotations use
degrees modulo 360. No interpolated direction or imputed value is used.

An independent run of the research notebook's original numerical cells
4, 5, 13 and 14, extending its exclusive endpoint to 46, was compared against
all 33 release-notebook updates. See
`visualization/animation_diagnostics/Hualien_original_comparison.json`. Maximum absolute
differences were 3.552713678800501e-15 in ln(PGA) and
9.237055564881302e-14 degrees in azimuth. μ and σ also agreed within 1e-10.
Thus no numerical inconsistency was found between the two inspected
Figure 2 program sources, apart from floating-point rounding. This comparison
does not claim to inspect numeric labels in a separate Word or assembled
manuscript figure.

Every animation run checks the stored 35 s release-notebook output at tolerance
1e-8, writes per-frame metrics, and records SHA-256 hashes of scientific and
plotting inputs. H.264 MP4 and GIF are decoded in full and checked for frame
count, resolution and GIF duration. Five PNG stills allow visual inspection.
Neither original input tables nor notebooks are overwritten.

The default playback is 1.5 frames/s (about 22 s for 33 frames).
The final cell of Run_KalmanFilter.ipynb calls the shared snapshot renderer
with the notebook's in-memory results at SNAPSHOT_SECOND, selected in the last cell. The default
Hualien evaluation is 35 s; its time curves use the configured history end
(40 s by default). No saved animation still is used as notebook output.

The map composition adds a vertical PGA colorbar at
`jBR+o1.0c/0.2c+v+w3c/0.3c` and the label PGA (gal) at
(121.99, 22.92), using 9 pt Helvetica. The colorbar follows the supplied original drawing code:
`intensity_colorbar.cpt`, `series=[0, 10]`, and
`color_model="+c0,2.5,8.0,15.0,25.0,40.0,80.0,150.0,250.0,400.0"`.
No equalsize or truncate override is applied.
The animation explicitly selects Hualien in memory regardless of the event
currently selected in the interactive notebook. If the notebook's saved
evaluation belongs to another event, the verified 35 s Hualien metrics
from the existing animation CSV supply the numerical comparison instead.

The map uses the supplied region [119.4, 122.3, 21.7, 25.35].
Time/direction annotations are placed at longitude 119.5 to remain inside
this region; their text, latitude, and fonts are retained.

The right-hand time-series panels omit the fixed 14 s and 25 s reference
lines. The red current-time cursor and the horizontal zero-residual line
remain. The final notebook cell reloads the shared renderer so rerunning
that cell uses the latest plotting changes in an existing kernel.
