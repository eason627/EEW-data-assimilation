"""Point- and line-source GMPE benchmarks against the Kalman-filter prediction.

For the three Taiwan events (IDs 1-3) this script evaluates Chao20 (Chao et al.,
2020; function Chao19H) at the AutoBATS Mw with the final CWA hypocenter, as
a point source and as a line source, and compares them with the KF prediction
at the event's evaluation time. 25-gal alert outcomes are given as percentages
of all stations with a positive final PGA.

Point source: Rrup = hypocentral distance, Ztor = hypocentral depth.
Line source: a horizontal trace starting at the epicenter along the given
strike, with Wells & Coppersmith (1994) length round(10**((Mw - 5.08) / 1.16))
km and width round(10**((Mw - 4.06) / 2.25)) km. The trace is placed at
Ztor = max(hypocentral depth - width / 2, 0) and Rrup is the shortest 3-D
distance to it, sqrt(Repi_line**2 + Ztor**2), using 1 km trace steps.
Both are converted from g to gal with 980 x 1.22.

The KF block reproduces the Taiwan branch of
seismic_data_preprocess/Run_KalmanFilter.ipynb.

Usage: python analysis/point_line_benchmark.py [output_dir]
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyproj import Geod

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "GMPE code"))
sys.path.insert(0, str(ROOT))
from GMPE_Chao19 import Chao19H
from vdist import vdist

OUTPUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "analysis" / "output"
INIT_SEC = 5  # time index whose station list is used

# AutoBATS Mw and line-source strike for the benchmarks
BENCHMARKS = {
    1: dict(name="Hualien", magnitude=7.3, strike=21),
    2: dict(name="Meinong", magnitude=6.4, strike=299),
    3: dict(name="Chihshang", magnitude=7.0, strike=26),
}

# KF settings, identical to Run_KalmanFilter.ipynb
EVALUATION_SECOND = {1: 35, 2: 25, 3: 25}
KF_MAGNITUDE = {1: 6.2, 2: 6.1, 3: 6.2}
KF_START = {1: 5, 2: 13, 3: 10}
HISTORY_END_SECOND = 40
OBS_VARIANCE = 1e-4
Q = np.diag([1e-8, 1e-7, 5e-5, 1e-5, 5e-5, 5e-5])
VS_TRUE = 3.0
V_APP = 2.6
MIN_STATIONS = 5


def chao20_gal(magnitude, rrup, ztor, vs30):
    """Chao20 PGA in gal with crustal strike-slip flags (FSS=1)."""
    pga_g, *_ = Chao19H(0, magnitude, rrup, ztor, vs30, -999,
                        0, 1, 0, 0, 0, 0, 0, 0)
    return pga_g.flatten() * 980 * 1.22


def design_matrix(frame, directional=True):
    ln_r = np.log(frame["SRhypo"].to_numpy())
    terms = [ln_r**2, ln_r, np.log(frame["Vs30"].to_numpy() / 760), np.ones(len(frame))]
    if directional:
        theta = np.deg2rad(frame["Azimuth"].to_numpy())
        terms.extend([np.cos(theta), np.sin(theta)])
    return np.column_stack(terms)


def run_kf(event_id, data, stations, depth):
    """Taiwan KF of Run_KalmanFilter.ipynb; returns predicted PGA by second."""
    gmpe_initial = chao20_gal(KF_MAGNITUDE[event_id], stations["SRhypo"].to_numpy(),
                              depth, stations["Vs30"].to_numpy())
    G_fit = design_matrix(stations, directional=False)
    fit_coefficients = np.linalg.lstsq(G_fit, np.log(gmpe_initial), rcond=None)[0]
    state = np.r_[fit_coefficients, 0.0, 0.0]
    covariance = np.zeros((6, 6))
    G_all = design_matrix(stations)
    predictions = {}
    for second in range(KF_START[event_id], HISTORY_END_SECOND + 1):
        if second not in data.index:
            continue
        current = data.loc[[second]]
        wave_front = V_APP * (second - depth / VS_TRUE)
        arrived = current[current["SRepi"] <= wave_front]
        if len(arrived) < MIN_STATIONS:
            continue
        arrived = arrived.sort_values("SRhypo")
        observed = arrived["RealPGA"].to_numpy()
        ln_observed = np.log(np.where(observed == 0, 1e-10, observed))
        H = design_matrix(arrived)
        R_obs = np.eye(len(arrived)) * OBS_VARIANCE
        covariance = covariance + Q
        PHt = covariance @ H.T
        innovation_covariance = H @ covariance @ H.T + R_obs
        gain = np.linalg.solve(innovation_covariance.T, PHt.T).T
        state = state + gain @ (ln_observed - H @ state)
        covariance = (np.eye(6) - gain @ H) @ covariance
        predictions[second] = np.exp(G_all @ state)
    return predictions


def alert_rates(pred, obs, threshold=25):
    """25-gal outcomes as % of all stations."""
    n = pred.size
    counts = dict(
        correct_alert=np.sum((pred >= threshold) & (obs >= threshold)),
        correct_no_alert=np.sum((pred < threshold) & (obs < threshold)),
        false_alert=np.sum((pred >= threshold) & (obs < threshold)),
        missed_alert=np.sum((pred < threshold) & (obs >= threshold)),
    )
    rates = {k: 100 * v / n for k, v in counts.items()}
    rates["accuracy"] = rates["correct_alert"] + rates["correct_no_alert"]
    return rates


def plot_pred_vs_obs(pred_ln, obs_pga, azimuth, epi_dist, title, save_path):
    """Prediction vs observation, residual vs azimuth, PGA vs distance."""
    log_ratio = pred_ln - np.log(obs_pga)
    median_val = np.median(log_ratio)
    std = np.std(log_ratio)
    me = np.sqrt(median_val**2 + std**2)
    pred = np.exp(pred_ln)
    rates = alert_rates(pred, obs_pga)

    fig, ax = plt.subplots(3, 1, figsize=(9, 22))
    for a in ax:
        a.tick_params(axis="both", which="major", direction="in", length=10, width=1.5, labelsize=20)
        a.tick_params(axis="both", which="minor", direction="in", length=5, width=1.0)
        a.minorticks_on()
        a.grid(which="both", alpha=0.5)
        for side in ["top", "right", "bottom", "left"]:
            a.spines[side].set_linewidth(1.5)
    ax[0].set_title(title, fontsize=26)

    max_ln = 7
    ax[0].axvline(np.log(25), color="r", lw=2, linestyle="--", alpha=0.6)
    ax[0].axhline(np.log(25), color="r", lw=2, linestyle="--", alpha=0.6)
    ax[0].plot([0, 1000], [0, 1000], "k-", lw=2, alpha=0.5)
    ax[0].plot(np.log(obs_pga), pred_ln, "bo", markersize=12, mec="k", alpha=0.8)
    ax[0].set_xlim([0.5, max_ln]); ax[0].set_ylim([0.5, max_ln])
    ax[0].set_aspect("equal", adjustable="box")
    ax[0].set_xlabel(r"ln($PGA_{obs}$)", fontsize=35, labelpad=10)
    ax[0].set_ylabel(r"ln($PGA_{pre}$)", fontsize=35, labelpad=10)
    ax[0].text(5.2, 2.9, f"μ = {median_val:.2f}", fontsize=22, color="k")
    ax[0].text(5.2, 2.5, f"σ = {std:.2f}", fontsize=22, color="k")
    ax[0].text(5.2, 2.1, f"ME = {me:.2f}", fontsize=22, color="k")
    labels = ["Correct alert", "Correct no alert", "False alert", "Missed alert"]
    keys = ["correct_alert", "correct_no_alert", "false_alert", "missed_alert"]
    colors = ["green", "green", "red", "red"]
    positions = [(5.1, 6.8), (1.9, 1.4), (1.9, 6.8), (5.1, 1.4)]
    for (x, y), label, key, color in zip(positions, labels, keys, colors):
        ax[0].text(x, y, label, fontsize=22, ha="center", va="center", color=color)
        ax[0].text(x, y - 0.5, f"({rates[key]:.1f} %)", fontsize=22, ha="center", va="center", color="k")

    ax[1].plot(azimuth, log_ratio, "bo", markersize=14, mec="k", alpha=0.8)
    ax[1].hlines(median_val, xmin=-5, xmax=700, colors="red", lw=2)
    ax[1].set_xlim([-5, 365]); ax[1].set_ylim([-3.5, 3.5])
    ax[1].set_xlabel("source-to-site azimuth (°)", fontsize=30, labelpad=10)
    ax[1].set_ylabel(r"ln($PGA_{pre}$)$-$ln($PGA_{obs}$)", fontsize=35, labelpad=10)

    ax[2].scatter(epi_dist, obs_pga, s=150, facecolor="lightgray", edgecolor="k",
                  linewidth=0.8, alpha=0.85, label="Observed final PGA", zorder=3)
    ax[2].scatter(epi_dist, pred, s=150, facecolor="blue", edgecolor="k",
                  linewidth=0.8, alpha=0.85, label="Predicted PGA", zorder=4)
    ax[2].axhline(25, color="deeppink", lw=1.5, linestyle="--", alpha=0.7)
    ax[2].set_xscale("log"); ax[2].set_yscale("log")
    ax[2].set_xlim([1, 300]); ax[2].set_ylim([1, 1200])
    ax[2].set_xlabel("Epicentral Distance (km)", fontsize=30, labelpad=10)
    ax[2].set_ylabel("PGA (gal)", fontsize=35, labelpad=10)
    ax[2].legend(fontsize=16, loc="lower left", framealpha=0.6)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(ROOT / "earthquake_index.csv", index_col="num")
    geod = Geod(ellps="WGS84")
    rows = []
    for event_id, bench in BENCHMARKS.items():
        event = events.loc[event_id]
        event_key = event["time"].replace("T", "")
        data = pd.read_csv(
            ROOT / "RuptureParaTime_new" / f"{event_key}_RuptureParaTimeByStation_new.csv",
            index_col=0,
        )
        depth = float(event["hypodepth"])
        hypo_lat, hypo_lon = float(event["hypolat"]), float(event["hypolon"])
        distance_m, azimuth, _ = vdist(
            np.full(len(data), hypo_lat), np.full(len(data), hypo_lon),
            data["Latitude"].to_numpy(), data["Longitude"].to_numpy(),
        )
        data["SRepi"] = distance_m / 1000
        data["SRhypo"] = np.sqrt(data["SRepi"] ** 2 + depth ** 2)
        data["Azimuth"] = azimuth
        stations = data.loc[[INIT_SEC]].copy()
        lat = stations["Latitude"].to_numpy()
        lon = stations["Longitude"].to_numpy()
        vs30 = stations["Vs30"].to_numpy()

        # Line-source geometry
        magnitude = bench["magnitude"]
        fault_width = round(10 ** ((magnitude - 4.06) / 2.25))
        fault_length = round(10 ** ((magnitude - 5.08) / 1.16))
        along_m = np.arange(0.0, fault_length + 1e-9, 1.0) * 1000.0
        trace_lon, trace_lat, _ = geod.fwd(
            np.full_like(along_m, hypo_lon), np.full_like(along_m, hypo_lat),
            np.full_like(along_m, bench["strike"]), along_m,
        )
        repi_line = np.min(np.column_stack([
            vdist(np.full_like(lat, trace_lat[i]), np.full_like(lon, trace_lon[i]), lat, lon)[0] / 1000.0
            for i in range(len(trace_lat))
        ]), axis=1)
        ztor_line = max(depth - fault_width / 2, 0)
        rrup_line = np.sqrt(repi_line**2 + ztor_line**2)

        predictions = {
            "point": chao20_gal(magnitude, stations["SRhypo"].to_numpy(), depth, vs30),
            "line": chao20_gal(magnitude, rrup_line, ztor_line, vs30),
        }
        evaluation_second = EVALUATION_SECOND[event_id]
        kf = run_kf(event_id, data, stations, depth)
        kf_second = max(t for t in kf if t <= evaluation_second)
        predictions["KF"] = kf[kf_second]

        obs = stations["EndRealPGA"].to_numpy()
        valid = obs > 0
        for method, pred in predictions.items():
            rates = alert_rates(pred[valid], obs[valid])
            residual = np.log(pred[valid]) - np.log(obs[valid])
            rows.append(dict(
                event_id=event_id, event=bench["name"], method=method,
                magnitude=KF_MAGNITUDE[event_id] if method == "KF" else magnitude,
                second=kf_second if method == "KF" else np.nan,
                n_stations=int(valid.sum()), mu=np.median(residual), sigma=np.std(residual),
                **rates,
            ))
            plot_pred_vs_obs(
                np.log(pred[valid]), obs[valid], stations["Azimuth"].to_numpy()[valid],
                stations["SRepi"].to_numpy()[valid], f"{bench['name']} {method}",
                OUTPUT / f"benchmark_{event_id}_{bench['name']}_{method}.png",
            )
        print(f"{bench['name']}: line length {fault_length} km, width {fault_width} km, "
              f"strike {bench['strike']}°, Ztor {ztor_line:.2f} km")

    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT / "benchmark_rates.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 200)
    print(table[["event", "method", "n_stations", "accuracy", "missed_alert",
                 "false_alert", "mu", "sigma"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
