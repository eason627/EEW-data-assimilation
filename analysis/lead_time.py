"""Warning lead time for the 2024 Hualien earthquake (event ID 1).

Lead time at a station = (first second at which the observed PGA reaches
25 gal) - (first second at which the KF-predicted PGA reaches 25 gal).
Only stations whose final PGA reaches 25 gal and with a positive lead time
are counted. Northern Taiwan uses TAP stations, southern Taiwan KAU stations.

The KF here is the one used for the lead-time analysis in the paper: Chao20
initial magnitude 6.2 at the final CWA hypocenter, updates from 5 to 54 s,
zero real-time PGA excluded and the simple covariance update.

Usage: python analysis/lead_time.py [output_dir]
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "GMPE code"))
sys.path.insert(0, str(ROOT))
from GMPE_Chao19 import Chao19H
from vdist import vdist

OUTPUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "analysis" / "output"
EVENT_ID = 1
M_INITIAL = 6.2
INIT_SEC = 5
END_TIME_STEP = 55  # exclusive
VS_TRUE = 3.0
V_APP = 2.6
Q = np.diag([1e-8, 1e-7, 5e-5, 1e-5, 5e-5, 5e-5])
THRESHOLD = 25
EXAMPLE_STATIONS = ["TAP020", "TAP065", "TAP127", "TAP128", "TCU083", "TCU017", "KAU111", "KAU123"]

# Additional northern lead times (s) included in the published northern mean
# and standard deviation. They were entered manually in the analysis notebook
# and are not computed from the TAP stations of the PGA table.
EXTRA_NORTHERN_SECONDS = [24, 18, 22, 22, 23, 24, 21, 21, 21, 20, 21, 21,
                          21, 19, 21, 21, 19, 19, 11, 21, 21, 18, 13]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    event = pd.read_csv(ROOT / "earthquake_index.csv", index_col="num").loc[EVENT_ID]
    event_key = event["time"].replace("T", "")
    event_data = pd.read_csv(
        ROOT / "RuptureParaTime_new" / f"{event_key}_RuptureParaTimeByStation_new.csv",
        index_col=0,
    )
    hypo_depth = float(event["hypodepth"])
    hypo_lat, hypo_lon = float(event["hypolat"]), float(event["hypolon"])
    lat = event_data["Latitude"].to_numpy()
    lon = event_data["Longitude"].to_numpy()
    distance_m, azimuth, _ = vdist(np.full_like(lat, hypo_lat), np.full_like(lon, hypo_lon), lat, lon)
    event_data["SRepi"] = distance_m / 1000
    event_data["SRhypo"] = np.sqrt(event_data["SRepi"] ** 2 + hypo_depth ** 2)
    event_data["Azimuth"] = azimuth

    # Initial GMPE (point source, strike-slip flags) and least-square fit
    stations = event_data.loc[INIT_SEC]
    r_hypo = stations["SRhypo"].to_numpy()
    vs30 = stations["Vs30"].to_numpy()
    pga_g, *_ = Chao19H(0, M_INITIAL, r_hypo, hypo_depth, vs30, -999,
                        0, 1, 0, 0, 0, 0, 0, 0)
    ln_gmpe = np.log(pga_g * 980 * 1.22).flatten()
    G_fit = np.stack([np.log(r_hypo) ** 2, np.log(r_hypo), np.log(vs30 / 760.0), np.ones_like(vs30)], axis=1)
    state = np.r_[np.linalg.lstsq(G_fit, ln_gmpe, rcond=None)[0], 0.0, 0.0]
    covariance = np.zeros((6, 6))
    theta = np.deg2rad(stations["Azimuth"].to_numpy())
    G_all = np.column_stack([G_fit, np.cos(theta), np.sin(theta)])

    # KF updates
    t0 = hypo_depth / VS_TRUE
    station_names = stations["Station"].reset_index(drop=True)
    example_index = station_names[station_names.isin(EXAMPLE_STATIONS)].index
    example_ln_pred = pd.DataFrame(index=range(INIT_SEC, END_TIME_STEP),
                                   columns=station_names[example_index], dtype=float)
    for second in range(INIT_SEC, END_TIME_STEP):
        current = event_data.loc[second]
        arrived = current[current["SRepi"] <= V_APP * (second - t0)]
        if len(arrived) < 5:
            continue
        arrived = arrived.sort_values(by="SRhypo")
        observed = arrived["RealPGA"].to_numpy()
        valid = observed > 0
        if valid.sum() < 5:
            continue
        a_r = arrived["SRhypo"].to_numpy()[valid]
        a_vs30 = arrived["Vs30"].to_numpy()[valid]
        a_theta = np.deg2rad(arrived["Azimuth"].to_numpy()[valid])
        H = np.column_stack([np.log(a_r) ** 2, np.log(a_r), np.log(a_vs30 / 760),
                             np.ones_like(a_vs30), np.cos(a_theta), np.sin(a_theta)])
        R_obs = np.diag(np.ones(valid.sum()) * 1e-4)
        covariance += Q
        S = H @ covariance @ H.T + R_obs
        gain = np.linalg.solve(S.T, (covariance @ H.T).T).T
        state += gain @ (np.log(observed[valid]) - H @ state)
        covariance = (np.eye(6) - gain @ H) @ covariance
        ln_pred = G_all @ state
        event_data.loc[second, "PGA_prediction"] = np.exp(ln_pred)
        example_ln_pred.loc[second] = ln_pred[example_index]

    # First exceedance of 25 gal: observed (final PGA >= 25) and predicted
    observed_first = event_data.copy()
    observed_first["Exceed_Second"] = np.where(
        (observed_first["EndRealPGA"] >= THRESHOLD) & (observed_first["RealPGA"] >= THRESHOLD),
        observed_first.index, np.nan,
    )
    observed_first = (observed_first[observed_first["RealPGA"] >= THRESHOLD]
                      .groupby("Station").first().reset_index())
    predicted_first = event_data.copy()
    predicted_first["prediction_Second"] = predicted_first.index
    predicted_first = (predicted_first[predicted_first["PGA_prediction"] >= THRESHOLD]
                       .groupby("Station").first().reset_index())
    lead = pd.merge(
        predicted_first[["Station", "prediction_Second", "Longitude", "Latitude"]],
        observed_first[["Station", "Exceed_Second"]], on="Station", how="left",
    ).dropna()
    difference = lead["Exceed_Second"] - lead["prediction_Second"]
    lead = lead[difference > 0].copy()
    lead["Lead_Time"] = (lead["Exceed_Second"] - lead["prediction_Second"]).round().astype(int)
    lead.to_csv(OUTPUT / "leadtime_hualien.csv", index=False)

    north = lead.loc[lead["Station"].str.startswith("TAP"), "Lead_Time"].to_numpy()
    south = lead.loc[lead["Station"].str.startswith("KAU"), "Lead_Time"]
    north_all = np.concatenate([north, EXTRA_NORTHERN_SECONDS])
    print(f"Northern (TAP, n={north.size}): {north.mean():.2f} ± {north.std(ddof=0):.2f} s")
    print(f"Northern (TAP + {len(EXTRA_NORTHERN_SECONDS)} additional, n={north_all.size}): "
          f"{north_all.mean():.2f} ± {north_all.std(ddof=0):.2f} s (population std)")
    print(f"Southern (KAU, n={south.size}): {south.mean():.2f} ± {south.std():.2f} s (sample std)")

    # Real-time observed vs KF-predicted PGA at example stations
    for name in EXAMPLE_STATIONS:
        row = lead[lead["Station"] == name]
        if row.empty:
            print(f"Skipping {name}: no lead time")
            continue
        record = event_data[event_data["Station"] == name]
        pred_second = row["prediction_Second"].values[0]
        exc_second = row["Exceed_Second"].values[0]
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(record.index, record["RealPGA"], "k-", lw=2, label="Observed PGA")
        ax.plot(example_ln_pred.index, np.exp(example_ln_pred[name]), "r-", lw=2, label="KF predicted PGA")
        ax.axvline(pred_second, color="gray", linestyle="--", linewidth=2, alpha=0.8)
        ax.annotate("", xy=(pred_second + 0.1, THRESHOLD), xytext=(exc_second - 0.5, THRESHOLD),
                    arrowprops=dict(arrowstyle="<->", color="blue", lw=3, mutation_scale=30))
        ax.text(0.5 * (pred_second + exc_second), THRESHOLD + 1, f"{row['Lead_Time'].values[0]} s",
                fontsize=30, color="blue", ha="center", va="bottom")
        ax.set_xlim(5, 70)
        ax.set_title(f"{name} (Repi = {round(record['SRepi'].values[0])} km, "
                     f"Azi. = {round(record['Azimuth'].values[0])}°)", fontsize=24)
        ax.set_xlabel("Second after origin time", fontsize=22)
        ax.set_ylabel("PGA (gal)", fontsize=22)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=14)
        fig.tight_layout()
        fig.savefig(OUTPUT / f"leadtime_{name}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
