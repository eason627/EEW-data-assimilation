#!/usr/bin/env python3
"""Replay Figure 2 by executing the existing KF notebook and plotting cells."""
import argparse
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "seismic_data_preprocess/Run_KalmanFilter.ipynb"
REFERENCE = Path(__file__).resolve().with_name("figure2_plotting_reference.json")
DIAGNOSTICS = ROOT / "visualization/animation_diagnostics"


def execute(nodes, ns, filename):
    tree = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(tree, filename, "exec"), ns)


def name(node):
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id


def load_results():
    """Run existing calculations, extending only HISTORY_END_SECOND to 45."""
    notebook = json.loads(NOTEBOOK.read_text())
    ns = {"__name__": "figure2_playback"}
    for index in (1, 3, 5, 7, 9, 11):
        nodes = ast.parse("".join(notebook["cells"][index]["source"])).body
        if index == 3:
            for node in nodes:
                if name(node) == "EVENT_ID":
                    node.value = ast.Constant(1)
                elif name(node) == "HISTORY_END_SECOND":
                    node.value = ast.Constant(45)
        elif index == 5:
            nodes = [n for n in nodes if (isinstance(n, ast.FunctionDef) and n.name == "predict_gmpe")
                     or name(n) in {"gmpe_initial", "initial_residual", "initial_output"}]
        elif index == 7:
            nodes = nodes[:2]
        elif index == 11:
            stop = next(i for i, n in enumerate(nodes) if name(n) == "eval_i")
            nodes = nodes[:stop]
        elif index == 9:
            stop = next(i for i, n in enumerate(nodes) if isinstance(n, ast.Assign)
                        and isinstance(n.targets[0], ast.Tuple)
                        and [t.id for t in n.targets[0].elts] == ["fig", "ax"])
            nodes = nodes[:stop]
        execute(nodes, ns, str(NOTEBOOK) + f":cell{index}")
    times = ns["pga_history"].index.to_numpy()
    if not np.array_equal(times, np.arange(int(times.min()), 46)):
        raise ValueError("Missing KF updates; refusing to interpolate or fill predictions")
    identifiers = ns["stations"][["Station", "Longitude", "Latitude"]].to_numpy()
    for second in times:
        current = ns["data"].loc[[second]]
        if not np.array_equal(current[["Station", "Longitude", "Latitude"]].to_numpy(), identifiers):
            raise ValueError(f"Station identity/order differs at {second} s")
    # Evaluate every actual update with the notebook's existing statements.
    nodes = ast.parse("".join(notebook["cells"][11]["source"])).body
    start = next(i for i, n in enumerate(nodes) if name(n) == "eval_i")
    stop = next(i for i, n in enumerate(nodes) if name(n) == "evaluation_output")
    rows = []
    for second in times:
        ns["evaluation_second"] = int(second)
        execute(nodes[start:stop + 1], ns, str(NOTEBOOK) + ":evaluation")
        row = ns["evaluation_output"].copy()
        row["arrived_stations"] = int(np.count_nonzero(
            ns["data"].loc[[second]]["SRepi"].to_numpy()
            <= ns["V_APP"] * (second - ns["depth"] / ns["VS_TRUE"])))
        rows.append(row)
    stored = None
    for output in notebook["cells"][11].get("outputs", []):
        for line in "".join(output.get("text", [])).splitlines():
            if line.startswith("Evaluation-time output: "):
                stored = ast.literal_eval(line.split(": ", 1)[1])
    if stored is None or stored["event_id"] != 1:
        # The interactive notebook may currently select a different event.
        # Validate Hualien against the previously verified animation metrics.
        import csv
        metrics_path = DIAGNOSTICS / "Hualien_metrics.csv"
        with metrics_path.open() as file:
            baseline = next(r for r in csv.DictReader(file)
                            if int(r["evaluation_second"]) == 35)
        stored = dict(event_id=1, evaluation_second=35,
                      **{key: float(baseline[key])
                         for key in ("mu", "sigma", "me", "azimuth")})
    actual = next(r for r in rows if r["evaluation_second"] == stored["evaluation_second"])
    delta = {k: actual[k] - stored[k] for k in ("mu", "sigma", "me", "azimuth")}
    if any(abs(v) > 1e-8 for v in delta.values()):
        raise ValueError(f"Results differ from stored notebook output: {delta}")
    return ns, rows, delta


def plotting_namespace(ns, rows):
    reference = json.loads(REFERENCE.read_text())
    ns.update(event_data=ns["data"], INIT_SEC=ns["settings"]["start"],
              hypo_lon=float(ns["event"]["hypolon"]), hypo_lat=float(ns["event"]["hypolat"]),
              hypo_depth=ns["depth"], Ztor=ns["depth"],
              event_num={1: 16, 2: 6, 3: 14}.get(ns["EVENT_ID"], -1), iT=0,
              FRO=0, FSS=1, FNO=0, Finter=0, Fintra=0, Fas=0, Fma=0, FVS30=0,
              ground_motion_chao20_initial=np.log(ns["gmpe_initial"]),
              azimuth_history=[r["azimuth"] for r in rows],
              kf_residual_history={"time": np.array([r["evaluation_second"] for r in rows]),
                                   "median": np.array([r["mu"] for r in rows]),
                                   "std": np.array([r["sigma"] for r in rows])},
              display=lambda *args: None, MultipleLocator=MultipleLocator)
    for kind in ("diagnostics", "map"):
        source = reference["cells"][kind]
        if kind == "map":
            source = source.replace("region = [119.3, 122.2, 21.7, 25.35]",
                                    "region = [119.4, 122.3, 21.7, 25.35]")
            source = source.replace("x=119.38", "x=119.5")
        source = source.replace('"../PGA_colorbar.cpt"', repr(str(ROOT / "PGA_colorbar.cpt")))
        source = source.replace('"/raid1/PLOTDATA/TW_city_boundary.txt"', repr(str(ROOT / "TW_city_boundary.txt")))
        nodes = ast.parse(source).body
        if kind == "map":
            # Keep all drawing calls, return the figure instead of saving outside.
            nodes[0].body[-1] = ast.Return(value=ast.Name(id="fig2", ctx=ast.Load()))
            colorbar = ast.parse(
                f'pygmt.makecpt(cmap={str(ROOT / "intensity_colorbar.cpt")!r}, '
                'series=[0, 10], '
                'color_model="+c0,2.5,8.0,15.0,25.0,40.0,80.0,150.0,250.0,400.0")\n'
                'fig2.colorbar(position="jBR+o1.0c/0.2c+v+w3c/0.3c")\n'
                'fig2.text(x=121.99, y=22.92, text="PGA (gal)", '
                'font="9p,Helvetica,black")'
            ).body
            nodes[0].body[-1:-1] = colorbar
        execute(nodes, ns, str(REFERENCE) + ":" + kind)
    timeseries = reference["cells"]["timeseries"].replace("plt.show()", "")
    timeseries = timeseries.replace("reference_times = [14,25]", "reference_times = []")
    execute(ast.parse(timeseries).body, ns, str(REFERENCE) + ":timeseries")
    plt.close(ns["fig_sigma"])
    plt.close(ns["fig2"])
    return reference


def raster(figure, axis=None):
    buffer = io.BytesIO()
    figure.canvas.draw()
    bbox = "tight" if axis is None else axis.get_tightbbox(figure.canvas.get_renderer()).transformed(
        figure.dpi_scale_trans.inverted()).expanded(1.025, 1.035)
    figure.savefig(buffer, dpi=160, bbox_inches=bbox, pad_inches=0.05, facecolor="white")
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def paste_fit(canvas, picture, box):
    x, y, width, height = box
    picture = picture.copy()
    picture.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas.paste(picture, (x + (width - picture.width) // 2, y + (height - picture.height) // 2))


def render(ns, rows, temporary, frames, selected_second=None):
    cursors = [ns[k].axvline(rows[0]["evaluation_second"], color="red", lw=2, zorder=8)
               for k in ("ax_az", "ax_mu")]
    dots = [ns[k].plot([], [], "o", ms=8, mfc="red", mec="white", zorder=9)[0]
            for k in ("ax_az", "ax_mu")]
    st = ns["stations"]
    for index, row in enumerate(rows):
        second = row["evaluation_second"]
        if selected_second is not None and second != selected_second:
            continue
        prediction = ns["pga_history"].loc[second].to_numpy()
        current = ns["data"].loc[[second]]
        wave_front = ns["V_APP"] * (second - ns["depth"] / ns["VS_TRUE"])
        arrived = current[current["SRepi"] <= wave_front].sort_values("SRhypo")
        ns["plot_pred_vs_obs"](
            np.log(prediction), ns["final_pga"], st["Azimuth"].to_numpy(),
            st["SRepi"].to_numpy(), real_pga=arrived["RealPGA"].to_numpy(),
            threshould_epi=wave_front, real_epi=arrived["SRepi"].to_numpy())
        diagnostics = plt.gcf()
        axes = diagnostics.axes
        axes[1].lines[0].set_label("Initial GMPE − final observed")
        axes[1].lines[1].set_label("Current KF − final observed")
        axes[1].legend(loc="lower left", fontsize=18, framealpha=0.85)
        residual_image = raster(diagnostics, axes[1])
        distance_image = raster(diagnostics, axes[2])
        plt.close(diagnostics)
        above = prediction >= 25
        map_figure = ns["Plot_shaking_map"](
            st["Latitude"].to_numpy()[above], st["Longitude"].to_numpy()[above],
            np.log(prediction), second, row["azimuth"])
        map_path = temporary / "map.png"
        map_figure.savefig(str(map_path), dpi=230)
        for cursor in cursors:
            cursor.set_xdata([second, second])
        dots[0].set_data([second], [ns["az_plot"][index]])
        dots[1].set_data([second], [row["mu"]])
        canvas = Image.new("RGB", (2100, 1260), "white")
        paste_fit(canvas, Image.open(map_path).convert("RGB"), (15, 105, 660, 1050))
        paste_fit(canvas, residual_image, (680, 105, 735, 515))
        paste_fit(canvas, distance_image, (680, 650, 735, 540))
        paste_fit(canvas, raster(ns["fig_az"]), (1425, 105, 650, 515))
        paste_fit(canvas, raster(ns["fig_mu"]), (1425, 670, 650, 520))
        # Trim margins formerly reserved for the header and footer.
        canvas = canvas.crop((0, 105, 2100, 1191))
        canvas.save(frames / f"frame_{index:03d}.png")
        print(f"Rendered {second} s ({index + 1}/{len(rows)})", flush=True)


def snapshot(notebook_namespace, second=None):
    """Compose a still from the notebook's in-memory predictions and settings."""
    ns = notebook_namespace.copy()
    times = ns["pga_history"].index.to_numpy()
    requested = ns["evaluation_second"] if second is None else second
    if requested < times.min() or requested > times.max():
        raise ValueError(f"Choose a second within the computed history: {times.min()}–{times.max()} s. "
                         "Increase HISTORY_END_SECOND and rerun the KF cell for later times.")
    available = times[times <= requested]
    if not len(available):
        raise ValueError(f"No KF update at or before {requested} s")
    selected = int(available[-1])
    rows = []
    for time in times:
        prediction = ns["pga_history"].loc[time].to_numpy()
        residual = np.log(prediction) - np.log(ns["final_pga"])
        state = ns["coefficient_history"].loc[time].to_numpy()
        mu, sigma = float(np.median(residual)), float(np.std(residual))
        rows.append(dict(evaluation_second=int(time), mu=mu, sigma=sigma,
                         me=float(np.hypot(mu, sigma)),
                         azimuth=float(np.degrees(np.arctan2(state[5], state[4])) % 360)))
    existing_figures = set(plt.get_fignums())
    try:
        plotting_namespace(ns, rows)
        if ns["EVENT_ID"] != 1:
            values = ns["az_plot"]
            ns["ax_az"].set_ylim(min(-90, float(values.min()) - 10),
                                 max(90, float(values.max()) + 10))
        if ns["settings"]["gmpe"] == "Zhao06":
            # Reuse the notebook's Japan map and inset for Japanese events.
            def japan_map(latitudes, longitudes, ln_pga, time, direction):
                figure = ns["plot_pga_map"](np.exp(ln_pga))
                figure.plot(x=float(ns["event"]["hypolon"]),
                            y=float(ns["event"]["hypolat"]),
                            style="v0.8c+e+a30+gblack+h0+p2p,black",
                            pen="3p,black", direction=[[(450-direction) % 360], [2]])
                figure.text(x=128.5, y=33.6, text=f"{time} s after O.T.",
                            justify="ML", font="14p,Helvetica-BoldOblique,red")
                figure.text(x=128.5, y=33.35, text=f"Dir. = {direction:.0f}°",
                            justify="ML", font="12p,Helvetica-BoldOblique,red")
                return figure
            ns["Plot_shaking_map"] = japan_map
        with tempfile.TemporaryDirectory(prefix="figure2-snapshot-") as directory:
            temporary = Path(directory)
            render(ns, rows, temporary, temporary, selected_second=selected)
            index = next(i for i, row in enumerate(rows)
                         if row["evaluation_second"] == selected)
            with Image.open(temporary / f"frame_{index:03d}.png") as picture:
                return picture.copy()
    finally:
        for number in set(plt.get_fignums()) - existing_figures:
            plt.close(number)


def encode_and_verify(frames, output, count, fps):
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    mp4 = output / "Hualien_playback.mp4"
    gif = output / "Hualien_playback.gif"
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(fps),
                    "-i", str(frames / "frame_%03d.png"), "-c:v", "libx264", "-crf", "19",
                    "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4)], check=True)
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(fps),
                    "-i", str(frames / "frame_%03d.png"), "-filter_complex",
                    "scale=1500:776:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3",
                    "-loop", "0", str(gif)], check=True)
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-xerror", "-i", str(mp4),
                    "-f", "null", "-"], check=True)
    reader = imageio_ffmpeg.read_frames(str(mp4), pix_fmt="rgb24")
    metadata = next(reader)
    decoded_count, first_hash, last_hash = 0, None, None
    for pixels in reader:
        last_hash = hashlib.sha256(pixels).hexdigest()
        if first_hash is None:
            first_hash = last_hash
        decoded_count += 1
    assert metadata["codec"] == "h264", metadata
    assert metadata["size"] == (2100, 1086), metadata
    assert decoded_count == count and first_hash != last_hash
    with Image.open(gif) as preview:
        assert preview.n_frames == count
        for index in range(count):
            preview.seek(index)
            preview.load()
            assert abs(preview.info["duration"] - 1000 / fps) < 11
    return {"mp4_bytes": mp4.stat().st_size, "gif_bytes": gif.stat().st_size,
            "mp4_decoded_frames": decoded_count, "gif_decoded_frames": count,
            "codec": metadata["codec"], "mp4_resolution": metadata["size"],
            "gif_resolution": [1500, 776], "fps": fps, "playback_seconds": count / fps}


def main():
    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "animations")
    parser.add_argument("--fps", type=float, default=1.5, help="Playback frames/s; scientific interval remains 1 s")
    args = parser.parse_args()
    if args.fps not in (1, 1.5, 2, 5, 10):
        parser.error("--fps must be 1, 1.5, 2, 5 or 10")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    ns, rows, delta = load_results()
    with tempfile.TemporaryDirectory(prefix="figure2-") as directory:
        temporary = Path(directory)
        frames = temporary / "frames"
        frames.mkdir()
        reference = plotting_namespace(ns, rows)
        render(ns, rows, temporary, frames)
        for second in (rows[0]["evaluation_second"], 14, 25, 35, 45):
            index = next(i for i, r in enumerate(rows) if r["evaluation_second"] == second)
            Image.open(frames / f"frame_{index:03d}.png").save(DIAGNOSTICS / f"Hualien_{second}s.png")
        verification = encode_and_verify(frames, output, len(rows), args.fps)
    ns["pd"].DataFrame(rows).to_csv(DIAGNOSTICS / "Hualien_metrics.csv", index=False)
    sources = [NOTEBOOK, REFERENCE, ROOT / "earthquake_index.csv",
               ROOT / "RuptureParaTime_new/202404022358_RuptureParaTimeByStation_new.csv",
               ROOT / "GMPE code/GMPE_Chao19.py", ROOT / "GMPE code/GMPE_Chao20_model_coefficients.csv",
               ROOT / "vdist.py", ROOT / "PGA_colorbar.cpt", ROOT / "intensity_colorbar.cpt",
               ROOT / "TW_city_boundary.txt"]
    report = dict(start_second=rows[0]["evaluation_second"], end_second=rows[-1]["evaluation_second"],
                  scientific_interval_seconds=1, frame_count=len(rows), stations=len(ns["stations"]),
                  stored_notebook_deltas=delta, original_plotting_source=reference["source"],
                  original_plotting_notebook_sha256=reference["source_sha256"],
                  sources_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  **verification)
    (DIAGNOSTICS / "Hualien_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    plt.close("all")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
