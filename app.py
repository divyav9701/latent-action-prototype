"""
Latent Action Dataset Generator — Streamlit UI.

**Important:** Latent actions here are **weak pseudo-labels** derived from
2D optical-flow clustering. They are **not** true robot motor commands,
end-effector poses, or task-success labels. Use only as a fast prototyping
baseline for manipulation video data.
"""

from __future__ import annotations

import io
import shutil
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from pipeline.action_clustering import cluster_actions
from pipeline.confidence import compute_confidence
from pipeline.export import export_jsonl, export_numpy_actions, export_summary_csv
from pipeline.optical_flow import compute_flow_features
from pipeline.video_io import extract_frames, save_uploaded_videos

ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs"


def _rel_posix(path: Path) -> str:
    """Path relative to project root, POSIX for JSON."""
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _status_from_confidence(avg: float) -> str:
    if avg >= 0.75:
        return "usable"
    if avg >= 0.5:
        return "review"
    return "low quality"


def _clear_outputs() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for child in OUTPUT_ROOT.iterdir():
        if child.name == ".gitkeep":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


@st.cache_data(show_spinner=False)
def _fig_timeline(indices: tuple[int, ...], actions: tuple[int, ...]) -> bytes:
    fig, ax = plt.subplots(figsize=(8, 2.2))
    ax.step(indices, actions, where="post")
    ax.set_xlabel("Frame pair index")
    ax.set_ylabel("Latent action")
    ax.set_title("Action timeline (pseudo-labels)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def _fig_confidence(indices: tuple[int, ...], confs: tuple[float, ...]) -> bytes:
    fig, ax = plt.subplots(figsize=(8, 2.2))
    ax.plot(indices, confs, linewidth=1.2)
    ax.axhline(0.75, color="g", linestyle="--", alpha=0.5, label="usable")
    ax.axhline(0.5, color="orange", linestyle="--", alpha=0.5, label="review")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Frame pair index")
    ax.set_ylabel("Confidence")
    ax.set_title("Per-pair confidence (proxy)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def main() -> None:
    st.set_page_config(
        page_title="Latent Action Dataset Generator",
        layout="wide",
    )
    st.title("Latent Action Dataset Generator")
    st.caption(
        "Prototype: optical-flow motion vectors → KMeans pseudo-actions. "
        "Not LAPA / not trained policies — labels are **weak** and **not** "
        "true robot actions."
    )

    uploaded = st.file_uploader(
        "Upload 5–10 short manipulation videos (.mp4 / .mov)",
        type=["mp4", "mov", "avi", "mkv"],
        accept_multiple_files=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        fps = st.slider("Extraction FPS", min_value=1, max_value=15, value=5)
    with c2:
        n_clusters = st.slider(
            "Number of latent action clusters (K)",
            min_value=2,
            max_value=24,
            value=8,
        )

    zip_full = st.checkbox("Zip download includes full `outputs/` tree", value=False)

    if st.button("Process videos", type="primary"):
        if not uploaded:
            st.warning("Please upload at least one video.")
            return
        if len(uploaded) > 15:
            st.error("Please limit to ~15 files for this prototype.")
            return

        _clear_outputs()
        progress = st.progress(0.0, text="Starting…")
        all_rows: list[dict] = []
        clip_meta: list[dict] = []

        try:
            saved = save_uploaded_videos(uploaded, OUTPUT_ROOT)
        except Exception as e:
            st.error(f"Could not save uploads: {e}")
            return

        n_clips = len(saved)
        feat_blocks: list[np.ndarray] = []
        clip_ids_order: list[str] = []

        for ci, (clip_id, video_path) in enumerate(saved):
            progress.progress((ci + 0.1) / max(n_clips, 1), text=f"Frames: {clip_id}")
            frame_dir = OUTPUT_ROOT / clip_id / "frames"
            frame_paths = extract_frames(video_path, frame_dir, fps=float(fps))
            n_frames = len(frame_paths)

            if n_frames < 2:
                clip_meta.append(
                    {
                        "clip_id": clip_id,
                        "frame_paths": frame_paths,
                        "features": np.zeros((0, 6)),
                        "skipped": True,
                    }
                )
                continue

            feats = compute_flow_features(frame_paths)
            feat_blocks.append(feats)
            clip_ids_order.extend([clip_id] * feats.shape[0])

            pair_rows: list[dict] = []
            for pi in range(feats.shape[0]):
                t_path = frame_paths[pi]
                t1_path = frame_paths[pi + 1]
                pair_rows.append(
                    {
                        "clip_id": clip_id,
                        "pair_index": pi,
                        "frame_t": _rel_posix(t_path),
                        "frame_t1": _rel_posix(t1_path),
                        "motion_magnitude": float(feats[pi, 2]),
                    }
                )

            clip_meta.append(
                {
                    "clip_id": clip_id,
                    "frame_paths": frame_paths,
                    "features": feats,
                    "pair_rows": pair_rows,
                    "skipped": False,
                }
            )

        X = (
            np.vstack(feat_blocks)
            if feat_blocks
            else np.zeros((0, 6), dtype=np.float64)
        )
        n_pairs_total = X.shape[0]

        if n_pairs_total == 0:
            progress.progress(1.0, text="Done (no motion pairs).")
            summary_rows = [
                {
                    "clip_id": m["clip_id"],
                    "num_frames": len(m["frame_paths"]),
                    "num_action_labels": 0,
                    "avg_confidence": 0.0,
                    "status": "skipped — too few frames",
                }
                for m in clip_meta
            ]
            bundle_dir = OUTPUT_ROOT / "_bundle"
            bundle_dir.mkdir(parents=True, exist_ok=True)
            jsonl_path = bundle_dir / "dataset.jsonl"
            csv_path = bundle_dir / "summary.csv"
            npy_path = bundle_dir / "latent_actions.npy"
            export_jsonl([], jsonl_path)
            export_summary_csv(summary_rows, csv_path)
            export_numpy_actions([], [], npy_path)
            st.session_state["summary_rows"] = summary_rows
            st.session_state["all_rows"] = []
            st.session_state["actions_conf"] = np.zeros((0, 2))
            st.session_state["clip_meta"] = clip_meta
            st.session_state["labels_full"] = np.array([], dtype=np.int32)
            st.session_state["conf_full"] = np.array([], dtype=np.float64)
            st.session_state["jsonl_path"] = str(jsonl_path)
            st.session_state["csv_path"] = str(csv_path)
            st.session_state["npy_path"] = str(npy_path)
            st.session_state["zip_full"] = zip_full
            st.warning("No frame pairs to cluster. Upload longer clips or raise FPS.")
            return

        progress.progress(0.55, text="Clustering motions (KMeans)…")
        labels, dists = cluster_actions(X, n_clusters=n_clusters, random_state=0)

        # Integer clip index per row for smoothness within clip (first-seen order)
        seen: dict[str, int] = {}
        clip_row_idx_list: list[int] = []
        for cid in clip_ids_order:
            if cid not in seen:
                seen[cid] = len(seen)
            clip_row_idx_list.append(seen[cid])
        clip_row_idx = np.array(clip_row_idx_list, dtype=np.int32)

        motion_s, cons_s, smooth_s, conf = compute_confidence(X, dists, clip_row_idx)

        progress.progress(0.85, text="Building dataset rows…")

        summary_rows: list[dict] = []
        row_cursor = 0
        for meta in clip_meta:
            if meta.get("skipped"):
                summary_rows.append(
                    {
                        "clip_id": meta["clip_id"],
                        "num_frames": len(meta["frame_paths"]),
                        "num_action_labels": 0,
                        "avg_confidence": 0.0,
                        "status": "skipped — too few frames",
                    }
                )
                continue
            feats = meta["features"]
            pair_rows: list[dict] = meta["pair_rows"]
            m = feats.shape[0]
            sl = slice(row_cursor, row_cursor + m)
            row_cursor += m
            labs = labels[sl]
            cfs = conf[sl]
            sms = smooth_s[sl]

            for i, pr in enumerate(pair_rows):
                all_rows.append(
                    {
                        "clip_id": pr["clip_id"],
                        "frame_t": pr["frame_t"],
                        "frame_t1": pr["frame_t1"],
                        "latent_action": int(labs[i]),
                        "confidence": float(cfs[i]),
                        "motion_magnitude": float(pr["motion_magnitude"]),
                        "smoothness_score": float(sms[i]),
                    }
                )

            avg_c = float(np.mean(cfs)) if m else 0.0
            summary_rows.append(
                {
                    "clip_id": meta["clip_id"],
                    "num_frames": len(meta["frame_paths"]),
                    "num_action_labels": m,
                    "avg_confidence": round(avg_c, 4),
                    "status": _status_from_confidence(avg_c),
                }
            )

        actions_conf = np.column_stack(
            [labels.astype(np.int32), conf.astype(np.float64)]
        )

        bundle_dir = OUTPUT_ROOT / "_bundle"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = bundle_dir / "dataset.jsonl"
        csv_path = bundle_dir / "summary.csv"
        npy_path = bundle_dir / "latent_actions.npy"

        try:
            export_jsonl(all_rows, jsonl_path)
            export_summary_csv(summary_rows, csv_path)
            export_numpy_actions(labels, conf, npy_path)
        except Exception as e:
            st.error(f"Export failed: {e}")
            return

        progress.progress(1.0, text="Done.")

        st.session_state["summary_rows"] = summary_rows
        st.session_state["all_rows"] = all_rows
        st.session_state["actions_conf"] = actions_conf
        st.session_state["jsonl_path"] = str(jsonl_path)
        st.session_state["csv_path"] = str(csv_path)
        st.session_state["npy_path"] = str(npy_path)
        st.session_state["clip_meta"] = clip_meta
        st.session_state["labels_full"] = labels
        st.session_state["conf_full"] = conf
        st.session_state["zip_full"] = zip_full
        st.success("Processing complete.")

    if "summary_rows" in st.session_state:
        st.subheader("Results")
        df = pd.DataFrame(st.session_state["summary_rows"])
        st.dataframe(df, use_container_width=True)

        if st.session_state.get("all_rows"):
            overall_avg = float(
                np.mean([r["confidence"] for r in st.session_state["all_rows"]])
            )
            st.metric("Average confidence (all pairs)", f"{overall_avg:.3f}")

        st.subheader("Per-clip charts")
        labels_full: np.ndarray = st.session_state.get("labels_full", np.array([]))
        conf_full: np.ndarray = st.session_state.get("conf_full", np.array([]))
        row_off = 0
        for meta in st.session_state.get("clip_meta", []):
            if meta.get("skipped"):
                st.write(f"**{meta['clip_id']}** — skipped (insufficient frames).")
                continue
            m = meta["features"].shape[0]
            sl = slice(row_off, row_off + m)
            row_off += m
            labs = labels_full[sl]
            cfs = conf_full[sl]
            idx = tuple(range(m))
            st.write(f"**{meta['clip_id']}**")
            c1, c2 = st.columns(2)
            with c1:
                st.image(
                    _fig_timeline(idx, tuple(int(x) for x in labs)),
                    use_container_width=True,
                )
            with c2:
                st.image(
                    _fig_confidence(idx, tuple(float(x) for x in cfs)),
                    use_container_width=True,
                )

        st.subheader("Downloads")
        d1, d2, d3, d4 = st.columns(4)
        jp = st.session_state.get("jsonl_path")
        cp = st.session_state.get("csv_path")
        np_p = st.session_state.get("npy_path")
        if jp and Path(jp).exists():
            d1.download_button(
                "dataset.jsonl",
                data=Path(jp).read_bytes(),
                file_name="dataset.jsonl",
                mime="application/x-ndjson",
            )
        if cp and Path(cp).exists():
            d2.download_button(
                "summary.csv",
                data=Path(cp).read_bytes(),
                file_name="summary.csv",
                mime="text/csv",
            )
        if np_p and Path(np_p).exists():
            d3.download_button(
                "latent_actions.npy",
                data=Path(np_p).read_bytes(),
                file_name="latent_actions.npy",
                mime="application/octet-stream",
            )

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            if jp and Path(jp).exists():
                zf.write(jp, arcname="dataset.jsonl")
            if cp and Path(cp).exists():
                zf.write(cp, arcname="summary.csv")
            if np_p and Path(np_p).exists():
                zf.write(np_p, arcname="latent_actions.npy")
            if st.session_state.get("zip_full"):
                for p in sorted(OUTPUT_ROOT.rglob("*")):
                    if p.is_file():
                        arc = p.relative_to(OUTPUT_ROOT).as_posix()
                        zf.write(p, arcname=f"outputs/{arc}")
        buf.seek(0)
        d4.download_button(
            "bundle.zip",
            data=buf.getvalue(),
            file_name="latent_action_bundle.zip",
            mime="application/zip",
        )


if __name__ == "__main__":
    main()
