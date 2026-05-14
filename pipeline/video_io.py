"""Video ingest and frame extraction using OpenCV."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Sequence

import cv2


def save_uploaded_videos(
    uploaded_files: Sequence,
    output_dir: str | Path,
    prefix: str = "clip",
) -> list[tuple[str, Path]]:
    """
    Persist Streamlit UploadedFile objects to disk.

    Returns list of (clip_id, video_path) in stable order.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[tuple[str, Path]] = []

    for idx, uf in enumerate(uploaded_files, start=1):
        clip_id = f"{prefix}_{idx:03d}"
        suffix = Path(uf.name).suffix.lower() or ".mp4"
        if suffix not in {".mp4", ".mov", ".avi", ".mkv"}:
            suffix = ".mp4"
        dest = output_dir / clip_id / f"source{suffix}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(uf.getbuffer())
        saved.append((clip_id, dest))

    return saved


def extract_frames(
    video_path: str | Path,
    frame_dir: str | Path,
    fps: float = 5.0,
) -> list[Path]:
    """
    Sample frames from a video at approximately ``fps`` and save as JPG.

    Frame files are named ``000000.jpg``, ``000001.jpg``, ...

    Returns ordered list of saved frame paths. Empty if the video could not
    be read or yielded no frames.
    """
    video_path = Path(video_path)
    frame_dir = Path(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if src_fps <= 1e-3:
        src_fps = 30.0

    step = max(1, int(round(src_fps / max(fps, 0.1))))

    paths: list[Path] = []
    frame_idx = 0
    saved_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            out = frame_dir / f"{saved_idx:06d}.jpg"
            cv2.imwrite(str(out), frame)
            paths.append(out)
            saved_idx += 1
        frame_idx += 1

    cap.release()
    return paths
