# Latent Action Dataset Generator (prototype)

A small **Streamlit** app that turns uploaded manipulation videos into a **weakly labeled** frame-pair dataset: optical flow between consecutive frames → feature vectors → **KMeans** “latent action” IDs, plus **proxy** confidence scores and downloadable artifacts.

## What it does

1. Saves uploaded videos under `outputs/clip_XXX/`.
2. Extracts frames at a user-chosen FPS (default 5).
3. Computes dense Farneback optical flow between consecutive frames.
4. Summarizes each motion interval with a fixed feature vector.
5. Clusters **all** intervals (across clips) into `K` pseudo-action classes.
6. Assigns heuristic confidence from motion level, distance to cluster center, and temporal smoothness of features **within each clip**.

## Why action-ish labels for robotics data

Learning from video often needs **pairs** `(observation_t, observation_{t+1})` together with **something like an action** or transition label. Real robot actions are high-dimensional and embodiment-specific; this tool is a V0 pipeline to allow for future prototyping of dataset formats and training loops before using true teleop or controller logs.

## V0 baseline (this repo)

This version **does not** integrate LAPA or train a model. Pseudo-actions are **2D image motion clusters**, not motor torques, gripper commands, or 3D object-relative motions. Treat cluster IDs as **weak, non-semantic** indices.

## Run locally

```bash
cd latent-action-prototype
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

## Example JSONL row

Each line is one consecutive frame pair:

```json
{
  "clip_id": "clip_001",
  "frame_t": "outputs/clip_001/frames/000012.jpg",
  "frame_t1": "outputs/clip_001/frames/000013.jpg",
  "latent_action": 4,
  "confidence": 0.82,
  "motion_magnitude": 0.77,
  "smoothness_score": 0.88
}
```

`latent_actions.npy` is a 2-column array: `[latent_action_id, confidence]` per row, in the same order as lines in `dataset.jsonl`.


## Project layout

```
latent-action-prototype/
  app.py
  requirements.txt
  README.md
  pipeline/
    __init__.py
    video_io.py
    optical_flow.py
    action_clustering.py
    confidence.py
    export.py
  outputs/          # cleared each successful “Process videos” run
```
