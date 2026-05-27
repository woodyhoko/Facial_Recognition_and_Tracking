# Real-Time Facial Recognition + ECO Tracking

**Machine Learning Group 14** — CSIE, National Central University

📄 **[Read the Final Report](https://github.com/woodyhoko/Facial_Recognition_and_Tracking/blob/main/facial%20recognition%20and%20tracking%20-%20final%20report.pdf)**

---

## Abstract

> *We combine Viola-Jones face detection, PCA-FLDA recognition, and the ECO visual object tracker to build a real-time identity-tracking system — enabling continuous surveillance tracking without re-running face recognition on every frame.*

Existing surveillance systems run face recognition independently on each frame, which is computationally expensive and breaks down when the subject turns away. This project fuses **recognition** (who is this person?) with **tracking** (where are they now?) — using the ECO tracker to maintain identity between recognition events, dramatically reducing the per-frame compute burden.

---

## System Pipeline

```
Video Frame
    │
    ▼
[Viola-Jones Face Detector]
    │  Bounding box of face region
    ▼
[PCA-FLDA Face Recognizer]
    │  Identity: "Person A / unknown"
    ▼
[ECO Visual Tracker]
    │  Tracks the bounding box across subsequent frames
    │  without re-running recognition every frame
    ▼
Identity-labeled bounding box overlay
```

The key insight: run recognition only when needed (new face appears, or tracker loses the target), and let ECO maintain the bounding box — and thus the identity label — between those events.

---

## Components

### Face Detection — Viola-Jones
The Viola-Jones algorithm detects frontal faces via a cascade of Haar-feature classifiers. It is fast enough to run in real-time but sensitive to pose and lighting variation — which is why the tracker is needed to bridge non-frontal frames.

### Face Recognition — PCA + FLDA
1. **PCA (Principal Component Analysis)** — reduces the face image to a compact eigenface representation (Eigenfaces method)
2. **FLDA (Fisher's Linear Discriminant Analysis)** — maximizes between-class variance and minimizes within-class variance for robust classification across subjects

### Visual Tracking — ECO (Efficient Convolution Operators)

ECO is built on C-COT (Continuous Convolution Operator Tracker) with three key optimizations:

| Optimization | Problem Solved |
|---|---|
| **Factorized Convolution Operator** | Reduces model parameter count via low-rank filter decomposition (P: D×C projection matrix) |
| **Compact Sample Space Model** | Replaces the growing training set with a generative Gaussian mixture model, reducing memory and training time |
| **Conservative Model Updates** | Reduces over-fitting by updating less frequently; prevents tracker drift |

This project uses **ECO_HC** — the handcrafted-features variant — for real-time performance without deep feature extraction overhead.

---

## Key Technical Detail: Conservative Update Strategy

Standard discriminative trackers (DFS-based) update on every frame, causing the model to overfit to recent appearance and drift over time. ECO's update strategy maintains a compact model of the *distribution* of training samples (Gaussian mixture in feature space), updating conservatively — which is particularly important for long-horizon tracking tasks like multi-camera surveillance.

---

## Files

| File | Description |
|---|---|
| `ECO_GUI.fig` / `ECO_GUI.m` | MATLAB GUI for standalone ECO tracking |
| `Final_GUI.m` | Integrated pipeline: detection + recognition + tracking |
| `demo_ECO.m` | ECO tracker demo script |
| `.gitmodules` | ECO submodule reference |
| `facial recognition and tracking - final report.pdf` | Full 4-page paper |

---

## Run

```matlab
% In MATLAB — launch the integrated system:
run('Final_GUI.m')

% Or demo the tracker alone:
run('demo_ECO.m')
```

**Initialize ECO submodule after cloning:**
```bash
git submodule update --init --recursive
```

**Requirements:** MATLAB with Image Processing Toolbox and Computer Vision Toolbox.

---

## Limitations & Future Work

The main open challenge (noted in the paper) is reliably determining whether a newly detected face is already in the database — i.e., re-identification after track loss. Current implementation handles this heuristically; a learned re-ID embedding would improve robustness.
