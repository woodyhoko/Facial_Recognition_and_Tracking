# Facial Recognition and Tracking

A real-time **face recognition and object tracking** system combining classical and deep learning approaches, with a MATLAB GUI interface.

📄 **[Read the Final Report](https://github.com/woodyhoko/Facial_Recognition_and_Tracking/blob/main/facial%20recognition%20and%20tracking%20-%20final%20report.pdf)**

---

## Overview

This project implements an end-to-end pipeline for:

1. **Face Detection** — locating face regions in each video frame
2. **Face Recognition** — identifying whose face it is against a registered database
3. **Object Tracking** — maintaining identity across frames using the ECO (Efficient Convolution Operators) tracker

The system is implemented in MATLAB with a graphical interface (`ECO_GUI.fig`, `Final_GUI.m`).

---

## Key Components

| File | Description |
|---|---|
| `ECO_GUI.fig` / `ECO_GUI.m` | MATLAB GUI for the ECO tracker |
| `Final_GUI.m` | Integrated recognition + tracking interface |
| `demo_ECO.m` | Standalone ECO tracker demonstration |
| `.gitmodules` | ECO tracker included as a submodule |

---

## ECO Tracker

The **ECO (Efficient Convolution Operators for Tracking)** algorithm is a state-of-the-art discriminative tracker that achieves high accuracy with low computational cost by factorizing the update step. It is included as a git submodule.

---

## Stack

- MATLAB (Image Processing Toolbox, Computer Vision Toolbox)
- ECO Tracker (C++ / MATLAB)

---

## Usage

```matlab
% In MATLAB:
run('Final_GUI.m')   % Launch the integrated GUI
% or
run('demo_ECO.m')    % Run the ECO tracker demo
```

> Initialize submodules after cloning:
> ```bash
> git submodule update --init --recursive
> ```

