# Real-Time Facial Recognition + ECO Tracking

*Fusing Viola-Jones detection, PCA-FLDA recognition, and the ECO visual tracker for continuous identity-aware surveillance without per-frame recognition overhead.*

**Machine Learning Group 14** — CSIE, National Central University

📄 **[Read the Final Report](https://github.com/woodyhoko/Facial_Recognition_and_Tracking/blob/main/facial%20recognition%20and%20tracking%20-%20final%20report.pdf)** | **[▶ Pipeline Demo](demo.html)**

---

## 1. Problem statement

Existing surveillance systems run face recognition independently on every video frame. This is computationally prohibitive at real-time frame rates and fails when the subject turns away or is partially occluded — conditions where the detector yields no valid output. The key insight of this project is that **recognition** (who is this person?) and **tracking** (where are they now?) are complementary problems that should be solved jointly:

- Recognition is accurate but expensive and requires a frontal face
- Tracking is cheap and works across pose/occlusion but carries no identity

By running recognition only when needed (new face appears, or tracker loses the target) and delegating spatial continuity to the ECO tracker, we achieve real-time identity-aware tracking at a fraction of the compute cost of per-frame recognition.

---

## 2. System pipeline

```
Video stream (30 fps)
        │
        ▼
[Viola-Jones Detector]          ← runs every N frames or on track failure
        │   bounding box
        ▼
[PCA → FLDA Recognizer]         ← runs once per new detection event
        │   identity label + confidence
        ▼
[ECO Visual Tracker]            ← runs every frame to propagate the box
        │   updated bounding box (with inherited identity label)
        ▼
Identity-labeled overlay on video
```

---

## 3. Component analysis

### 3.1 Face detection — Viola-Jones

The Viola-Jones algorithm (Viola & Jones 2004) detects frontal faces using a **cascade of weak classifiers** (boosted Haar-feature detectors):

1. **Integral image** — O(1) computation of rectangular area sums
2. **Haar-like features** — rectangular intensity difference patterns sensitive to face structure (eye–cheek, nose bridge, mouth region)
3. **AdaBoost** — selects the most discriminative features and combines them into a strong classifier
4. **Cascade structure** — early rejection of non-face regions at coarse stages; only face candidates reach the final stage

The cascade achieves real-time performance (>30 fps on CPU) at high precision for frontal faces. It is the detection engine of choice when computational resources are constrained. Its limitation — sensitivity to non-frontal poses — is precisely what the ECO tracker compensates for.

### 3.2 Face recognition — PCA + FLDA (Eigenfaces + Fisherfaces)

Recognition maps the detected face crop to an identity via two sequential dimensionality reduction steps:

**Step 1 — PCA (Eigenfaces, Turk & Pentland 1991):**

The face image (flattened to a vector **x** ∈ ℝᵈ) is projected onto the top *k* eigenvectors of the face covariance matrix:

```
C = (1/N) Σ (xᵢ − μ)(xᵢ − μ)ᵀ
W_PCA = top-k eigenvectors of C
z = W_PCA^T (x − μ)                    z ∈ ℝᵏ
```

This compresses the 256×256 face image (~65K dimensions) to a compact eigenface representation (~128 dimensions) while retaining most variance.

**Step 2 — FLDA (Fisherfaces, Belhumeur et al. 1997):**

PCA does not optimize for class separation. FLDA finds the projection **W_LDA** that simultaneously **maximizes between-class scatter** and **minimizes within-class scatter**:

```
W_LDA = argmax  |W^T S_B W|
                ──────────
                |W^T S_W W|
```

where S_B is the between-class scatter matrix and S_W is the within-class scatter. The combined PCA→FLDA pipeline (Fisherfaces) outperforms PCA alone under illumination variation because FLDA explicitly penalizes within-identity appearance variation.

### 3.3 Visual tracking — ECO (Efficient Convolution Operators)

ECO (Danelljan et al. 2017) is a discriminative correlation filter tracker built on C-COT (Danelljan et al. 2016). It addresses three limitations of prior discriminative trackers:

**Problem 1: Model bloat.** Training sets grow unboundedly as frames accumulate, making model updates increasingly expensive.

**ECO solution — Compact Sample Space Model:** Instead of storing all training samples, ECO maintains a **Gaussian Mixture Model (GMM)** in feature space that approximates the distribution of past appearances. New samples are incorporated by updating the GMM rather than appending to a list:

```
GMM: p(x) = Σᵢ wᵢ · 𝒩(x; μᵢ, Σᵢ)     (K components, K ≪ T frames)
```

**Problem 2: Over-parameterized filters.** Standard correlation filter banks have one filter per feature channel, producing redundant parameters.

**ECO solution — Factorized Convolution Operator:** The filter is decomposed as **f = P · h**, where **P** ∈ ℝ^{D×C} is a learned projection matrix (D compressed channels, C original channels) and **h** is the correlation filter in the compressed space. This reduces filter parameters by ~90% while preserving discriminative capacity.

**Problem 3: Overfitting to recent frames.** Updating the model on every frame causes tracker drift (the model "forgets" what the target originally looked like).

**ECO solution — Conservative Model Updates:** The model is updated only every *n* frames (default n = 6), and the update step size is much smaller than in standard CF trackers. This is equivalent to a high-momentum gradient descent that smoothly incorporates new appearance information without catastrophic forgetting.

This project uses **ECO-HC** (Handcrafted features variant) — HOG + Color Names features — which achieves near state-of-the-art tracking without deep feature extraction, making it suitable for real-time deployment.

---

## 4. Re-identification challenge

The main open problem noted in the paper: when the tracker loses the target (occlusion, leaving frame) and the face detector fires again, the system must determine whether the newly detected face is the same identity as before or a new person. Current implementation handles this heuristically (proximity + appearance threshold). A learned **re-ID embedding** (e.g. ArcFace, Deng et al. 2019) would provide a principled solution.

---

## 5. Files

| File | Description |
|---|---|
| `ECO_GUI.fig` / `ECO_GUI.m` | MATLAB GUI for standalone ECO tracking |
| `Final_GUI.m` | Integrated pipeline: detection + recognition + ECO tracking |
| `demo_ECO.m` | ECO tracker demo script |
| `.gitmodules` | ECO submodule reference |
| `facial recognition and tracking - final report.pdf` | Full 4-page paper |
| `demo.html` | Interactive pipeline visualizer (browser) |

---

## 6. Run

```matlab
% Integrated pipeline:
run('Final_GUI.m')

% ECO tracker alone:
run('demo_ECO.m')
```

```bash
# Initialize ECO submodule after cloning:
git submodule update --init --recursive
```

**Requirements:** MATLAB + Image Processing Toolbox + Computer Vision Toolbox.

---

## 7. References

1. P. Viola and M. Jones. "Robust Real-Time Face Detection." *IJCV*, 57(2):137–154, 2004.
2. M. Turk and A. Pentland. "Eigenfaces for Recognition." *J. Cognitive Neuroscience*, 3(1):71–86, 1991.
3. P. N. Belhumeur, J. P. Hespanha, D. J. Kriegman. "Eigenfaces vs. Fisherfaces." *IEEE TPAMI*, 19(7):711–720, 1997.
4. M. Danelljan et al. "ECO: Efficient Convolution Operators for Tracking." *CVPR 2017*.
5. M. Danelljan et al. "Beyond Correlation Filters: Learning Continuous Convolution Operators (C-COT)." *ECCV 2016*.
6. J. Deng et al. "ArcFace: Additive Angular Margin Loss for Deep Face Recognition." *CVPR 2019*.
