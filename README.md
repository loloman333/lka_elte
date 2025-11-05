# Lane Keep Assist (LKA) — Video Lane Detection & Annotation

A compact, classical-computer-vision-based lane detector that detects left/right lane boundaries in road videos, overlays polylines on frames, and logs per-frame confidence metrics to CSV. The code is written in C++17 with OpenCV and runs in real time on common clips.

___

## Table of Contents

1. Project Goals
2. Pipeline
3. Outputs and Quality Metrics
4. Build and Run
6. File Layout

---

## 1) Project Goals

The objective of this project is to develop a system that detects and tracks lane markings from driving video footage, similar to the perception component of a lane-keeping assist (LKA) system. The system aims to determine, for each frame, whether the left and right lane boundaries are detected, visualize them as overlaid polylines on the road, and estimate a confidence level for each side. The final outputs include an annotated video with lane overlays and a heads-up display (HUD), as well as a CSV file logging the per-frame detection status and confidence values.

---

## 2) Build and Run

### Build 

Option A — Dev Container / Docker (Ubuntu 22.04, OpenCV preinstalled):
- Container definition: [`Dockerfile`](Dockerfile)
- Just open in VS Code Dev Containers and build.

Option B — Local
- Requirements: C++17 compiler, OpenCV 4 (pkg-config preferred).
- Build with the provided [`Makefile`](Makefile):

### Run

Run on file or camera input:
```sh
# Run with a video file as input
bin/lane_detector path/to/video.mp4
# Run with webcam as input
bin/lane_detector
```

**CLI Options:** 
- `--no-debug`: disable the debug mosaic and per-step writers
- `--scale=F`: resize factor (e.g., 0.5 for half-size)
- `--stride=N`: process every Nth frame
- `--fast`: short-hand for fast preview (no debug, stride=2, scale=0.5)

**Examples:**
```sh
# Quick preview (every 2nd frame, half size):
bin/lane_detector --fast path/to/video.mp4

# Full debug, 80% scale:
bin/lane_detector --scale=0.8 path/to/video.mp4
```

**Artifacts** (auto-created in `output/`):
- `00_input.mp4`: resized input stream
- `99_output.mp4`: final overlay video
- `metrics.csv`: per-frame metrics
- Optional per-step MP4s when debug is enabled (see Debug UI)

Windows and Linux are supported. Video writers attempt H.264 (`avc1`) then fall back to `mp4v`.

---

## 3) File Layout

- Entry point and UI:
  - [`src/main.cpp`](src/main.cpp)
- Lane detector API and implementation:
  - Header: [`include/lane_detector.h`](include/lane_detector.h)
  - Source: [`src/lane_detector.cpp`](src/lane_detector.cpp)
- Data
  - Example Input Videos: ['data/*'](data/)
- Utilities (text box, mosaics, crosshair, etc.):
  - Header: [`include/utils.h`](include/utils.h)
  - Source: [`src/utils.cpp`](src/utils.cpp)
- Build:
  - [`Makefile`](Makefile)
  - [`Dockerfile`](Dockerfile)
  - [`docker-compose.yml`](docker-compose.yml)
- Saved results (CSV):
  - [`results/1-basic.csv`](results/1-basic.mp4)
  - [`results/1-basic.csv`](results/1-basic.csv)
  - ...

---

## 4) Lane Detection Pipeline

flowchart TD
    A[1. Convert to HLS] --> A1[1A. Apply CLAHE on L channel]
    A1 --> B[2. Edge Detection]
    B --> C[3. Apply ROI Mask]
    C --> D[4. Detect Hough Lines]
    D --> E[5. Pick Seed Lines and Classify]
    E --> F[6. Build Lane Masks]
    F --> G[7. Fit Polynomials]
    G --> H[8. Temporal Smoothing, Confidence & Output]

    subgraph Step1[Step 1: Preprocessing]
        A --> A1
    end

    subgraph Step2[Step 2: Edge Extraction]
        B
    end

    subgraph Step3[Step 3: ROI Masking]
        C
    end

    subgraph Step4[Step 4: Line Detection]
        D
    end

    subgraph Step5[Step 5: Line Classification]
        E
    end

    subgraph Step6[Step 6: Lane Mask Building]
        F
    end

    subgraph Step7[Step 7: Polynomial Fitting]
        G
    end

    subgraph Step8[Step 8: Temporal & Output]
        H
    end


**Step 1 – Convert to HLS**  
- Convert the input BGR frame to HLS color space.  

**Step 1A – Apply CLAHE on L channel**  
- Extract the L (lightness) channel.  
- Apply *Contrast Limited Adaptive Histogram Equalization (CLAHE)* to improve contrast and stabilize edges under varying lighting.  

**Step 2 – Edge Detection**  
- Apply Gaussian blur to suppress noise.  
- Use the **Canny** edge detector to obtain crisp edges.  

**Step 3 – Apply ROI Mask**  
- Generate a region of interest mask keeping the lower `roiKeepRatio` portion of the image.  
- Optionally cut side regions based on `roiAngleDeg`.  
- Apply the mask to retain only relevant road areas.  

**Step 4 – Detect Hough Lines**  
- Run the Hough Transform on the masked edge image to find line segments.  
- Visualize the raw detected lines for debugging.  

**Step 5 – Pick Seed Lines and Classify**  
- Use raycasting to identify initial left and right seed lines.  
- Apply **RANSAC** to filter outliers and refine the seed lines.  
- Group all lines that agree in slope and position with each seed.  
- Visualize classified left and right line sets.  

**Step 6 – Build Lane Masks**  
- For each side, sweep perpendicularly around the line segments to collect nearby edge pixels.  
- Construct binary masks for left and right lanes.  
- Overlay the two lane masks in color for visualization.  

**Step 7 – Fit Polynomials**  
- Extract lane pixel coordinates from each mask.  
- Fit a quadratic function \( x(y) = a y^2 + b y + c \) for each side.  
- Draw the raw polynomial fits for inspection.  
- Apply temporal smoothing to stabilize coefficients across frames.  

**Step 8 – Temporal Smoothing, Confidence, and Output**  
- Compute smoothed polynomial curves using exponential averaging.  
- Estimate per-side confidence based on line consistency and mask quality.  
- Write per-frame results (status + confidence) to CSV.  
- Draw final overlays: solid curves for detected lanes, dashed or fallback guides otherwise.  
- Add a HUD showing detection states and confidence levels.  
- Optionally store debug views of all intermediate steps.


---

## 5) Results
---


## 6) Design choices & Limitations: TODO

Design choices:
- Work in HLS and apply CLAHE on L to stabilize edges across lighting.
- Prefer vertical-ish segments; suppress near-horizontal Hough lines early.
- Use a quadratic $x(y)$ model: sufficient for mild curvature on most clips.
- Confidence blends multiple light-weight signals (mask density, line agreement, slope stability, coverage, temporal stability, curvature).

Common failure modes to watch for:
- Night and rain: specular highlights create spurious edges; paint is faint.
- Strong shadows or occlusions: broken masks reduce fit stability.
- Curbs/guardrails: occasionally mistaken for lane boundaries.
- Very sharp curves: quadratic may underfit; use piecewise fits or splines as an extension.

When in doubt, the system degrades gracefully: dashed gray guides and “NO” status make uncertainty explicit.