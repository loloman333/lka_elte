# Lane Detector (Simplified)

A minimal C++17 + OpenCV project to detect lane lines from a camera or video file using a simple pipeline:

Input → HLS → Edge Detection (Canny) → ROI Mask → Hough Lines → Bin Left/Right → Overlay Output

Planned: polynomial fitting and temporal smoothing.

## Build and Run (Host)

Prereqs: OpenCV dev packages, build-essential, pkg-config.

```sh
make -j$(nproc)
./bin/lane_detector               # uses default camera
./bin/lane_detector path/to/video # or a video file
```

## Build and Run (Docker)

Build image and run:

```sh
# Build
docker build -t lane-detector:dev .

# Run with X11 on Linux (adjust if using Wayland)
# Allow X from local docker
xhost +local:root

# Camera (may vary by device), display, and optional video mount
docker run --rm \
  --device=/dev/video0 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$PWD":/app \
  lane-detector:dev ./bin/lane_detector

# Or run on a file inside the container
# docker run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix lane-detector:dev ./bin/lane_detector /app/data/sample.mp4
```

Notes:
- On Wayland, consider `xhost` alternatives or running with `--ipc=host` and Xwayland.
- For NVIDIA GPUs, you can add `--gpus all` if available.

## Project Layout

- `include/` headers
- `src/` sources
- `data/` optional videos/images
- `bin/` output executable (ignored)
- `build/` object files (ignored)

## Controls

- q or ESC: quit
- space: pause frame

## Tuning

Adjust thresholds in `LaneDetector::Params` (`include/lane_detector.h`).

## Next steps

- Polynomial lane fitting (curves)
- Temporal smoothing (EMA or Kalman)
- Unit tests for geometry helpers
