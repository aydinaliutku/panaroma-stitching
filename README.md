# Feature-Based Panorama Stitching

A compact computer-vision pipeline for stitching overlapping images into a panorama. The project implements **normalized DLT homography estimation** and **RANSAC-based outlier rejection**, then uses OpenCV/NumPy for feature extraction, warping, blending, and visualization.

## Highlights

- ORB or SIFT feature extraction
- Lowe ratio-test feature matching
- Homography estimation with normalized Direct Linear Transform (DLT)
- Custom RANSAC loop for robust geometric estimation
- Perspective warping and feather blending
- Optional match visualizations for debugging
- CLI with portable paths; no machine-specific configuration

## Project structure

```text
panorama-stitching/
├── src/
│   └── panorama.py
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Put overlapping images in one directory, ordered by filename:

```text
data/office/
├── 01.jpg
├── 02.jpg
└── 03.jpg
```

Then run:

```bash
python src/panorama.py --input data/office --output outputs/office_panorama.jpg --feature orb
```

Use `--feature sift` for SIFT. Add `--save-matches outputs/matches` to export pairwise match visualizations.

## Pipeline

1. Detect local features and descriptors.
2. Match consecutive image pairs and filter matches with Lowe's ratio test.
3. Estimate pairwise homographies with custom RANSAC + normalized DLT.
4. Warp each new image into the current panorama coordinate frame.
5. Blend overlapping regions with distance-based feather weights.
6. Crop empty borders and save the final panorama.

## Notes

RANSAC and homography estimation are intentionally implemented explicitly to make the geometry easy to inspect. OpenCV is used for feature detectors, perspective warping, and basic image I/O.
