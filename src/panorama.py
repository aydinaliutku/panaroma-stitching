from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class MatchResult:
    keypoints_a: list[cv2.KeyPoint]
    keypoints_b: list[cv2.KeyPoint]
    matches: list[cv2.DMatch]
    homography: np.ndarray
    inlier_mask: np.ndarray


def load_images(folder: Path) -> tuple[list[np.ndarray], list[Path]]:
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    images = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in paths]
    valid = [(img, p) for img, p in zip(images, paths) if img is not None]
    if len(valid) < 2:
        raise ValueError(f"Need at least two readable images in {folder}")
    return [x[0] for x in valid], [x[1] for x in valid]


def create_feature_extractor(name: str):
    if name == "orb":
        return cv2.ORB_create(nfeatures=5000), cv2.NORM_HAMMING
    if name == "sift":
        return cv2.SIFT_create(nfeatures=5000), cv2.NORM_L2
    raise ValueError(f"Unsupported feature backend: {name}")


def normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    centroid = points.mean(axis=0)
    shifted = points - centroid
    mean_distance = np.linalg.norm(shifted, axis=1).mean()
    if mean_distance < 1e-12:
        raise ValueError("Degenerate point configuration")
    scale = np.sqrt(2.0) / mean_distance
    transform = np.array(
        [[scale, 0.0, -scale * centroid[0]],
         [0.0, scale, -scale * centroid[1]],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    homogeneous = np.column_stack([points, np.ones(len(points))])
    normalized = (transform @ homogeneous.T).T[:, :2]
    return normalized, transform


def estimate_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    if len(src) < 4 or len(dst) < 4:
        raise ValueError("At least four correspondences are required")

    src_n, t_src = normalize_points(src)
    dst_n, t_dst = normalize_points(dst)

    a_rows: list[list[float]] = []
    for (x, y), (u, v) in zip(src_n, dst_n):
        a_rows.append([-x, -y, -1, 0, 0, 0, x * u, y * u, u])
        a_rows.append([0, 0, 0, -x, -y, -1, x * v, y * v, v])

    _, _, vt = np.linalg.svd(np.asarray(a_rows, dtype=np.float64))
    h_normalized = vt[-1].reshape(3, 3)
    h = np.linalg.inv(t_dst) @ h_normalized @ t_src
    if abs(h[2, 2]) > 1e-12:
        h /= h[2, 2]
    return h


def project_points(points: np.ndarray, h: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(len(points))])
    projected = (h @ homogeneous.T).T
    denom = projected[:, 2:3]
    denom[np.abs(denom) < 1e-12] = 1e-12
    return projected[:, :2] / denom


def ransac_homography(
    src: np.ndarray,
    dst: np.ndarray,
    iterations: int = 2000,
    threshold: float = 3.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if len(src) < 4:
        raise ValueError("RANSAC needs at least four correspondences")

    rng = np.random.default_rng(seed)
    best_mask = np.zeros(len(src), dtype=bool)
    best_h: np.ndarray | None = None

    for _ in range(iterations):
        sample_idx = rng.choice(len(src), size=4, replace=False)
        try:
            h = estimate_homography(src[sample_idx], dst[sample_idx])
        except (ValueError, np.linalg.LinAlgError):
            continue

        error = np.linalg.norm(project_points(src, h) - dst, axis=1)
        mask = error < threshold
        if mask.sum() > best_mask.sum():
            best_mask = mask
            best_h = h

    if best_h is None or best_mask.sum() < 4:
        raise RuntimeError("Could not estimate a stable homography")

    refined = estimate_homography(src[best_mask], dst[best_mask])
    return refined, best_mask


def match_pair(
    image_a: np.ndarray,
    image_b: np.ndarray,
    feature: str,
    ratio: float,
    ransac_iterations: int,
    ransac_threshold: float,
) -> MatchResult:
    extractor, norm_type = create_feature_extractor(feature)
    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
    kp_a, desc_a = extractor.detectAndCompute(gray_a, None)
    kp_b, desc_b = extractor.detectAndCompute(gray_b, None)
    if desc_a is None or desc_b is None:
        raise RuntimeError("Feature extraction produced no descriptors")

    matcher = cv2.BFMatcher(norm_type)
    knn = matcher.knnMatch(desc_a, desc_b, k=2)
    good = [m for pair in knn if len(pair) == 2 for m, n in [pair] if m.distance < ratio * n.distance]
    if len(good) < 4:
        raise RuntimeError(f"Only {len(good)} good matches found; at least 4 are required")

    src = np.float64([kp_a[m.queryIdx].pt for m in good])
    dst = np.float64([kp_b[m.trainIdx].pt for m in good])
    h, inliers = ransac_homography(src, dst, ransac_iterations, ransac_threshold)
    return MatchResult(kp_a, kp_b, good, h, inliers)


def feather_blend(base: np.ndarray, warped: np.ndarray) -> np.ndarray:
    base_mask = np.any(base > 0, axis=2).astype(np.uint8)
    warped_mask = np.any(warped > 0, axis=2).astype(np.uint8)

    only_base = (base_mask == 1) & (warped_mask == 0)
    only_warped = (base_mask == 0) & (warped_mask == 1)
    overlap = (base_mask == 1) & (warped_mask == 1)

    out = np.zeros_like(base, dtype=np.float32)
    out[only_base] = base[only_base]
    out[only_warped] = warped[only_warped]

    if np.any(overlap):
        d_base = cv2.distanceTransform(base_mask, cv2.DIST_L2, 3)
        d_warped = cv2.distanceTransform(warped_mask, cv2.DIST_L2, 3)
        denom = d_base + d_warped + 1e-6
        w_base = d_base / denom
        w_warped = d_warped / denom
        out[overlap] = (
            base[overlap].astype(np.float32) * w_base[overlap, None]
            + warped[overlap].astype(np.float32) * w_warped[overlap, None]
        )
    return np.clip(out, 0, 255).astype(np.uint8)


def warp_into_panorama(panorama: np.ndarray, image: np.ndarray, h_panorama_to_image: np.ndarray) -> np.ndarray:
    h_image_to_panorama = np.linalg.inv(h_panorama_to_image)
    h_p, w_p = panorama.shape[:2]
    h_i, w_i = image.shape[:2]

    corners_p = np.array([[0, 0], [w_p, 0], [w_p, h_p], [0, h_p]], dtype=np.float64)
    corners_i = np.array([[0, 0], [w_i, 0], [w_i, h_i], [0, h_i]], dtype=np.float64)
    warped_corners_i = project_points(corners_i, h_image_to_panorama)
    all_corners = np.vstack([corners_p, warped_corners_i])

    min_xy = np.floor(all_corners.min(axis=0)).astype(int)
    max_xy = np.ceil(all_corners.max(axis=0)).astype(int)
    tx, ty = -min(0, min_xy[0]), -min(0, min_xy[1])
    out_w, out_h = int(max_xy[0] + tx), int(max_xy[1] + ty)

    translation = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    canvas[ty:ty + h_p, tx:tx + w_p] = panorama
    warped = cv2.warpPerspective(image, translation @ h_image_to_panorama, (out_w, out_h))
    return feather_blend(canvas, warped)


def crop_black_border(image: np.ndarray) -> np.ndarray:
    mask = np.any(image > 0, axis=2).astype(np.uint8)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return image
    x, y, w, h = cv2.boundingRect(coords)
    return image[y:y + h, x:x + w]


def save_match_visualization(
    image_a: np.ndarray,
    image_b: np.ndarray,
    result: MatchResult,
    destination: Path,
) -> None:
    inlier_matches = [m for m, keep in zip(result.matches, result.inlier_mask) if keep]
    drawn = cv2.drawMatches(
        image_a,
        result.keypoints_a,
        image_b,
        result.keypoints_b,
        inlier_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), drawn)


def stitch_images(
    images: Iterable[np.ndarray],
    feature: str,
    ratio: float,
    ransac_iterations: int,
    ransac_threshold: float,
    match_dir: Path | None = None,
) -> np.ndarray:
    images = list(images)
    panorama = images[0]

    for idx, next_image in enumerate(images[1:], start=1):
        result = match_pair(
            panorama,
            next_image,
            feature,
            ratio,
            ransac_iterations,
            ransac_threshold,
        )
        if match_dir is not None:
            save_match_visualization(panorama, next_image, result, match_dir / f"pair_{idx:02d}.jpg")
        panorama = crop_black_border(warp_into_panorama(panorama, next_image, result.homography))
        print(f"Stitched image {idx + 1}/{len(images)} with {int(result.inlier_mask.sum())} inliers")

    return panorama


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature-based panorama stitching")
    parser.add_argument("--input", type=Path, required=True, help="Directory containing ordered overlapping images")
    parser.add_argument("--output", type=Path, default=Path("outputs/panorama.jpg"))
    parser.add_argument("--feature", choices=["orb", "sift"], default="orb")
    parser.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio-test threshold")
    parser.add_argument("--ransac-iterations", type=int, default=2000)
    parser.add_argument("--ransac-threshold", type=float, default=3.0)
    parser.add_argument("--save-matches", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images, paths = load_images(args.input)
    print(f"Loaded {len(images)} images: {', '.join(p.name for p in paths)}")
    panorama = stitch_images(
        images,
        feature=args.feature,
        ratio=args.ratio,
        ransac_iterations=args.ransac_iterations,
        ransac_threshold=args.ransac_threshold,
        match_dir=args.save_matches,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), panorama):
        raise OSError(f"Could not write {args.output}")
    print(f"Saved panorama to {args.output}")


if __name__ == "__main__":
    main()
