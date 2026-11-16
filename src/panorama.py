from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAX_CANVAS_PIXELS = 20_000_000


@dataclass(frozen=True)
class MatchResult:
    keypoints_a: list[cv2.KeyPoint]
    keypoints_b: list[cv2.KeyPoint]
    matches: list[cv2.DMatch]
    homography: np.ndarray
    inlier_mask: np.ndarray


def load_images(folder: Path) -> tuple[list[np.ndarray], list[Path]]:
    images, paths = [], []
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Skipping unreadable image: {path}", file=sys.stderr)
            continue
        images.append(image)
        paths.append(path)
    if len(images) < 2:
        raise ValueError(f"Need at least two readable images in {folder}")
    return images, paths


def create_feature_extractor(name: str):
    if name == "orb":
        return cv2.ORB_create(nfeatures=5000), cv2.NORM_HAMMING
    if name == "sift":
        return cv2.SIFT_create(nfeatures=5000), cv2.NORM_L2
    raise ValueError(f"Unsupported feature backend: {name}")


def normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroid = points.mean(axis=0)
    shifted = points - centroid
    mean_distance = np.linalg.norm(shifted, axis=1).mean()
    if mean_distance < 1e-12:
        raise ValueError("Degenerate point configuration")
    scale = np.sqrt(2.0) / mean_distance
    transform = np.array(
        [[scale, 0.0, -scale * centroid[0]],
         [0.0, scale, -scale * centroid[1]],
         [0.0, 0.0, 1.0]]
    )
    return shifted * scale, transform


def estimate_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src, dst = np.asarray(src, dtype=np.float64), np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2 or len(src) < 4:
        raise ValueError("Expected at least four paired 2D correspondences")
    if not np.isfinite(src).all() or not np.isfinite(dst).all():
        raise ValueError("Correspondences must be finite")

    src_n, t_src = normalize_points(src)
    dst_n, t_dst = normalize_points(dst)
    rows = []
    for (x, y), (u, v) in zip(src_n, dst_n):
        rows.append([-x, -y, -1, 0, 0, 0, x * u, y * u, u])
        rows.append([0, 0, 0, -x, -y, -1, x * v, y * v, v])

    matrix = np.array(rows, dtype=np.float64)
    # Four correspondences give an 8x9 matrix, so request the extra null-space vector.
    # For larger fits, reduced SVD avoids allocating a square matrix of size 2N.
    _, values, vt = np.linalg.svd(matrix, full_matrices=len(src) == 4)
    if values[7] <= values[0] * max(matrix.shape) * np.finfo(matrix.dtype).eps:
        raise ValueError("Correspondences do not constrain a homography")

    h = np.linalg.solve(t_dst, vt[-1].reshape(3, 3) @ t_src)
    if np.linalg.matrix_rank(h) < 3:
        raise ValueError("Homography is singular")
    return h / (h[2, 2] if abs(h[2, 2]) > 1e-12 else np.linalg.norm(h))


def project_points(points: np.ndarray, h: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(len(points))])
    projected = homogeneous @ h.T
    denominator = projected[:, 2:3]
    return np.divide(
        projected[:, :2], denominator,
        out=np.full((len(points), 2), np.inf),
        where=np.abs(denominator) > 1e-12,
    )


def ransac_homography(
    src: np.ndarray,
    dst: np.ndarray,
    iterations: int = 2000,
    threshold: float = 3.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if len(src) < 4 or src.shape != dst.shape:
        raise ValueError("RANSAC needs at least four paired correspondences")
    if iterations < 1 or not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("RANSAC iterations and threshold must be positive")

    rng = np.random.default_rng(seed)
    best_mask = np.zeros(len(src), dtype=bool)
    best_h = None
    for _ in range(iterations):
        sample = rng.choice(len(src), size=4, replace=False)
        try:
            h = estimate_homography(src[sample], dst[sample])
        except (ValueError, np.linalg.LinAlgError):
            continue
        mask = np.linalg.norm(project_points(src, h) - dst, axis=1) < threshold
        if mask.sum() > best_mask.sum():
            best_h, best_mask = h, mask
            if best_mask.all():
                break
    if best_h is None or best_mask.sum() < 4:
        raise RuntimeError("Could not estimate a stable homography")

    refined = estimate_homography(src[best_mask], dst[best_mask])
    refined_mask = np.linalg.norm(project_points(src, refined) - dst, axis=1) < threshold
    if refined_mask.sum() >= best_mask.sum():
        return refined, refined_mask
    return best_h, best_mask


def match_pair(
    image_a: np.ndarray,
    image_b: np.ndarray,
    feature: str,
    ratio: float,
    ransac_iterations: int,
    ransac_threshold: float,
) -> MatchResult:
    if not 0 < ratio < 1:
        raise ValueError("The match ratio must be between 0 and 1")
    extractor, norm_type = create_feature_extractor(feature)
    kp_a, desc_a = extractor.detectAndCompute(cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY), None)
    kp_b, desc_b = extractor.detectAndCompute(cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY), None)
    if desc_a is None or desc_b is None:
        raise RuntimeError("Feature extraction produced no descriptors")

    matcher = cv2.BFMatcher(norm_type)
    good = []
    for pair in matcher.knnMatch(desc_a, desc_b, k=2):
        if len(pair) == 2:
            first, second = pair
            if first.distance < ratio * second.distance:
                good.append(first)
    if len(good) < 4:
        raise RuntimeError(f"Only {len(good)} good matches found; at least 4 are required")

    src = np.float64([kp_a[m.queryIdx].pt for m in good])
    dst = np.float64([kp_b[m.trainIdx].pt for m in good])
    h, inliers = ransac_homography(src, dst, ransac_iterations, ransac_threshold)
    return MatchResult(kp_a, kp_b, good, h, inliers)


def feather_blend(
    base: np.ndarray, warped: np.ndarray, base_mask: np.ndarray, warped_mask: np.ndarray,
) -> None:
    overlap = (base_mask != 0) & (warped_mask != 0)
    incoming = (base_mask == 0) & (warped_mask != 0)
    base[incoming] = warped[incoming]
    if overlap.any():
        distances = [
            cv2.distanceTransform(np.pad(mask, 1), cv2.DIST_L2, 3)[1:-1, 1:-1][overlap]
            for mask in (base_mask, warped_mask)
        ]
        weight = (distances[0] / (distances[0] + distances[1]))[:, None]
        base[overlap] = np.rint(base[overlap] * weight + warped[overlap] * (1 - weight))
    cv2.bitwise_or(base_mask, warped_mask, dst=base_mask)


def image_corners(image: np.ndarray, transform: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64)
    denominator = corners @ transform[2, :2] + transform[2, 2]
    if not ((denominator > 1e-12).all() or (denominator < -1e-12).all()):
        raise RuntimeError("A projected image crosses the homography horizon")
    return project_points(corners, transform)


def compose_panorama(images: list[np.ndarray], transforms: list[np.ndarray]) -> np.ndarray:
    corners = np.vstack([image_corners(img, h) for img, h in zip(images, transforms, strict=True)])
    minimum = np.floor(corners.min(axis=0))
    maximum = np.ceil(corners.max(axis=0))
    width, height = maximum - minimum
    if not np.isfinite(corners).all() or width * height > MAX_CANVAS_PIXELS or max(width, height) >= 32_767:
        raise RuntimeError("Stitching would create an oversized canvas; check overlap or resize the inputs")

    size = (int(width), int(height))
    translation = np.eye(3)
    translation[:2, 2] = -minimum
    panorama = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    occupied = np.zeros((size[1], size[0]), dtype=np.uint8)
    for image, transform in zip(images, transforms, strict=True):
        transform = translation @ transform
        warped = cv2.warpPerspective(image, transform, size, borderMode=cv2.BORDER_REPLICATE)
        footprint = np.ones(image.shape[:2], dtype=np.uint8)
        mask = cv2.warpPerspective(footprint, transform, size, flags=cv2.INTER_NEAREST)
        feather_blend(panorama, warped, occupied, mask)

    x, y, width, height = cv2.boundingRect(occupied)
    if not width or not height:
        raise RuntimeError("The projected canvas is empty")
    return panorama[y:y + height, x:x + width]


def save_match_visualization(
    image_a: np.ndarray,
    image_b: np.ndarray,
    result: MatchResult,
    destination: Path,
) -> None:
    inliers = [m for m, keep in zip(result.matches, result.inlier_mask) if keep]
    drawn = cv2.drawMatches(
        image_a, result.keypoints_a, image_b, result.keypoints_b, inliers, None,
        matchColor=(81, 218, 145),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    save_image(destination, drawn)


def save_image(destination: Path, image: np.ndarray) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), image):
        raise OSError(f"Could not write {destination}")


def stitch_images(
    images: Iterable[np.ndarray],
    feature: str,
    ratio: float,
    ransac_iterations: int,
    ransac_threshold: float,
    match_dir: Path | None = None,
) -> np.ndarray:
    images = list(images)
    if len(images) < 2:
        raise ValueError("Stitching requires at least two images")
    transforms = [np.eye(3)]
    for index, (previous, current) in enumerate(zip(images, images[1:]), start=1):
        try:
            result = match_pair(previous, current, feature, ratio, ransac_iterations, ransac_threshold)
            transform = transforms[-1] @ np.linalg.inv(result.homography)
        except (RuntimeError, np.linalg.LinAlgError) as exc:
            raise RuntimeError(f"Pair {index}–{index + 1}: {exc}") from exc
        transforms.append(transform / np.max(np.abs(transform)))
        if match_dir is not None:
            save_match_visualization(previous, current, result, match_dir / f"pair_{index:02d}.jpg")
        print(f"Pair {index}–{index + 1}: {int(result.inlier_mask.sum())}/{len(result.matches)} inlier matches")
    return compose_panorama(images, transforms)


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature-based panorama stitching")
    parser.add_argument("--input", type=Path, required=True, help="Directory containing ordered overlapping images")
    parser.add_argument("--output", type=Path, default=Path("outputs/panorama.jpg"))
    parser.add_argument("--feature", choices=["orb", "sift"], default="orb")
    parser.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio-test threshold")
    parser.add_argument("--ransac-iterations", type=int, default=2000)
    parser.add_argument("--ransac-threshold", type=float, default=3.0, help="Inlier reprojection error in pixels")
    parser.add_argument("--save-matches", type=Path)
    args = parser.parse_args()
    try:
        images, paths = load_images(args.input)
        if args.output.resolve() in {path.resolve() for path in paths}:
            raise ValueError("The output path points to one of the input images")
        print(f"Loaded {len(images)} images: {', '.join(p.name for p in paths)}")
        panorama = stitch_images(
            images, args.feature, args.ratio, args.ransac_iterations,
            args.ransac_threshold, args.save_matches,
        )
        save_image(args.output, panorama)
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        parser.error(str(exc))
    print(f"Saved {args.output} ({panorama.shape[1]} × {panorama.shape[0]} px)")


if __name__ == "__main__":
    main()
