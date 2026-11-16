import unittest

import numpy as np

from src.panorama import compose_panorama, estimate_homography, project_points, ransac_homography


class PanoramaTests(unittest.TestCase):
    def test_dlt_recovers_four_point_projective_transform(self):
        points = np.array([[0, 0], [120, 0], [120, 100], [0, 100]])
        expected = np.array([[1.1, 0.03, -24], [0.08, 0.95, 17], [0.001, -0.0005, 1]])
        destination = project_points(points, expected)
        np.testing.assert_allclose(project_points(points, estimate_homography(points, destination)), destination)

    def test_dlt_rejects_collinear_points(self):
        line = np.column_stack([np.arange(4), np.arange(4)])
        with self.assertRaises(ValueError):
            estimate_homography(line, line + 10)

    def test_ransac_reports_outlier_after_refit(self):
        points = np.array([[0, 0], [120, 0], [120, 100], [0, 100], [10, 30], [50, 20]])
        destination = points + 8
        destination[-1] += 100
        h, inliers = ransac_homography(points, destination, iterations=100, threshold=0.5)
        np.testing.assert_array_equal(inliers, [True, True, True, True, True, False])
        np.testing.assert_allclose(project_points(points[:-1], h), destination[:-1], atol=1e-8)

    def test_black_pixels_keep_their_coverage(self):
        dark = np.zeros((12, 12, 3), np.uint8)
        light = np.full_like(dark, 180)
        shift = np.eye(3)
        shift[0, 2] = 6
        panorama = compose_panorama([dark, light], [np.eye(3), shift])
        self.assertEqual(panorama.shape, (12, 18, 3))
        self.assertEqual(panorama[5, 0, 0], 0)
        self.assertLess(panorama[5, 7, 0], 180)
        self.assertEqual(panorama[5, -1, 0], 180)

    def test_unbounded_warp_is_rejected(self):
        image = np.zeros((12, 12, 3), np.uint8)
        h = np.eye(3)
        h[2, 0] = -0.1
        with self.assertRaisesRegex(RuntimeError, "horizon"):
            compose_panorama([image, image], [np.eye(3), h])
