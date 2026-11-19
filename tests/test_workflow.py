import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]


class PanoramaWorkflowTests(unittest.TestCase):
    def test_both_feature_backends_write_a_panorama_and_match_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            for feature in ("orb", "sift"):
                with self.subTest(feature=feature):
                    output = Path(temporary) / f"{feature}.jpg"
                    matches = Path(temporary) / feature
                    completed = subprocess.run(
                        [
                            sys.executable, "src/panorama.py", "--input", "examples/tiles",
                            "--output", str(output), "--feature", feature,
                            "--save-matches", str(matches),
                        ],
                        cwd=ROOT, capture_output=True, text=True, timeout=60,
                    )
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    image = cv2.imread(str(output))
                    self.assertIsNotNone(image)
                    self.assertGreater(image.shape[1], 600)
                    self.assertEqual(len(list(matches.glob("pair_*.jpg"))), 2)
