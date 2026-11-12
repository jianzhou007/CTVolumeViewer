import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ct_volume_viewer.scv import SCVVolume, read_scv_header
from ct_volume_viewer.render3d import render_surface_preview


def write_test_volume(path: Path, values: np.ndarray, header_size: int = 1024) -> None:
    z_size, y_size, x_size = values.shape
    header = bytearray(header_size)
    struct.pack_into("<6I", header, 0, header_size, 0, 16, x_size, y_size, z_size)
    struct.pack_into("<3d", header, 24, 0.14844, 0.14844, 0.14844)
    path.write_bytes(header + values.astype("<u2").tobytes())


class SCVVolumeTests(unittest.TestCase):
    def test_reads_header_metadata_and_three_slice_directions(self) -> None:
        values = np.arange(3 * 4 * 5, dtype=np.uint16).reshape((3, 4, 5))
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "sample.uint16_scv"
            write_test_volume(path, values)

            header = read_scv_header(path)
            with SCVVolume(path) as volume:
                np.testing.assert_array_equal(volume.get_slice("axial", 1), values[1, :, :])
                np.testing.assert_array_equal(volume.get_slice("coronal", 2), values[:, 2, :])
                np.testing.assert_array_equal(volume.get_slice("sagittal", 3), values[:, :, 3])

        self.assertEqual(header.dimensions, (5, 4, 3))
        self.assertEqual(header.header_size, 1024)
        self.assertEqual(header.bit_depth, 16)
        self.assertAlmostEqual(header.spacing_mm[0], 0.14844)

    def test_rejects_file_size_mismatch(self) -> None:
        values = np.zeros((2, 2, 2), dtype=np.uint16)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "truncated.scv"
            write_test_volume(path, values)
            path.write_bytes(path.read_bytes()[:-1])

            with self.assertRaisesRegex(ValueError, "file size mismatch"):
                read_scv_header(path)

    def test_rejects_non_16_bit_volume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "unsupported.scv"
            header = bytearray(1024)
            struct.pack_into("<6I", header, 0, 1024, 0, 8, 1, 1, 1)
            struct.pack_into("<3d", header, 24, 0.1, 0.1, 0.1)
            path.write_bytes(header + b"\x00")

            with self.assertRaisesRegex(ValueError, "bit depth"):
                read_scv_header(path)


class SurfacePreviewTests(unittest.TestCase):
    def test_renders_thresholded_surface_and_changes_with_rotation(self) -> None:
        volume = np.zeros((24, 24, 24), dtype=np.uint16)
        volume[5:19, 6:18, 7:17] = 50_000

        front = render_surface_preview(volume, 30_000, 0.0, 0.0, 160, 160)
        rotated = render_surface_preview(volume, 30_000, 0.7, 0.2, 160, 160)
        empty = render_surface_preview(volume, 60_000, 0.0, 0.0, 160, 160)

        self.assertEqual(front.size, (160, 160))
        self.assertGreater(np.asarray(front).sum(), 0)
        self.assertFalse(np.array_equal(front, rotated))
        front_pixels = np.asarray(front)
        np.testing.assert_array_equal(front_pixels[:, :, 0], front_pixels[:, :, 1])
        np.testing.assert_array_equal(front_pixels[:, :, 1], front_pixels[:, :, 2])
        expected_background = np.full((160, 160, 3), (13, 16, 18), dtype=np.uint8)
        np.testing.assert_array_equal(np.asarray(empty), expected_background)

if __name__ == "__main__":
    unittest.main()