from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

Axis = Literal["axial", "coronal", "sagittal"]
HEADER_FIELDS = struct.Struct("<6I")
SPACING_FIELDS = struct.Struct("<3d")


@dataclass(frozen=True)
class SCVHeader:
    header_size: int
    bit_depth: int
    dimensions: tuple[int, int, int]
    spacing_mm: tuple[float, float, float]

    @property
    def voxel_count(self) -> int:
        x_size, y_size, z_size = self.dimensions
        return x_size * y_size * z_size


def read_scv_header(path: str | Path) -> SCVHeader:
    volume_path = Path(path)
    with volume_path.open("rb") as stream:
        header = stream.read(max(HEADER_FIELDS.size, 48))

    if len(header) < 48:
        raise ValueError("File is too small to contain an SCV header")

    header_size, _, bit_depth, x_size, y_size, z_size = (
        HEADER_FIELDS.unpack_from(header)
    )
    spacing = SPACING_FIELDS.unpack_from(header, 24)

    if header_size < 48:
        raise ValueError(f"Invalid SCV header size: {header_size}")
    if bit_depth != 16:
        raise ValueError(f"Unsupported SCV bit depth: {bit_depth}; expected 16")
    if min(x_size, y_size, z_size) <= 0:
        raise ValueError("SCV dimensions must all be positive")
    if not all(np.isfinite(value) and value > 0 for value in spacing):
        raise ValueError("SCV voxel spacing must contain three positive finite values")

    expected_size = header_size + x_size * y_size * z_size * (bit_depth // 8)
    actual_size = volume_path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"SCV file size mismatch: expected {expected_size:,} bytes from its header, "
            f"found {actual_size:,}"
        )

    return SCVHeader(
        header_size=header_size,
        bit_depth=bit_depth,
        dimensions=(x_size, y_size, z_size),
        spacing_mm=spacing,
    )


class SCVVolume:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.header = read_scv_header(self.path)
        x_size, y_size, z_size = self.header.dimensions
        self.data: np.memmap | None = np.memmap(
            self.path,
            dtype="<u2",
            mode="r",
            offset=self.header.header_size,
            shape=(z_size, y_size, x_size),
        )

    def get_slice(self, axis: Axis, index: int) -> np.ndarray:
        if self.data is None:
            raise RuntimeError("The SCV volume has been closed")

        limits = {"axial": self.data.shape[0], "coronal": self.data.shape[1], "sagittal": self.data.shape[2]}
        if axis not in limits:
            raise ValueError(f"Unknown slice axis: {axis}")
        if not 0 <= index < limits[axis]:
            raise IndexError(f"{axis} slice index must be between 0 and {limits[axis] - 1}")

        if axis == "axial":
            return self.data[index, :, :]
        if axis == "coronal":
            return self.data[:, index, :]
        return self.data[:, :, index]

    def close(self) -> None:
        if self.data is not None:
            self.data._mmap.close()
            self.data = None

    def __enter__(self) -> SCVVolume:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()