from __future__ import annotations

import math

import numpy as np
from PIL import Image


def render_surface_preview(
    volume: np.ndarray,
    threshold: int,
    yaw: float,
    pitch: float,
    width: int,
    height: int,
) -> Image.Image:
    occupied = volume >= threshold
    interior = occupied.copy()
    interior[1:, :, :] &= occupied[:-1, :, :]
    interior[:-1, :, :] &= occupied[1:, :, :]
    interior[:, 1:, :] &= occupied[:, :-1, :]
    interior[:, :-1, :] &= occupied[:, 1:, :]
    interior[:, :, 1:] &= occupied[:, :, :-1]
    interior[:, :, :-1] &= occupied[:, :, 1:]
    interior[[0, -1], :, :] = False
    interior[:, [0, -1], :] = False
    interior[:, :, [0, -1]] = False

    surface = occupied & ~interior
    coordinates = np.argwhere(surface)
    image = Image.new("RGB", (width, height), "#0d1012")
    if coordinates.size == 0:
        return image

    values = volume[surface]
    point_step = max(1, math.ceil(len(coordinates) / 100_000))
    coordinates = coordinates[::point_step]
    values = values[::point_step].astype(np.float32)

    z_size, y_size, x_size = volume.shape
    extent = max(x_size, y_size, z_size)
    x = (coordinates[:, 2] - (x_size - 1) / 2) * (2.0 / extent)
    y = (coordinates[:, 1] - (y_size - 1) / 2) * (2.0 / extent)
    z = (coordinates[:, 0] - (z_size - 1) / 2) * (2.0 / extent)

    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    rotated_x = cos_yaw * x + sin_yaw * z
    rotated_z = -sin_yaw * x + cos_yaw * z
    cos_pitch, sin_pitch = math.cos(pitch), math.sin(pitch)
    projected_y = cos_pitch * y - sin_pitch * rotated_z
    depth = sin_pitch * y + cos_pitch * rotated_z

    scale = min(width, height) * 0.43
    pixel_x = np.rint(width / 2 + rotated_x * scale).astype(np.intp)
    pixel_y = np.rint(height / 2 - projected_y * scale).astype(np.intp)
    intensity_range = max(1, int(volume.max()) - threshold)
    brightness = np.clip((values - threshold) / intensity_range, 0, 1)
    depth_shade = 0.65 + 0.35 * (depth - depth.min()) / max(1e-6, float(np.ptp(depth)))
    brightness = (90 + 165 * brightness * depth_shade).astype(np.uint8)

    buffer = np.zeros((height, width), dtype=np.uint8)
    for offset_x, offset_y in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
        point_x = pixel_x + offset_x
        point_y = pixel_y + offset_y
        visible = (
            (point_x >= 0)
            & (point_x < width)
            & (point_y >= 0)
            & (point_y < height)
        )
        np.maximum.at(buffer, (point_y[visible], point_x[visible]), brightness[visible])

    rendered = np.zeros((height, width, 3), dtype=np.uint8)
    rendered[:, :, 0] = buffer
    rendered[:, :, 1] = buffer
    rendered[:, :, 2] = buffer
    return Image.fromarray(rendered)