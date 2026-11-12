from __future__ import annotations

import argparse
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from .render3d import render_surface_preview
from .scv import Axis, SCVVolume

DEFAULT_VOLUME = Path(
    r"C:\Users\U7JIZHOU\CT_V6X\ZADDTraining\Alu_Demopart_Medium_148µm.uint16_scv"
)
VIEW_NAMES: tuple[tuple[Axis, str], ...] = (
    ("axial", "AXIAL · Z"),
    ("coronal", "CORONAL · Y"),
    ("sagittal", "SAGITTAL · X"),
)
VIEW_POSITIONS = {"axial": (2, 0), "coronal": (2, 1), "sagittal": (3, 0)}


class CTVolumeViewer:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("CT Volume Viewer")
        self.root.geometry("1440x880")
        self.root.minsize(1000, 680)
        self.root.configure(bg="#171b1e")
        self.volume: SCVVolume | None = None
        self.canvases: dict[Axis, tk.Canvas] = {}
        self.scales: dict[Axis, ttk.Scale] = {}
        self.slice_values: dict[Axis, tk.IntVar] = {}
        self.slice_labels: dict[Axis, tk.StringVar] = {}
        self.photos: dict[Axis, ImageTk.PhotoImage] = {}
        self.surface_photo: ImageTk.PhotoImage | None = None
        self.preview_volume: np.ndarray | None = None
        self.threshold_value = tk.IntVar(value=0)
        self.threshold_label = tk.StringVar(value="Threshold: -")
        self.threshold_scale: ttk.Scale | None = None
        self.render3d_after: str | None = None
        self.rotation_yaw = -0.65
        self.rotation_pitch = -0.3
        self.drag_origin: tuple[int, int] | None = None
        self.window_center = tk.StringVar(value="32768")
        self.window_width = tk.StringVar(value="65536")
        self.metadata = tk.StringVar(value="No volume loaded")
        self.status = tk.StringVar(value="Open a 16-bit SCV volume to begin")

        self._configure_style()
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background="#171b1e")
        style.configure("Panel.TFrame", background="#202629")
        style.configure("TLabel", background="#171b1e", foreground="#e7ecee")
        style.configure("Panel.TLabel", background="#202629", foreground="#e7ecee")
        style.configure("Muted.TLabel", foreground="#9eaaae")
        style.configure("TButton", padding=(12, 7))
        style.configure("TScale", background="#202629", troughcolor="#384246")

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=18)
        container.pack(fill="both", expand=True)
        container.columnconfigure((0, 1), weight=1, uniform="views")
        container.rowconfigure((2, 3), weight=1, uniform="views")

        heading = ttk.Frame(container)
        heading.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 14))
        ttk.Label(heading, text="CT VOLUME VIEWER", font=("Segoe UI", 18, "bold")).pack(side="left")
        ttk.Button(heading, text="Open volume...", command=self._choose_volume).pack(side="right")

        ttk.Label(container, textvariable=self.metadata, style="Muted.TLabel").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(0, 12)
        )

        for axis, title in VIEW_NAMES:
            row, view_column = VIEW_POSITIONS[axis]
            self._build_view(container, row, view_column, axis, title)
        self._build_3d_view(container)

        controls = ttk.Frame(container, style="Panel.TFrame", padding=(14, 10))
        controls.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(14, 0))
        ttk.Label(controls, text="WINDOW / LEVEL", style="Panel.TLabel", font=("Segoe UI", 9, "bold")).pack(side="left")
        ttk.Label(controls, text="Center", style="Panel.TLabel").pack(side="left", padx=(18, 5))
        ttk.Spinbox(controls, from_=-65535, to=131070, textvariable=self.window_center, width=9).pack(side="left")
        ttk.Label(controls, text="Width", style="Panel.TLabel").pack(side="left", padx=(12, 5))
        ttk.Spinbox(controls, from_=1, to=131070, textvariable=self.window_width, width=9).pack(side="left")
        ttk.Button(controls, text="Apply", command=self._apply_window).pack(side="left", padx=(8, 0))
        ttk.Button(controls, text="Auto", command=self._auto_window).pack(side="left", padx=6)
        ttk.Label(controls, textvariable=self.status, style="Muted.TLabel").pack(side="right")

    def _build_view(
        self, parent: ttk.Frame, row: int, column: int, axis: Axis, title: str
    ) -> None:
        panel = ttk.Frame(parent, style="Panel.TFrame", padding=10)
        panel.grid(row=row, column=column, sticky="nsew", padx=(0 if column == 0 else 6, 6 if column == 0 else 0), pady=(0, 6))
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=1)

        title_row = ttk.Frame(panel, style="Panel.TFrame")
        title_row.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(title_row, text=title, style="Panel.TLabel", font=("Segoe UI", 10, "bold")).pack(side="left")
        slice_label = tk.StringVar(value="- / -")
        self.slice_labels[axis] = slice_label
        ttk.Label(title_row, textvariable=slice_label, style="Panel.TLabel").pack(side="right")

        canvas = tk.Canvas(panel, background="#0d1012", highlightthickness=0, height=300)
        canvas.grid(row=1, column=0, sticky="nsew")
        canvas.bind("<Configure>", lambda _, view=axis: self._render_view(view))
        canvas.bind("<MouseWheel>", lambda event, view=axis: self._scroll_slice(view, event))
        self.canvases[axis] = canvas

        slice_value = tk.IntVar(value=0)
        self.slice_values[axis] = slice_value
        scale = ttk.Scale(
            panel,
            from_=0,
            to=0,
            orient="horizontal",
            command=lambda value, view=axis: self._on_slice_change(view, value),
        )
        scale.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        scale.configure(variable=slice_value)
        self.scales[axis] = scale

    def _build_3d_view(self, parent: ttk.Frame) -> None:
        panel = ttk.Frame(parent, style="Panel.TFrame", padding=10)
        panel.grid(row=3, column=1, sticky="nsew", padx=(6, 0), pady=(0, 6))
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=1)

        title_row = ttk.Frame(panel, style="Panel.TFrame")
        title_row.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(
            title_row, text="3D SURFACE", style="Panel.TLabel", font=("Segoe UI", 10, "bold")
        ).pack(side="left")
        ttk.Label(title_row, textvariable=self.threshold_label, style="Panel.TLabel").pack(
            side="left", padx=(12, 0)
        )
        ttk.Button(title_row, text="Reset", command=self._reset_3d_view).pack(side="right")

        canvas = tk.Canvas(panel, background="#0d1012", highlightthickness=0, height=300)
        canvas.grid(row=1, column=0, sticky="nsew")
        canvas.bind("<Configure>", lambda _: self._schedule_3d_render())
        canvas.bind("<ButtonPress-1>", self._start_3d_drag)
        canvas.bind("<B1-Motion>", self._drag_3d)
        canvas.bind("<MouseWheel>", self._scroll_3d_threshold)
        self.surface_canvas = canvas

        self.threshold_scale = ttk.Scale(
            panel,
            from_=0,
            to=65535,
            orient="horizontal",
            variable=self.threshold_value,
            command=self._on_threshold_change,
        )
        self.threshold_scale.grid(row=2, column=0, sticky="ew", pady=(8, 0))

    def _choose_volume(self) -> None:
        initial_dir = str(DEFAULT_VOLUME.parent) if DEFAULT_VOLUME.parent.is_dir() else str(Path.home())
        path = filedialog.askopenfilename(
            title="Open 16-bit SCV volume",
            initialdir=initial_dir,
            filetypes=(("SCV volumes", "*.scv *.uint16_scv"), ("All files", "*.*")),
        )
        if path:
            self._open_volume(Path(path))

    def _open_volume(self, path: Path) -> None:
        try:
            candidate = SCVVolume(path)
        except (OSError, ValueError, MemoryError) as error:
            messagebox.showerror("Open volume", str(error), parent=self.root)
            self.status.set("Could not open volume")
            return

        previous = self.volume
        self.volume = candidate
        self.preview_volume = candidate.data[::6, ::6, ::6]
        if previous is not None:
            previous.close()

        x_size, y_size, z_size = candidate.header.dimensions
        ranges = {"axial": z_size, "coronal": y_size, "sagittal": x_size}
        for axis, _ in VIEW_NAMES:
            middle = ranges[axis] // 2
            self.slice_values[axis].set(middle)
            scale = self.scales[axis]
            scale.configure(to=ranges[axis] - 1)
            scale.set(middle)

        sx, sy, sz = candidate.header.spacing_mm
        self.metadata.set(
            f"{path.name}    ·    {x_size} × {y_size} × {z_size} voxels    ·    "
            f"{sx:.5f} × {sy:.5f} × {sz:.5f} mm/voxel    ·    memory mapped"
        )
        threshold = int(np.percentile(self.preview_volume, 90))
        self.threshold_value.set(threshold)
        self.threshold_label.set(f"Threshold: {threshold:,}")
        self.threshold_scale.set(threshold)
        self.status.set("Volume ready")
        self._auto_window()
        for axis, _ in VIEW_NAMES:
            self._render_view(axis)
        self._schedule_3d_render()

    def _on_slice_change(self, axis: Axis, value: str) -> None:
        if self.volume is None:
            return
        index = int(float(value))
        self.slice_values[axis].set(index)
        self._render_view(axis)

    def _scroll_slice(self, axis: Axis, event: tk.Event) -> str:
        if self.volume is not None:
            direction = -1 if event.delta > 0 else 1
            data_shape = self.volume.data.shape
            size = {"axial": data_shape[0], "coronal": data_shape[1], "sagittal": data_shape[2]}[axis]
            value = max(0, min(size - 1, self.slice_values[axis].get() + direction))
            self.slice_values[axis].set(value)
            scale = self.scales[axis]
            scale.set(value)
        return "break"

    def _apply_window(self) -> None:
        try:
            width = int(self.window_width.get())
            int(self.window_center.get())
            if width < 1:
                raise ValueError("Window width must be at least 1")
        except ValueError as error:
            messagebox.showerror("Window / level", str(error), parent=self.root)
            return
        for axis, _ in VIEW_NAMES:
            self._render_view(axis)

    def _auto_window(self) -> None:
        if self.volume is None:
            return
        samples = [
            self.volume.get_slice("axial", self.slice_values["axial"].get()).ravel(),
            self.volume.get_slice("coronal", self.slice_values["coronal"].get()).ravel(),
            self.volume.get_slice("sagittal", self.slice_values["sagittal"].get()).ravel(),
        ]
        low, high = np.percentile(np.concatenate(samples), (1, 99))
        width = max(1, int(round(high - low)))
        center = int(round((high + low) / 2))
        self.window_center.set(str(center))
        self.window_width.set(str(width))
        for axis, _ in VIEW_NAMES:
            self._render_view(axis)

    def _render_view(self, axis: Axis) -> None:
        if self.volume is None:
            return
        canvas = self.canvases[axis]
        canvas_width, canvas_height = canvas.winfo_width(), canvas.winfo_height()
        if canvas_width < 2 or canvas_height < 2:
            return

        index = self.slice_values[axis].get()
        slice_data = np.flipud(self.volume.get_slice(axis, index))
        center = int(self.window_center.get())
        width = max(1, int(self.window_width.get()))
        low = center - width / 2
        pixels = np.clip((slice_data.astype(np.float32) - low) * (255.0 / width), 0, 255).astype(np.uint8)

        image = Image.fromarray(pixels)
        scale = min(canvas_width / image.width, canvas_height / image.height)
        target_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.photos[axis] = photo
        canvas.delete("all")
        canvas.create_image(canvas_width // 2, canvas_height // 2, image=photo, anchor="center")

        limits = {"axial": self.volume.data.shape[0], "coronal": self.volume.data.shape[1], "sagittal": self.volume.data.shape[2]}
        self.slice_labels[axis].set(f"{index + 1} / {limits[axis]}")

    def _on_threshold_change(self, value: str) -> None:
        threshold = int(float(value))
        self.threshold_value.set(threshold)
        self.threshold_label.set(f"Threshold: {threshold:,}")
        self._schedule_3d_render()

    def _schedule_3d_render(self) -> None:
        if self.render3d_after is None:
            self.render3d_after = self.root.after(40, self._render_3d_view)

    def _render_3d_view(self) -> None:
        self.render3d_after = None
        width, height = self.surface_canvas.winfo_width(), self.surface_canvas.winfo_height()
        if width < 2 or height < 2:
            return
        if self.preview_volume is None:
            image = Image.new("RGB", (width, height), "#0d1012")
        else:
            image = render_surface_preview(
                self.preview_volume,
                self.threshold_value.get(),
                self.rotation_yaw,
                self.rotation_pitch,
                width,
                height,
            )
        photo = ImageTk.PhotoImage(image)
        self.surface_photo = photo
        self.surface_canvas.delete("all")
        self.surface_canvas.create_image(0, 0, image=photo, anchor="nw")
        if self.preview_volume is None:
            self.surface_canvas.create_text(
                width // 2,
                height // 2,
                text="Open a volume to see its 3D surface",
                fill="#9eaaae",
                font=("Segoe UI", 10),
            )

    def _start_3d_drag(self, event: tk.Event) -> None:
        self.drag_origin = (event.x, event.y)

    def _drag_3d(self, event: tk.Event) -> None:
        if self.drag_origin is None:
            return
        previous_x, previous_y = self.drag_origin
        self.rotation_yaw += (event.x - previous_x) * 0.01
        self.rotation_pitch = float(
            np.clip(self.rotation_pitch + (event.y - previous_y) * 0.01, -1.5, 1.5)
        )
        self.drag_origin = (event.x, event.y)
        self._schedule_3d_render()

    def _scroll_3d_threshold(self, event: tk.Event) -> str:
        direction = 1 if event.delta > 0 else -1
        threshold = max(0, min(65535, self.threshold_value.get() + direction * 500))
        self.threshold_value.set(threshold)
        self._on_threshold_change(str(threshold))
        return "break"

    def _reset_3d_view(self) -> None:
        self.rotation_yaw = -0.65
        self.rotation_pitch = -0.3
        self._schedule_3d_render()

    def _close(self) -> None:
        if self.volume is not None:
            self.volume.close()
        self.preview_volume = None
        self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description="View 16-bit SCV CT volumes")
    parser.add_argument("volume", nargs="?", type=Path, help="Path to a uint16 SCV volume")
    args = parser.parse_args()

    root = tk.Tk()
    app = CTVolumeViewer(root)
    path = args.volume or DEFAULT_VOLUME
    if path.is_file():
        root.after(100, lambda: app._open_volume(path))
    root.mainloop()


if __name__ == "__main__":
    main()