# CT Volume Viewer

A desktop viewer for little-endian, 16-bit SCV CT volumes. The volume is opened as a read-only memory map, so the image data is not copied into RAM at load time.

The viewer opens the supplied file by default:

```text
C:\Users\U7JIZHOU\CT_V6X\ZADDTraining\Alu_Demopart_Medium_148µm.uint16_scv
```

It shows axial, coronal, and sagittal slices plus a thresholded 3D surface preview. Use each slice slider or the mouse wheel over a slice view to move through the volume. Drag the 3D preview to rotate it, use its slider or mouse wheel to change the surface threshold, and select **Reset** to restore the initial orientation. Window center and width adjust grayscale contrast; **Auto** estimates a useful range from the three current slices.

## Setup

In PowerShell, from this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Launch the default volume:

```powershell
ct-volume-viewer
```

Or open a specific file:

```powershell
ct-volume-viewer "C:\path\to\volume.uint16_scv"
```

## SCV layout supported

The loader reads a 1,024-byte header whose first six little-endian unsigned integers are header size, reserved value, bit depth, X size, Y size, and Z size. Three little-endian 64-bit floats at byte offset 24 contain voxel spacing in millimeters. The remainder is contiguous little-endian `uint16` data with X changing fastest. Files with another bit depth or a size inconsistent with the header are rejected.

The supplied volume was verified to have dimensions `526 × 879 × 883`, spacing `0.14844056 mm` per axis, and an exact on-disk size of 816,518,188 bytes.

## Tests

```powershell
python -m unittest discover -s tests -v
```