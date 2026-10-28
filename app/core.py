"""Local image geometry and measurements. No network calls."""
from dataclasses import dataclass, field
from pathlib import Path
import io
import json
import math
import numpy as np
import nibabel as nib
import pandas as pd
import psutil
from scipy import ndimage
from skimage.measure import marching_cubes

ROOT = Path(__file__).resolve().parents[1]
MAX_VOXELS = 128_000_000


@dataclass
class Volume:
    data: np.ndarray
    affine: np.ndarray
    name: str
    notes: list = field(default_factory=list)

    @property
    def spacing(self):
        return np.linalg.norm(self.affine[:3, :3], axis=0)

    @property
    def voxel_mm3(self):
        return float(abs(np.linalg.det(self.affine[:3, :3])))


def guard(shape, bytes_per_voxel=4):
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError("A non-empty, three-dimensional image is required.")
    n = math.prod(shape)
    if n > MAX_VOXELS:
        raise ValueError("Image exceeds this prototype's 128-million-voxel limit.")
    if n * max(bytes_per_voxel, 4) * 3 > psutil.virtual_memory().available:
        raise ValueError("Insufficient free RAM for safe loading. Close other applications first.")


def validate(volume, mask=False):
    a = volume.affine
    if a.shape != (4, 4) or not np.all(np.isfinite(a)) or abs(np.linalg.det(a[:3, :3])) < 1e-12:
        raise ValueError("Invalid or singular image geometry.")
    if not np.allclose(a[3], [0, 0, 0, 1]):
        raise ValueError("Invalid affine transformation.")
    if volume.data.ndim != 3 or volume.data.dtype.kind not in "buif":
        raise ValueError("Only real-valued 3D volumes are supported.")
    labels = set()
    for k in range(volume.data.shape[2]):
        slab = volume.data[:, :, k]
        if not np.isfinite(slab).all():
            raise ValueError("Image contains non-finite values.")
        if mask:
            if np.any(slab < 0) or np.any(slab > 65535) or not np.equal(slab, np.rint(slab)).all():
                raise ValueError("Masks must contain non-negative integer label IDs, not scan intensities or probabilities.")
            labels.update(np.unique(slab).tolist())
            if len(labels) > 256:
                raise ValueError("More than 256 labels found. This may be a raw scan, not a mask.")
    if mask:
        volume.data = volume.data.astype(np.uint16, copy=False)
    return volume


def from_sitk(img, name):
    import SimpleITK as sitk
    if img.GetDimension() != 3 or img.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError("Only scalar 3D images are supported.")
    guard(img.GetSize())
    lps_to_ras = np.diag([-1., -1., 1.])
    affine = np.eye(4)
    affine[:3, :3] = lps_to_ras @ np.asarray(img.GetDirection()).reshape(3, 3) @ np.diag(img.GetSpacing())
    affine[:3, 3] = lps_to_ras @ np.asarray(img.GetOrigin())
    data = sitk.GetArrayFromImage(img).transpose(2, 1, 0)
    return Volume(data, affine, name, ["SimpleITK physical coordinates interpreted in millimetres; verify source calibration."])


def load_volume(path, mask=False, assume_mm=False):
    path = Path(path).expanduser()
    if not path.is_file():
        raise ValueError("Image file was not found.")
    lower = path.name.lower()
    if lower.endswith((".nii", ".nii.gz")):
        img = nib.load(str(path))
        guard(img.shape, img.get_data_dtype().itemsize)
        unit = img.header.get_xyzt_units()[0]
        scales = {"mm": 1., "meter": 1000., "micron": .001}
        notes = []
        if unit not in scales:
            if not assume_mm:
                raise ValueError("NIfTI spatial units are unknown. Confirm millimetres explicitly before loading.")
            notes.append("ASSUMED: unknown spatial units interpreted as millimetres by user.")
        if int(img.header['sform_code']) == 0 and int(img.header['qform_code']) == 0:
            raise ValueError("NIfTI lacks a coded spatial transform. Supply an image with verified geometry.")
        affine = np.array(img.affine, dtype=float)
        affine[:3, :] *= scales.get(unit, 1.)
        vol = Volume(np.asanyarray(img.dataobj), affine, path.name, notes)
    elif lower.endswith(".mha"):
        import SimpleITK as sitk
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(path))
        reader.ReadImageInformation()
        guard(reader.GetSize())
        vol = from_sitk(reader.Execute(), path.name)
    else:
        raise ValueError("Use .nii, .nii.gz or self-contained .mha files.")
    return validate(vol, mask)


def dicom_series(folder):
    import SimpleITK as sitk
    if not Path(folder).is_dir():
        raise ValueError("DICOM folder was not found.")
    return list(sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(folder)) or [])


def load_dicom(folder, series_id):
    import SimpleITK as sitk
    if series_id not in dicom_series(folder):
        raise ValueError("Select an available series.")
    files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(folder), series_id)
    if len(files) < 2:
        raise ValueError("This importer requires a conventional multi-file 3D DICOM series.")
    positions = []
    first_orientation = None
    for filename in files:
        reader = sitk.ImageFileReader()
        reader.SetFileName(filename)
        reader.ReadImageInformation()
        if not reader.HasMetaDataKey("0020|0032") or not reader.HasMetaDataKey("0020|0037"):
            raise ValueError("DICOM position/orientation metadata is missing.")
        p = np.array([float(x) for x in reader.GetMetaData("0020|0032").split("\\")])
        orientation = np.array([float(x) for x in reader.GetMetaData("0020|0037").split("\\")])
        if first_orientation is None:
            first_orientation = orientation
            guard((reader.GetSize()[0], reader.GetSize()[1], len(files)))
        elif not np.allclose(orientation, first_orientation, atol=1e-4):
            raise ValueError("Inconsistent DICOM slice orientations.")
        positions.append(p)
    offsets = np.diff(np.array(positions), axis=0)
    normal = np.cross(first_orientation[:3], first_orientation[3:])
    projected = offsets @ normal
    if np.any(np.abs(projected) < 1e-5) or not np.allclose(projected, projected[0], atol=.01, rtol=.01):
        raise ValueError("Duplicate or irregularly spaced DICOM slices; conversion needs specialist review.")
    if np.max(np.linalg.norm(offsets - projected[:, None] * normal, axis=1)) > .01:
        raise ValueError("Tilted DICOM series is not supported by this importer.")
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    vol = validate(from_sitk(reader.Execute(), "DICOM series"))
    vol.notes.append("DICOM metadata is not exported. This is not an anonymization service.")
    return vol


def same_grid(a, b):
    return a.data.shape == b.data.shape and np.allclose(a.affine, b.affine, atol=1e-4, rtol=0)


def label_names(path=None):
    if not path:
        return {}
    obj = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    entries = obj.get("labels", obj)
    if not isinstance(entries, dict):
        raise ValueError("Label definitions must be a JSON object.")
    out = {}
    for key, value in entries.items():
        if isinstance(value, int):
            out[value] = str(key)
        elif str(key).isdigit() and isinstance(value, str):
            out[int(key)] = value
        else:
            raise ValueError("Use name: integer-ID or integer-ID: name label definitions.")
    return out


def analyse(mask, names=None):
    """Bounds include complete voxel cells in the source image RAS frame."""
    names = names or {}
    rows = []
    ids, counts = np.unique(mask.data, return_counts=True)
    half_cell = .5 * np.abs(mask.affine[:3, :3]).sum(axis=1)
    for label, count in zip(ids, counts):
        if label == 0:
            continue
        lo, hi, sums = np.full(3, np.inf), np.full(3, -np.inf), np.zeros(3)
        touching = False
        # Slab processing avoids a large full-volume coordinate array.
        for k in range(mask.data.shape[2]):
            x, y = np.where(mask.data[:, :, k] == label)
            if len(x) == 0:
                continue
            xyz = np.column_stack((x, y, np.full(len(x), k)))
            world = nib.affines.apply_affine(mask.affine, xyz)
            lo = np.minimum(lo, world.min(axis=0))
            hi = np.maximum(hi, world.max(axis=0))
            sums += world.sum(axis=0)
            touching |= bool(k in (0, mask.data.shape[2] - 1) or (x == 0).any() or (y == 0).any() or (x == mask.data.shape[0]-1).any() or (y == mask.data.shape[1]-1).any())
        extents = hi - lo + 2 * half_cell
        centroid = sums / count
        rows.append({"label_id": int(label), "structure": names.get(int(label), f"Label {label}"),
                     "voxels": int(count), "volume_mm3": float(count * mask.voxel_mm3),
                     "extent_R_mm": float(extents[0]), "extent_A_mm": float(extents[1]), "extent_S_mm": float(extents[2]),
                     "centroid_R_mm": float(centroid[0]), "centroid_A_mm": float(centroid[1]), "centroid_S_mm": float(centroid[2]),
                     "touches_image_boundary": touching})
    return pd.DataFrame(rows)


def mesh(mask, label, step=1):
    binary = mask.data == label
    coords = np.nonzero(binary)
    if len(coords[0]) == 0:
        raise ValueError("Selected label is absent.")
    lo = np.array([c.min() for c in coords])
    hi = np.array([c.max() for c in coords]) + 1
    del coords
    cut = binary[tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))]
    padded = np.pad(cut, 1).astype(np.uint8)
    vertices, faces, _, _ = marching_cubes(padded, level=.5, step_size=step, allow_degenerate=False)
    vertices = nib.affines.apply_affine(mask.affine, vertices + lo - 1)
    return vertices, faces


def obj_bytes(vertices, faces):
    lines = ["# Research surface in millimetres, source RAS coordinates"]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices]
    lines += [f"f {a+1} {b+1} {c+1}" for a, b, c in faces]
    return ("\n".join(lines) + "\n").encode()


def nifti_bytes(volume):
    img = nib.Nifti1Image(volume.data, volume.affine)
    img.header.set_xyzt_units("mm")
    img.set_sform(volume.affine, code=1)
    return img.to_bytes()


def compare(predicted, reference, names=None):
    if not same_grid(predicted, reference):
        raise ValueError("Reference and prediction must have matching shape and physical geometry. No automatic resampling is performed.")
    names = names or {}
    rows = []
    ids = np.union1d(np.unique(predicted.data), np.unique(reference.data))
    for label in ids:
        if label == 0:
            continue
        p, r = predicted.data == label, reference.data == label
        np_, nr = int(p.sum()), int(r.sum())
        intersection = int(np.count_nonzero(p & r))
        rows.append({"label_id": int(label), "structure": names.get(int(label), f"Label {label}"),
                     "dice": 2 * intersection / (np_ + nr),
                     "iou": intersection / (np_ + nr - intersection),
                     "predicted_volume_mm3": np_ * predicted.voxel_mm3,
                     "reference_volume_mm3": nr * reference.voxel_mm3,
                     "signed_volume_error_mm3": (np_ - nr) * reference.voxel_mm3,
                     "relative_volume_error_percent": 100 * (np_ - nr) / nr if nr else None})
    return pd.DataFrame(rows)


def demo():
    """Illustrative ellipsoids only: not teeth, training data, or validation data."""
    shape = (112, 96, 72)
    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    masks = np.zeros(shape, np.uint16)
    names = {}
    for label, angle in enumerate(np.linspace(.18, np.pi-.18, 10), start=1):
        cx, cy = 56 + 36*np.cos(angle), 28 + 34*np.sin(angle)
        body = ((x-cx)/5.2)**2 + ((y-cy)/6.0)**2 + ((z-36)/17.)**2 < 1
        masks[body] = label
        names[label] = f"Example object {label}"
    rng = np.random.default_rng(2026)
    scan = (rng.normal(0, 12, shape) + (masks > 0) * 650).astype(np.float32)
    affine = np.diag([.4, .4, .6, 1.])
    affine[:3, 3] = [-22.4, -19.2, -21.6]
    note = ["SIMULATED geometric demonstration. Ellipsoids and Gaussian intensity noise; no patient anatomy or medical accuracy claim."]
    return Volume(masks, affine, "Synthetic demonstration", note), Volume(scan, affine, "Synthetic intensity image", note), names
