"""Physical geometry and explicit research QA; no diagnostic inference."""
import math
import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
from core import same_grid, guard


METHODS = {
    'classification': 'CALCULATED from supplied image and segmentation; not clinical validation',
    'surface_area_mm2': 'Full-resolution marching cubes at level 0.5, padded crop, full affine to RAS mm; sum of triangle areas. An estimate of the segmentation surface.',
    'voxel_face_area_mm2': 'Exposed voxel faces including image edges; each face area is the norm of the cross product of its affine basis vectors. Staircase surface, distinct from a smooth anatomical surface.',
    'components': '6-neighbour connectivity on the source voxel grid; fragmentation is a review flag, not a diagnosis.',
    'intensity': 'Min, max, mean and population standard deviation use all selected voxels. Median and histogram use a deterministic sample capped at 200,000. Values are source units, not assumed HU.',
    'pair_distance': 'Closest 6-connected boundary voxel centres in physical RAS mm, using exact nearest-neighbour search. This is not the continuous surface gap or anatomical clearance.',
    'limitations': 'Calibration, voxel resolution and segmentation errors limit accuracy. Edge contact can truncate a structure. Model accuracy requires independent reference data.'
}


def intensity_summary(data, selected=None):
    count = int(np.count_nonzero(selected)) if selected is not None else int(data.size)
    if not count:
        return None
    stride = max(1, math.ceil(count / 200_000))
    n = 0; mean = 0.; m2 = 0.; low = math.inf; high = -math.inf; samples = []
    for k in range(data.shape[2]):
        slab = data[:, :, k]
        values = np.asarray(slab[selected[:, :, k]] if selected is not None else slab.ravel(), dtype=np.float64)
        if not values.size:
            continue
        size = values.size; batch_mean = float(values.mean()); delta = batch_mean - mean
        m2 += float(np.sum((values - batch_mean) ** 2)) + delta * delta * n * size / (n + size)
        mean += delta * size / (n + size)
        low = min(low, float(values.min())); high = max(high, float(values.max()))
        samples.append(values[(-n) % stride::stride]); n += size
    sample = np.concatenate(samples)
    hist, edges = np.histogram(sample, bins=48)
    return {'count': n, 'min': low, 'max': high, 'mean': mean, 'std': float(math.sqrt(max(0., m2/n))),
            'median_sampled': float(np.median(sample)), 'sample_count': int(sample.size),
            'histogram': {'counts': hist.tolist(), 'edges': edges.tolist()}, 'units': 'source intensity units (not assumed HU)'}


def crop_label(mask, label):
    if not 1 <= label <= 65535:
        raise ValueError('Choose a non-background structure.')
    lo = np.array(mask.data.shape); hi = np.zeros(3, dtype=int)
    for k in range(mask.data.shape[2]):
        x, y = np.where(mask.data[:, :, k] == label)
        if len(x):
            lo = np.minimum(lo, [x.min(), y.min(), k]); hi = np.maximum(hi, [x.max()+1, y.max()+1, k+1])
    if np.any(hi <= lo):
        raise ValueError('Selected label is absent.')
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    return mask.data[sl] == label, lo, sl


def shape_metrics(binary, affine):
    guard(tuple(n+2 for n in binary.shape), bytes_per_voxel=12)
    padded = np.pad(binary, 1).astype(np.uint8)
    v, f, _, _ = marching_cubes(padded, .5, allow_degenerate=False)
    # Translation does not affect area. Use the full linear transform, including shear.
    v = v @ affine[:3, :3].T
    triangles = v[f]
    area = float(np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0]), axis=1).sum()/2)
    face_area = 0.
    for axis in range(3):
        other = [a for a in range(3) if a != axis]
        area_per_face = np.linalg.norm(np.cross(affine[:3,other[0]], affine[:3,other[1]]))
        face_area += int(np.count_nonzero(np.diff(padded.astype(np.int8), axis=axis))) * area_per_face
    components, n = ndimage.label(binary, structure=ndimage.generate_binary_structure(3,1))
    sizes = np.bincount(components.ravel())[1:]
    return {'surface_area_mm2': area, 'voxel_face_area_mm2': float(face_area), 'components': int(n),
            'largest_component_percent': float(100 * sizes.max() / sizes.sum()),
            'detached_voxels': int(sizes.sum() - sizes.max()), 'surface_triangles': int(len(f))}


def analyse_case(mask, scan, rows, progress=None):
    if mask is not None and scan is not None and not same_grid(mask,scan):
        raise ValueError('Scan and mask must share a physical grid.')
    output = []
    # A single bounding-box pass avoids scanning the whole volume for every label.
    boxes = ndimage.find_objects(mask.data) if mask is not None else []
    for i, source in enumerate(rows):
        label = source['label_id']; sl = boxes[label-1]
        binary = mask.data[sl] == label
        row = dict(source); row.update(shape_metrics(binary,mask.affine))
        row['intensity'] = intensity_summary(scan.data[sl],binary) if scan is not None else None
        row['review_flags'] = ([f"{row['components']} disconnected components"] if row['components'] > 1 else [])
        if row['touches_image_boundary']:
            row['review_flags'].append('Touches image boundary; may be truncated')
        output.append(row)
        if progress: progress(i+1,len(rows))
    return {'methods': METHODS, 'structures': output, 'scan_intensity': intensity_summary(scan.data) if scan is not None else None,
            'summary': {'structures': len(output), 'total_volume_mm3': sum(r['volume_mm3'] for r in output),
                        'flagged_structures': sum(bool(r['review_flags']) for r in output),
                        'fragmented_structures': sum(r['components'] > 1 for r in output),
                        'boundary_structures': sum(r['touches_image_boundary'] for r in output)}}


def boundary_points(mask, label):
    binary, lo, _ = crop_label(mask,label)
    edge = binary & ~ndimage.binary_erosion(binary, structure=ndimage.generate_binary_structure(3,1), border_value=0)
    count = np.count_nonzero(edge)
    if count > 1_000_000:
        raise ValueError('Boundary exceeds the one-million-point resource limit. No approximate distance was substituted.')
    return nib.affines.apply_affine(mask.affine, np.argwhere(edge) + lo)


def pair_distance(mask, first, second):
    if first == second: raise ValueError('Choose two different structures.')
    a, b = boundary_points(mask,first), boundary_points(mask,second)
    distances, indices = cKDTree(b).query(a, k=1, workers=1)
    i = int(distances.argmin())
    return {'first_label': first, 'second_label': second, 'boundary_centres_distance_mm': float(distances[i]),
            'points_ras_mm': [a[i].tolist(), b[int(indices[i])].tolist()], 'method': METHODS['pair_distance']}


def surface_agreement(predicted, reference, label):
    if not same_grid(predicted,reference): raise ValueError('Physical grids must match.')
    if not np.any(predicted.data == label) or not np.any(reference.data == label):
        return {'surface_hd95_mm': None, 'surface_mean_distance_mm': None, 'surface_status': 'Undefined: label absent from one mask'}
    a, b = boundary_points(predicted,label), boundary_points(reference,label)
    ab = cKDTree(b).query(a,k=1,workers=1)[0]; ba = cKDTree(a).query(b,k=1,workers=1)[0]
    return {'surface_hd95_mm': float(max(np.percentile(ab,95),np.percentile(ba,95))),
            'surface_mean_distance_mm': float((ab.sum()+ba.sum())/(len(ab)+len(ba))),
            'surface_status': 'Boundary voxel centres; HD95 is max of directed 95th percentiles'}
