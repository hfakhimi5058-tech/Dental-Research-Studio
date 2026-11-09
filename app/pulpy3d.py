"""Read Pulpy3D annotations without treating them as predictions or whole teeth.

Format evidence: https://github.com/mahmoudgamal0/Pulpy3D
datasets/Pulpy3D/dataset.json and dataloader/Transformation.py define raw FDI
and compact training IDs. No upstream executable code or weights are bundled.
"""
from pathlib import Path
import numpy as np
from core import load_volume, same_grid, label_names

SOURCE = 'https://github.com/mahmoudgamal0/Pulpy3D'
VARIANTS = {
    'gt_instance_ian': 'Individual mandibular pulp cavities + inferior alveolar nerve',
    'gt_instance': 'Individual pulp cavities',
    'gt_pulp_mandible': 'Mandibular pulp mask',
    'gt_pulp': 'Pulp mask',
    'gt_ian': 'Inferior alveolar nerve mask',
}
FDI = list(range(31, 39)) + list(range(41, 49))


def _nifti(folder, stem):
    return next((p for ext in ('.nii.gz', '.nii') if (p := folder / (stem + ext)).is_file()), None)


def inspect_folder(folder):
    """Bounded, one-level inventory; does not decompress images or copy data."""
    if not str(folder).strip():
        raise ValueError('Enter the extracted Pulpy3D folder first.')
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise ValueError('Choose an extracted Pulpy3D folder or a single patient folder.')
    candidates = [root] if _nifti(root, 'data') else sorted(p for p in root.iterdir() if p.is_dir())
    cases = []
    for patient in candidates:
        scan = _nifti(patient, 'data')
        if not scan:
            continue
        masks = [{'key':key, 'name':name, 'path':str(path)} for key,name in VARIANTS.items()
                 if (path := _nifti(patient, key))]
        if masks:
            cases.append({'id':patient.name, 'scan':str(scan), 'masks':masks, 'layout':'patient'})
    # Also accept an explicitly paired, single-channel nnU-Net-style release.
    if not cases and (root / 'imagesTr').is_dir() and (root / 'labelsTr').is_dir():
        for mask in sorted((root / 'labelsTr').iterdir()):
            if not mask.name.endswith(('.nii.gz', '.nii')):
                continue
            stem = mask.name.removesuffix('.gz').removesuffix('.nii')
            scan = _nifti(root / 'imagesTr', stem + '_0000') or _nifti(root / 'imagesTr', stem)
            if scan:
                cases.append({'id':stem, 'scan':str(scan), 'layout':'nnunet',
                              'masks':[{'key':'release', 'name':'Release label mask', 'path':str(mask)}]})
    if not cases:
        raise ValueError('No paired Pulpy3D cases found. Use a folder containing patient folders with data.nii.gz and gt_*.nii.gz, or imagesTr + labelsTr. An archive must be extracted first.')
    metadata = next((p for p in (root/'dataset.json', root.parent/'dataset.json') if p.is_file()), None)
    return {'folder':str(root), 'cases':cases, 'metadata':str(metadata) if metadata else '',
            'note':'Imported dataset annotations, not a new model prediction. Annotation accuracy and manual versus generated origin require source review.'}


def names_for(variant, encoding, metadata=None):
    if encoding == 'custom':
        if not metadata:
            raise ValueError('Select the matching label-definitions JSON for this release.')
        return label_names(metadata)
    if encoding == 'binary':
        if variant not in ('gt_pulp', 'gt_pulp_mandible', 'gt_ian'):
            raise ValueError('Binary mapping is only for a pulp or nerve mask, not an instance mask.')
        return {0:'Background', 1:'Inferior alveolar nerve' if variant == 'gt_ian' else 'Pulp cavity (all foreground)'}
    if encoding not in ('fdi', 'compact') or variant not in ('gt_instance', 'gt_instance_ian', 'release'):
        raise ValueError('Choose the label encoding that matches the annotation variant.')
    names = {0:'Background'}
    if variant != 'gt_instance':
        names[1] = 'Inferior alveolar nerve'
    for index, fdi in enumerate(FDI, start=2):
        names[fdi if encoding == 'fdi' else index] = f'Pulp cavity · FDI {fdi}'
    names[50 if encoding == 'fdi' else 18] = 'Pulp · anomaly label (source definition)'
    return names


def load_annotation(data):
    if data.get('mapping_confirmed') is not True:
        raise ValueError('Review and confirm the annotation variant and label encoding first.')
    inventory = inspect_folder(data.get('folder', ''))
    case = next((c for c in inventory['cases'] if c['id'] == data.get('patient')), None)
    if case is None:
        raise ValueError('Patient is no longer present; inspect the dataset folder again.')
    selected = next((m for m in case['masks'] if m['key'] == data.get('variant')), None)
    if selected is None:
        raise ValueError('The selected annotation file is not available for this patient.')
    names = names_for(selected['key'], data.get('encoding'), data.get('labels'))
    mask = load_volume(selected['path'], mask=True, assume_mm=data.get('assume_mm', False))
    ids = {int(x) for x in np.unique(mask.data)}
    if not ids.issubset(names):
        raise ValueError('Mask IDs are not defined by this encoding: ' + ', '.join(map(str, sorted(ids-set(names)))) + '. Choose the correct encoding or matching custom JSON; labels were not remapped.')
    if not ids - {0}:
        raise ValueError('This annotation contains no foreground structures.')
    scan = load_volume(case['scan'], assume_mm=data.get('assume_mm', False))
    if not same_grid(mask, scan):
        raise ValueError('Pulpy3D scan and annotation have different physical grids; no registration or resampling was applied.')
    details = {'dataset':'Pulpy3D', 'patient':case['id'], 'variant':selected['key'],
               'encoding':data['encoding'], 'kind':'imported_dataset_annotation', 'source_url':SOURCE,
               'annotation_origin':'Source annotation; manual versus model-generated origin not verified',
               'family':'maxillo_toothfairy',
               'limitations':'Pulp cavities are not whole-tooth segments. Review calibration, source annotations and shared ToothFairy patients before evaluation or training.'}
    source = {'scan':case['scan'], 'mask':selected['path'], 'assume_mm':data.get('assume_mm',False),
              'label_definitions':{str(k):v for k,v in names.items()}, 'dataset_import':details}
    return mask, scan, names, source, data.get('title') or f"Pulpy3D {case['id']} · {selected['name']}"
