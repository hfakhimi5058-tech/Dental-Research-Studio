"""Installed files and runtime availability, distinct from a research catalogue."""
import importlib.util
from pathlib import Path
from pipeline import model_info


def discover_models(root):
    root = Path(root)
    models, issues = [], []
    composite = root/'models'/'toothseg'
    if (composite/'pipeline.json').is_file():
        try:
            dataset, folds = model_info(composite)
            models.append({'path':str(composite), 'name':'ToothSeg · permanent teeth · public pretrained',
                           'folds':folds, 'labels':dataset['labels'], 'checkpoint':'checkpoint_final.pth',
                           'source':'https://doi.org/10.5281/zenodo.14893540', 'recommended':True})
        except (ValueError, OSError, KeyError, TypeError) as exc:
            issues.append({'model':'ToothSeg', 'message':str(exc)})
    for p in sorted((root/'models').rglob('plans.json')):
        if p.is_relative_to(composite):
            continue
        try:
            dataset, folds = model_info(p.parent)
            checkpoint = next((n for n in ('checkpoint_final.pth','checkpoint_best.pth','checkpoint_latest.pth')
                               if all((p.parent/f'fold_{f}'/n).is_file() for f in folds)), None)
            if not checkpoint:
                raise ValueError('No common checkpoint across the discovered folds.')
            models.append({'path':str(p.parent), 'name':p.parent.name, 'folds':folds,
                           'labels':dataset['labels'], 'checkpoint':checkpoint, 'recommended':False})
        except (ValueError, OSError, KeyError, TypeError) as exc:
            issues.append({'model':p.parent.name, 'message':str(exc)})
    return models, issues


def status(root):
    models, issues = discover_models(root)
    runtime = all(importlib.util.find_spec(name) is not None for name in ('torch','nnunetv2'))
    toothseg = any(m.get('recommended') for m in models)
    return {'models':models, 'issues':issues, 'runtime_installed':runtime, 'toothseg_weights':toothseg,
            'toothseg_status':('Files installed; hardware checked when run' if runtime and toothseg else
                               'Install segmentation runtime' if toothseg else 'Model weights not installed'),
            'pulpy3d':{'annotation_import':True, 'automatic_segmentation':False,
                       'status':'Annotation import available; no integrated pulp prediction model'},
            'notice':'Dataset entries describe research sources. They do not mean a dataset, model or anatomy is installed or trained on this computer.'}
