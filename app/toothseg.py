"""Local dual-branch ToothSeg inference with attributed, safely loaded weights.

Follows MIC-DKFZ/ToothSeg's ToothFairy2 script (majority tooth assignment).
The supplied source and Apache-2.0 license are preserved in models/pretrained.
Model weights are CC-BY-4.0: https://doi.org/10.5281/zenodo.14893540.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'models' / 'toothseg'
UPSTREAM = ROOT / 'models' / 'pretrained' / 'ToothSeg' / 'ToothSeg-main'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def safe_checkpoint(path):
    import numpy as np
    import torch
    allowed = {'numpy.dtype', 'numpy.core.multiarray.scalar', 'numpy._core.multiarray.scalar'}
    unknown = set(torch.serialization.get_unsafe_globals_in_checkpoint(path)) - allowed
    if unknown:
        raise ValueError(f'Unsupported checkpoint objects: {sorted(unknown)}')
    # Only NumPy numeric scalar/dtype constructors are admitted, never model code.
    types = [np.dtype, (np._core.multiarray.scalar, 'numpy.core.multiarray.scalar'),
             (np._core.multiarray.scalar, 'numpy._core.multiarray.scalar')]
    types += [type(np.dtype(t)) for t in (np.float64, np.float32, np.int64, np.int32, np.bool_)]
    with torch.serialization.safe_globals(types):
        return torch.load(path, map_location='cpu', weights_only=True, mmap=True)


def predict_branch(branch, source, output):
    import numpy as np
    import psutil
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.inference.export_prediction import export_prediction_from_logits
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('This configured pipeline requires a CUDA GPU.')
    free_gpu, _ = torch.cuda.mem_get_info()
    if free_gpu < 8 * 2**30 or psutil.virtual_memory().available < 3 * 2**30:
        raise RuntimeError('At least 8 GiB free GPU memory and 3 GiB available RAM are required before inference.')
    folder = MODEL / branch
    checkpoint = folder / 'fold_5' / 'checkpoint_final.pth'
    weights = safe_checkpoint(checkpoint)
    dataset = json.loads((folder / 'dataset.json').read_text())
    plans = PlansManager(json.loads((folder / 'plans.json').read_text()))
    configuration = plans.get_configuration(weights['init_args']['configuration'])
    network = nnUNetTrainer.build_network_architecture(
        plans, configuration, determine_num_input_channels(plans, configuration, dataset),
        plans.get_label_manager(dataset).num_segmentation_heads, enable_deep_supervision=False)
    network.load_state_dict(weights['network_weights'], strict=True)
    predictor = nnUNetPredictor(tile_step_size=.5, use_gaussian=True, use_mirroring=True,
                               perform_everything_on_device=False, device=torch.device('cuda'),
                               verbose=False, verbose_preprocessing=False, allow_tqdm=True)
    predictor.manual_initialization(network, plans, configuration, [weights['network_weights']], dataset,
                                    weights['trainer_name'], weights['inference_allowed_mirroring_axes'])
    record = {'branch': branch, 'checkpoint_sha256': hashlib.file_digest(checkpoint.open('rb'), 'sha256').hexdigest(),
              'configuration': weights['init_args']['configuration'], 'epoch': weights.get('current_epoch'),
              'patch_size': configuration.patch_size, 'spacing': configuration.spacing,
              'mirror_axes': weights['inference_allowed_mirroring_axes'], 'tile_step': .5,
              'aggregation': 'CPU float16; CUDA patch inference; original patch size and test-time mirroring',
              'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(), 'started': time.time()}
    del weights
    gc.collect()
    print(json.dumps(record), flush=True)
    preprocessor = configuration.preprocessor_class(verbose=False)
    data, _, properties = preprocessor.run_case([str(source)], None, plans, configuration, dataset)
    print(f'Preprocessed {branch}: {data.shape}', flush=True)
    logits = predictor.predict_logits_from_preprocessed_data(torch.from_numpy(data))
    del data, predictor, network
    gc.collect()
    torch.cuda.empty_cache()
    output.parent.mkdir(parents=True, exist_ok=True)
    export_prediction_from_logits(logits, properties, configuration, plans, dataset,
                                  str(output)[:-7], False, num_threads_torch=2)
    record['finished'] = time.time()
    record['seconds'] = record['finished'] - record['started']
    write_json(output.with_suffix('.json'), record)


def resize_scan(source, output):
    import SimpleITK as sitk
    import numpy as np
    import torch
    from nnunetv2.preprocessing.resampling.resample_torch import resample_torch_simple
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import compute_new_shape
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
    image = sitk.ReadImage(str(source))
    data = sitk.GetArrayFromImage(image).astype(np.float32)
    spacing = (.2, .2, .2)
    shape = compute_new_shape(data.shape, list(image.GetSpacing())[::-1], spacing)
    resized = resample_torch_simple(data[None], shape, is_seg=False, num_threads=2, device=torch.device('cpu'))[0]
    result = sitk.GetImageFromArray(resized)
    result.SetSpacing(spacing)
    result.SetDirection(image.GetDirection())
    result.SetOrigin(image.GetOrigin())
    sitk.WriteImage(result, str(output), True)


def postprocess(run, source):
    import SimpleITK as sitk
    import numpy as np
    import torch
    from nnunetv2.preprocessing.resampling.resample_torch import resample_torch_simple
    sys.path.insert(0, str(UPSTREAM))
    from toothseg.toothseg.postprocess_predictions.border_core_to_instances import load_convert_semantic_to_instance_save
    from toothseg.toothseg.postprocess_predictions.assign_majority_tooth_labels import assign_correct_tooth_labels_to_instanceseg
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
    instance_file = run / 'instances_02.nii.gz'
    load_convert_semantic_to_instance_save(str(run / 'instance.nii.gz'), str(instance_file),
                                         small_center_threshold=16, isolated_border_as_separate_instance_threshold=0,
                                         min_instance_size=16)
    reference = sitk.ReadImage(str(source))
    inst = sitk.GetArrayFromImage(sitk.ReadImage(str(instance_file)))
    target = list(reference.GetSize())[::-1]
    # Same per-label interpolation/argmax as upstream; CPU avoids GPU contention.
    if int(inst.max()) > 255:
        raise ValueError('Too many tooth instances for the upstream majority-label implementation; review this scan.')
    resized = resize_instances(inst, target)
    image = sitk.GetImageFromArray(resized.astype(np.uint16))
    image.CopyInformation(reference)
    resized_file = run / 'instances_original.nii.gz'
    sitk.WriteImage(image, str(resized_file), True)
    del inst, resized, image, reference
    gc.collect()
    assign_correct_tooth_labels_to_instanceseg(str(run / 'semantic.nii.gz'), str(resized_file),
                                               str(run / 'output' / 'case.nii.gz'), min_tooth_volume=0,
                                               isolated_semsegs_as_new_instances=False, allow_background_label=False)


def resize_instances(inst, target):
    """Upstream label-wise linear interpolation/float16 argmax, streamed by label.

    Avoids unsupported uint32 Torch sorting and an entire classes×volume buffer.
    Ascending labels and strict > preserve the upstream argmax tie rule.
    """
    import numpy as np
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(2)
    if list(inst.shape) == list(target):
        return inst.astype(np.uint16)
    best = np.full(target, -1, dtype=np.float16)
    result = np.zeros(target, dtype=np.uint16)
    with torch.inference_mode():
        for value in np.unique(inst):
            data = torch.from_numpy((inst == value).astype(np.float32))[None, None] * 1000
            scores = F.interpolate(data, list(target), mode='trilinear', align_corners=False)[0, 0].to(torch.float16).numpy()
            update = scores > best
            result[update] = value
            best[update] = scores[update]
    return result


def run_pipeline(source, run):
    from pipeline import environment
    from core import load_volume, same_grid
    source, run = Path(source).resolve(), Path(run).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    run.mkdir(parents=True, exist_ok=True)
    (run / 'output').mkdir(exist_ok=True)
    started = time.time()
    write_json(run / 'request.json', {'scan': str(source), 'started': started})
    stages = [('semantic', source), ('resize', source), ('instance', run / 'scan_02.nii.gz'), ('postprocess', source)]
    for stage, input_path in stages:
        write_json(run / 'status.json', {'stage': stage, 'started': started, 'updated': time.time(), 'pid': os.getpid()})
        print(f'ToothSeg stage: {stage}', flush=True)
        args = [sys.executable, '-B', str(Path(__file__).resolve()), '--stage', stage,
                '--scan', str(input_path), '--run', str(run)]
        with (run / f'{stage}.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(args, cwd=ROOT, env=environment(), stdout=log, stderr=subprocess.STDOUT,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), timeout=3600)
        if result.returncode:
            write_json(run / 'status.json', {'stage': 'failed', 'failed_stage': stage, 'log': str(run / f'{stage}.log')})
            raise RuntimeError(f'ToothSeg {stage} failed; see {run / (stage + ".log")}')
    return finalize(run, source, started)


def finalize(run, source, started=None):
    from core import load_volume, same_grid
    if started is None:
        started = json.loads((run / 'semantic.nii.json').read_text())['started']
    scan = load_volume(source)
    mask = load_volume(run / 'output' / 'case.nii.gz', mask=True)
    if not same_grid(scan, mask):
        raise RuntimeError('Prediction does not retain input image geometry.')
    provenance = {'model': 'MIC-DKFZ ToothSeg — ToothFairy2 dual branch, fold 5 final',
                  'source': 'https://doi.org/10.5281/zenodo.14893540', 'license': 'CC-BY-4.0',
                  'training': 'Public pretrained model; not trained by this app', 'scan': str(source),
                  'scan_sha256': hashlib.file_digest(Path(source).open('rb'), 'sha256').hexdigest(),
                  'method': 'Official TF2 pipeline: semantic + border/core, 0.2 mm instance grid, 16 mm3 core and instance threshold, majority tooth labels; no mincost in-house self-correction',
                  'branches': [json.loads((run / f'{b}.nii.json').read_text()) for b in ['semantic', 'instance']],
                  'clinical_validation': False, 'seconds': time.time() - started,
                  'geometry_matches_input': True}
    write_json(run / 'provenance.json', provenance)
    write_json(run / 'status.json', {'stage': 'completed', 'seconds': provenance['seconds'], 'output': str(run / 'output' / 'case.nii.gz')})
    return provenance


if __name__ == '__main__':
    from pipeline import environment
    os.environ.update(environment())
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['all', 'semantic', 'instance', 'resize', 'postprocess', 'finalize'], default='all')
    parser.add_argument('--scan', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    if args.stage in ('semantic', 'instance'):
        predict_branch(args.stage, args.scan, args.run / (args.stage + '.nii.gz'))
    elif args.stage == 'resize':
        resize_scan(args.scan, args.run / 'scan_02.nii.gz')
    elif args.stage == 'postprocess':
        postprocess(args.run, args.scan)
    elif args.stage == 'finalize':
        finalize(args.run, args.scan)
    else:
        run_pipeline(args.scan, args.run)
