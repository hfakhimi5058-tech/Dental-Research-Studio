"""Isolated nnU-Net integration. Training is never started automatically."""
from pathlib import Path
import importlib.util
import json
import os
import subprocess
import sys
import time
import uuid
from core import ROOT, load_volume, same_grid, nifti_bytes


def environment(training=False):
    env = dict(os.environ)
    paths = {"nnUNet_raw": ROOT / "data" / "nnUNet_raw",
             "nnUNet_preprocessed": ROOT / "data" / "nnUNet_preprocessed",
             "nnUNet_results": ROOT / "models" / "nnUNet_results",
             "TEMP": ROOT / "work" / "temp", "TMP": ROOT / "work" / "temp",
             "TORCH_HOME": ROOT / "models" / "cache", "XDG_CACHE_HOME": ROOT / "work" / "cache"}
    for key, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    # Planning passes this value to torch.set_num_threads, which rejects zero.
    # Only training uses zero to select single-process augmentation.
    env["nnUNet_n_proc_DA"] = "0" if training else "1"
    env["OMP_NUM_THREADS"] = "2"
    env["MPLCONFIGDIR"] = str(ROOT / "work" / "matplotlib")
    env["nnUNet_compile"] = "false"
    return env


def executable(name):
    suffix = ".exe" if os.name == "nt" else ""
    path = Path(sys.executable).parent / (name + suffix)
    if not path.is_file():
        raise ValueError("nnU-Net is not installed in this app's environment. Dataset/model setup remains pending.")
    return str(path)


def model_info(folder):
    root = Path(folder).expanduser()
    if (root / 'pipeline.json').is_file():
        if root.resolve() != (ROOT / 'models' / 'toothseg').resolve():
            raise ValueError('Unknown composite model pipeline.')
        for branch in ('semantic', 'instance'):
            if not (root / branch / 'fold_5' / 'checkpoint_final.pth').is_file():
                raise ValueError(f'ToothSeg {branch} checkpoint is missing.')
        dataset = json.loads((root / 'semantic' / 'dataset.json').read_text())
        return dataset, ['5']
    for name in ("dataset.json", "plans.json"):
        if not (root / name).is_file():
            raise ValueError(f"The model folder needs {name}.")
    dataset = json.loads((root / "dataset.json").read_text())
    if len(dataset.get("channel_names", {})) != 1:
        raise ValueError("Only single-channel CBCT models are supported.")
    ending = dataset.get("file_ending")
    if ending not in (".nii.gz", ".nii", ".mha"):
        raise ValueError("This model's input format is not supported.")
    folds = [p.name.removeprefix("fold_") for p in root.glob("fold_*") if p.is_dir() and any(p.glob('checkpoint_*.pth'))]
    if not folds:
        raise ValueError("No model folds were found.")
    return dataset, sorted(folds)


def predict(scan, model_folder, folds, checkpoint="checkpoint_final.pth"):
    if (Path(model_folder) / 'pipeline.json').is_file():
        import nibabel as nib
        from toothseg import run_pipeline
        dataset, available = model_info(model_folder)
        if folds != ['5'] or checkpoint != 'checkpoint_final.pth':
            raise ValueError('The audited ToothSeg pipeline uses fold 5 final checkpoints for both branches.')
        run = ROOT / 'results' / ('toothseg-' + uuid.uuid4().hex[:10])
        run.mkdir(parents=True)
        image = nib.Nifti1Image(scan.data, scan.affine)
        image.header.set_xyzt_units('mm')
        image.set_sform(scan.affine, code=1)
        source = run / 'scan.nii.gz'
        nib.save(image, str(source))
        run_pipeline(source, run)
        return load_volume(run / 'output' / 'case.nii.gz', mask=True), dataset, run
    command = executable("nnUNetv2_predict_from_modelfolder")
    dataset, available = model_info(model_folder)
    if not folds or not set(folds).issubset(available):
        raise ValueError("Choose at least one available model fold.")
    for fold in folds:
        if not (Path(model_folder) / f"fold_{fold}" / checkpoint).is_file():
            raise ValueError(f"Fold {fold} lacks {checkpoint}.")
    run = ROOT / "results" / ("segmentation-" + uuid.uuid4().hex[:10])
    inp, out = run / "input", run / "output"
    inp.mkdir(parents=True)
    out.mkdir()
    ending = dataset["file_ending"]
    if ending in (".nii", ".nii.gz"):
        import nibabel as nib
        img = nib.Nifti1Image(scan.data, scan.affine)
        img.header.set_xyzt_units("mm")
        img.set_sform(scan.affine, code=1)
        nib.save(img, str(inp / ("case_0000" + ending)))
    else:
        import SimpleITK as sitk
        import numpy as np
        spacing = scan.spacing
        direction = scan.affine[:3, :3] / spacing
        if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-4):
            raise ValueError("MHA model input cannot represent this image's sheared affine.")
        image = sitk.GetImageFromArray(scan.data.transpose(2, 1, 0))
        lps = np.diag([-1., -1., 1.])
        image.SetSpacing(tuple(spacing))
        image.SetDirection(tuple((lps @ direction).ravel()))
        image.SetOrigin(tuple(lps @ scan.affine[:3, 3]))
        sitk.WriteImage(image, str(inp / "case_0000.mha"))
    args = [command, "-i", str(inp), "-o", str(out), "-m", str(Path(model_folder).resolve()),
            "-f", *folds, "-chk", checkpoint, "-npp", "1", "-nps", "1"]
    log = run / "inference.log"
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(args, env=environment(), stdout=handle, stderr=subprocess.STDOUT,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), timeout=3600)
    if result.returncode:
        raise ValueError(f"Segmentation failed. Read {log}. No prediction was loaded.")
    output = out / ("case" + ending)
    prediction = load_volume(output, mask=True)
    if not same_grid(scan, prediction):
        raise ValueError("Model output geometry differs from the scan; review the saved output before measurement.")
    return prediction, dataset, run


def normalized_metadata(metadata):
    """Normalize documented TF2 field spelling without altering source metadata."""
    metadata = dict(metadata)
    if 'channel_names' not in metadata and 'channels_names' in metadata:
        metadata['channel_names'] = metadata['channels_names']
    labels = dict(metadata.get('labels', {}))
    if 'Background' in labels and 'background' not in labels:
        labels['background'] = labels.pop('Background')
    metadata['labels'] = labels
    return metadata


def inspect_dataset(folder):
    folder = Path(folder).expanduser().resolve()
    source = folder / "dataset.json"
    if not source.is_file():
        raise ValueError("Choose the extracted dataset folder containing dataset.json, imagesTr and labelsTr.")
    metadata = normalized_metadata(json.loads(source.read_text(encoding="utf-8-sig")))
    ending = metadata.get("file_ending")
    if ending not in (".mha", ".nii.gz", ".nii"):
        raise ValueError("Unsupported or missing dataset file_ending.")
    labels = metadata.get("labels", {})
    if not labels or any(not isinstance(x, int) for x in labels.values()) or labels.get("background") != 0:
        raise ValueError("Dataset needs integer name-to-ID labels including background=0.")
    if len(set(labels.values())) != len(labels):
        raise ValueError("Dataset has duplicate label IDs.")
    if len(metadata.get("channel_names", {})) != 1:
        raise ValueError("Only single-channel datasets are supported.")
    cases = []
    for image in sorted((folder / "imagesTr").glob("*_0000" + ending)):
        case = image.name[:-len("_0000" + ending)]
        label = folder / "labelsTr" / (case + ending)
        if not label.is_file():
            raise ValueError(f"Missing reference mask for {case}.")
        cases.append((case, image, label))
    if not cases:
        raise ValueError("No paired training images and masks found.")
    return metadata, cases


def prepare_dataset(folder, dataset_id=701):
    """Convert original label IDs to contiguous nnU-Net IDs with a retained map."""
    import nibabel as nib
    import numpy as np
    metadata, cases = inspect_dataset(folder)
    dest = ROOT / "data" / "nnUNet_raw" / f"Dataset{int(dataset_id):03d}_DentalResearch"
    if dest.exists():
        raise ValueError("Prepared dataset already exists. It is never overwritten automatically.")
    names = {v: k for k, v in metadata["labels"].items()}
    # All declared IDs are retained, including reserved names, avoiding silent anatomy loss.
    mapping = {old: new for new, old in enumerate(sorted(names))}
    dest.mkdir(parents=True)
    (dest / "imagesTr").mkdir()
    (dest / "labelsTr").mkdir()
    (dest / "PREPARATION_INCOMPLETE").write_text("Do not train until preparation completes.")
    for case, image_path, mask_path in cases:
        image, mask = load_volume(image_path), load_volume(mask_path, mask=True)
        if not same_grid(image, mask):
            raise ValueError(f"Scan and mask geometry mismatch: {case}.")
        found = set(int(x) for x in np.unique(mask.data))
        if not found.issubset(mapping):
            raise ValueError(f"Undeclared labels in {case}: {sorted(found-set(mapping))}.")
        lookup = np.zeros(max(mapping) + 1, dtype=np.uint16)
        for old, new in mapping.items():
            lookup[old] = new
        for data, name in ((image.data.astype(np.float32), f"imagesTr/{case}_0000.nii.gz"),
                           (lookup[mask.data], f"labelsTr/{case}.nii.gz")):
            img = nib.Nifti1Image(data, image.affine)
            img.header.set_xyzt_units("mm")
            img.set_sform(image.affine, code=1)
            nib.save(img, str(dest / name))
    newmeta = {"name": "DentalResearch", "channel_names": {"0": "CBCT"},
               "labels": {names[old]: new for old, new in mapping.items()},
               "numTraining": len(cases), "file_ending": ".nii.gz"}
    (dest / "dataset.json").write_text(json.dumps(newmeta, indent=2), encoding="utf-8")
    (dest / "source_label_mapping.json").write_text(json.dumps({"original_to_training_id": mapping,
        "source_metadata": metadata, "patient_split_review_required": True}, indent=2), encoding="utf-8")
    (dest / "PREPARATION_INCOMPLETE").unlink()
    return dest


def run_training(stage, dataset_id=701, fold=0):
    """Separate bounded commands; no implicit training loops or installations."""
    dest = ROOT / "data" / "nnUNet_raw" / f"Dataset{dataset_id:03d}_DentalResearch"
    if not (dest / "dataset.json").is_file() or (dest / "PREPARATION_INCOMPLETE").exists():
        raise ValueError("A completely prepared dataset is required.")
    if stage == "preprocess":
        args = [executable("nnUNetv2_plan_and_preprocess"), "-d", str(dataset_id),
                "--verify_dataset_integrity", "-c", "3d_fullres", "-np", "1", "-npfp", "1",
                "-gpu_memory_target", "8"]
    elif stage == "train":
        args = [executable("nnUNetv2_train"), str(dataset_id), "3d_fullres", str(fold)]
    else:
        raise ValueError("Unknown training stage.")
    return subprocess.call(args, env=environment(training=stage == "train"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare and train a local dental research model.")
    parser.add_argument("stage", choices=["inspect", "prepare", "preprocess", "train"])
    parser.add_argument("--source")
    parser.add_argument("--dataset-id", type=int, default=701)
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--patient-split-reviewed", action="store_true")
    args = parser.parse_args()
    try:
        if args.stage == "inspect":
            meta, pairs = inspect_dataset(args.source)
            print(json.dumps({"paired_cases": len(pairs), "labels": meta['labels'], "file_ending": meta['file_ending']}, indent=2))
        elif args.stage == "prepare":
            print(prepare_dataset(args.source, args.dataset_id))
        else:
            if args.stage == "train" and not args.patient_split_reviewed:
                raise ValueError("Review patient-level split independence before training; then use --patient-split-reviewed.")
            raise SystemExit(run_training(args.stage, args.dataset_id, args.fold))
    except (ValueError, OSError) as exc:
        parser.exit(1, str(exc) + "\n")
