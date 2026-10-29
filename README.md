# Dental Research Studio

A local research application for dental CBCT review, automated tooth segmentation, interactive 3D visualization and quantitative analysis.

**Research prototype. Not clinically validated.** Automated masks and measurements require review. Segmentation uses the public pretrained [MIC-DKFZ ToothSeg model](https://doi.org/10.5281/zenodo.14893540); training a separate model remains experimental.

## What is included

- NIfTI (`.nii`, `.nii.gz`), MHA and conventional multi-file DICOM import.
- Linked slices, window/level, interactive 3D surfaces, rotation controls and label selection.
- Individual-tooth controls and visual size cards with volume, voxel counts, physical bounds and CSV export.
- Volumes, physical bounds, centroids, surfaces, component checks and reference-mask agreement.
- Local tooth segmentation, saved cases, slice corrections, undo and NIfTI/OBJ/CSV/JSON/report exports.
- Optional bounded, resumable nnU-Net training tools and a source-specific dataset registry.

Scans, datasets, model weights, local environments, credentials and previous users' results are **not included**. A fresh installation starts with a clearly labelled synthetic geometry example.

## Quick start — Windows

Prerequisite: **64-bit Python 3.12** with the Python launcher (`py`). Use a short writable folder, for example `C:\DentalResearchStudio`, to avoid Windows path-length problems. Do not place confidential scans in a synced folder unless that sync is intended.

```powershell
git clone https://github.com/hfakhimi5058-tech/Dental-Research-Studio.git
cd Dental-Research-Studio
.\Setup.cmd
.\Start.cmd
```

The repository is private; cloning requires access to it. Alternatively use GitHub's **Code → Download ZIP**, extract it, then double-click `Setup.cmd` and `Start.cmd`.

Open **http://127.0.0.1:8517/**. Setup creates an isolated `.venv`; it does not install packages globally. Viewer/measurement mode does not require a GPU or trained model. If the `py` launcher is unavailable, run `Setup.ps1 -Python "C:\path\to\python.exe"` with a Python 3.12 interpreter.

For usage instructions, follow the [user guide](docs/HOW_TO_USE.md), or select **?** in the app for its quick guide. Import an unsegmented image in **Raw scan**, leaving **Existing segmentation mask** empty: segmentation creates labels and does not require them beforehand.

In **Workspace**, choose a tooth to view it separately, use the arrows to move between labels, or select **Show all teeth**. The selected segment's measurements appear above the images. **Segment sizes** opens searchable visual cards with a CSV export. Processing progress is shown above the images and beneath the segmentation button; 3D surface preparation has its own count.

## Enable automated segmentation

This optional setup downloads PyTorch/CUDA dependencies and the approximately 920 MB official model archive. It needs an NVIDIA GPU with a compatible driver. The inference code checks for at least 8 GiB free GPU memory and 3 GiB available RAM; larger scans may require more. CPU-only model inference is not implemented in this configured pipeline.

```powershell
.\Setup.cmd -Segmentation -DownloadModels
```

Restart the app if it was open during installation. Then:

1. Choose **Import images** and supply the raw scan. An existing segmentation mask is optional.
2. Open **Models & jobs**, select **ToothSeg · dual-branch tooth segmentation**, and keep `checkpoint_final.pth`.
3. Review the model source and label definitions, then choose **Run local segmentation**.
4. Inspect the result in **Workspace**, calculate additional measurements in **Analysis**, and export from **Reports**.

The model produces 32 permanent-tooth IDs. It has no separate jaw, canal, sinus, implant or restoration labels. Predicted tooth regions may include crowns. Not every scan contains all 32 teeth. Installing this model does not train a new one.

The downloader pins the upstream source revision, verifies the model archive's published MD5 and safely extracts both model branches into short paths. Downloaded code/weights stay in ignored `models/`. See [model attribution](docs/TOOTHSEG_ATTRIBUTION.txt) and [third-party notices](THIRD_PARTY_NOTICES.md).

## Linux/macOS — manual viewer setup

These launch instructions are provided for the Python app; the supplied one-click launchers target Windows. GPU inference was exercised on Windows, not validated here on other platforms.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
mkdir -p work/temp work/pycache
TMPDIR="$PWD/work/temp" PYTHONPYCACHEPREFIX="$PWD/work/pycache" python app/app.py
```

For Linux with a compatible NVIDIA GPU, install the CUDA packages and `requirements-segmentation.txt`, then run the model downloader:

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-segmentation.txt
python scripts/download_models.py
```

## Data, training and limitations

Obtain datasets directly from their publishers and review their access and usage conditions. The [ToothFairy2 portal](https://ditto.ing.unimore.it/toothfairy2/) provides the source used by the managed training workflow. A dataset is not needed for viewer mode or for inference on your own authorized scan.

For experimental training, place the official archive at `data/ToothFairy2_Dataset.zip`. With segmentation dependencies installed, run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\Start-Research.ps1 -Stage all -Minutes 120
# Later, resume the prepared run and its saved checkpoint:
powershell -NoProfile -ExecutionPolicy Bypass -File .\Start-Research.ps1 -Stage resume -Minutes 120
```

This is a time budget checked at epoch boundaries, **not a complete training schedule**. Preparation checks exact duplicates and preserves a held-out partition. Cropped/resampled duplicates, annotation meaning and patient overlap can require further review. Additional datasets are not merged automatically. The managed workflow reserves 100 GiB free disk and requires at least 9 GiB free GPU memory for training.

Reference evaluation requires matching physical grids and matching label definitions. Volumes use labelled voxel count × the absolute determinant of the spatial transform. Bounds are not anatomical tooth lengths. A high mean Dice can hide a failed tooth. No diagnostic accuracy or clinical readiness is claimed. Prior local pilot records are excluded from this source repository.

## Check installation and run software tests

```powershell
.\.venv\Scripts\python.exe scripts/check_install.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests use synthetic fixtures, not clinical evidence. The optional model downloader requires Internet access; normal image review and inference use local files and a loopback server.

## Troubleshooting

- **Port 8517 is in use:** stop another running copy with its `Stop.cmd`, or close that copy. The launcher does not kill unrelated processes.
- **Python not found:** install Python 3.12 or pass its path to `Setup.ps1 -Python ...`.
- **No model listed:** finish `Setup.cmd -Segmentation -DownloadModels`; a viewer-only installation intentionally has no weights.
- **CUDA unavailable/out of memory:** check the NVIDIA driver and CUDA-enabled PyTorch installation; free GPU/RAM and retry the failed job once after resolving the cause.
- **Dataset has no paired images:** point to an extracted dataset containing `dataset.json`, `imagesTr` and `labelsTr`, or use the managed archive workflow above.
- **Long-path extraction failure:** move a fresh source checkout to a shorter folder and rerun setup. Do not move a running app with open cases.
- **Where are errors?** See the job log in **Models & jobs** and the local `work/` and `results/` folders.
- **Is segmentation still working?** The job card and floating status panel show the current stage, elapsed time, worker status and time since the last log update. When the model reports block counts, the progress bar shows the current prediction pass only. Stages without a counter remain indeterminate; a pass reaching 100% does not mean the whole job is finished. Expand **Live stage log** for details. Completed and failed outcomes remain visible, and refreshing the page reconnects to a running job.

## Evidence & Uncertainty Checker

In **Reports**, choose **Generate draft & check** to create measurement statements with source locations. Add individual claims from a draft report, inspect their source slices, and record visibility, anatomical grounding, image quality and segmentation/model support. Unresolved findings are flagged for specialist review. Changes are saved locally with review history; old assessments become stale after image or evidence changes. JSON and printable reports retain the flags.

This stage uses deterministic checks and reviewer assessments. It does not yet run the supplied paper's fine-tuned VLM, automatically verify diagnoses or produce calibrated confidence scores. See the [checker guide and limitations](docs/EVIDENCE_CHECKER.md).

## Stop, storage and removal

Run `Stop.cmd` to stop this app and its owned workers. On a manual terminal launch, use Ctrl+C. Source files are preserved; generated data lives under `data/`, `models/`, `results/` and `work/`. These paths and `.venv/` are ignored by Git. Keep the server on `127.0.0.1`; this prototype has no multi-user deployment/authentication system.

To remove the local installation, stop it and delete its project folder. Browser downloads, external input files and cloud-sync recovery copies are separate. No project-wide open-source license has been assigned to the original application code; third-party components retain their own licenses.
