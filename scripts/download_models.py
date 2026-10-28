"""Fetch and register the official public ToothSeg models; no scan upload."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'models' / 'pretrained' / 'ToothSeg'
REVISION = 'b29d1017fa124f89645fa56f98649e3f3f43bdb0'
MODEL_URL = 'https://zenodo.org/api/records/14893540/files/ToothSeg.zip/content'
MODEL_MD5 = '5d8dd061cce9529943567aeba3271143'


def download(url, name, expected_md5=None):
    DEST.mkdir(parents=True, exist_ok=True)
    target = DEST / name
    if not target.exists():
        if shutil.disk_usage(DEST).free < 4 * 2**30:
            raise RuntimeError('Keep at least 4 GiB free for archives and extraction.')
        partial = target.with_suffix(target.suffix + '.partial')
        request = urllib.request.Request(url, headers={'User-Agent': 'Dental-Research-Studio-model-setup'})
        print(f'Downloading {url}', flush=True)
        with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as output:
            while chunk := response.read(4 * 1024 * 1024):
                output.write(chunk)
        partial.replace(target)
    with target.open('rb') as source:
        digest = hashlib.file_digest(source, 'md5').hexdigest()
    if expected_md5 and digest != expected_md5:
        raise RuntimeError(f'Checksum mismatch: {target.name}. No model was registered; remove the bad archive before retrying.')
    return target, digest


def safe_parts(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe archive path')
    return path.parts


def extract(archive, source=False):
    with zipfile.ZipFile(archive) as content:
        for item in content.infolist():
            parts = safe_parts(item.filename)
            if item.is_dir():
                continue
            if source:
                if parts[0] != 'ToothSeg-' + REVISION:
                    raise ValueError('Unexpected source revision folder')
                target = DEST / 'ToothSeg-main' / Path(*parts[1:])
            else:
                if len(parts) < 4 or parts[0] != 'ToothSeg':
                    raise ValueError('Unexpected model layout')
                branch = 'semantic' if parts[1].startswith('Dataset121_') else 'instance' if parts[1].startswith('Dataset123_') else None
                if branch is None:
                    raise ValueError('Unexpected model branch')
                target = ROOT / 'models' / 'toothseg' / branch / Path(*parts[3:])
            if not target.resolve().is_relative_to((ROOT / 'models').resolve()):
                raise ValueError('Extraction escaped the model directory')
            target.parent.mkdir(parents=True, exist_ok=True)
            with content.open(item) as incoming, target.open('wb') as outgoing:
                shutil.copyfileobj(incoming, outgoing)


def main():
    # Registration is written last: an interrupted setup never advertises missing weights.
    source, source_md5 = download(f'https://codeload.github.com/MIC-DKFZ/ToothSeg/zip/{REVISION}', 'source-pinned.zip')
    extract(source, source=True)
    models, models_md5 = download(MODEL_URL, 'ToothSeg.zip', MODEL_MD5)
    extract(models)
    for branch in ('semantic', 'instance'):
        folder = ROOT / 'models' / 'toothseg' / branch
        for name in ('dataset.json', 'plans.json', 'fold_5/checkpoint_final.pth'):
            if not (folder / name).is_file():
                raise RuntimeError(f'Model setup incomplete: {branch}/{name}')
    shutil.copy2(ROOT / 'config/toothseg-pipeline.json', ROOT / 'models/toothseg/pipeline.json')
    shutil.copy2(ROOT / 'docs/TOOTHSEG_ATTRIBUTION.txt', ROOT / 'models/toothseg/ATTRIBUTION.txt')
    (DEST / 'acquisition.json').write_text(json.dumps({'source_revision': REVISION, 'source_archive_md5': source_md5,
        'model_source': MODEL_URL, 'model_archive_md5': models_md5,
        'model_license': 'CC-BY-4.0', 'code_license': 'Apache-2.0'}, indent=2))
    print('ToothSeg installed. Restart the app, then open Models & jobs. No training was started.')


if __name__ == '__main__':
    main()
