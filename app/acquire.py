"""Finish the explicitly authorized browser download inside the isolated folder."""
from pathlib import Path
import argparse
import json
import os
import shutil
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / 'data' / 'acquisition.json'


def record(**values):
    values['updated'] = time.time()
    tmp = STATUS.with_suffix('.tmp')
    tmp.write_text(json.dumps(values, indent=2), encoding='utf-8')
    tmp.replace(STATUS)


def main(partial, expected):
    partial, expected = Path(partial).resolve(), Path(expected).resolve()
    if partial.parent != expected.parent or partial.suffix != '.crdownload':
        raise ValueError('Expected one authorized browser download and its completed filename.')
    target = (ROOT / 'data' / 'ToothFairy2_Dataset.zip').resolve()
    target.relative_to(ROOT.resolve())
    stop = ROOT / 'work' / 'stop-acquisition'
    deadline = time.monotonic() + 6 * 3600
    while time.monotonic() < deadline:
        if stop.exists():
            record(stage='stopped', message='Transfer watcher stopped; browser download was not cancelled.')
            return
        if target.exists():
            record(stage='downloaded', archive=str(target), bytes=target.stat().st_size)
            return
        if partial.exists():
            if shutil.disk_usage(ROOT).free < 100 * 2**30:
                raise ValueError('Disk reserve reached. Cancel the browser download before disk space is exhausted.')
            record(stage='downloading', bytes=partial.stat().st_size,
                   message='Official ToothFairy2 archive is downloading; total size is not reported by the browser.')
        elif expected.exists():
            # Only recognize the archive by its official internal root, not its filename alone.
            with zipfile.ZipFile(expected) as archive:
                names = archive.namelist()
                metadata = json.loads(archive.read('Dataset112_ToothFairy2/dataset.json'))
                if metadata.get('name') != 'ToothFairy 2':
                    raise ValueError('Completed file is not the expected dataset.')
                total = sum(i.file_size for i in archive.infolist())
            # Both resolved paths were checked above; never move an in-progress file.
            if target.exists():
                raise ValueError('Destination exists; no file was overwritten.')
            shutil.move(str(expected), str(target))
            record(stage='downloaded', archive=str(target), bytes=target.stat().st_size,
                   uncompressed_bytes=total, members=len(names), licence=metadata.get('license'),
                   message='Moved into the isolated folder. Full member CRC checks occur during extraction.')
            return
        else:
            record(stage='waiting_for_completed_name', message='Waiting for the browser to rename the completed archive.')
        time.sleep(15)
    raise TimeoutError('Download watcher reached its six-hour limit; no training started.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--partial', required=True)
    p.add_argument('--completed', required=True)
    args = p.parse_args()
    try:
        main(args.partial, args.completed)
    except Exception as exc:
        record(stage='failed', message=str(exc))
        raise
