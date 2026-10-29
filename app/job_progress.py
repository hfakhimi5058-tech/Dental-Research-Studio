"""Read real stage logs and worker activity; never estimate whole-job completion."""
import json
import os
from pathlib import Path
import re
import time

import psutil

STAGES = [
    ('prepare', 'Preparing scan'),
    ('semantic', 'Identifying teeth'),
    ('resize', 'Resampling scan'),
    ('instance', 'Separating individual teeth'),
    ('postprocess', 'Assembling tooth masks'),
    ('finalize', 'Checking and loading result'),
]
COUNTER = re.compile(r'\d{1,3}%\|[^\r\n]*?\|\s*(\d+)\s*/\s*(\d+)\s*\[')


def track_run(folder, kind):
    """Link this child run to its parent job, including nested branch logs."""
    target = os.environ.get('DENTAL_JOB_PROGRESS')
    if target:
        path = Path(target)
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps({'folder': str(folder), 'kind': kind}), encoding='utf-8')
        temp.replace(path)


def read_json(path):
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def tail(path):
    try:
        with path.open('rb') as stream:
            info = os.fstat(stream.fileno())
            stream.seek(max(0, info.st_size - 16000))
            return stream.read().decode('utf-8', errors='replace'), info.st_mtime
    except OSError:
        return '', None


def counter(text):
    matches = list(COUNTER.finditer(text))
    if not matches:
        return None
    done, total = map(int, matches[-1].groups())
    if total <= 0 or not 0 <= done <= total:
        return None
    return {'done': done, 'total': total, 'percent': round(done * 100 / total, 1)}


def worker(item):
    if not item.get('pid') or item['status'] not in ('running', 'cancelling'):
        return None
    try:
        process = psutil.Process(item['pid'])
        # Do not mistake a reused PID for the job's child.
        if process.create_time() < item.get('started', item['created']) - 2:
            return {'alive': False}
        cpu = 0.0
        for child in [process, *process.children(recursive=True)]:
            try:
                times = child.cpu_times()
                cpu += times.user + times.system
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return {'alive': process.is_running(), 'cpu_seconds': round(cpu, 2)}
    except psutil.NoSuchProcess:
        return {'alive': False}
    except psutil.AccessDenied:
        return None


def snapshot(item, root, now=None):
    now = time.time() if now is None else now
    end = item.get('finished', now)
    progress = {'observed_at': now, 'elapsed_seconds': max(0, end - item.get('started', item['created'])),
                'stage': 'Preparing local worker', 'steps': [], 'counter': None,
                'last_output_at': None, 'worker': worker(item)}
    main_text, main_time = tail(Path(item['log'])) if item.get('log') else ('', None)
    progress['last_output_at'] = main_time
    detail_text = ''
    descriptor = read_json(root / 'work' / 'jobs' / (item['id'] + '.progress.json'))
    folder = Path(descriptor.get('folder', root)).resolve()
    # Read only this application's result directory, never an arbitrary log path.
    if folder.is_relative_to((root / 'results').resolve()):
        if descriptor.get('kind') == 'toothseg':
            status = read_json(folder / 'status.json')
            key = status.get('stage', 'prepare')
            if key == 'failed':
                key = status.get('failed_stage', 'prepare')
            if key == 'completed':
                key = 'finalize'
            keys = [key for key, _ in STAGES]
            if key not in keys:
                key = 'prepare'
            position = keys.index(key)
            progress.update(stage=STAGES[position][1], stage_index=position + 1, stage_total=len(STAGES))
            for index, (_, label) in enumerate(STAGES):
                state = 'complete' if index < position or item['status'] == 'completed' else 'pending'
                if index == position and item['status'] != 'completed':
                    state = 'active' if item['status'] in ('running', 'queued') else item['status']
                progress['steps'].append({'label': label, 'state': state})
            if key in ('semantic', 'resize', 'instance', 'postprocess'):
                detail_text, detail_time = tail(folder / (key + '.log'))
                if detail_time:
                    progress['last_output_at'] = max(main_time or 0, detail_time)
                if key in ('semantic', 'instance'):
                    progress['counter'] = counter(detail_text)
                    progress['counter_label'] = 'Blocks in this prediction pass'
        else:
            progress['stage'] = 'Running segmentation model'
            detail_text, detail_time = tail(folder / 'inference.log')
            progress['last_output_at'] = max(main_time or 0, detail_time or 0) or None
            progress['counter'] = counter(detail_text)
            progress['counter_label'] = 'Blocks in the current prediction pass'
    elif main_text:
        stages = re.findall(r'ToothSeg stage: (\w+)', main_text)
        progress['stage'] = dict(STAGES).get(stages[-1], 'Processing locally') if stages else 'Processing locally'
        progress['counter'] = counter(main_text)
        progress['counter_label'] = 'Items in the current logged step'
    if item['status'] == 'queued':
        progress['stage'] = 'Waiting for local worker'
    elif item['status'] == 'completed':
        progress['stage'] = 'Completed'
    elif item['status'] in ('failed', 'cancelled', 'interrupted') and not progress['steps']:
        progress['stage'] = item['status'].capitalize()
    progress['detail_tail'] = detail_text
    return progress
