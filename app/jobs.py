"""One local job at a time; persistent status and cancellable child processes."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import os
import subprocess
import threading
import time
import uuid
import psutil
from core import ROOT
from pipeline import environment
from job_progress import snapshot

FOLDER = ROOT / 'work' / 'jobs'
FOLDER.mkdir(parents=True, exist_ok=True)
POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix='dental-job')
LOCK = threading.RLock()
JOBS = {}
for path in FOLDER.glob('*.json'):
    try:
        item = json.loads(path.read_text())
        if item['status'] in ('queued', 'running', 'cancelling'):
            item['status'] = 'interrupted'
            item['message'] = 'Application restarted; this job has not been resumed.'
        JOBS[item['id']] = item
    except (ValueError, KeyError):
        pass


def save(item):
    serial = {k: v for k, v in item.items() if k not in ('event', 'process')}
    target = FOLDER / (item['id'] + '.json')
    temp = target.with_suffix('.tmp')
    temp.write_text(json.dumps(serial, indent=2, default=str), encoding='utf-8')
    temp.replace(target)


def submit(name, function, cancellable=False):
    with LOCK:
        if any(j['status'] in ('queued','running','cancelling') for j in JOBS.values()):
            raise ValueError('A local job is already running. Finish or cancel it before starting another.')
        ident = uuid.uuid4().hex[:12]
        item = {'id': ident, 'name': name, 'status':'queued', 'created':time.time(),
                'message':'Waiting for local worker', 'cancellable':cancellable,
                'event':threading.Event(), 'result':None}
        JOBS[ident] = item
        save(item)

    def execute():
        try:
            item.update(status='running', started=time.time(), message='Running locally')
            save(item)
            result = function(item)
            item.update(result=result, status='cancelled' if item['event'].is_set() else 'completed',
                        message='Cancelled' if item['event'].is_set() else 'Completed')
        except Exception as exc:
            item.update(status='cancelled' if item['event'].is_set() else 'failed', message=str(exc))
        finally:
            item['finished'] = time.time()
            save(item)
    POOL.submit(execute)
    return ident


def run_process(item, args, minutes=60):
    log = FOLDER / (item['id'] + '.log')
    item['log'] = str(log)
    with log.open('w', encoding='utf-8') as handle:
        env = environment()
        env['DENTAL_JOB_PROGRESS'] = str(FOLDER / (item['id'] + '.progress.json'))
        env['PYTHONUNBUFFERED'] = '1'
        process = subprocess.Popen(args, env=env, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        item['pid'] = process.pid
        save(item)
        deadline = time.monotonic() + float(minutes)*60
        while process.poll() is None:
            if item['event'].wait(.5) or time.monotonic() > deadline:
                try:
                    parent = psutil.Process(process.pid)
                    for child in parent.children(recursive=True):
                        child.terminate()
                    parent.terminate()
                    process.wait(timeout=15)
                except (psutil.NoSuchProcess, subprocess.TimeoutExpired):
                    if process.poll() is None:
                        process.kill()
                if item['event'].is_set():
                    return {'cancelled':True}
                raise ValueError('Time limit reached. Review the local log and any saved model checkpoint before resuming.')
        if process.returncode:
            raise ValueError('The process failed. Open the job log for its error; no successful result is claimed.')
    return {'exit_code':process.returncode, 'log':str(log)}


def listing():
    with LOCK:
        result=[]
        for item in sorted(JOBS.values(),key=lambda x:x['created'],reverse=True):
            entry={k:v for k,v in item.items() if k not in ('event','process')}
            if item.get('log') and Path(item['log']).is_file():
                with Path(item['log']).open('rb') as f:
                    f.seek(max(0,Path(item['log']).stat().st_size-16000))
                    entry['tail']=f.read().decode('utf-8',errors='replace')
            entry['progress'] = snapshot(item, ROOT)
            result.append(entry)
        return result[:40]


def cancel(ident):
    item=JOBS.get(ident)
    if not item or not item.get('cancellable') or item['status'] not in ('queued','running'):
        raise ValueError('This job cannot be cancelled at its current stage.')
    item['event'].set()
    item['status']='cancelling'
    save(item)
