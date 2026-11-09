"""Loopback-only imaging workspace; all assets and processing are local."""
from pathlib import Path
from datetime import datetime, timezone
import gzip
import copy
import hashlib
import html
import importlib.metadata
import importlib.util
import io
import json
import os
import secrets
import subprocess
import sys
import threading
import time
import uuid
import numpy as np
import nibabel as nib
from PIL import Image
import psutil
from scipy import ndimage
from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response, FileResponse
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware
from core import ROOT, Volume, demo, load_volume, load_dicom, dicom_series, label_names, validate, analyse, same_grid, mesh, obj_bytes, nifti_bytes, compare
from pipeline import inspect_dataset, model_info, executable, environment
from analysis import analyse_case, pair_distance, surface_agreement
import evidence as report_evidence
import jobs
import pulpy3d
import capabilities

STATIC = Path(__file__).parent / 'static'
CASES = ROOT / 'data' / 'cases'
CASES.mkdir(parents=True,exist_ok=True)
TOKEN = secrets.token_urlsafe(32)
PALETTE = ['#42d3bc','#a78bfa','#60a5fa','#f2bd62','#f48dab','#83c98c','#dc9c73','#62c9dc','#d7a1ef','#c6d06c']
LOCK = threading.RLock()
CURRENT = None


def display_title(title):
    return str(title).replace('actual ToothSeg AI prediction','ToothSeg segmentation').replace('AI prediction','model prediction')


def now():
    return datetime.now(timezone.utc).isoformat()


def set_current(mask, scan, names, title, ident=None, synthetic=False, source=None):
    global CURRENT
    if mask is None and scan is None:
        raise ValueError('A scan or a segmentation mask is required.')
    if mask is not None and scan is not None and not same_grid(mask,scan):
        raise ValueError('Scan and mask do not share the same physical grid. No automatic registration was applied.')
    rows=analyse(mask,names) if mask is not None else None
    with LOCK:
        CURRENT={'id':ident or uuid.uuid4().hex[:12],'revision':uuid.uuid4().hex[:8], 'title':title[:120],
                 'mask':mask,'scan':scan,'names':names,'synthetic':synthetic,'created':now(),
                 'rows':rows,'meshes':{},'analysis':None,'reference':None,'evaluation':None,'undo':[],'annotations':[],'dirty':False,'source':source or {}}
    return public_case()


def get_current():
    global CURRENT
    if CURRENT is None:
        active=ROOT/'work'/'active-case.json'
        if active.is_file():
            try:
                ident=json.loads(active.read_text())['id']
                if ident.isalnum():
                    info=json.loads((CASES/ident/'case.json').read_text())
                    load_case(info['source'],saved=info)
            except (OSError,ValueError,KeyError):
                pass
    if CURRENT is None:
        m,s,n=demo()
        set_current(m,s,n,'Geometry demonstration','demo',True)
    return CURRENT


def public_case():
    c=get_current()
    base=c['mask'] if c['mask'] is not None else c['scan']
    sample=(c['scan'].data if c['scan'] is not None else base.data)[::4,::4,::4]
    lo,hi=np.percentile(sample,[1,99])
    return {'id':c['id'],'revision':c['revision'],'title':display_title(c['title']),'synthetic':c['synthetic'],
            'shape':list(base.data.shape),'spacing':base.spacing.tolist(),'affine':base.affine.tolist(),
            'orientation':list(nib.aff2axcodes(base.affine)), 'voxel_mm3':base.voxel_mm3,
            'has_mask':c['mask'] is not None,'has_scan':c['scan'] is not None,
            'rows':[] if c['rows'] is None else c['rows'].to_dict(orient='records'),
            'notes':base.notes,'window':[float(lo),float(max(hi,lo+1))],
            'dirty':c['dirty'],'undo_count':len(c['undo']),'annotations':c['annotations'],
            'analysis':c.get('analysis'),'has_reference':c.get('reference') is not None,'evaluation':c.get('evaluation'),
            'dataset_import':c['source'].get('dataset_import'),
            'provenance':c['source'].get('provenance')}


def current_matches(data):
    c=get_current()
    if data.get('case_id')!=c['id'] or data.get('revision')!=c['revision']:
        raise ValueError('The active case changed. Refresh before applying this action.')
    return c


def evidence_record(c):
    if 'evidence_record' not in c:
        path=CASES/c['id']/'evidence.json'
        record=json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None
        if record is not None and (record.get('schema_version')!=1 or record.get('case_id')!=c['id']):
            raise ValueError('Saved evidence does not match this case or schema.')
        c['evidence_record']=record
    return c['evidence_record']


def update_evidence(data, action):
    with LOCK:
        c=current_matches(data)
        previous=evidence_record(c)
        if data.get('evidence_version')!=(previous or {}).get('version'):
            raise ValueError('The evidence report changed in another view. Reload the checker.')
        if action=='evidence-generate':
            record=report_evidence.new_record(c,copy.deepcopy(previous))
            detail={'action':'regenerate_measurement_draft','replaced_measurements':[f for f in (previous or {}).get('findings',[]) if f['kind']=='calculated_volume']}
        else:
            if previous is None:raise ValueError('Generate the measurement draft and checker first.')
            record=copy.deepcopy(previous)
            if action=='evidence-add':
                finding=report_evidence.add_finding(c,record,data);detail={'action':'add_draft','finding':copy.deepcopy(finding)}
            elif action=='evidence-review':
                finding=report_evidence.review_finding(c,record,data);detail={'action':'record_review','finding_id':finding['id'],'review':copy.deepcopy(finding['review'])}
            else:
                finding=next((f for f in record['findings'] if f['id']==data.get('finding_id')),None)
                if finding is None:raise ValueError('Finding no longer exists.')
                record['findings'].remove(finding);detail={'action':'remove_finding','finding':copy.deepcopy(finding)}
        record['version']=uuid.uuid4().hex
        record['audit'].append({**detail,'at':now(),'case_revision':c['revision'],'version':record['version']})
        if not c['id'].isalnum():raise ValueError('Invalid case identifier.')
        folder=CASES/c['id'];folder.mkdir(parents=True,exist_ok=True)
        target=folder/'evidence.json';temp=folder/'evidence.json.tmp'
        temp.write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf-8');temp.replace(target)
        c['evidence_record']=record
        return report_evidence.check_record(c,record)


def save_manifest(c):
    folder=CASES/c['id']
    folder.mkdir(parents=True,exist_ok=True)
    info={k:c[k] for k in ('id','title','created','synthetic','source','annotations')}
    (folder/'case.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    (ROOT/'work'/'active-case.json').write_text(json.dumps({'id':c['id']}))


def load_case(data, saved=None):
    m=load_volume(data['mask'],True,data.get('assume_mm',False)) if data.get('mask') else None
    if data.get('dicom_folder'):
        s=load_dicom(data['dicom_folder'],data['series_id'])
    else:
        s=load_volume(data['scan'],False,data.get('assume_mm',False)) if data.get('scan') else None
    names=label_names(data.get('labels')) if data.get('labels') else {int(k):str(v) for k,v in data.get('label_definitions',{}).items()}
    result=set_current(m,s,names,(saved or {}).get('title') or data.get('title') or 'Imported research case',
                       ident=(saved or {}).get('id'),synthetic=(saved or {}).get('synthetic',False),source=dict(data))
    if saved:
        get_current()['created']=saved['created']
        get_current()['annotations']=saved.get('annotations',[])
    if m is not None and data.get('evaluation_file') and data.get('reference'):
        evidence=json.loads(Path(data['evaluation_file']).read_text())
        with Path(data['mask']).open('rb') as f:
            unchanged=hashlib.file_digest(f,'sha256').hexdigest()==evidence.get('prediction_sha256')
        if unchanged:
            ref=load_volume(data['reference'],True)
            if same_grid(m,ref):
                get_current()['reference']=ref
                get_current()['evaluation']={**evidence,'case_id':get_current()['id'],'revision':get_current()['revision']}
    save_manifest(get_current())
    return public_case()


def refresh_mask(c):
    c['rows']=analyse(c['mask'],c['names'])
    c['revision']=uuid.uuid4().hex[:8]
    c['meshes'].clear()
    c['analysis']=None
    c['reference']=None
    c['evaluation']=None
    c['dirty']=True


def perform_edit(data):
    c=current_matches(data)
    m=c['mask']
    if m is None:
        raise ValueError('Open a label mask first.')
    axis=int(data['axis'])
    if axis not in (0,1,2):
        raise ValueError('Invalid plane.')
    point=np.asarray(data['point'],dtype=int)
    if point.shape!=(3,) or np.any(point<0) or np.any(point>=m.data.shape):
        raise ValueError('Point is outside the volume.')
    label=int(data.get('label',0))
    if not 0<=label<=65535:
        raise ValueError('Invalid label ID.')
    radius=float(data.get('radius',1))
    if not .1<=radius<=10:
        raise ValueError('Brush radius must be between 0.1 and 10 mm.')
    axes=[i for i in range(3) if i!=axis]
    basis=m.affine[:3,axes]
    min_scale=np.linalg.svd(basis,compute_uv=False).min()
    span=int(np.ceil(radius/min_scale))
    indices=[]
    for a in range(max(0,point[axes[0]]-span),min(m.data.shape[axes[0]],point[axes[0]]+span+1)):
        for b in range(max(0,point[axes[1]]-span),min(m.data.shape[axes[1]],point[axes[1]]+span+1)):
            if np.linalg.norm(basis @ np.array([a-point[axes[0]],b-point[axes[1]]]))<=radius:
                p=point.copy();p[axes[0]]=a;p[axes[1]]=b;indices.append(p)
    xyz=np.array(indices)
    index=tuple(xyz.T)
    old=m.data[index].copy()
    with LOCK:
        c['undo'].append((xyz,old))
        c['undo']=c['undo'][-30:]
        m.data[index]=label
        refresh_mask(c)
    return public_case()


def export_report(c):
    b=c['mask'] if c['mask'] is not None else c['scan']
    result={'case':display_title(c['title']),'created':now(),'classification':'SIMULATED illustration' if c['synthetic'] else 'CALCULATED research measurements',
            'source_geometry_ras_mm':b.affine.tolist(),'shape':list(b.data.shape),'voxel_volume_mm3':b.voxel_mm3,
            'volume_method':'N_label × abs(det(A)); A is the spatial affine in mm',
            'bounds_method':'Axis-aligned RAS bounds of complete labelled voxel cells; not anatomical tooth lengths',
            'structures':report_evidence.rows_for(c),'annotations':c['annotations'],'notes':b.notes,
            'limitations':'Input calibration and segmentation quality determine accuracy. No clinical validation.',
            'software':{k:importlib.metadata.version(k) for k in ('numpy','nibabel','SimpleITK','scikit-image')},
            'modified_mask':c['dirty'],'advanced_analysis':c.get('analysis'),'reference_evaluation':c.get('evaluation'),
            'dataset_import':c['source'].get('dataset_import'),
            'model_provenance':c['source'].get('provenance'),
            'evidence_uncertainty':report_evidence.check_record(c,evidence_record(c))}
    if c['synthetic']:
        result['simulation_conditions']={'purpose':'Software illustration only; no patient anatomy or training evidence',
            'geometry':'10 ellipsoids, centres (56+36 cos(t), 28+34 sin(t), 36) voxels; t=linspace(0.18, pi-0.18, 10); radii (5.2,6,17) voxels',
            'grid':'112×96×72; spacing (0.4,0.4,0.6) mm; origin (-22.4,-19.2,-21.6) mm',
            'intensity':'Foreground 650 plus Gaussian noise mean 0, standard deviation 12; RNG seed 2026',
            'method':'Analytical rasterization; no physical solver, material, loading, boundary or convergence model',
            'validation':'Synthetic software checks only; discretization affects boundaries; clinical accuracy unknown',
            'source':'app/core.py demo()'}
    return result


async def home(request):
    return Response((STATIC/'index.html').read_text(encoding='utf-8').replace('__TOKEN__',TOKEN),media_type='text/html')


async def api_get(request):
    action=request.path_params['action']
    if action=='evidence':
        c=get_current()
        return JSONResponse(await run_in_threadpool(report_evidence.check_record,c,evidence_record(c)))
    if action=='case':
        return JSONResponse(await run_in_threadpool(public_case))
    if action=='jobs':
        return JSONResponse(jobs.listing())
    if action=='cases':
        rows=[]
        for path in CASES.glob('*/case.json'):
            try:
                item=json.loads(path.read_text());item['title']=display_title(item['title']);rows.append({k:item[k] for k in ('id','title','created','synthetic')})
            except (ValueError,KeyError):
                pass
        return JSONResponse(rows)
    if action=='system':
        def saved_status(path):
            try:
                value=json.loads(path.read_text())
                if value.get('pid') and value.get('stage') in ('waiting_for_download','preparing','preprocessing','training'):
                    try:
                        proc=psutil.Process(value['pid'])
                        if str(ROOT/'app'/'research_run.py') not in proc.cmdline():raise psutil.NoSuchProcess(value['pid'])
                    except psutil.NoSuchProcess:
                        value={**value,'stage':'interrupted','message':'Worker exited. Read the training log before resuming.'}
                if value.get('stage')=='preprocessing':
                    log=ROOT/'work'/'training'/'preprocessing.log'
                    if log.exists():
                        import re
                        with log.open('rb') as stream:
                            stream.seek(max(0,log.stat().st_size-12000));tail=stream.read().decode('utf-8',errors='replace')
                        matches=re.findall(r'Preprocessing cases:.*?(\d+)/(\d+)',tail)
                        if matches:value.update(completed=int(matches[-1][0]),total=int(matches[-1][1]))
                return value
            except (OSError,ValueError):return None
        return JSONResponse({'free_ram_gib':round(psutil.virtual_memory().available/2**30,1),
            'runtime_ready':importlib.util.find_spec('torch') is not None and importlib.util.find_spec('nnunetv2') is not None,
            'models':len(capabilities.discover_models(ROOT)[0]),'workspace':str(ROOT),
            'version':'0.3 research','dataset_archives':[{'name':p.name,'bytes':p.stat().st_size} for p in (ROOT/'data').glob('*.zip')],
            'acquisition':saved_status(ROOT/'data'/'acquisition.json'),
            'training':saved_status(ROOT/'work'/'training'/'status.json')})
    if action=='datasets':
        path=ROOT/'config'/'datasets'/'dataset_registry.json'
        return JSONResponse(json.loads(path.read_text(encoding='utf-8-sig')) if path.is_file() else {'datasets':[],'status':'Research in progress'})
    if action=='capabilities':
        return JSONResponse(await run_in_threadpool(capabilities.status,ROOT))
    if action=='models':
        return JSONResponse((await run_in_threadpool(capabilities.discover_models,ROOT))[0])
    if action=='ai-evidence':
        path=ROOT/'results'/'toothseg-pilot'/'evaluation.json'
        if not path.is_file():return JSONResponse({'cases':[],'completed_cases':0})
        evidence=json.loads(path.read_text())
        return JSONResponse({k:v for k,v in evidence.items() if k!='manifest'})
    raise ValueError('Unknown API action.')


async def api_post(request):
    data=await request.json()
    action=request.path_params['action']
    if action in ('evidence-generate','evidence-add','evidence-review','evidence-remove'):
        if any(j['status'] in ('queued','running','cancelling') for j in jobs.JOBS.values()):
            raise ValueError('Wait for the current image or analysis job before changing the evidence report.')
        return JSONResponse(await run_in_threadpool(update_evidence,data,action))
    if action in ('edit','undo') and any(j['status'] in ('queued','running','cancelling') for j in jobs.JOBS.values()):
        raise ValueError('Wait for the current analysis or image job before modifying the mask.')
    if action=='analyse':
        c=current_matches(data);revision=c['revision']
        def analysis_job(item):
            def progress(done,total):
                item['message']=f'Analysing structure {done} of {total}';jobs.save(item)
            result=analyse_case(c['mask'],c['scan'],[] if c['rows'] is None else c['rows'].to_dict(orient='records'),progress)
            if c['revision']!=revision:raise ValueError('Mask changed; analysis was discarded.')
            result.update(case_id=c['id'],revision=revision,created=now(),synthetic=c['synthetic'])
            if c['synthetic']:result['methods']={**result['methods'],'classification':'SIMULATED geometry; calculated software demonstration only'}
            path=ROOT/'results'/('analysis-'+item['id']+'.json')
            path.write_text(json.dumps(result,indent=2),encoding='utf-8')
            c['analysis']=result
            return {'analysis':True,'case_id':c['id'],'revision':revision,'file':str(path)}
        return JSONResponse({'job':jobs.submit('Quantitative structure analysis',analysis_job)})
    if action=='pair-distance':
        c=current_matches(data)
        if c['mask'] is None:raise ValueError('Load a segmentation mask first.')
        def distance_job(item):
            result=pair_distance(c['mask'],int(data['first']),int(data['second']))
            result.update(case_id=c['id'],revision=c['revision'])
            target=ROOT/'results'/('distance-'+item['id']+'.json')
            target.write_text(json.dumps(result,indent=2),encoding='utf-8')
            return {'pair_distance':result,'file':str(target)}
        return JSONResponse({'job':jobs.submit('Structure boundary distance',distance_job)})
    if action in ('pipeline','infer'):
        managed=ROOT/'work'/'training'/'research.pid'
        if managed.exists():
            try:
                proc=psutil.Process(int(managed.read_text()))
                if str(ROOT/'app'/'research_run.py') in proc.cmdline():
                    raise ValueError('The managed research worker is active. Stop it before starting another segmentation job.')
            except psutil.NoSuchProcess:pass
    if action=='stop-research':
        folder=ROOT/'work'/'training';folder.mkdir(parents=True,exist_ok=True)
        (folder/'STOP').write_text('Requested through the local workspace.')
        return JSONResponse({'ok':True,'message':'Stop requested. The worker will stop at its next safe check; saved work is retained.'})
    if action=='resume-research':
        minutes=int(data.get('minutes',120))
        if not 1<=minutes<=720:raise ValueError('Training budget must be 1–720 minutes.')
        if any(j['status'] in ('queued','running','cancelling') for j in jobs.JOBS.values()):
            raise ValueError('Finish the current local job before resuming training.')
        folder=ROOT/'work'/'training';pidfile=folder/'research.pid'
        if pidfile.exists():
            try:
                process=psutil.Process(int(pidfile.read_text()))
                if str(ROOT/'app'/'research_run.py') in process.cmdline():raise ValueError('The research worker is already active.')
            except psutil.NoSuchProcess:pass
        raw=ROOT/'data'/'nnUNet_raw'/'Dataset701_DentalResearch'
        if not (raw/'dataset.json').exists() or (raw/'PREPARATION_INCOMPLETE').exists() or not (folder/'splits_final.json').exists():
            raise ValueError('The managed dataset must finish preparation first.')
        (folder/'STOP').unlink(missing_ok=True)
        logpath=folder/('resume-'+uuid.uuid4().hex[:10]+'.log')
        with logpath.open('w',encoding='utf-8') as log:
            process=subprocess.Popen([sys.executable,str(ROOT/'app'/'research_run.py'),'--stage','resume','--minutes',str(minutes)],
                cwd=ROOT,env=environment(),stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        pidfile.write_text(str(process.pid))
        return JSONResponse({'message':'Resumed prepared data through preprocessing and bounded training. Check status for actual progress.'})
    if action=='demo':
        m,s,n=demo()
        return JSONResponse(await run_in_threadpool(set_current,m,s,n,'Geometry demonstration','demo',True))
    if action=='pulpy-inspect':
        return JSONResponse(await run_in_threadpool(pulpy3d.inspect_folder,data.get('folder','')))
    if action=='pulpy-import':
        if CURRENT is not None and CURRENT.get('dirty'):
            raise ValueError('Save the current corrections before importing another case.')
        def import_pulpy(item):
            item['message']='Loading Pulpy3D annotation and checking its label encoding and physical grid';jobs.save(item)
            mask,scan,names,source,title=pulpy3d.load_annotation(data)
            result=set_current(mask,scan,names,title,source=source)
            save_manifest(get_current())
            return result
        return JSONResponse({'job':jobs.submit('Import Pulpy3D annotation',import_pulpy)})
    if action=='import':
        return JSONResponse({'job':jobs.submit('Import and analyse case',lambda item:load_case(data))})
    if action=='series':
        return JSONResponse(await run_in_threadpool(dicom_series,data['folder']))
    if action=='open':
        ident=str(data['id'])
        if not ident.isalnum():raise ValueError('Invalid case ID.')
        info=json.loads((CASES/ident/'case.json').read_text())
        def reopen(item):
            return load_case(info['source'],saved=info)
        return JSONResponse({'job':jobs.submit('Open saved case',reopen)})
    if action=='edit':
        return JSONResponse(await run_in_threadpool(perform_edit,data))
    if action=='undo':
        c=current_matches(data)
        if not c['undo']:raise ValueError('No mask edit to undo.')
        xyz,old=c['undo'].pop();c['mask'].data[tuple(xyz.T)]=old
        await run_in_threadpool(refresh_mask,c)
        return JSONResponse(public_case())
    if action=='save':
        c=current_matches(data)
        folder=CASES/c['id'];folder.mkdir(exist_ok=True,parents=True)
        if c['mask'] is not None:
            path=folder/('mask-'+c['revision']+'.nii.gz')
            path.write_bytes(gzip.compress(nifti_bytes(c['mask'])))
            c['source']['mask']=str(path)
        if c['scan'] is not None and not c['source'].get('scan') and not c['source'].get('dicom_folder'):
            path=folder/'scan.nii.gz';path.write_bytes(gzip.compress(nifti_bytes(c['scan'])))
            c['source']['scan']=str(path)
        lp=folder/'labels.json';lp.write_text(json.dumps(c['names']),encoding='utf-8');c['source']['labels']=str(lp)
        c['source']['title']=c['title'];c['dirty']=False
        save_manifest(c)
        return JSONResponse(public_case())
    if action=='measure':
        c=current_matches(data);b=c['mask'] if c['mask'] is not None else c['scan']
        p=np.asarray(data['points'],float)
        if p.shape!=(2,3) or not np.all(np.isfinite(p)) or np.any(p<0) or np.any(p>=np.asarray(b.data.shape)):
            raise ValueError('Two image-space points inside the volume are required.')
        world=nib.affines.apply_affine(b.affine,p)
        result={'id':uuid.uuid4().hex[:8],'kind':'point distance','points_ijk':p.tolist(),
                'points_ras_mm':world.tolist(),'distance_mm':float(np.linalg.norm(world[1]-world[0])),
                'method':'Euclidean distance between selected voxel centres after full affine transform',
                'note':'Point-placement and image resolution limit accuracy; not anatomical tooth length.'}
        c['annotations'].append(result)
        if (CASES/c['id']).exists():save_manifest(c)
        return JSONResponse(result)
    if action=='compare':
        c=current_matches(data)
        if c['mask'] is None:raise ValueError('Load a prediction mask first.')
        if not data.get('labels_confirmed'):raise ValueError('Confirm matching label definitions.')
        def evaluation(item):
            reference=load_volume(data['reference'],True,data.get('assume_mm',False))
            result=compare(c['mask'],reference,c['names'])
            result=json.loads(result.to_json(orient='records'))
            if data.get('surface_metrics'):
                for row in result:
                    row.update(surface_agreement(c['mask'],reference,row['label_id']))
            c['reference']=reference
            c['evaluation']={'case_id':c['id'],'revision':c['revision'],'reference_name':reference.name,'rows':result,
                'surface_method':'Boundary voxel centres; HD95 is max of directed 95th percentiles; mean pools both directions.'}
            target=ROOT/'results'/('evaluation-'+item['id']+'.json')
            target.write_text(json.dumps({'case_id':c['id'],'revision':c['revision'],'method':'Voxel Dice, IoU and volume error on identical physical grids; optional boundary-voxel-centre surface distances','results':result},indent=2))
            return {'evaluation':result,'case_id':c['id'],'revision':c['revision'],'file':str(target)}
        return JSONResponse({'job':jobs.submit('Evaluate mask agreement',evaluation)})
    if action=='cancel':
        jobs.cancel(data['id']);return JSONResponse({'ok':True})
    if action=='dataset-inspect':
        def inspect(item):
            meta,pairs=inspect_dataset(data['source'])
            return {'paired_cases':len(pairs),'labels':meta['labels'],'ending':meta['file_ending']}
        return JSONResponse({'job':jobs.submit('Inspect dataset pairs',inspect)})
    if action=='pipeline':
        stage=data['stage']
        if stage not in ('prepare','preprocess','train'):raise ValueError('Invalid pipeline stage.')
        ident=int(data.get('dataset_id',701));fold=int(data.get('fold',0));minutes=int(data.get('minutes',120))
        if not 1<=ident<=999 or fold not in range(5) or not 1<=minutes<=1440:raise ValueError('Invalid job settings.')
        args=[sys.executable,str(Path(__file__).parent/'pipeline.py'),stage,'--dataset-id',str(ident)]
        if stage=='prepare':args+=['--source',data['source']]
        if stage=='train':
            if not data.get('split_reviewed'):raise ValueError('Review patient-level split independence first.')
            args+=['--fold',str(fold),'--patient-split-reviewed']
        return JSONResponse({'job':jobs.submit(stage.title()+' dataset '+str(ident),lambda item:jobs.run_process(item,args,minutes),True)})
    if action=='infer':
        c=current_matches(data)
        if c['scan'] is None or c['synthetic']:raise ValueError('Import a real scan first.')
        if not data.get('trusted'):raise ValueError('Confirm the model source is trusted.')
        dataset,folds=model_info(data['model'])
        chosen=data.get('folds') or folds[:1]
        if not set(chosen).issubset(folds):raise ValueError('Unknown model fold.')
        # Run the existing guarded integration in a child so it can be stopped safely.
        run=ROOT/'results'/('inference-'+uuid.uuid4().hex[:10]);run.mkdir(parents=True)
        scan_path=run/'scan.nii.gz';scan_path.write_bytes(gzip.compress(nifti_bytes(c['scan'])))
        config={'scan':str(scan_path),'model':data['model'],'folds':chosen,'checkpoint':data.get('checkpoint','checkpoint_final.pth')}
        cp=run/'request.json';cp.write_text(json.dumps(config))
        args=[sys.executable,str(Path(__file__).parent/'run_inference.py'),str(cp)]
        def inference(item):
            jobs.run_process(item,args,int(data.get('minutes',60)))
            if item['event'].is_set():return None
            result=json.loads((run/'result.json').read_text())
            m=load_volume(result['mask'],True)
            if not same_grid(m,c['scan']):raise ValueError('Prediction geometry mismatch.')
            new=set_current(m,c['scan'],{v:k for k,v in dataset['labels'].items()},display_title(c['title'])+' · model prediction',source={'mask':result['mask'],'scan':str(scan_path),'provenance':result.get('provenance',{'model':data['model'],'checkpoint':config['checkpoint'],'folds':chosen,'clinical_validation':False})})
            lp=CASES/new['id'];lp.mkdir(parents=True,exist_ok=True)
            labels=lp/'labels.json';labels.write_text(json.dumps(get_current()['names']));get_current()['source']['labels']=str(labels)
            save_manifest(get_current())
            return public_case()
        return JSONResponse({'job':jobs.submit('Segment scan with local model',inference,True)})
    raise ValueError('Unknown action.')


async def upload(request):
    folder=ROOT/'work'/'uploads'/uuid.uuid4().hex[:12];folder.mkdir(parents=True)
    form=await request.form(max_files=3,max_fields=6,max_part_size=512*1024*1024)
    data={'title':str(form.get('title','Imported case')),'assume_mm':form.get('assume_mm')=='true'}
    for key in ('mask','scan','labels'):
        f=form.get(key)
        if f and getattr(f,'filename',None):
            name=Path(f.filename).name
            ending='.nii.gz' if name.lower().endswith('.nii.gz') else Path(name).suffix.lower()
            if ending not in ('.nii','.nii.gz','.mha','.json'):raise ValueError('Unsupported upload type.')
            path=folder/(key+ending);total=0
            with path.open('wb') as out:
                while chunk:=await f.read(1024*1024):
                    total+=len(chunk)
                    if total>512*1024*1024:raise ValueError('File exceeds 512 MiB; use a local path instead.')
                    out.write(chunk)
            data[key]=str(path)
    return JSONResponse({'job':jobs.submit('Import uploaded images',lambda item:load_case(data))})


def render_slice(params):
    if 'case_id' in params or 'revision' in params:
        current_matches(params)
    c=get_current();b=c['mask'] if c['mask'] is not None else c['scan']
    axis=int(params.get('axis',2))
    if axis not in (0,1,2):raise ValueError('Invalid slice plane.')
    index=int(params.get('index',b.data.shape[axis]//2))
    if not 0<=index<b.data.shape[axis]:raise ValueError('Invalid slice.')
    data=np.take(c['scan'].data,index,axis=axis).T if c['scan'] is not None else np.zeros(tuple(b.data.shape[i] for i in range(3) if i!=axis)[::-1])
    low=float(params.get('low',0));high=float(params.get('high',1000))
    grey=(np.clip((data-low)/max(high-low,1e-9),0,1)*255).astype(np.uint8)
    rgb=np.stack([grey]*3,axis=-1)
    alpha=np.clip(float(params.get('alpha',.45)),0,1)
    selected={int(x) for x in params.get('labels','').split(',') if x.isdigit()}
    if c['mask'] is not None and alpha:
        labels=np.take(c['mask'].data,index,axis=axis).T
        for label in selected:
            hit=labels==label;h=PALETTE[label%len(PALETTE)].lstrip('#');col=np.array([int(h[i:i+2],16) for i in (0,2,4)])
            rgb[hit]=((1-alpha)*rgb[hit]+alpha*col).astype(np.uint8)
    if params.get('mode')=='disagreement':
        if c.get('reference') is None:raise ValueError('Compare with a reference mask before displaying disagreement.')
        predicted=np.take(c['mask'].data,index,axis=axis).T
        reference=np.take(c['reference'].data,index,axis=axis).T
        rgb=np.stack([grey]*3,axis=-1)
        for hit,col in (( (predicted>0)&(reference==0),[245,146,63]),((predicted==0)&(reference>0),[72,190,245]),
                        ((predicted>0)&(reference>0)&(predicted!=reference),[231,97,193])):
            rgb[hit]=((1-alpha)*rgb[hit]+alpha*np.array(col)).astype(np.uint8)
    output=io.BytesIO();Image.fromarray(np.flipud(rgb)).save(output,format='PNG')
    return output.getvalue()


async def slice_image(request):
    return Response(await run_in_threadpool(render_slice,request.query_params),media_type='image/png',headers={'Cache-Control':'no-store'})


def get_mesh(label,step):
    c=get_current()
    if c['mask'] is None:raise ValueError('No segmentation mask.')
    key=(int(label),int(step))
    if key not in c['meshes']:
        v,f=mesh(c['mask'],int(label),int(step))
        c['meshes'][key]={'vertices':np.round(v,4).tolist(),'faces':f.tolist(),'label':int(label),'step':int(step)}
    return c['meshes'][key]


async def mesh_data(request):
    step=int(request.query_params.get('step',2))
    if step not in (1,2,3,4):raise ValueError('Invalid surface detail.')
    return JSONResponse(await run_in_threadpool(get_mesh,int(request.path_params['label']),step))


async def download(request):
    c=get_current();kind=request.path_params['kind']
    if kind in ('analysis.json','analysis.csv'):
        result=c.get('analysis')
        if not result:raise ValueError('Run quantitative analysis for this mask revision first.')
        if kind=='analysis.json':
            content=json.dumps(result,indent=2);media='application/json'
        else:
            import csv
            stream=io.StringIO();rows=[]
            for source in result['structures']:
                row={k:v for k,v in source.items() if k not in ('intensity','review_flags')}
                row['review_flags']='; '.join(source['review_flags'])
                row.update({'intensity_'+k:v for k,v in (source.get('intensity') or {}).items() if k not in ('histogram','units')})
                rows.append(row)
            if rows:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            content=stream.getvalue();media='text/csv'
    elif kind=='report.json':
        content=json.dumps(export_report(c),indent=2);media='application/json'
    elif kind=='evidence.json':
        content=json.dumps(report_evidence.check_record(c,evidence_record(c)),indent=2);media='application/json'
    elif kind=='measurements.csv':
        if c['rows'] is None:raise ValueError('No mask measurements.')
        content=c['rows'].to_csv(index=False);media='text/csv'
    elif kind=='report.html':
        r=export_report(c)
        table=c['rows'].to_html(index=False,escape=True,float_format=lambda x:f'{x:.4g}') if c['rows'] is not None else '<p>No mask measurements.</p>'
        table=report_evidence.html_section(r['evidence_uncertainty'])+table
        content='<!doctype html><meta charset="utf-8"><title>Dental research report</title><style>body{font:15px system-ui;color:#203343;margin:42px}table{font-size:11px;border-collapse:collapse}td,th{padding:6px;border:1px solid #dde4e9}pre{white-space:pre-wrap;font-size:12px}h1{color:#167d75}@media print{button{display:none}}</style><button onclick="window.print()">Print / save PDF</button><h1>'+html.escape(display_title(c['title']))+'</h1><p>'+html.escape(r['classification'])+' · Research use only</p>'+table+'<h2>Methods, provenance and limitations</h2><pre>'+html.escape(json.dumps({k:v for k,v in r.items() if k!='structures'},indent=2))+'</pre>'
        media='text/html'
    elif kind=='mask.nii.gz':
        if c['mask'] is None:raise ValueError('No mask.')
        content=gzip.compress(nifti_bytes(c['mask']));media='application/gzip'
    elif kind=='surface.obj':
        m=await run_in_threadpool(get_mesh,int(request.query_params['label']),1)
        content=obj_bytes(np.asarray(m['vertices']),np.asarray(m['faces']));media='text/plain'
    else:raise ValueError('Unknown export.')
    return Response(content,media_type=media,headers={'Content-Disposition':f'attachment; filename="{kind}"'})


class LocalGuard(BaseHTTPMiddleware):
    async def dispatch(self,request,call_next):
        host=request.headers.get('host','')
        if host not in ('127.0.0.1:8517','localhost:8517'):
            return JSONResponse({'error':'Local access only'},403)
        if request.method not in ('GET','HEAD','OPTIONS'):
            origin=request.headers.get('origin')
            if origin and origin not in ('http://127.0.0.1:8517','http://localhost:8517','http://testserver'):
                return JSONResponse({'error':'Cross-origin writes are blocked'},403)
            if not secrets.compare_digest(request.headers.get('x-local-token',''),TOKEN):
                return JSONResponse({'error':'Reload the application before making changes'},403)
        try:
            response=await call_next(request)
        except (ValueError,KeyError,FileNotFoundError,TypeError) as exc:
            return JSONResponse({'error':str(exc)},400)
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Cache-Control']='no-store' if request.url.path.startswith('/api') else 'no-cache'
        return response


app=Starlette(routes=[Route('/',home),Route('/api/{action}',api_get,methods=['GET']),
    Route('/api/{action}',api_post,methods=['POST']),Route('/upload',upload,methods=['POST']),
    Route('/slice',slice_image),Route('/mesh/{label:int}',mesh_data),Route('/export/{kind}',download),
    Mount('/static',app=StaticFiles(directory=STATIC),name='static')])
app.add_middleware(LocalGuard)
