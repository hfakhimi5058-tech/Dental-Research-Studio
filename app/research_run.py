"""Audited TF2 ingestion and a bounded, resumable nnU-Net research run.

No external requests, synthetic training data, downloads or hidden retries.
The official archive remains the immutable source; conversions are checkpointed.
"""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
from pipeline import environment, executable, normalized_metadata

WORK = ROOT / 'work' / 'training'
STATUS = WORK / 'status.json'
STOP = WORK / 'STOP'
RAW = ROOT / 'data' / 'nnUNet_raw' / 'Dataset701_DentalResearch'
SOURCE = ROOT / 'data' / 'ToothFairy2'
HOLDOUT = ROOT / 'data' / 'holdout' / 'ToothFairy2'
RESERVE = 100 * 2**30


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    try:
        temp.replace(path)
    except PermissionError:
        # Windows readers may allow writes but deny delete/rename sharing.
        # The complete temporary copy is retained if replacement is unavailable.
        with path.open('w',encoding='utf-8') as output:
            output.write(temp.read_text(encoding='utf-8'));output.flush();os.fsync(output.fileno())


def status(stage, **values):
    write_json(STATUS, {'stage':stage, 'updated':time.time(), 'pid':os.getpid(), **values})
    print(json.dumps({'stage':stage, **values}), flush=True)


def check():
    if STOP.exists():
        raise InterruptedError('Stopped by the user; completed conversions and checkpoints are preserved.')
    if shutil.disk_usage(ROOT).free < RESERVE:
        raise RuntimeError('Less than the configured 100 GiB disk reserve remains; stopped before further writes.')


def wait_archive():
    target = ROOT / 'data' / 'ToothFairy2_Dataset.zip'
    deadline = time.monotonic() + 6*3600
    status('waiting_for_download', message='Waiting for the completed, recognized archive to be moved into data/.')
    while not target.is_file():
        check()
        acquisition = ROOT / 'data' / 'acquisition.json'
        if acquisition.exists():
            info = json.loads(acquisition.read_text())
            if info.get('stage') in ('failed','stopped'):
                raise RuntimeError('Acquisition requires attention: '+info.get('message',''))
        if time.monotonic() > deadline:
            raise TimeoutError('Archive wait reached its six-hour limit.')
        time.sleep(15)
    return target


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(8*1024*1024):
            check()
            h.update(chunk)
    return h.hexdigest()


def resolve_duplicates(manifest):
    """Retain one identical annotation; quarantine every conflicting duplicate."""
    import nibabel as nib
    import numpy as np
    groups={}
    for case,entry in manifest.items():groups.setdefault(entry['voxel_sha256'],[]).append(case)
    review=[]
    for digest,cases in groups.items():
        if len(cases)<2:continue
        cases=sorted(cases);reference=nib.load(manifest[cases[0]]['label'])
        data=np.asanyarray(reference.dataobj)
        identical=True
        for case in cases[1:]:
            candidate=nib.load(manifest[case]['label'])
            if candidate.shape!=reference.shape or not np.allclose(candidate.affine,reference.affine) or not np.array_equal(data,np.asanyarray(candidate.dataobj)):
                identical=False;break
        excluded=cases[1:] if identical else cases
        reason='Redundant identical image and annotation' if identical else 'Identical image content with conflicting annotations or physical geometry'
        for case in cases:
            manifest[case]['duplicate_group']='image-sha256:'+digest
        for case in excluded:
            entry=manifest[case]
            if entry['split']=='quarantined':continue
            for role in ('image','label'):
                source=Path(entry[role]).resolve()
                target=(ROOT/'data'/'quarantine'/'ToothFairy2'/role/source.name).resolve()
                # Explicitly validate both resolved paths before every file move.
                source.relative_to(ROOT.resolve());target.relative_to(ROOT.resolve())
                if not (source.is_relative_to(RAW.resolve()) or source.is_relative_to(HOLDOUT.resolve())):
                    raise ValueError('Unexpected source location during duplicate quarantine.')
                if target.exists():raise ValueError('Quarantine target exists; no file overwritten.')
                target.parent.mkdir(parents=True,exist_ok=True);source.replace(target);entry[role]=str(target)
            entry.update(previous_split=entry['split'],split='quarantined',fold=None,exclusion_reason=reason)
        review.append({'cases':cases,'kept':cases[:1] if identical else [],'excluded':excluded,'reason':reason})
    write_json(WORK/'duplicate-groups.json',review)
    return manifest


def prepare(archive):
    """Extract only each paired case while converting; never train on the held-out set."""
    import nibabel as nib
    import numpy as np
    import SimpleITK as sitk
    from core import load_volume, same_grid
    SOURCE.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        prefix = 'Dataset112_ToothFairy2/'
        original = json.loads(z.read(prefix+'dataset.json'))
        meta = normalized_metadata(original)
        if meta.get('license') != 'CC-BY-SA 4.0':
            raise ValueError('Archive licence differs from the reviewed release; review before ingestion.')
        write_json(SOURCE/'dataset.json', original)
        members = {i.filename:i for i in z.infolist()}
        for name in members:
            p = PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or ':' in name or '\\' in name or not name.startswith(prefix):
                raise ValueError('Unexpected or unsafe archive member: '+name)
        pairs = []
        for name in sorted(members):
            match = re.fullmatch(re.escape(prefix)+r'imagesTr/(ToothFairy2([PF])_(\d+))_0000\.mha', name)
            if match:
                case,cohort,number = match.groups()
                label = prefix+'labelsTr/'+case+'.mha'
                if label not in members:
                    raise ValueError('Reference mask is missing: '+case)
                pairs.append((case,cohort,number,name,label))
        if not pairs:
            raise ValueError('No recognized paired TF2 cases in the archive.')
        label_members = {n for n in members if n.startswith(prefix+'labelsTr/') and n.endswith('.mha')}
        if label_members != {p[4] for p in pairs}:
            raise ValueError('Unexpected or unpaired masks require source inspection.')
        # Cohort-stratified deterministic assignment. All releases of a P/F identity share a group.
        heldout=set(); folds={}
        for cohort in sorted({p[1] for p in pairs}):
            group=sorted([p for p in pairs if p[1]==cohort], key=lambda p:hashlib.sha256(('dental-2026:'+p[0]).encode()).hexdigest())
            reserve=max(1,round(len(group)*.1))
            heldout.update(p[0] for p in group[:reserve])
            for i,p in enumerate(group[reserve:]):folds[p[0]]=i%5
        mapping={old:new for new,old in enumerate(sorted(meta['labels'].values()))}
        lookup=np.zeros(max(mapping)+1,dtype=np.uint16)
        for old,new in mapping.items():lookup[old]=new
        manifest_path=WORK/'cases.json'
        manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        contract={'source':str(archive),'source_bytes':archive.stat().st_size,'source_metadata':original,
                  'original_to_training_id':{str(k):v for k,v in mapping.items()},'seed':'dental-2026','heldout_fraction':.1,
                  'source_url':'https://ditto.ing.unimore.it/toothfairy2/',
                  'grouping':'Publisher P/F patient IDs; shared group across ToothFairy/Maxillo releases.',
                  'independence_limit':'Exact image hashes and source IDs detect known duplicates; unknown repeated/resampled acquisitions need further audit.'}
        contract_path=WORK/'contract.json'
        if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
            raise ValueError('Source or training contract changed. Existing preparation was not modified.')
        write_json(contract_path,contract)
        RAW.mkdir(parents=True,exist_ok=True)
        (RAW/'PREPARATION_INCOMPLETE').write_text('Do not train until ingestion and split audit finish.')
        for index,(case,cohort,number,image_member,label_member) in enumerate(pairs):
            check()
            if case in manifest:
                entry=manifest[case]
                if all(Path(entry[k]).is_file() for k in ('image','label')):
                    continue
                raise ValueError('Completed conversion is missing a file: '+case)
            status('preparing',case=case,completed=index,total=len(pairs))
            scratch=WORK/'current-case';scratch.mkdir(parents=True,exist_ok=True)
            paths=[];hashes=[]
            for role,member in (('image',image_member),('label',label_member)):
                target=scratch/(role+'.mha');digest=hashlib.sha256()
                with z.open(member) as src,target.open('wb') as dst:
                    while chunk:=src.read(8*1024*1024):
                        check();digest.update(chunk);dst.write(chunk)
                # zipfile verifies the member CRC while reading to EOF.
                paths.append(target);hashes.append(digest.hexdigest())
            image,mask=load_volume(paths[0]),load_volume(paths[1],mask=True)
            if not same_grid(image,mask):raise ValueError('Image/mask grid mismatch: '+case)
            labels=set(int(v) for v in np.unique(mask.data))
            if not labels.issubset(mapping):raise ValueError('Undeclared labels in '+case+': '+str(labels-set(mapping)))
            destination=HOLDOUT if case in heldout else RAW
            ip,lp=destination/'imagesTr'/(case+'_0000.nii.gz'),destination/'labelsTr'/(case+'.nii.gz')
            for path,values in ((ip,image.data.astype(np.float32)),(lp,lookup[mask.data])):
                path.parent.mkdir(parents=True,exist_ok=True)
                nii=nib.Nifti1Image(values,image.affine);nii.header.set_xyzt_units('mm');nii.set_sform(image.affine,code=1)
                temporary=path.with_name(path.name.replace('.nii.gz','.partial.nii.gz'))
                nib.save(nii,str(temporary));temporary.replace(path)
            # Header-independent voxel hash detects identical image content despite wrapper metadata.
            voxel_hash=hashlib.sha256()
            for k in range(image.data.shape[2]):voxel_hash.update(np.ascontiguousarray(image.data[:,:,k],dtype=np.float32).tobytes())
            digest=voxel_hash.hexdigest()
            previous=next((c for c,e in manifest.items() if e['voxel_sha256']==digest),None)
            manifest[case]={'group':'maxillo_toothfairy:'+cohort+str(int(number)), 'cohort':cohort,
                'split':'holdout' if case in heldout else 'development','fold':folds.get(case),
                'image':str(ip),'label':str(lp),'source_sha256':hashes,'voxel_sha256':digest,
                'source_labels':sorted(labels),'shape':list(image.data.shape),'affine_ras_mm':image.affine.tolist()}
            if previous:manifest[case]['duplicate_of']=previous
            write_json(manifest_path,manifest)
            del image,mask,values,nii
            for p in paths:p.unlink()  # Only our two bounded scratch files; archive remains unchanged.
        manifest=resolve_duplicates(manifest)
        write_json(manifest_path,manifest)
        development=sorted(c for c,e in manifest.items() if e['split']=='development')
        heldout={c for c,e in manifest.items() if e['split']=='holdout'}
        newmeta={'name':'DentalResearch','channel_names':{'0':'CBCT'},'labels':{n:mapping[i] for n,i in meta['labels'].items()},
                 'numTraining':len(development),'file_ending':'.nii.gz','licence':meta['license'],
                 'reference':meta['reference'],'description':'TF2 development set; separate held-out cases are excluded.'}
        write_json(RAW/'dataset.json',newmeta)
        write_json(RAW/'source_label_mapping.json',contract)
        splits=[{'train':[c for c in development if folds[c]!=f],'val':[c for c in development if folds[c]==f]} for f in range(5)]
        if any(not s['train'] or not s['val'] or set(s['train'])&set(s['val']) for s in splits):
            raise ValueError('Invalid split partition.')
        write_json(WORK/'splits_final.json',splits)
        write_json(HOLDOUT/'dataset.json',{**newmeta,'numTraining':len(heldout),'description':'HELD OUT: excluded from planning, training and model selection.'})
        (RAW/'PREPARATION_INCOMPLETE').unlink()
        status('prepared',cases=len(manifest),development=len(development),heldout=len(heldout),
               quarantined=sum(e['split']=='quarantined' for e in manifest.values()),
               message='Prepared paired real data, retained source labels/checksums, and excluded the hold-out cohort.')


def preprocess():
    check()
    marker=WORK/'preprocessing-complete.json'
    if marker.exists():return
    status('preprocessing',message='One worker, 8 GiB planner target; timing is not yet measured.')
    args=[executable('nnUNetv2_plan_and_preprocess'),'-d','701','--verify_dataset_integrity','-c','3d_fullres',
          '-np','1','-npfp','1','-gpu_memory_target','8']
    with (WORK/'preprocessing.log').open('a',encoding='utf-8') as log:
        process=subprocess.Popen(args,env=environment(),stdout=log,stderr=subprocess.STDOUT,
                                 creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        start=time.monotonic()
        try:
            while process.poll() is None:
                check()
                if time.monotonic()-start>12*3600:raise TimeoutError('Preprocessing exceeded the twelve-hour bound.')
                time.sleep(5)
            if process.returncode:raise RuntimeError('Preprocessing failed; read work/training/preprocessing.log.')
        except BaseException:
            import psutil
            try:
                parent=psutil.Process(process.pid)
                for child in parent.children(recursive=True):child.terminate()
                parent.terminate()
            except psutil.NoSuchProcess:pass
            raise
    preprocessed=ROOT/'data'/'nnUNet_preprocessed'/'Dataset701_DentalResearch'
    shutil.copyfile(WORK/'splits_final.json',preprocessed/'splits_final.json')
    write_json(marker,{'completed':time.time(),'configuration':'3d_fullres','planner_gpu_target_gib':8})


def train(minutes):
    os.environ.update(environment(training=True))
    import torch
    from nnunetv2.run.run_training import get_trainer_from_args,maybe_load_checkpoint
    check()
    if not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable; no CPU training fallback was started.')
    free,total=torch.cuda.mem_get_info()
    if free < 9*2**30:raise RuntimeError('Less than 9 GiB GPU memory is free; training did not start.')
    torch.set_num_threads(2)
    trainer=get_trainer_from_args('701','3d_fullres',0,device=torch.device('cuda'))
    trainer.save_every=1
    latest=Path(trainer.output_folder)/'checkpoint_latest.pth'
    final=Path(trainer.output_folder)/'checkpoint_final.pth'
    if final.exists():
        status('training_schedule_finished',message='Existing final checkpoint retained; independent evaluation remains pending.')
        return
    if latest.exists():maybe_load_checkpoint(trainer,True,False)
    status('training',budget_minutes=minutes,model_folder=str(Path(trainer.output_folder).parent),
           message='Training fold 0 on real data. Patch validation is not independent full-volume evaluation.')
    trainer.on_train_start()
    started=time.monotonic();epochs=0
    # The upstream training steps/optimizer/augmentation are unchanged. Stop only at epoch boundaries.
    try:
        while trainer.current_epoch < trainer.num_epochs:
            check()
            if epochs and time.monotonic()-started >= minutes*60:break
            trainer.on_epoch_start();trainer.on_train_epoch_start()
            outputs=[]
            for _ in range(trainer.num_iterations_per_epoch):
                check();outputs.append(trainer.train_step(next(trainer.dataloader_train)))
            trainer.on_train_epoch_end(outputs)
            with torch.no_grad():
                trainer.on_validation_epoch_start();outputs=[]
                for _ in range(trainer.num_val_iterations_per_epoch):
                    check();outputs.append(trainer.validation_step(next(trainer.dataloader_val)))
                trainer.on_validation_epoch_end(outputs)
            trainer.on_epoch_end();epochs+=1
            status('training',completed_epochs=trainer.current_epoch,budget_minutes=minutes,
                   elapsed_seconds=round(time.monotonic()-started),checkpoint=str(latest),
                   message='Checkpoint saved; training metrics are provisional patch metrics.')
        if trainer.current_epoch >= trainer.num_epochs:
            trainer.on_train_end()
            status('training_schedule_finished',checkpoint=str(final),message='Full-volume held-out evaluation remains pending.')
        else:
            status('budget_complete',completed_epochs=trainer.current_epoch,checkpoint=str(latest),
                   message='Initial GPU time budget reached. Checkpoint is resumable; model training/validation is unfinished.')
    finally:
        for loader in (getattr(trainer,'dataloader_train',None),getattr(trainer,'dataloader_val',None)):
            if hasattr(loader,'_finish'):loader._finish()


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['prepare','preprocess','train','resume','all'],default='all')
    parser.add_argument('--minutes',type=int,default=120)
    args=parser.parse_args()
    if not 1<=args.minutes<=720:parser.error('Training budget must be 1–720 minutes.')
    WORK.mkdir(parents=True,exist_ok=True)
    os.environ.update(environment())
    try:
        if args.stage in ('prepare','all'):prepare(wait_archive())
        if args.stage == 'resume' and ((RAW/'PREPARATION_INCOMPLETE').exists() or not (WORK/'splits_final.json').exists()):
            raise RuntimeError('Complete preparation before resuming preprocessing and training.')
        if args.stage in ('preprocess','resume','all'):preprocess()
        if args.stage in ('train','resume','all'):train(args.minutes)
    except BaseException as exc:
        status('stopped' if isinstance(exc,InterruptedError) else 'failed',message=str(exc),exception=type(exc).__name__)
        raise
