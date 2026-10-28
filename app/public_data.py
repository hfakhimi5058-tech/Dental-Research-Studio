"""Fetch labelled 3D STS members from public split ZIP ranges, with CRC checks.

The archive index and publisher metadata were retrieved from Zenodo. This avoids
downloading unrelated 2D images and unlabelled scans that this supervised pipeline
cannot yet use. No user data or authentication is transmitted.
"""
from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import shutil
import struct
import time
import zlib
import httpx

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/'config'/'datasets'
DEST=ROOT/'data'/'STS-Tooth'
STATUS=DEST/'acquisition.json'


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2))
    try:tmp.replace(path)
    except PermissionError:
        with path.open('w',encoding='utf-8') as output:
            output.write(tmp.read_text());output.flush();os.fsync(output.fileno())


def save(stage,**kwargs):
    write(STATUS,{'stage':stage,'updated':time.time(),**kwargs})
    print(json.dumps({'stage':stage,**kwargs}),flush=True)


def check():
    if (ROOT/'work'/'stop-public-data').exists():raise InterruptedError('Public data acquisition stopped.')
    if shutil.disk_usage(ROOT).free < 100*2**30:raise RuntimeError('Disk reserve reached.')


class SplitArchive:
    def __init__(self,files):
        self.files=sorted(files,key=lambda f:int(f['key'].rsplit('.',1)[1]))
        self.client=httpx.Client(timeout=60,follow_redirects=True)

    def chunks(self,offset,length):
        for part in self.files:
            if offset >= part['size']:
                offset-=part['size'];continue
            count=min(length,part['size']-offset)
            if count<=0:break
            end=offset+count-1;received=0
            with self.client.stream('GET',part['links']['self'],headers={'Range':f'bytes={offset}-{end}'}) as r:
                if r.status_code!=206 or r.headers.get('content-range')!=f'bytes {offset}-{end}/{part["size"]}':
                    raise RuntimeError('Server did not honor the exact archive range; no full-file fallback.')
                for chunk in r.iter_bytes(1024*1024):
                    check();received+=len(chunk)
                    if received>count:raise RuntimeError('Range response exceeded its declared length.')
                    yield chunk
            if received!=count:raise RuntimeError('Truncated range response.')
            length-=count;offset=0
            if length==0:return
        if length:raise ValueError('Requested offset exceeds the split archive.')

    def read(self,offset,length):return b''.join(self.chunks(offset,length))


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    metadata=json.loads((EVIDENCE/'zenodo-10597292.json').read_text())
    if metadata['metadata']['license']['id']!='cc-by-4.0':raise ValueError('Unreviewed licence.')
    rows=json.loads((EVIDENCE/'sts-zip-index.json').read_text())
    chosen=[r for r in rows if '/STS-3D-Tooth/' in r['name'] and '/Labeled/' in r['name'] and r['name'].endswith('.nii.gz')]
    for row in chosen:
        # ZIP64 extra values occur only for fields whose normal value overflowed.
        blob=bytes.fromhex(row['extra']);p=0
        while p+4<=len(blob):
            kind,length=struct.unpack_from('<HH',blob,p);p+=4;payload=blob[p:p+length];p+=length
            if kind==1:
                q=0
                for key in ('size','compressed','offset'):
                    if row[key]==0xffffffff:row[key]=struct.unpack_from('<Q',payload,q)[0];q+=8
        if row['disk']!=0 or row['method'] not in (0,8):raise ValueError('Unsupported ZIP member encoding.')
    source={'record':'https://zenodo.org/records/10597292','licence':'CC-BY-4.0','publisher_metadata':metadata['metadata'],
            'scope':'Only labelled 3D Integrity and ROI members. Cropped ROI/full-volume patient correspondence is not established.',
            'training_status':'Not yet admitted to training; geometry, label semantics and patient overlap require audit.',
            'excluded':'Unlabelled and 2D data: no compatible training objective is implemented.'}
    write(DEST/'source.json',source)
    manifest_path=DEST/'files.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    stream=SplitArchive(metadata['files'])
    try:
        for index,row in enumerate(chosen):
            check();name=row['name'];parts=PurePosixPath(name).parts
            if '..' in parts or ':' in name or '\\' in name:raise ValueError('Unsafe archive path.')
            target=DEST.joinpath(*parts[2:]).resolve();target.relative_to(DEST.resolve())
            if name in manifest and target.is_file() and target.stat().st_size==row['size']:continue
            save('downloading',file='/'.join(parts[2:]),completed=index,total=len(chosen),selected_bytes=sum(r['compressed'] for r in chosen))
            head=stream.read(row['offset'],30)
            if head[:4]!=b'PK\x03\x04':raise ValueError('Archive member offset does not match a ZIP header.')
            fields=struct.unpack('<IHHHHHIIIHH',head)
            if fields[2]&1 or fields[3]!=row['method']:raise ValueError('Unexpected encrypted/encoded member.')
            local_name=stream.read(row['offset']+30,fields[9]).decode()
            if local_name!=name:raise ValueError('Archive index/header filename mismatch.')
            start=row['offset']+30+fields[9]+fields[10]
            target.parent.mkdir(parents=True,exist_ok=True);temp=target.with_suffix(target.suffix+'.part')
            decoder=zlib.decompressobj(-15) if row['method']==8 else None
            crc=0;size=0;digest=hashlib.sha256()
            with temp.open('wb') as out:
                for chunk in stream.chunks(start,row['compressed']):
                    raw=decoder.decompress(chunk) if decoder else chunk
                    size+=len(raw)
                    if size>row['size']:raise ValueError('Decompressed data exceeds declared size.')
                    crc=zlib.crc32(raw,crc);digest.update(raw);out.write(raw)
                if decoder:
                    raw=decoder.flush();size+=len(raw);crc=zlib.crc32(raw,crc);digest.update(raw);out.write(raw)
                    if not decoder.eof:raise ValueError('Incomplete compressed stream.')
            if size!=row['size'] or crc!=row['crc']:raise ValueError('ZIP integrity check failed; file not admitted.')
            temp.replace(target)
            manifest[name]={'path':str(target),'bytes':size,'sha256':digest.hexdigest(),'zip_crc32':f'{crc:08x}'}
            write(manifest_path,manifest)
        save('downloaded',files=len(manifest),message='Labelled 3D sources acquired and CRC checked. Training admission remains pending a label/overlap audit.')
    finally:stream.client.close()


if __name__=='__main__':
    try:main()
    except Exception as exc:
        DEST.mkdir(parents=True,exist_ok=True);save('stopped' if isinstance(exc,InterruptedError) else 'failed',message=str(exc));raise
