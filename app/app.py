from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('TEMP', str(ROOT / 'work' / 'temp'))
os.environ.setdefault('TMP', str(ROOT / 'work' / 'temp'))
os.environ.setdefault('TORCH_HOME', str(ROOT / 'models' / 'cache'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'work' / 'matplotlib'))
os.environ.setdefault('nnUNet_raw', str(ROOT / 'data' / 'nnUNet_raw'))
os.environ.setdefault('nnUNet_preprocessed', str(ROOT / 'data' / 'nnUNet_preprocessed'))
os.environ.setdefault('nnUNet_results', str(ROOT / 'models' / 'nnUNet_results'))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('server:app', host='127.0.0.1', port=8517, access_log=False)
