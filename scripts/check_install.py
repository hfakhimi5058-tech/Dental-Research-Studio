"""Small offline installation check; no real images or models needed."""
from pathlib import Path
import os, sys
ROOT=Path(__file__).resolve().parents[1]
for folder in ('work/temp','work/pycache'):(ROOT/folder).mkdir(parents=True,exist_ok=True)
os.environ['TEMP']=os.environ['TMP']=str(ROOT/'work/temp')
sys.path.insert(0,str(ROOT/'app'))
from starlette.testclient import TestClient
import server
with TestClient(server.app,base_url='http://127.0.0.1:8517') as client:
    assert client.get('/').status_code==200
    case=client.get('/api/case'); assert case.status_code==200,case.text
    assert client.get('/api/models').status_code==200
    assert client.get('/static/plotly.min.js').status_code==200
    assert client.post('/api/demo',json={}).status_code==403
print('PASS: application, example case, local assets, model listing and request guard')
