import json
from pathlib import Path
import sys
from core import load_volume
from pipeline import predict

if __name__=='__main__':
    request=Path(sys.argv[1])
    data=json.loads(request.read_text())
    scan=load_volume(data['scan'])
    mask,meta,folder=predict(scan,data['model'],data['folds'],data['checkpoint'])
    result={'mask':str(folder/'output'/('case'+meta['file_ending']))}
    if (folder/'provenance.json').is_file():
        result['provenance']=json.loads((folder/'provenance.json').read_text())
    (request.parent/'result.json').write_text(json.dumps(result))
    print('Segmentation completed; inspect the prediction before using measurements.')
