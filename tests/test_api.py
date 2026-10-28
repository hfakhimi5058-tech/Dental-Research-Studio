import io
import sys
from pathlib import Path
import unittest
import numpy as np
from PIL import Image
from starlette.testclient import TestClient

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'app'))
import server


class WorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client=TestClient(server.app,base_url='http://127.0.0.1:8517')
        cls.headers={'X-Local-Token':server.TOKEN}
        cls.case=cls.client.get('/api/case').json()

    def setUp(self):
        response=self.client.post('/api/demo',json={},headers=self.headers)
        self.assertEqual(response.status_code,200)
        self.case=self.client.get('/api/case').json()

    def test_home_offline_assets(self):
        page=self.client.get('/')
        self.assertEqual(page.status_code,200)
        self.assertIn('Research Studio',page.text)
        self.assertIn(server.TOKEN,page.text)
        self.assertEqual(self.client.get('/static/app.js').status_code,200)

    def test_write_requires_token_and_same_origin(self):
        self.assertEqual(self.client.post('/api/demo',json={}).status_code,403)
        self.assertEqual(self.client.post('/api/demo',json={},headers={**self.headers,'Origin':'https://example.org'}).status_code,403)

    def test_case_geometry(self):
        self.assertEqual(self.case['shape'],[112,96,72])
        self.assertEqual(len(self.case['rows']),10)
        self.assertTrue(self.case['synthetic'])

    def test_slice_and_mesh(self):
        response=self.client.get('/slice?axis=2&index=36&labels=1,2&low=0&high=700')
        self.assertEqual(response.status_code,200)
        self.assertEqual(Image.open(io.BytesIO(response.content)).size,(112,96))
        result=self.client.get('/mesh/1?step=2').json()
        self.assertGreater(len(result['faces']),0)
        self.assertEqual(self.client.get('/slice?axis=8').status_code,400)

    def test_distance_uses_affine(self):
        c=self.client.get('/api/case').json()
        response=self.client.post('/api/measure',json={'case_id':c['id'],'revision':c['revision'],'points':[[1,1,1],[4,5,1]]},headers=self.headers)
        self.assertEqual(response.status_code,200)
        self.assertAlmostEqual(response.json()['distance_mm'],2.0)

    def test_paint_and_undo_preserve_original(self):
        c=self.client.get('/api/case').json()
        original=server.CURRENT['mask'].data.copy()
        response=self.client.post('/api/edit',json={'case_id':c['id'],'revision':c['revision'],'point':[2,2,2],'axis':2,'radius':.5,'label':42},headers=self.headers)
        self.assertEqual(response.status_code,200)
        edited=response.json()
        self.assertEqual(server.CURRENT['mask'].data[2,2,2],42)
        response=self.client.post('/api/undo',json={'case_id':edited['id'],'revision':edited['revision']},headers=self.headers)
        self.assertEqual(response.status_code,200)
        np.testing.assert_array_equal(server.CURRENT['mask'].data,original)

    def test_stale_mutation_rejected(self):
        response=self.client.post('/api/edit',json={'case_id':'wrong','revision':'wrong'},headers=self.headers)
        self.assertEqual(response.status_code,400)

    def test_export_provenance(self):
        report=self.client.get('/export/report.json').json()
        self.assertIn('simulation_conditions',report)
        self.assertIn('software',report)
        self.assertEqual(self.client.get('/export/measurements.csv').status_code,200)
        self.assertIn('Simulation',self.client.get('/export/report.html').text.title())

    def test_invalid_training_stage_does_not_launch(self):
        response=self.client.post('/api/pipeline',json={'stage':'invented'},headers=self.headers)
        self.assertEqual(response.status_code,400)


if __name__=='__main__':
    unittest.main(verbosity=2)
