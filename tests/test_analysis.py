"""Analytical geometry fixtures and API regression checks; no clinical claims."""
import io,json,sys,time,unittest
import tempfile
from unittest.mock import patch,Mock
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'app'))
from core import Volume,analyse
from analysis import shape_metrics,intensity_summary,pair_distance,surface_agreement,analyse_case
from pipeline import environment
import server
from starlette.testclient import TestClient


class AnalysisChecks(unittest.TestCase):
    def test_exposed_faces_anisotropic_box(self):
        # 3x4x5 cells at 2x3x4 mm => side lengths 6,12,20 mm.
        result=shape_metrics(np.ones((3,4,5),bool),np.diag([2.,3.,4.,1.]))
        self.assertAlmostEqual(result['voxel_face_area_mm2'],2*(6*12+6*20+12*20))
        self.assertEqual(result['components'],1)
        self.assertEqual(result['largest_component_percent'],100)

    def test_full_affine_face_area_and_mesh_scaling(self):
        a=np.eye(4);a[:3,:3]=[[2,.5,0],[0,3,0],[0,0,4]]
        cube=np.ones((1,1,1),bool)
        r=shape_metrics(cube,a)
        self.assertAlmostEqual(r['voxel_face_area_mm2'],2*(4*np.sqrt(9.25)+8+6))
        b=a.copy();b[:3,:3]*=2
        self.assertAlmostEqual(shape_metrics(cube,b)['surface_area_mm2'],4*r['surface_area_mm2'])

    def test_fragmentation_and_all_voxel_intensity(self):
        labels=np.zeros((6,6,6),np.uint16);labels[1:3,1:3,1:3]=1;labels[5,5,5]=1
        volume=Volume(labels,np.eye(4),'fixture');scan=Volume(np.arange(216).reshape(6,6,6),np.eye(4),'scan')
        r=analyse_case(volume,scan,analyse(volume).to_dict(orient='records'))['structures'][0]
        self.assertEqual(r['components'],2);self.assertEqual(r['detached_voxels'],1)
        self.assertAlmostEqual(r['largest_component_percent'],800/9)
        self.assertEqual(len(r['review_flags']),2)
        expected=scan.data[labels==1]
        self.assertAlmostEqual(r['intensity']['mean'],expected.mean())
        self.assertAlmostEqual(r['intensity']['std'],expected.std())

    def test_sample_limit_and_exact_moments(self):
        d=np.arange(600_000,dtype=np.float64).reshape(100,100,60)
        r=intensity_summary(d)
        self.assertLessEqual(r['sample_count'],200_000)
        self.assertEqual(r['count'],d.size);self.assertEqual(r['min'],0);self.assertEqual(r['max'],599999)
        self.assertAlmostEqual(r['mean'],d.mean(),places=6);self.assertAlmostEqual(r['std'],d.std(),places=6)
        self.assertEqual(sum(r['histogram']['counts']),r['sample_count'])

    def test_distances_known_points_and_missing_label(self):
        labels=np.zeros((8,8,8),np.uint16);labels[1,1,1]=1;labels[4,1,1]=2
        a=np.diag([2.,3.,4.,1.]);volume=Volume(labels,a,'fixture')
        self.assertEqual(pair_distance(volume,1,2)['boundary_centres_distance_mm'],6)
        ref=np.zeros_like(labels);ref[4,1,1]=1
        result=surface_agreement(volume,Volume(ref,a,'reference'),1)
        self.assertEqual(result['surface_hd95_mm'],6);self.assertEqual(result['surface_mean_distance_mm'],6)
        self.assertIsNone(surface_agreement(volume,Volume(ref,a,'reference'),2)['surface_hd95_mm'])

    def test_planning_and_training_environment_separation(self):
        self.assertGreater(int(environment()['nnUNet_n_proc_DA']),0)
        self.assertEqual(environment(training=True)['nnUNet_n_proc_DA'],'0')

    def test_analysis_export_and_revision_invalidation(self):
        data=np.zeros((6,6,6),np.uint16);data[1:4,1:4,1:4]=1
        c=server.set_current(Volume(data,np.eye(4),'test'),None,{1:'Fixture'},'test','analysisfixture',True)
        current=server.get_current();current['analysis']=analyse_case(current['mask'],None,current['rows'].to_dict(orient='records'))
        client=TestClient(server.app,base_url='http://127.0.0.1:8517');headers={'X-Local-Token':server.TOKEN}
        self.assertEqual(client.get('/export/analysis.json').status_code,200)
        self.assertIn('surface_area_mm2',client.get('/export/analysis.csv').text)
        response=client.post('/api/edit',headers=headers,json={'case_id':c['id'],'revision':c['revision'],'point':[4,4,4],'axis':2,'label':1,'radius':.1})
        self.assertEqual(response.status_code,200);self.assertIsNone(response.json()['analysis'])
        self.assertEqual(client.get('/export/analysis.json').status_code,400)

    def test_reference_disagreement_colours(self):
        data=np.zeros((5,5,5),np.uint16);ref=data.copy()
        data[1,1,2]=1;ref[2,1,2]=1;data[3,1,2]=1;ref[3,1,2]=2
        server.set_current(Volume(data,np.eye(4),'prediction'),None,{},'fixture')
        server.get_current()['reference']=Volume(ref,np.eye(4),'reference')
        image=np.array(Image.open(io.BytesIO(server.render_slice({'axis':'2','index':'2','mode':'disagreement','alpha':'1'}))))
        np.testing.assert_array_equal(image[3,1],[245,146,63])
        np.testing.assert_array_equal(image[3,2],[72,190,245])
        np.testing.assert_array_equal(image[3,3],[231,97,193])

    def test_resume_rejects_invalid_budget_and_existing_worker(self):
        client=TestClient(server.app,base_url='http://127.0.0.1:8517');headers={'X-Local-Token':server.TOKEN}
        self.assertEqual(client.post('/api/resume-research',headers=headers,json={'minutes':0}).status_code,400)
        with tempfile.TemporaryDirectory(dir=ROOT/'work'/'temp') as folder:
            test_root=Path(folder);pidfile=test_root/'work'/'training'/'research.pid'
            pidfile.parent.mkdir(parents=True);pidfile.write_text('123')
            proc=Mock();proc.cmdline.return_value=[str(test_root/'app'/'research_run.py')]
            with patch.object(server,'ROOT',test_root),patch.object(server.psutil,'Process',return_value=proc),patch.object(server.subprocess,'Popen') as launch:
                response=client.post('/api/resume-research',headers=headers,json={'minutes':120})
                self.assertEqual(response.status_code,400);self.assertIn('already active',response.json()['error'])
                launch.assert_not_called()


if __name__=='__main__':unittest.main(verbosity=2)
