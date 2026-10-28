"""Synthetic evidence lifecycle/security tests, not diagnostic validation."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from starlette.testclient import TestClient

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'app'))
from core import Volume
import evidence
import server


class EvidenceChecks(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory(dir=ROOT/'work'/'temp')
        self.patch=patch.object(server,'CASES',Path(self.folder.name));self.patch.start()
        self.jobs=patch.dict(server.jobs.JOBS,{},clear=True);self.jobs.start()
        data=np.zeros((8,8,8),np.uint16);data[2:5,2:5,2:5]=1
        scan=Volume(np.arange(512,dtype=np.float32).reshape(8,8,8),np.diag([2.,3.,4.,1.]),'scan')
        mask=Volume(data,scan.affine.copy(),'mask')
        self.case=server.set_current(mask,scan,{1:'Tooth fixture'},'Fixture','evidencetest')
        self.c=server.CURRENT
        self.client=TestClient(server.app,base_url='http://127.0.0.1:8517')
        self.headers={'X-Local-Token':server.TOKEN}

    def tearDown(self):
        self.jobs.stop();self.patch.stop();self.folder.cleanup()

    def post(self,action,version=None,**data):
        return self.client.post('/api/'+action,headers=self.headers,json={
            'case_id':self.c['id'],'revision':self.c['revision'],'evidence_version':version,**data})

    def generate(self):
        r=self.post('evidence-generate');self.assertEqual(r.status_code,200);return r.json()

    def review(self):
        return dict(visibility='visible',image_quality='adequate',model_support='adequate',
                    disposition='accepted_for_research',reviewer='Test reviewer',note='Synthetic test assessment, not a patient finding.')

    def test_missing_checker_is_explicit_in_exports(self):
        report=self.client.get('/export/report.json').json()['evidence_uncertainty']
        self.assertEqual(report['status'],'not_run')
        self.assertIn('Checker has not been run',self.client.get('/export/report.html').text)

    def test_draft_uses_real_label_voxel_and_affine_without_auto_acceptance(self):
        r=self.generate();f=r['findings'][0];p=f['point_ijk']
        self.assertEqual(int(self.c['mask'].data[tuple(p)]),1)
        self.assertAlmostEqual(f['measurement']['value'],27*24)
        self.assertEqual(f['evidence']['point_ras_mm'],[p[0]*2,p[1]*3,p[2]*4])
        self.assertEqual(f['checks']['anatomical_link'],'linked')
        self.assertEqual(f['status'],'specialist_review');self.assertIsNone(r['model_confidence'])

    def test_review_then_edit_becomes_stale(self):
        r=self.generate();f=r['findings'][0]
        r=self.post('evidence-review',r['version'],finding_id=f['id'],review=self.review()).json()
        self.assertEqual(r['findings'][0]['status'],'review_recorded')
        self.c['mask'].data[5,5,5]=1;server.refresh_mask(self.c)
        r=self.client.get('/api/evidence').json()
        self.assertTrue(r['findings'][0]['stale']);self.assertEqual(r['findings'][0]['status'],'specialist_review')
        self.assertEqual(self.post('evidence-review',r['version'],finding_id=f['id'],review=self.review()).status_code,400)

    def test_changed_reference_invalidates_previous_review(self):
        r=self.generate();f=r['findings'][0]
        self.post('evidence-review',r['version'],finding_id=f['id'],review=self.review())
        self.c['evaluation']={'revision':self.c['revision'],'rows':[{'label_id':1,'dice':.5}]}
        f=self.client.get('/api/evidence').json()['findings'][0]
        self.assertFalse(f['review_current']);self.assertEqual(f['evidence']['reference_agreement']['dice'],.5)

    def test_missing_image_or_synthetic_cannot_be_cleared_by_review(self):
        for mode in ('mask_only','synthetic'):
            with self.subTest(mode=mode):
                self.c['synthetic']=mode=='synthetic'
                if mode=='mask_only':self.c['scan']=None
                record=evidence.new_record(self.c)
                evidence.review_finding(self.c,record,{'finding_id':record['findings'][0]['id'],'review':self.review()})
                self.assertEqual(evidence.check_record(self.c,record)['findings'][0]['status'],'specialist_review')

    def test_missing_label_and_mismatched_anchor_are_flagged(self):
        r=self.generate()
        for label,point in ((999,None),(1,[0,0,0])):
            r=self.post('evidence-add',r['version'],text='Unverified draft',label_id=label,point_ijk=point).json()
            self.assertEqual(r['findings'][-1]['checks']['anatomical_link'],'unresolved')
        self.assertEqual(self.post('evidence-add',r['version'],text='Bad point',point_ijk=[8,2,3]).status_code,400)

    def test_named_region_on_raw_scan_and_claim_escaping(self):
        self.c['mask']=None;self.c['rows']=None
        r=self.generate();self.assertEqual(r['summary']['total'],0)
        r=self.post('evidence-add',r['version'],text='<script>alert(1)</script>',anatomical_region='Named region',point_ijk=[3,3,3]).json()
        self.assertEqual(r['findings'][0]['checks']['anatomical_link'],'linked')
        html=self.client.get('/export/report.html').text
        self.assertNotIn('<script>alert(1)</script>',html);self.assertIn('&lt;script&gt;',html)

    def test_boundaries_components_and_calibration_remain_flags(self):
        self.c['mask'].data[0,0,0]=1;server.refresh_mask(self.c)
        self.c['mask'].notes=['ASSUMED calibration']
        self.c['analysis']={'structures':[{'label_id':1,'components':2}]}
        record=evidence.new_record(self.c)
        evidence.review_finding(self.c,record,{'finding_id':record['findings'][0]['id'],'review':self.review()})
        f=evidence.check_record(self.c,record)['findings'][0]
        self.assertEqual(f['status'],'specialist_review')
        for term in ('boundary','disconnected','calibration'):
            self.assertTrue(any(term in flag for flag in f['flags']))

    def test_version_auth_and_audit_persistence(self):
        self.assertEqual(self.client.post('/api/evidence-generate',json={}).status_code,403)
        r=self.generate()
        self.assertEqual(self.post('evidence-add','wrong',text='stale tab').status_code,400)
        r=self.post('evidence-add',r['version'],text='Audit retained',label_id=1).json();f=r['findings'][-1]
        r=self.post('evidence-remove',r['version'],finding_id=f['id']).json()
        self.assertEqual(r['audit'][-1]['finding']['text'],'Audit retained')
        saved=json.loads((server.CASES/self.c['id']/'evidence.json').read_text())
        self.assertEqual(saved['version'],r['version'])
        self.c.pop('evidence_record');self.c['revision']='reopened'
        self.assertTrue(self.client.get('/api/evidence').json()['findings'][0]['stale'])

    def test_stale_source_slice_is_rejected(self):
        r=self.client.get('/slice?case_id=evidencetest&revision=old&axis=2&index=3')
        self.assertEqual(r.status_code,400)


if __name__=='__main__':unittest.main()
