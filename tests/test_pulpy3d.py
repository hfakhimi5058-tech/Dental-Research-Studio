import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
from starlette.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'app'))
import pulpy3d
import capabilities
import server
import pipeline
from core import analyse


class PulpyImportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.patient = self.root/'P1'
        self.patient.mkdir()
        self.affine = np.diag([.5,.5,.5,1.])
        self.data = np.zeros((12,12,12),dtype=np.uint16)
        self.data[2:4,2:4,2:4] = 31
        self.data[7:9,7:9,7:9] = 1
        self.write('data', np.zeros_like(self.data))
        self.write('gt_instance_ian', self.data)
        self.request = dict(folder=str(self.root),patient='P1',variant='gt_instance_ian',
                            encoding='fdi',mapping_confirmed=True)

    def write(self, stem, data, affine=None, unit='mm'):
        a = self.affine if affine is None else affine
        image = nib.Nifti1Image(data,a)
        image.header.set_xyzt_units(unit)
        image.set_sform(a,code=1)
        nib.save(image, self.patient/(stem+'.nii.gz'))

    def test_raw_fdi_names_and_physical_size_preserved(self):
        mask,scan,names,source,title = pulpy3d.load_annotation(self.request)
        np.testing.assert_array_equal(mask.data,self.data)
        self.assertEqual(names[31],'Pulp cavity · FDI 31')
        self.assertEqual(names[1],'Inferior alveolar nerve')
        rows=analyse(mask,names).set_index('label_id')
        self.assertAlmostEqual(rows.loc[31,'volume_mm3'],1.)
        self.assertAlmostEqual(rows.loc[31,'extent_R_mm'],1.)
        self.assertEqual(source['dataset_import']['kind'],'imported_dataset_annotation')
        self.assertNotIn('provenance',source)

    def test_compact_encoding_does_not_reinterpret_fdi_mask(self):
        with self.assertRaisesRegex(ValueError,'IDs are not defined'):
            pulpy3d.load_annotation({**self.request,'encoding':'compact'})
        self.data[self.data==31]=2
        self.write('gt_instance_ian',self.data)
        mask,_,names,_,_=pulpy3d.load_annotation({**self.request,'encoding':'compact'})
        self.assertEqual(names[2],'Pulp cavity · FDI 31')
        self.assertEqual(set(np.unique(mask.data)),{0,1,2})

    def test_binary_mask_cannot_silently_collapse_instance_labels(self):
        self.write('gt_pulp',self.data)
        with self.assertRaisesRegex(ValueError,'IDs are not defined'):
            pulpy3d.load_annotation({**self.request,'variant':'gt_pulp','encoding':'binary'})
        binary=(self.data==31).astype(np.uint16)
        self.write('gt_pulp',binary)
        _,_,names,_,_=pulpy3d.load_annotation({**self.request,'variant':'gt_pulp','encoding':'binary'})
        self.assertEqual(names[1],'Pulp cavity (all foreground)')

    def test_geometry_and_units_must_match(self):
        shifted=self.affine.copy();shifted[0,3]=2
        self.write('gt_instance_ian',self.data,shifted)
        with self.assertRaisesRegex(ValueError,'different physical grids'):
            pulpy3d.load_annotation(self.request)
        self.write('gt_instance_ian',self.data,unit='unknown')
        with self.assertRaisesRegex(ValueError,'units are unknown'):
            pulpy3d.load_annotation(self.request)
        mask,_,_,_,_=pulpy3d.load_annotation({**self.request,'assume_mm':True})
        self.assertIn('ASSUMED',mask.notes[0])

    def test_confirm_missing_variant_and_empty_mask(self):
        with self.assertRaisesRegex(ValueError,'confirm'):
            pulpy3d.load_annotation({**self.request,'mapping_confirmed':False})
        with self.assertRaisesRegex(ValueError,'not available'):
            pulpy3d.load_annotation({**self.request,'variant':'gt_ian'})
        self.write('gt_instance_ian',np.zeros_like(self.data))
        with self.assertRaisesRegex(ValueError,'no foreground'):
            pulpy3d.load_annotation(self.request)

    def test_custom_metadata_covers_every_id(self):
        meta=self.root/'labels.json'
        meta.write_text(json.dumps({'0':'Background','1':'Nerve','31':'Custom pulp 31'}))
        _,_,names,_,_=pulpy3d.load_annotation({**self.request,'encoding':'custom','labels':str(meta)})
        self.assertEqual(names[31],'Custom pulp 31')
        meta.write_text(json.dumps({'0':'Background','31':'Custom pulp 31'}))
        with self.assertRaisesRegex(ValueError,'IDs are not defined by this encoding: 1'):
            pulpy3d.load_annotation({**self.request,'encoding':'custom','labels':str(meta)})

    def test_inventory_single_patient_and_release_layout(self):
        inv=pulpy3d.inspect_folder(self.patient)
        self.assertEqual(inv['cases'][0]['id'],'P1')
        release=self.root/'release';(release/'imagesTr').mkdir(parents=True);(release/'labelsTr').mkdir()
        (release/'imagesTr'/'P1_0000.nii.gz').write_bytes((self.patient/'data.nii.gz').read_bytes())
        (release/'labelsTr'/'P1.nii.gz').write_bytes((self.patient/'gt_instance_ian.nii.gz').read_bytes())
        inv=pulpy3d.inspect_folder(release)
        self.assertEqual(inv['cases'][0]['layout'],'nnunet')
        with self.assertRaisesRegex(ValueError,'Enter'):
            pulpy3d.inspect_folder('')

    def test_reopen_keeps_annotation_names_and_report_provenance(self):
        # Isolated source/state; never read or change an actual saved patient case.
        with patch.object(server,'ROOT',self.root), patch.object(server,'CASES',self.root/'cases'), patch.object(server,'CURRENT',None):
            (self.root/'work').mkdir()
            m,s,n,source,title=pulpy3d.load_annotation(self.request)
            server.set_current(m,s,n,title,source=source)
            server.save_manifest(server.CURRENT)
            ident=server.CURRENT['id']
            saved=json.loads((self.root/'cases'/ident/'case.json').read_text())
            server.CURRENT=None
            case=server.load_case(saved['source'],saved=saved)
            self.assertFalse(case['synthetic'])
            self.assertIsNone(case['provenance'])
            self.assertEqual(case['dataset_import']['dataset'],'Pulpy3D')
            self.assertIn('Pulp cavity · FDI 31',[r['structure'] for r in case['rows']])
            report=server.export_report(server.CURRENT)
            self.assertEqual(report['dataset_import']['encoding'],'fdi')
            self.assertIsNone(report['model_provenance'])

    def test_readonly_inventory_api_and_no_model_installation_state(self):
        client=TestClient(server.app,base_url='http://127.0.0.1:8517')
        with patch.object(server,'ROOT',self.root):
            self.assertEqual(client.post('/api/pulpy-inspect',json={'folder':str(self.root)}).status_code,403)
            r=client.post('/api/pulpy-inspect',json={'folder':str(self.root)},headers={'X-Local-Token':server.TOKEN})
            self.assertEqual(r.status_code,200)
            self.assertEqual(len(r.json()['cases']),1)
            caps=client.get('/api/capabilities').json()
            self.assertFalse(caps['toothseg_weights'])
            self.assertTrue(caps['pulpy3d']['annotation_import'])
            self.assertFalse(caps['pulpy3d']['automatic_segmentation'])

    def test_broken_model_does_not_break_catalogue(self):
        folder=self.root/'models'/'toothseg';folder.mkdir(parents=True)
        (folder/'pipeline.json').write_text('{}')
        with patch.object(pipeline,'ROOT',self.root):
            result=capabilities.status(self.root)
        self.assertFalse(result['toothseg_weights'])
        self.assertEqual(len(result['issues']),1)
        self.assertIn('checkpoint is missing',result['issues'][0]['message'])


if __name__=='__main__':
    unittest.main(verbosity=2)
