import sys
from pathlib import Path
import tempfile
import unittest
import numpy as np
import nibabel as nib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
from core import Volume, analyse, compare, validate, load_volume, from_sitk, mesh, nifti_bytes, demo


class GeometryTests(unittest.TestCase):
    def test_anisotropic_cube(self):
        data = np.zeros((7, 8, 9), np.uint16)
        data[1:3, 2:5, 3:7] = 1  # 24 voxels; each 2*3*4 = 24 mm³
        vol = Volume(data, np.diag([2., 3., 4., 1.]), 'known cuboid')
        row = analyse(vol).iloc[0]
        self.assertEqual(row.voxels, 24)
        self.assertAlmostEqual(row.volume_mm3, 576)
        np.testing.assert_allclose(row[['extent_R_mm','extent_A_mm','extent_S_mm']].astype(float), [4,9,16])
        np.testing.assert_allclose(row[['centroid_R_mm','centroid_A_mm','centroid_S_mm']].astype(float), [3,9,18])
        self.assertFalse(row.touches_image_boundary)

    def test_rotated_sheared_single_voxel(self):
        a = np.array([[2.,1.,0.,10.],[1.,3.,0.,-2.],[0.,0.,4.,5.],[0.,0.,0.,1.]])
        data = np.zeros((3,3,3),np.uint16)
        data[1,1,1] = 3
        row = analyse(Volume(data,a,'sheared')).iloc[0]
        self.assertAlmostEqual(row.volume_mm3,20)
        np.testing.assert_allclose(row[['extent_R_mm','extent_A_mm','extent_S_mm']].astype(float),[3,4,4])

    def test_fractional_labels_rejected(self):
        data = np.zeros((2,2,2),np.float32)
        data[0,0,0] = .25
        with self.assertRaises(ValueError):
            validate(Volume(data,np.eye(4),'probability'),True)

    def test_overlap_and_missing_label(self):
        p = np.zeros((3,3,3),np.uint16)
        r = p.copy()
        p[0,0,0] = 1
        r[0,0,0] = r[1,0,0] = 1
        p[2,2,2] = 2
        result = compare(Volume(p,np.eye(4),'p'),Volume(r,np.eye(4),'r')).set_index('label_id')
        self.assertAlmostEqual(result.loc[1,'dice'],2/3)
        self.assertAlmostEqual(result.loc[1,'iou'],.5)
        self.assertEqual(result.loc[2,'dice'],0)
        self.assertTrue(np.isnan(result.loc[2,'relative_volume_error_percent']))
        with self.assertRaises(ValueError):
            compare(Volume(p,np.eye(4),'p'),Volume(r,np.diag([2,1,1,1]),'r'))

    def test_unit_conversion_unknown_units_and_roundtrip(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'work'/'temp') as folder:
            path = Path(folder)/'mask.nii.gz'
            a = np.diag([.002,.003,.004,1.])
            im = nib.Nifti1Image(np.ones((2,2,2),np.uint16),a)
            im.header.set_xyzt_units('meter')
            im.set_sform(a,code=1)
            nib.save(im,path)
            v = load_volume(path,True)
            np.testing.assert_allclose(v.spacing,[2,3,4],rtol=1e-6)
            self.assertAlmostEqual(analyse(v).iloc[0].volume_mm3,192,places=4)
            im.header.set_xyzt_units('unknown')
            nib.save(im,path)
            with self.assertRaises(ValueError):
                load_volume(path,True)
            self.assertIn('ASSUMED',load_volume(path,True,True).notes[0])
            back = nib.Nifti1Image.from_bytes(nifti_bytes(v))
            np.testing.assert_array_equal(np.asanyarray(back.dataobj),v.data)
            np.testing.assert_allclose(back.affine,v.affine)

    def test_lps_ras_conversion(self):
        import SimpleITK as sitk
        image = sitk.GetImageFromArray(np.zeros((3,4,5),np.int16))
        image.SetSpacing([2.,3.,4.])
        image.SetOrigin([10.,20.,30.])
        v = from_sitk(image,'mha')
        self.assertEqual(v.data.shape,(5,4,3))
        np.testing.assert_allclose(v.affine[:3,:3],np.diag([-2,-3,4]))
        np.testing.assert_allclose(v.affine[:3,3],[-10,-20,30])

    def test_mesh_includes_boundary_voxels(self):
        v = Volume(np.ones((2,3,4),np.uint16),np.diag([2.,3.,4.,1.]),'edge')
        verts,faces = mesh(v,1,1)
        np.testing.assert_allclose(verts.min(axis=0),[-1,-1.5,-2])
        np.testing.assert_allclose(verts.max(axis=0),[3,7.5,14])
        self.assertGreater(len(faces),0)
        self.assertTrue(analyse(v).iloc[0].touches_image_boundary)

    def test_demo_has_no_empty_labels(self):
        m,s,n = demo()
        rows = analyse(m,n)
        self.assertEqual(len(rows),10)
        self.assertTrue((rows.volume_mm3 > 0).all())
        self.assertEqual(m.data.shape,s.data.shape)


if __name__ == '__main__':
    unittest.main(verbosity=2)
