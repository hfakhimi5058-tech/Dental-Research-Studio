import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
from job_progress import counter, snapshot, track_run


class ProgressTests(unittest.TestCase):
    def setUp(self):
        (ROOT / 'work/temp').mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'work/temp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / 'results/toothseg-test'
        self.run.mkdir(parents=True)
        self.jobs = self.root / 'work/jobs'
        self.jobs.mkdir(parents=True)
        self.log = self.jobs / 'test.log'
        self.log.write_text('ToothSeg stage: instance\n')
        self.item = {'id': 'test', 'created': 100, 'started': 110, 'status': 'running', 'log': str(self.log)}
        with patch.dict(os.environ, {'DENTAL_JOB_PROGRESS': str(self.jobs / 'test.progress.json')}):
            track_run(self.run, 'toothseg')

    def stage(self, name):
        (self.run / 'status.json').write_text(json.dumps({'stage': name}))

    def test_reads_nested_log_and_current_pass(self):
        self.stage('instance')
        (self.run / 'instance.log').write_text('100%|###| 12/12 [00:10<00:00]\r 41%|####| 75/180 [01:09<01:33]\r')
        p = snapshot(self.item, self.root, now=200)
        self.assertEqual(p['counter']['done'], 75)
        self.assertEqual(p['counter']['total'], 180)
        self.assertEqual(p['stage'], 'Separating individual teeth')
        self.assertEqual(p['elapsed_seconds'], 90)
        self.assertEqual(p['steps'][3]['state'], 'active')
        self.assertEqual(p['steps'][4]['state'], 'pending')
        self.assertIn('75/180', p['detail_tail'])

    def test_old_pass_counter_is_not_used_for_new_stage(self):
        self.stage('postprocess')
        (self.run / 'instance.log').write_text('100%|###| 180/180 [02:00<00:00]')
        p = snapshot(self.item, self.root)
        self.assertIsNone(p['counter'])
        self.assertEqual(p['stage'], 'Assembling tooth masks')
        self.assertEqual(p['steps'][3]['state'], 'complete')

    def test_100_percent_does_not_finish_job(self):
        self.stage('instance')
        (self.run / 'instance.log').write_text('100%|###| 180/180 [02:00<00:00]')
        p = snapshot(self.item, self.root)
        self.assertEqual(p['counter']['percent'], 100)
        self.assertEqual(p['steps'][3]['state'], 'active')
        self.assertEqual(p['steps'][5]['state'], 'pending')

    def test_loading_result_remains_active_until_parent_completes(self):
        self.stage('completed')
        p = snapshot(self.item, self.root)
        self.assertEqual(p['stage'], 'Checking and loading result')
        self.assertEqual(p['steps'][-1]['state'], 'active')
        self.item.update(status='completed', finished=260)
        p = snapshot(self.item, self.root, now=500)
        self.assertEqual(p['elapsed_seconds'], 150)
        self.assertTrue(all(s['state'] == 'complete' for s in p['steps']))

    def test_failure_preserves_failed_stage(self):
        (self.run / 'status.json').write_text(json.dumps({'stage': 'failed', 'failed_stage': 'instance'}))
        self.item['status'] = 'failed'
        p = snapshot(self.item, self.root)
        self.assertEqual(p['steps'][3]['state'], 'failed')
        self.assertEqual(p['steps'][4]['state'], 'pending')

    def test_missing_or_partial_log_has_no_invented_counter(self):
        self.stage('semantic')
        (self.run / 'semantic.log').write_text('Loading model\n 20%|##| 5/')
        self.assertIsNone(snapshot(self.item, self.root)['counter'])
        self.assertIsNone(counter('100%|##| 5/0 [bad]'))
        self.assertIsNone(counter('100%|##| 6/5 [bad]'))

    def test_descriptor_outside_results_is_ignored(self):
        (self.jobs / 'test.progress.json').write_text(json.dumps({'kind': 'toothseg', 'folder': str(self.jobs)}))
        (self.jobs / 'instance.log').write_text('100%|##| 5/5 [bad]')
        self.assertIsNone(snapshot(self.item, self.root)['counter'])

    def test_malformed_sidecar_does_not_break_job_status(self):
        (self.jobs / 'test.progress.json').write_text('[')
        self.assertEqual(snapshot(self.item, self.root)['stage'], 'Separating individual teeth')


if __name__ == '__main__':
    unittest.main()
