"""Local, deterministic evidence bookkeeping. No diagnostic or VLM inference."""
import hashlib
import html
import json
import uuid
from datetime import datetime, timezone

import nibabel as nib
import numpy as np
from core import same_grid


LIMITATIONS = (
    'Image visibility and claim-specific quality are reviewer judgments. A label or image '
    'location does not establish that a finding is true. No VLM, calibrated probability, '
    'automated pathology verification or clinical clearance is provided. Missing labels '
    'do not prove missing teeth. Review flags are retained in exported reports.'
)
METHODS = {
    'grounding': 'Check an integer voxel location against image bounds and the assigned label; transform it with the full RAS affine.',
    'quality': 'Check source-image availability, grid agreement, boundary contact, available component analysis and calibration notes. Diagnostic adequacy requires a reviewer.',
    'uncertainty': 'Categorical missing-evidence and review flags, not model probabilities. Dice is reference-mask overlap, not confidence in a clinical finding.',
    'revision': 'Mask edits and case reopenings invalidate prior finding/review revisions. Changed analysis or reference evidence invalidates prior review context.',
    'paper_context': 'From Voxels to Views to Reports: A Segmentation-Guided VLM Pipeline for CBCT Report Generation, sections 2.1–2.3. This module adds an explicit review stage; it does not reproduce that paper’s fine-tuned VLM or validation-based suppression rules.',
}
CHOICES = {
    'visibility': {'not_assessed', 'visible', 'not_visible', 'uncertain'},
    'image_quality': {'not_assessed', 'adequate', 'limited', 'unusable'},
    'model_support': {'not_assessed', 'adequate', 'limited', 'not_applicable'},
    'disposition': {'specialist_review', 'accepted_for_research', 'rejected'},
}


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def context_stamp(c):
    value = {k: c.get(k) for k in ('id', 'revision', 'synthetic', 'analysis', 'evaluation')}
    value['provenance'] = c['source'].get('provenance')
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def rows_for(c):
    return [] if c['rows'] is None else c['rows'].to_dict(orient='records')


def anchor_for(c, row):
    """Choose an actual labelled voxel near the centroid, never a void centroid."""
    mask = c['mask']
    target = nib.affines.apply_affine(np.linalg.inv(mask.affine), [row['centroid_R_mm'], row['centroid_A_mm'], row['centroid_S_mm']])
    for z in sorted(range(mask.data.shape[2]), key=lambda k: abs(k-target[2])):
        x, y = np.where(mask.data[:, :, z] == row['label_id'])
        if x.size:
            pick = int(np.argmin((x-target[0])**2 + (y-target[1])**2))
            return [int(x[pick]), int(y[pick]), int(z)]
    return None


def new_record(c, previous=None):
    findings = []
    for row in rows_for(c):
        findings.append({
            'id': uuid.uuid4().hex[:12], 'kind': 'calculated_volume',
            'text': f"{row['structure']} (label {row['label_id']}): segmented volume {row['volume_mm3']:.6g} mm³.",
            'label_id': row['label_id'], 'anatomical_region': row['structure'],
            'point_ijk': anchor_for(c, row), 'case_revision': c['revision'],
            'measurement': {'value': row['volume_mm3'], 'units': 'mm3', 'voxel_count': row['voxels'],
                            'voxel_volume_mm3': c['mask'].voxel_mm3, 'method': 'voxel_count × abs(det(spatial affine))'},
            'review': None,
        })
    # Draft claims keep their old revision; regeneration cannot silently re-ground them.
    findings.extend(f for f in (previous or {}).get('findings', []) if f['kind'] == 'draft_claim')
    return {'schema_version': 1, 'case_id': c['id'], 'version': uuid.uuid4().hex,
            'generated_at': timestamp(), 'findings': findings, 'audit': (previous or {}).get('audit', [])}


def add_finding(c, record, data):
    text = str(data.get('text', '')).strip()
    region = str(data.get('anatomical_region', '')).strip()
    if not text or len(text) > 2000 or len(region) > 160:
        raise ValueError('Enter one draft finding (up to 2,000 characters) and a short region name.')
    if len(record['findings']) >= 512:
        raise ValueError('This report has reached its 512-finding limit.')
    label = data.get('label_id')
    if label is not None and (type(label) is not int or not 1 <= label <= 65535):
        raise ValueError('Label must be an integer from 1 to 65535, or absent.')
    point = data.get('point_ijk')
    base = c['scan'] if c['scan'] is not None else c['mask']
    if point is not None and (not isinstance(point, list) or len(point) != 3 or
            any(type(v) is not int or not 0 <= v < base.data.shape[i] for i, v in enumerate(point))):
        raise ValueError('The evidence location must contain three integer voxel indices inside the current image.')
    if point is None and label is not None:
        row = next((r for r in rows_for(c) if r['label_id'] == label), None)
        point = anchor_for(c, row) if row else None
    finding = {'id': uuid.uuid4().hex[:12], 'kind': 'draft_claim', 'text': text,
               'label_id': label, 'anatomical_region': region, 'point_ijk': point,
               'case_revision': c['revision'], 'review': None}
    record['findings'].append(finding)
    return finding


def review_finding(c, record, data):
    finding = next((f for f in record['findings'] if f['id'] == data.get('finding_id')), None)
    if finding is None:
        raise ValueError('Finding no longer exists. Reload the checker.')
    if finding['case_revision'] != c['revision']:
        raise ValueError('This finding belongs to an older image revision. Regenerate measurements or replace the draft finding.')
    review = data.get('review')
    if not isinstance(review, dict):
        raise ValueError('Review details are required.')
    clean = {}
    for key, choices in CHOICES.items():
        if review.get(key) not in choices:
            raise ValueError('Invalid review choice: ' + key)
        clean[key] = review[key]
    clean['reviewer'] = str(review.get('reviewer', '')).strip()
    clean['note'] = str(review.get('note', '')).strip()
    if not clean['reviewer'] or len(clean['reviewer']) > 120 or len(clean['note']) > 2000:
        raise ValueError('Provide a reviewer name/identifier (up to 120 characters) and a note up to 2,000 characters.')
    clean.update(recorded_at=timestamp(), context=context_stamp(c), revision=c['revision'])
    finding['review'] = clean
    return finding


def check_record(c, record):
    output = {'schema_version': 1, 'case_id': c['id'], 'case_revision': c['revision'],
              'version': (record or {}).get('version'), 'checked_at': timestamp(), 'findings': [],
              'limitations': LIMITATIONS, 'methods': METHODS, 'clinical_validation': False,
              'model_confidence': None, 'status': 'not_run' if record is None else 'checked'}
    if record is None:
        output['summary'] = {'total': 0, 'specialist_review': 0, 'review_recorded': 0, 'rejected': 0, 'stale': 0}
        return output
    rows = {r['label_id']: r for r in rows_for(c)}
    advanced = {r['label_id']: r for r in (c.get('analysis') or {}).get('structures', [])}
    evaluation = c.get('evaluation') or {}
    reference = {r['label_id']: r for r in evaluation.get('rows', [])} if evaluation.get('revision') == c['revision'] else {}
    scan, mask = c['scan'], c['mask']
    base = scan if scan is not None else mask
    stamp = context_stamp(c)
    grid_ok = mask is None or scan is None or same_grid(mask, scan)
    for finding in record['findings']:
        flags = []
        label, point = finding.get('label_id'), finding.get('point_ijk')
        row = rows.get(label)
        stale = finding['case_revision'] != c['revision']
        located = point is not None and len(point) == 3 and all(type(v) is int and 0 <= v < base.data.shape[i] for i, v in enumerate(point))
        label_at_point = int(mask.data[tuple(point)]) if located and mask is not None and grid_ok else None
        grounded = not stale and located and grid_ok and ((label is not None and row is not None and label_at_point == label) or
                                                       (label is None and bool(finding.get('anatomical_region'))))
        if stale: flags.append('Finding belongs to an earlier image revision; its location and review must be renewed.')
        if c['synthetic']: flags.append('Synthetic geometry is not patient evidence.')
        if scan is None: flags.append('No source scan is loaded; a segmentation alone cannot show the claimed finding.')
        if not grid_ok: flags.append('Scan and mask physical grids do not agree.')
        if not grounded: flags.append('No valid link to the assigned structure or named image region.')
        if label is not None and row is None: flags.append('Assigned label is absent. This does not establish anatomical absence.')
        if row and row['touches_image_boundary']: flags.append('Assigned structure touches the image boundary and may be truncated.')
        quality = advanced.get(label)
        if quality and quality['components'] > 1: flags.append('Assigned structure has disconnected segmentation components.')
        calibration_notes = [str(n) for v in (scan, mask) if v is not None for n in v.notes]
        if any('ASSUMED' in n.upper() for n in calibration_notes): flags.append('Spatial calibration includes an explicit assumption.')
        review = finding.get('review')
        review_current = bool(review and review.get('revision') == c['revision'] and review.get('context') == stamp and not stale)
        current = review if review_current else {}
        if review and not review_current: flags.append('Recorded review is stale after an image or evidence-context change.')
        if current.get('visibility') != 'visible': flags.append('A reviewer has not confirmed that this finding is visible on the source image.')
        if current.get('image_quality') != 'adequate': flags.append('Image quality has not been judged adequate for this specific claim.')
        model_sufficient = current.get('model_support') == 'adequate' if mask is not None else current.get('model_support') == 'not_applicable'
        if not model_sufficient: flags.append('Segmentation/model support has not been reviewed for this claim.')
        if current.get('disposition') != 'accepted_for_research': flags.append('Specialist review is pending or the finding was rejected.')
        status = 'rejected' if current.get('disposition') == 'rejected' else ('specialist_review' if flags else 'review_recorded')
        ref = reference.get(label)
        output['findings'].append({**finding, 'status': status, 'stale': stale, 'review_current': review_current,
            'specialist_review_required': status == 'specialist_review', 'flags': flags,
            'checks': {'visible_in_source': current.get('visibility', 'not_assessed'),
                       'anatomical_link': 'linked' if grounded else 'unresolved',
                       'image_quality': current.get('image_quality', 'not_assessed'),
                       'model_support': current.get('model_support', 'not_assessed')},
            'evidence': {'source_image_available': scan is not None, 'point_ijk': point,
                         'point_ras_mm': nib.affines.apply_affine(base.affine, point).tolist() if located and not stale else None,
                         'label_at_point': label_at_point, 'structure': (row or {}).get('structure'),
                         'source_image_name': scan.name if scan is not None else None,
                         'segmentation_name': mask.name if mask is not None else None,
                         'component_analysis_available': quality is not None,
                         'components': (quality or {}).get('components'), 'reference_agreement': ref,
                         'calibration_notes': list(dict.fromkeys(calibration_notes)),
                         'model_provenance': c['source'].get('provenance')}})
    output['summary'] = {'total': len(output['findings']), **{key: sum(f['status'] == key for f in output['findings'])
                         for key in ('specialist_review', 'review_recorded', 'rejected')},
                         'stale': sum(f['stale'] or (f.get('review') is not None and not f['review_current']) for f in output['findings'])}
    output['audit'] = record.get('audit', [])
    return output


def html_section(report):
    esc = lambda value: html.escape(str(value))
    parts = ['<h2>Evidence &amp; Uncertainty Checker</h2><p>' + esc(LIMITATIONS) + '</p>']
    if report['status'] == 'not_run':
        parts.append('<p><strong>Checker has not been run. No findings have been reviewed.</strong></p>')
    for finding in report['findings']:
        parts.append('<article><h3>' + esc(finding['status'].replace('_', ' ')) + '</h3><p>' + esc(finding['text']) + '</p>')
        parts.append('<p>Source voxel (IJK): ' + esc(finding.get('point_ijk')) + ' · label: ' + esc(finding.get('label_id')) + '</p>')
        parts.append('<ul>' + ''.join('<li>' + esc(flag) + '</li>' for flag in finding['flags']) + '</ul>')
        if finding.get('review'):
            parts.append('<p>Reviewer: ' + esc(finding['review']['reviewer']) + ' · ' + esc(finding['review']['recorded_at']) +
                         ' · current: ' + esc(finding['review_current']) + '</p><p>' + esc(finding['review']['note']) + '</p>')
        parts.append('</article>')
    return ''.join(parts)
