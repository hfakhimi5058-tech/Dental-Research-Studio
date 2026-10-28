'use strict';
let checkedReport=null, checkedSelection=null, evidenceRequest=0, capturedEvidencePoint=null;
const reviewLabels={not_assessed:'Not assessed',visible:'Visible on source image',not_visible:'Not visible',uncertain:'Uncertain',adequate:'Adequate for this claim',limited:'Limited / uncertain',unusable:'Unusable',not_applicable:'Not applicable (no segmentation)',specialist_review:'Specialist review required',accepted_for_research:'Record research review',rejected:'Reject finding'};
const statusLabels={specialist_review:'Specialist review',review_recorded:'Research review recorded',rejected:'Rejected'};
const evidencePanel=document.createElement('section');
evidencePanel.className='evidence-checker card';
evidencePanel.innerHTML=`<div class="evidence-heading"><div><span class="eyebrow">DRAFT → EVIDENCE → REVIEW</span><h2>Evidence &amp; Uncertainty Checker</h2></div><button id="evidence-generate" class="primary">Generate draft &amp; check</button></div>
<p class="small-copy">Check each claim against the source image and its anatomical location. Geometry checks are automatic; visibility and claim-specific quality require a reviewer. This is research review, not clinical clearance.</p>
<div id="evidence-summary" role="status"></div>
<div class="evidence-actions"><label class="check-label"><input type="checkbox" id="evidence-flags-only"> Needs review only</label><button id="evidence-refresh" class="secondary">Refresh evidence</button><a class="secondary" href="/export/evidence.json">Evidence + audit JSON</a></div>
<details class="evidence-add"><summary>Add a draft finding</summary><form id="evidence-add-form"><label>One claim from a draft report<textarea id="finding-text" maxlength="2000" required placeholder="Enter the finding to review; the checker does not infer diagnoses."></textarea></label><div class="evidence-fields"><label>Segmented structure<select id="finding-label"></select></label><label>Anatomical region (required if no label)<input id="finding-region" maxlength="160" placeholder="Named region in the source image"></label></div><div class="evidence-actions"><button id="finding-capture" class="secondary" type="button">Use workspace crosshair</button><button id="finding-clear-point" class="text-button" type="button">Clear location</button><span id="finding-point" class="small-copy">A selected structure supplies a labelled voxel if no point is chosen.</span></div><button class="primary" type="submit">Add for review</button></form></details>
<div class="evidence-grid"><div id="evidence-findings" class="evidence-findings"></div><div id="evidence-detail"></div></div>`;
$('measurement-table').parentElement.before(evidencePanel);

function evidenceMatches(r){return r&&r.case_id===state.c?.id&&r.case_revision===state.c?.revision}
async function renderEvidenceChecker(){
 if(state.page!=='reports'||!state.c)return;
 const request=++evidenceRequest, key=state.c.id+state.c.revision;
 try{const r=await api('evidence');if(request!==evidenceRequest||key!==state.c.id+state.c.revision||!evidenceMatches(r))return;checkedReport=r;paintEvidence()}catch(e){$('evidence-summary').textContent=e.message}
}
function paintEvidence(){
 if(!evidenceMatches(checkedReport))return;
 const r=checkedReport,s=r.summary;
 $('evidence-summary').textContent=r.status==='not_run'?'Checker not run. Generate the measurement draft, then add and review any report claims.':`${s.total} findings · ${s.specialist_review} need specialist review · ${s.review_recorded} research reviews recorded · ${s.rejected} rejected · ${s.stale} stale`;
 $('evidence-generate').textContent=r.status==='not_run'?'Generate draft & check':'Regenerate measurement draft';
 $('evidence-generate').title='Recalculate measurement claims and reset their reviews. Added draft claims retain their original image revision.';
 const previousLabel=$('finding-label').value;
 $('finding-label').innerHTML='<option value="">No segmented label / use a named region</option>'+state.c.rows.map(row=>`<option value="${row.label_id}">${esc(row.structure)} · ${row.label_id}</option>`).join('');
 if(previousLabel)$('finding-label').value=previousLabel;
 const findings=r.findings.filter(f=>!$('evidence-flags-only').checked||f.status==='specialist_review');
 if(!findings.some(f=>f.id===checkedSelection))checkedSelection=findings[0]?.id??null;
 $('evidence-findings').innerHTML=findings.map(f=>`<button class="evidence-item ${f.id===checkedSelection?'selected':''}" data-finding="${esc(f.id)}"><span class="evidence-badge ${f.status}">${esc(statusLabels[f.status])}${f.stale?' · stale':''}</span><span>${esc(f.text)}</span><small>${esc(f.kind==='calculated_volume'?'Calculated from mask':'Draft claim · not automatically verified')}</small></button>`).join('')||'<p class="small-copy">No findings in this view. A raw scan can use manually added, located draft claims.</p>';
 $$('[data-finding]').forEach(b=>b.onclick=()=>{checkedSelection=b.dataset.finding;paintEvidence()});
 paintEvidenceDetail();
}
function choiceField(id,label,choices,value){return `<label>${label}<select id="${id}">${choices.map(k=>`<option value="${k}" ${k===value?'selected':''}>${esc(reviewLabels[k])}</option>`).join('')}</select></label>`}
function paintEvidenceDetail(){
 const f=checkedReport.findings.find(f=>f.id===checkedSelection),host=$('evidence-detail');
 if(!f){host.innerHTML='';return}
 const ev=f.evidence,review=f.review_current?f.review:{},point=ev.point_ijk;
 const hasViews=ev.source_image_available&&point&&!f.stale;
 const low=+$('window-center').value-(+$('window-width').value)/2, high=low+(+$('window-width').value);
 const views=hasViews?[0,1,2].map(axis=>{
  const query=new URLSearchParams({...identity(),axis,index:point[axis],low,high,alpha:0,labels:''});
  return `<figure><img loading="lazy" src="/slice?${query}" alt="Source grid plane ${axis+1}, slice ${point[axis]}, grayscale scan without segmentation overlay"><figcaption>Grid ${axis+1} (${esc(state.c.orientation[axis])}) · slice ${point[axis]}</figcaption></figure>`;
 }).join(''):'<p class="review-flag">Current source views unavailable. Load a scan or renew the stale finding.</p>';
 host.innerHTML=`<h3>${esc(f.text)}</h3><p class="small-copy">${esc(f.kind==='calculated_volume'?'CALCULATED mask geometry':'DRAFT claim; semantic correctness is not automatically checked')}</p><div class="evidence-views">${views}</div>
 <p class="small-copy">Grey = source intensity. These are source-grid planes and may be oblique. Window ${fmt(low,0)} to ${fmt(high,0)} in source units, not assumed HU. Use the workspace to zoom, change the window and inspect the full region.</p>
 <div class="evidence-actions"><button id="evidence-locate" class="secondary" ${!point||f.stale?'disabled':''}>Locate in source images</button><span class="small-copy">IJK: ${esc(point?.join(', ')||'not recorded')} · Label: ${esc(f.label_id??'none')}</span></div>
 <dl class="evidence-checks"><dt>Visible on the source image?</dt><dd>${esc(reviewLabels[f.checks.visible_in_source])}</dd><dt>Linked to anatomy?</dt><dd>${esc(f.checks.anatomical_link)}</dd><dt>Image quality sufficient?</dt><dd>${esc(reviewLabels[f.checks.image_quality])}</dd><dt>Model output sufficient?</dt><dd>${esc(reviewLabels[f.checks.model_support])}</dd><dt>Route for specialist review?</dt><dd>${f.specialist_review_required?'Yes — unresolved evidence':f.status==='rejected'?'Finding rejected':'Reviewer recorded research acceptance'}</dd></dl>
 <ul class="evidence-flags">${f.flags.map(x=>`<li>${esc(x)}</li>`).join('')||'<li>No unresolved bookkeeping flags. This is not clinical validation.</li>'}</ul>
 <p class="small-copy">${ev.components==null?'Component analysis has not been run.':`Segmentation components: ${ev.components}.`} ${ev.reference_agreement?`Reference Dice: ${fmt(ev.reference_agreement.dice,4)} (overlap, not diagnostic confidence).`:'No current reference-mask comparison for this label.'} Model confidence: unavailable.</p>
 ${f.review&&!f.review_current?`<p class="review-flag">Previous review by ${esc(f.review.reviewer)} is stale and does not support this revision.</p>`:''}
 <form id="evidence-review-form"><h4>Record a reviewer’s assessment</h4><div class="evidence-fields">
 ${choiceField('review-visible','Finding visible',['not_assessed','visible','not_visible','uncertain'],review?.visibility)}
 ${choiceField('review-quality','Image quality for this claim',['not_assessed','adequate','limited','unusable'],review?.image_quality)}
 ${choiceField('review-model','Segmentation / model support',['not_assessed','adequate','limited','not_applicable'],review?.model_support)}
 ${choiceField('review-disposition','Decision',['specialist_review','accepted_for_research','rejected'],review?.disposition)}
 </div><label>Reviewer name or identifier<input id="evidence-reviewer" maxlength="120" required value="${esc(f.review?.reviewer||'')}"></label><label>Evidence / uncertainty note<textarea id="evidence-review-note" maxlength="2000" placeholder="Describe the visual evidence, image limitations and reason for referral.">${esc(review?.note||'')}</textarea></label><div class="evidence-actions"><button class="primary" ${f.stale?'disabled':''}>Save assessment</button><button type="button" class="text-button" id="evidence-remove">Remove finding (kept in audit)</button></div></form>
 <p class="small-copy">Assessments are self-recorded, not authenticated specialist sign-off. ${esc(checkedReport.limitations)}</p>`;
 $('evidence-locate').onclick=()=>{if(f.label_id!=null&&state.c.rows.some(r=>r.label_id===f.label_id))locateStructure(f.label_id);state.cursor=point.slice();setPage('workspace');updateCursorText();renderSlices();toast('Inspect the claim at this location, then return to Reports.')};
 $('evidence-review-form').onsubmit=handle(async event=>{event.preventDefault();await mutateEvidence('evidence-review',{finding_id:f.id,review:{visibility:$('review-visible').value,image_quality:$('review-quality').value,model_support:$('review-model').value,disposition:$('review-disposition').value,reviewer:$('evidence-reviewer').value,note:$('evidence-review-note').value}});toast('Assessment saved locally; unresolved flags remain in the report.')});
 $('evidence-remove').onclick=handle(()=>mutateEvidence('evidence-remove',{finding_id:f.id}));
}
async function mutateEvidence(action,details={}){
 if(!evidenceMatches(checkedReport))throw Error('Reload the checker for the current case.');
 const r=await api(action,{...identity(),evidence_version:checkedReport.version,...details});
 if(!evidenceMatches(r)){await renderEvidenceChecker();return}
 checkedReport=r;paintEvidence();
}
$('evidence-generate').onclick=handle(()=>mutateEvidence('evidence-generate'));
$('evidence-refresh').onclick=renderEvidenceChecker;
$('evidence-flags-only').onchange=paintEvidence;
$('finding-capture').onclick=()=>{capturedEvidencePoint={...identity(),point:state.cursor.slice()};$('finding-point').textContent='Captured source voxel IJK: '+state.cursor.join(', ')};
$('finding-clear-point').onclick=()=>{capturedEvidencePoint=null;$('finding-point').textContent='A selected structure supplies a labelled voxel if no point is chosen.'};
$('evidence-add-form').onsubmit=handle(async event=>{
 event.preventDefault();
 if(capturedEvidencePoint&&(capturedEvidencePoint.case_id!==state.c.id||capturedEvidencePoint.revision!==state.c.revision))throw Error('Captured location is stale. Capture it again or clear it.');
 await mutateEvidence('evidence-add',{text:$('finding-text').value,label_id:$('finding-label').value?+$('finding-label').value:null,anatomical_region:$('finding-region').value,point_ijk:capturedEvidencePoint?.point??null});
 $('finding-text').value='';checkedSelection=checkedReport.findings.at(-1)?.id;paintEvidence();
});
if(state.c&&state.page==='reports')renderEvidenceChecker();
