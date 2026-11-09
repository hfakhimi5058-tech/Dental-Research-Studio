'use strict';

async function loadDatasetWorkspace() {
  try {
    const [registry, caps] = await Promise.all([api('datasets'), api('capabilities')]);
    const list = registry.datasets || [];
    $('dataset-summary').innerHTML = `<article class="capability-card"><span class="eyebrow">AVAILABLE WITHOUT A MODEL</span><h3>Review existing annotations</h3><p>Import a labelled mask, inspect each structure in 3D, calculate volumes and export measurements. Names must match the source label definitions.</p><button class="secondary" id="dataset-import-images">Import images</button></article><article class="capability-card"><span class="eyebrow">RAW SCAN → PERMANENT TEETH</span><h3>ToothSeg</h3><p class="capability-status">${esc(caps.toothseg_status)}</p><p>Whole-tooth labels only. No separate pulp, jaw, nerve or sinus predictions.</p><button class="secondary" id="dataset-open-models">${caps.toothseg_weights?'Models & jobs':'Show setup instructions'}</button></article><article class="capability-card"><span class="eyebrow">PULPY3D ANNOTATIONS</span><h3>Pulp & nerve review</h3><p>Dedicated import, individual pulp views and size cards. Requires locally downloaded scans and annotations.</p><p class="capability-status">Automatic pulp prediction is not installed by this feature.</p><button class="primary" id="dataset-open-pulpy">Open Pulpy3D importer</button></article>`;
    $('dataset-import-images').onclick = openImport;
    $('dataset-open-models').onclick = () => setPage('models');
    $('dataset-open-pulpy').onclick = openPulpyImport;
    $('dataset-list').innerHTML = `<h2>Research sources · ${list.length} entries</h2><p class="small-copy">These describe the publishers’ annotations and possible future work. Their anatomy lists are not supported prediction outputs. Downloading data does not train or install a model.</p>` + list.map(d => `<article class="dataset-card"><header><h3>${esc(d.name)}</h3><span class="pill">${d.id==='pulpy3d'?'Annotation importer available':'Research catalogue'}</span></header><p><b>Source annotations:</b> ${esc(d.labels)}</p><p><b>Access conditions:</b> ${esc(d.access)}<br><b>Licence statement:</b> ${esc(d.licence)}<br><b>Research plan:</b> ${esc(d.route)}</p><details><summary>Source details & unresolved requirements</summary><p>${esc(d.format)} · Cohort family: ${esc(d.family)}</p><ul>${(d.blockers||[]).map(x=>`<li>${esc(x)}</li>`).join('')}</ul>${(d.source_urls||[]).map((url,i)=>`<a href="${esc(url)}" target="_blank" rel="noopener">Source ${i+1} ↗</a>`).join(' · ')}</details></article>`).join('');
  } catch (e) { toast(e.message, true); }
}

let pulpyInventory = null;
function openPulpyImport() {
  openImport();
  document.querySelector('[data-import="pulpy"]').click();
}
function selectedPulpyCase() { return pulpyInventory?.cases.find(c=>c.id===$('pulpy-patient').value); }
function pulpyVariantChanged() {
  const key = $('pulpy-variant').value;
  $('pulpy-encoding').value = key==='release' ? 'custom' : key.startsWith('gt_instance') ? 'fdi' : 'binary';
  $('pulpy-confirm').checked = false;
  updatePulpyMapping();
}
function updatePulpyMapping() {
  const value=$('pulpy-encoding').value;
  const descriptions={fdi:'Original Pulpy3D encoding: FDI 31–38 and 41–48 identify mandibular pulp cavities; 50 is the source anomaly label. The combined variant also uses 1 for inferior alveolar nerve. IDs are preserved.',compact:'Training encoding: 2–9 map to pulp FDI 31–38; 10–17 map to pulp FDI 41–48; 18 is the source anomaly label. The combined variant also uses 1 for nerve. IDs are preserved.',binary:'Only background 0 and foreground 1 are accepted. All foreground is one pulp or nerve segment; individual teeth cannot be distinguished in this binary annotation.',custom:'Supply the exact label-definitions JSON for this mask. Every observed ID must be defined; metadata for another release can misname structures.'};
  $('pulpy-mapping-help').textContent=descriptions[value];
  $('pulpy-labels-row').classList.toggle('hidden',value!=='custom');
}
function pulpyPatientChanged() {
  const c=selectedPulpyCase();
  $('pulpy-variant').innerHTML=(c?.masks||[]).map(m=>`<option value="${esc(m.key)}">${esc(m.name)} · ${esc(m.key)}</option>`).join('');
  pulpyVariantChanged();
}
async function inspectPulpy() {
  const button=$('pulpy-inspect');button.disabled=true;
  $('pulpy-status').textContent='Reading local filenames…';
  try {
    pulpyInventory=await api('pulpy-inspect',{folder:$('pulpy-folder').value});
    $('pulpy-patient').innerHTML=pulpyInventory.cases.map(c=>`<option value="${esc(c.id)}">${esc(c.id)}</option>`).join('');
    $('pulpy-labels').value=pulpyInventory.metadata;
    $('pulpy-options').classList.remove('hidden');
    $('pulpy-status').textContent=`${pulpyInventory.cases.length} paired cases found. Select one annotation to open. ${pulpyInventory.note}`;
    pulpyPatientChanged();
  } catch(e) {
    pulpyInventory=null;$('pulpy-options').classList.add('hidden');$('pulpy-status').textContent=e.message;throw e;
  } finally { button.disabled=false; }
}
async function importPulpy(common) {
  if(!pulpyInventory)throw Error('Inspect the Pulpy3D folder first.');
  if(!$('pulpy-confirm').checked)throw Error('Review and confirm the label encoding first.');
  return api('pulpy-import',{...common,title:common.title==='Research case'?'':common.title,
    folder:pulpyInventory.folder,patient:$('pulpy-patient').value,variant:$('pulpy-variant').value,
    encoding:$('pulpy-encoding').value,labels:$('pulpy-labels').value,mapping_confirmed:true});
}

async function renderModelSetup() {
  const caps=await api('capabilities');
  const host=$('model-setup');
  host.innerHTML=`<h3>Segmentation available on this computer</h3><p><b>ToothSeg:</b> ${esc(caps.toothseg_status)}. Runtime ${caps.runtime_installed?'found':'not installed'}. ${caps.models.filter(m=>!m.recommended).length} other model folders discovered; their labels and quality require separate review.</p><p>For permanent-tooth prediction, run <code>Setup.cmd -Segmentation -DownloadModels</code> once with internet, restart the app, then import a raw scan and select ToothSeg below. A compatible NVIDIA GPU and sufficient memory are required. Downloaded datasets are not needed to segment your own scan.</p><p><b>Pulp, jaws, canals and sinuses:</b> ToothSeg does not predict these as separate structures. Pulpy3D annotations can be imported and measured; no compatible pulp prediction weights are bundled.</p>${caps.issues.map(i=>`<p class="setup-issue"><b>${esc(i.model)}:</b> ${esc(i.message)}</p>`).join('')}`;
}

document.addEventListener('DOMContentLoaded',()=>{
  $('pulpy-inspect').onclick=handle(inspectPulpy);
  $('pulpy-patient').onchange=pulpyPatientChanged;
  $('pulpy-variant').onchange=pulpyVariantChanged;
  $('pulpy-encoding').onchange=()=>{$('pulpy-confirm').checked=false;updatePulpyMapping();};
  $('pulpy-labels').oninput=()=>{$('pulpy-confirm').checked=false;};
  $('pulpy-folder').oninput=()=>{pulpyInventory=null;$('pulpy-options').classList.add('hidden');$('pulpy-status').textContent='Inspect this folder to list its cases.';};
});
