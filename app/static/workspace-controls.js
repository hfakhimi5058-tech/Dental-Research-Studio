'use strict';

function syncToothControls() {
  const select = $('tooth-select');
  if (!select || !state.c) return;
  const rows = state.c.rows;
  select.innerHTML = rows.map(r => `<option value="${r.label_id}">${esc(r.structure)} · ${r.label_id}</option>`).join('');
  select.value = String(state.selected ?? '');
  for (const id of ['tooth-select','tooth-only','tooth-previous','tooth-next','teeth-all']) $(id).disabled = !rows.length;
  $('tooth-view-status').textContent = !rows.length ? 'Run segmentation to create tooth labels.' :
    state.visible.size === 1 ? 'Showing one predicted label' : `Showing ${state.visible.size} of ${rows.length} labels`;
  renderToothMeasurements();
}

function showOneTooth(id) {
  if (!state.c?.rows.some(r => r.label_id === id)) return;
  state.selected = id;
  state.visible = new Set([id]);
  state.camera = null;
  $('show-planes').checked = false;
  locateStructure(id);
}

function stepTooth(direction) {
  const ids = state.c?.rows.map(r => r.label_id) || [];
  if (!ids.length) return;
  const index = Math.max(0, ids.indexOf(state.selected));
  showOneTooth(ids[(index + direction + ids.length) % ids.length]);
}

function showWorkspaceProgress(job) {
  const host = $('workspace-progress'), inline = $('inference-progress');
  const running = job && JobProgress.active(job);
  const preparing = state.preparingSegmentation;
  const content = running ? `<span class="workspace-job-title">${esc(job.name)}</span>${JobProgress.render(job, true)}` :
    preparing ? '<strong>Preparing segmentation request…</strong><progress aria-label="Preparing scan; no numerical counter available"></progress><span>The scan is being prepared for the local worker.</span>' : '';
  for (const node of [host, inline]) {
    node.classList.toggle('hidden', !content);
    node.innerHTML = content;
  }
  $('infer-button').disabled = !!(running || preparing);
}

function showSurfaceProgress(done, total, drawing = false) {
  const host = $('surface-progress');
  host.classList.toggle('hidden', !total);
  if (!total) return;
  const percent = Math.floor(done * 100 / total);
  host.innerHTML = `<span>${drawing ? 'Drawing 3D view…' : 'Preparing 3D view'} · ${done}/${total} structures processed · ${percent}%</span><progress max="${total}" value="${done}" aria-label="3D surface preparation"></progress>`;
}

function sizeFacts(row) {
  return `<div class="size-facts"><div><small>Volume</small><strong>${fmt(row.volume_mm3, 1)} <em>mm³</em></strong></div><div><small>Labelled voxels</small><strong>${fmt(row.voxels, 0)}</strong></div><div><small>Bounds · R × A × S</small><strong class="bounds-value">${[row.extent_R_mm,row.extent_A_mm,row.extent_S_mm].map(v => fmt(v,2)).join(' × ')} <em>mm</em></strong></div></div>`;
}

function sizeWarning(row) {
  const components = state.c?.analysis?.structures.find(r => r.label_id === row.label_id)?.components;
  const notes = [];
  if (row.touches_image_boundary) notes.push('Touches scan edge');
  if (components > 1) notes.push(`${components} disconnected regions`);
  return notes.length ? `<p class="size-warning">Review size: ${notes.join(' · ')}</p>` : '';
}

function renderToothMeasurements() {
  const row = state.c?.rows.find(r => r.label_id === state.selected);
  const host = $('selected-tooth-size');
  if (!host) return;
  host.classList.toggle('hidden', !row);
  $('segment-sizes').disabled = !state.c?.rows.length;
  if (!row) return;
  host.innerHTML = `<div class="size-heading"><span><i class="swatch" style="background:${colours[row.label_id % colours.length]}"></i>${esc(row.structure)} <small>Label ${row.label_id}</small></span><button class="text-button" id="open-size-cards">All segment sizes ↗</button></div>${sizeFacts(row)}${sizeWarning(row)}`;
  $('open-size-cards').onclick = openSegmentSizes;
}

function renderSizeCards() {
  const query = $('segment-size-search').value.toLowerCase();
  let rows = (state.c?.rows || []).filter(r => `${r.structure} ${r.label_id}`.toLowerCase().includes(query));
  rows = [...rows].sort($('segment-size-sort').value === 'volume' ? (a,b) => b.volume_mm3-a.volume_mm3 : (a,b) => a.label_id-b.label_id);
  $('segment-size-cards').innerHTML = rows.length ? rows.map(row => `<article class="segment-size-card"><div class="size-card-heading"><i class="swatch" style="background:${colours[row.label_id % colours.length]}"></i><span>Label ${row.label_id}</span></div><h3>${esc(row.structure)}</h3>${sizeFacts(row)}${sizeWarning(row)}<button class="secondary full" data-size-tooth="${row.label_id}">View this tooth separately</button></article>`).join('') : '<p>No matching segments.</p>';
  $$('[data-size-tooth]').forEach(button => button.onclick = () => {$('segment-sizes-dialog').close(); showOneTooth(+button.dataset.sizeTooth);});
}

function openSegmentSizes() {
  renderSizeCards();
  $('segment-sizes-dialog').showModal();
}

document.addEventListener('DOMContentLoaded', () => {
$('tooth-select').onchange = () => showOneTooth(+$('tooth-select').value);
$('tooth-only').onclick = () => showOneTooth(state.selected);
$('tooth-previous').onclick = () => stepTooth(-1);
$('tooth-next').onclick = () => stepTooth(1);
$('teeth-all').onclick = () => {state.camera = null; $('select-all').click();};
$('isolate').onclick = () => showOneTooth(state.selected);
$('segment-sizes').onclick = openSegmentSizes;
$('segment-size-search').oninput = renderSizeCards;
$('segment-size-sort').onchange = renderSizeCards;
$('close-segment-sizes').onclick = () => $('segment-sizes-dialog').close();
});
