'use strict';
const JobProgress = (() => {
  const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const active = job => ['queued', 'running', 'cancelling'].includes(job.status);
  function duration(seconds) {
    const n = Math.max(0, Math.floor(seconds));
    return n >= 3600 ? `${Math.floor(n / 3600)}h ${Math.floor(n % 3600 / 60)}m` : n >= 60 ? `${Math.floor(n / 60)}m ${n % 60}s` : `${n}s`;
  }
  function data(job) {
    if (job.progress) return job.progress;
    const stages = {semantic:'Identifying teeth', resize:'Resampling scan', instance:'Separating individual teeth', postprocess:'Assembling tooth masks'};
    const matches = [...(job.tail || '').matchAll(/ToothSeg stage: (\w+)/g)];
    return {stage: stages[matches.at(-1)?.[1]] || job.message,
      elapsed_seconds: (job.finished || Date.now()/1000) - (job.started || job.created), steps: []};
  }
  function render(job, compact = false) {
    const p = data(job), running = active(job), count = p.counter;
    const valid = count && count.total > 0 && count.done >= 0 && count.done <= count.total;
    const percent = valid ? Math.floor(count.done * 100 / count.total) : null;
    const age = p.last_output_at ? Math.max(0, (p.observed_at || Date.now()/1000) - p.last_output_at) : null;
    const activity = !running ? '' : p.worker?.alive === false ? 'Worker exited; waiting for final status.' :
      age !== null ? `${p.worker?.alive ? 'Worker running · ' : ''}Last log update ${duration(age)} ago${age > 120 ? '. No new output; processing may still be underway.' : ''}` :
      p.worker?.alive ? 'Worker running; waiting for its next output.' : 'Waiting for worker output.';
    const label = valid ? `${percent}% · ${count.done} / ${count.total} ${compact ? 'blocks' : escape(p.counter_label || 'items')}` :
      running ? 'This stage has no numerical counter yet.' : '';
    const bar = valid ? `<progress max="${count.total}" value="${count.done}" aria-label="Progress of the current prediction pass"></progress>` :
      running && job.status !== 'queued' ? '<progress aria-label="Stage in progress; percentage unavailable"></progress>' : '';
    const steps = !compact && p.steps?.length ? `<ol class="job-steps">${p.steps.map(s => `<li class="step-${escape(s.state)}"><span>${escape(s.label)}</span><small>${escape(s.state)}</small></li>`).join('')}</ol>` : '';
    return `<div class="job-progress ${compact ? 'compact' : ''}"><div class="job-stage"><strong>${escape(p.stage)}</strong><span>${p.stage_index ? `Stage ${p.stage_index}/${p.stage_total}` : ''}</span></div>${bar}<div class="job-count">${label}${valid && count.done === count.total && running ? ' · finishing stage' : ''}</div><div class="job-timing">${job.status === 'queued' ? 'Queued' : 'Elapsed'} ${duration(p.elapsed_seconds)}${valid && running ? ' · current pass only' : ''}</div>${activity ? `<p class="job-activity">${escape(activity)}</p>` : ''}${steps}</div>`;
  }
  return {render, active};
})();
