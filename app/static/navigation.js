'use strict';

function activeCamera(){
 const c=$('scene').layout?.scene?.camera||state.camera||{};
 return {eye:{x:1.6,y:-1.7,z:1.15,...c.eye},center:{x:0,y:0,z:0,...c.center},up:{x:0,y:0,z:1,...c.up},projection:c.projection||{type:'perspective'}};
}
function updateCamera(camera){
 state.camera=camera;
 $$('[data-camera]').forEach(b=>b.classList.remove('active'));
 return Plotly.relayout('scene',{'scene.camera':camera});
}
function turnCamera(yaw,pitch){
 if(!$('scene').data)return;
 const c=activeCamera(),v=['x','y','z'].map(k=>c.eye[k]-c.center[k]);
 const r=Math.hypot(...v),azimuth=Math.atan2(v[1],v[0])+yaw;
 const elevation=Math.max(-Math.PI/2+.04,Math.min(Math.PI/2-.04,Math.atan2(v[2],Math.hypot(v[0],v[1]))+pitch));
 const next=[r*Math.cos(elevation)*Math.cos(azimuth),r*Math.cos(elevation)*Math.sin(azimuth),r*Math.sin(elevation)];
 c.eye=Object.fromEntries(['x','y','z'].map((k,i)=>[k,c.center[k]+next[i]]));c.up={x:0,y:0,z:1};
 return updateCamera(c);
}
function zoomCamera(factor){
 if(!$('scene').data)return;
 const c=activeCamera(),v=['x','y','z'].map(k=>c.eye[k]-c.center[k]),r=Math.hypot(...v);
 const scale=Math.max(.2,Math.min(12,r*factor))/(r||1);
 c.eye=Object.fromEntries(['x','y','z'].map((k,i)=>[k,c.center[k]+v[i]*scale]));
 return updateCamera(c);
}
function setDragMode(mode){
 state.dragMode=mode;
 $$('[data-drag-mode]').forEach(b=>{const active=b.dataset.dragMode===mode;b.classList.toggle('active',active);b.setAttribute('aria-pressed',String(active))});
 $('scene').style.cursor=mode==='pan'?'move':'grab';
 $('rotation-help').textContent=mode==='pan'?'Click and hold inside the 3D view, then drag to pan.':'Click and hold inside the 3D view, then drag to rotate. Or use the arrow buttons.';
 if($('scene').data)Plotly.relayout('scene',{'scene.dragmode':mode});
}
function setupSceneNavigation(){
 const graph=$('scene');if(graph._navigationInstalled)return;graph._navigationInstalled=true;
 graph.tabIndex=0;graph.setAttribute('aria-label','Interactive 3D model. Drag to rotate; use arrow keys when focused.');
 graph.on('plotly_relayout',event=>{if(Object.keys(event).some(k=>k==='scene.camera'||k.startsWith('scene.camera.')))state.camera=JSON.parse(JSON.stringify(event['scene.camera']||graph.layout.scene.camera))});
 graph.addEventListener('keydown',e=>{
  const moves={ArrowLeft:[-.2,0],ArrowRight:[.2,0],ArrowUp:[0,.2],ArrowDown:[0,-.2]};
  if(moves[e.key]){e.preventDefault();turnCamera(...moves[e.key])}
  else if(e.key==='+'||e.key==='='){e.preventDefault();zoomCamera(.8)}
  else if(e.key==='-'){e.preventDefault();zoomCamera(1.25)}
 });
}
const nav=document.createElement('div');nav.className='scene-navigation';nav.setAttribute('role','group');nav.setAttribute('aria-label','3D movement controls');
nav.innerHTML='<div class="rotation-mode"><button class="active" data-drag-mode="orbit" aria-pressed="true">Rotate</button><button data-drag-mode="pan" aria-pressed="false">Pan</button></div><button data-turn="left" aria-label="Rotate model left">← Left</button><button data-turn="right" aria-label="Rotate model right">Right →</button><button data-turn="up" aria-label="Rotate model upward">↑ Up</button><button data-turn="down" aria-label="Rotate model downward">↓ Down</button><button id="zoom-in-3d" aria-label="Zoom 3D model in">＋</button><button id="zoom-out-3d" aria-label="Zoom 3D model out">−</button>';
document.querySelector('.view-3d .panel-heading').after(nav);
const help=document.createElement('div');help.id='rotation-help';help.className='rotation-help';nav.after(help);
$$('[data-drag-mode]').forEach(b=>b.onclick=()=>setDragMode(b.dataset.dragMode));
const turns={left:[-.25,0],right:[.25,0],up:[0,.2],down:[0,-.2]};
$$('[data-turn]').forEach(b=>b.onclick=()=>turnCamera(...turns[b.dataset.turn]));
$('zoom-in-3d').onclick=()=>zoomCamera(.8);$('zoom-out-3d').onclick=()=>zoomCamera(1.25);
setDragMode('orbit');
