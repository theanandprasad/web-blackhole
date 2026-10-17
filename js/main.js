// WebGPU Kerr black hole: device setup, resources, camera, UI and the frame loop.
// Physics on the CPU side (observer frame, disk temperature, blackbody colours) comes from
// physics.js; the per-pixel geodesic integration lives in wgsl/trace.js.
import * as PH from './physics.js';
import { TRACE } from './wgsl/trace.js';
import { RESOLVE, BLOOM_DOWN, BLOOM_UP, COMPOSITE } from './wgsl/post.js';
import { SKY_BAKE, CUBE_MIP, SKY_CUBE_SIZE } from './wgsl/sky.js';

const qs = new URLSearchParams(location.search);
const D2R = Math.PI / 180;
const GU = GPUBufferUsage;
const TU = GPUTextureUsage;

// ----------------------------------------------------------------------------------------
// Tunable parameters (the settings panel edits these; any key also works as a URL param)
// ----------------------------------------------------------------------------------------
export const P = {
  spin: 0.94,          // a / M
  diskOuter: 18,       // M
  tPeak: 3150,         // K: peak gas temperature a non-spinning hole's disk would have at the
                       // same accretion rate (faster spins reach hotter inner disks)
  doppler: 1.0,        // 1 = physical Doppler + gravitational shift, 0 = none (film look)
  diskEV: 0.25,         // disk brightness, stops
  opacity: 1.0,
  thickness: 0.05,     // disk atmosphere scale height H/r
  atmosphere: 0.1,     // vertical optical depth of the disk atmosphere
  turbulence: 0.75,
  gaps: 0.0,
  timeSpeed: 0.4,      // x 25 M per second
  stars: 1.0,
  milkyWay: 1.3,
  skyTilt: 60,         // deg, galactic plane vs disk plane
  skyYaw: 210,          // deg
  exposure: 0.0,       // EV
  bloom: 0.08,
  contrast: 1.05,      // AgX look power
  hueKeep: 0.65,        // 0 = plain AgX, 1 = fully hue-preserving highlights
  limb: 0.7,           // electron-scattering limb darkening amount
  fov: 37,             // vertical, deg
  observer: 0,         // 0 static (hovering), 1 free-falling (Kerr-Schild normal observer)
  stepK: 0.15,         // ray step, fraction of r per step
  view: 0,             // 0 normal, 1 redshift map, 2 integration steps
};
const UI_SPEC = [
  ['spin', 'Spin a/M', -0.998, 0.998, 0.001],
  ['diskOuter', 'Disk outer radius (M)', 6, 40, 0.5],
  ['tPeak', 'Disk temperature scale (K)', 1000, 20000, 50],
  ['doppler', 'Doppler + gravitational shift', 0, 1, 0.01],
  ['diskEV', 'Disk brightness (EV)', -4, 4, 0.1],
  ['turbulence', 'Turbulence', 0, 1.5, 0.01],
  ['gaps', 'Gaps in the gas', 0, 1, 0.01],
  ['opacity', 'Disk opacity', 0, 1, 0.01],
  ['limb', 'Limb darkening', 0, 1, 0.01],
  ['thickness', 'Atmosphere height H/r', 0.005, 0.2, 0.005],
  ['atmosphere', 'Atmosphere optical depth', 0, 2, 0.01],
  ['timeSpeed', 'Time speed', 0, 5, 0.05],
  ['stars', 'Stars', 0, 3, 0.05],
  ['milkyWay', 'Milky Way', 0, 3, 0.05],
  ['skyTilt', 'Galaxy tilt (deg)', 0, 180, 1],
  ['skyYaw', 'Galaxy rotation (deg)', 0, 360, 1],
  ['exposure', 'Exposure (EV)', -6, 6, 0.1],
  ['bloom', 'Bloom', 0, 0.4, 0.005],
  ['contrast', 'Contrast', 0.8, 1.6, 0.01],
  ['hueKeep', 'Keep colour in highlights', 0, 1, 0.01],
  ['fov', 'Field of view (deg)', 10, 110, 1],
  ['stepK', 'Ray step (accuracy ↔ speed)', 0.03, 0.3, 0.005],
  ['observer', 'Observer', ['hovering (static)', 'free-falling']],
  ['view', 'View', ['normal', 'redshift map', 'integration steps']],
];
const DEFAULTS = { ...P };
// Presets start from the defaults and override a few settings.
const PRESETS = {
  'Physical (default)': {},
  'Film look: no Doppler shift': { doppler: 0, tPeak: 2700 },
  'Hot disk, 25 000 K peak': { tPeak: 9900, diskEV: -3.5, exposure: 0.3 },
  'Near-extremal spin, a = 0.998': { spin: 0.998 },
  'No spin (Schwarzschild), brighter disk': { spin: 0, tPeak: 7000 },
  'Retrograde disk, a = -0.9, brighter disk': { spin: -0.9, tPeak: 9000, diskEV: -1.0 },
  'Thick glowing atmosphere': { thickness: 0.09, atmosphere: 0.6 },
};
for (const [k, v] of qs) if (k in P && v !== '' && !Number.isNaN(Number(v))) P[k] = Number(v);

const SPEED_M_PER_S = 25;
const FLOW_PERIOD = 0.5;     // restart period of the disk texture advection, in local orbits
const TAA_BLEND = 0.2;       // weight of the newest frame in the temporal accumulation

function fail(msg) {
  const el = document.getElementById('err');
  el.style.display = 'block';
  el.textContent = msg;
  log('FATAL ' + msg);
  throw new Error(msg);
}
function log(msg) {
  console.log(msg);
  if (location.protocol.startsWith('http')) fetch('/__log', { method: 'POST', body: String(msg) }).catch(() => {});
}

async function main() {
  if (!navigator.gpu) fail('WebGPU is not available in this browser. Try a recent Chrome, Edge or Safari.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) fail('No WebGPU adapter found.');
  const hasTS = adapter.features.has('timestamp-query');
  const device = await adapter.requestDevice({ requiredFeatures: hasTS ? ['timestamp-query'] : [] });
  device.lost.then((i) => log('device lost: ' + i.message));
  device.onuncapturederror = (e) => log('GPU error: ' + e.error.message);

  const canvas = document.getElementById('c');
  const ctx = canvas.getContext('webgpu');
  const swapFormat = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format: swapFormat, alphaMode: 'opaque' });
  const srgbEncode = swapFormat.endsWith('-srgb') ? 0 : 1;

  async function shader(code, label) {
    const m = device.createShaderModule({ code, label });
    const info = await m.getCompilationInfo();
    let bad = false;
    for (const msg of info.messages) {
      if (msg.type === 'error') bad = true;
      if (msg.type !== 'info') {
        const line = code.split('\n')[msg.lineNum - 1];
        log(`[${label}] ${msg.type} ${msg.lineNum}:${msg.linePos} ${msg.message}\n    ${line}`);
      }
    }
    if (bad) fail('Shader compile error in ' + label + ' (see console)');
    return m;
  }

  // --------------------------------------------------------------------------------------
  // Pipelines
  // --------------------------------------------------------------------------------------
  const FS = GPUShaderStage.FRAGMENT;
  const traceBGL = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: FS, buffer: { type: 'uniform' } },
      { binding: 1, visibility: FS, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: FS, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: FS, texture: { sampleType: 'float', viewDimension: 'cube' } },
      { binding: 5, visibility: FS, sampler: { type: 'filtering' } },
    ],
  });
  const resolveBGL = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: FS, buffer: { type: 'uniform' } },
      { binding: 1, visibility: FS, texture: { sampleType: 'unfilterable-float' } },
      { binding: 2, visibility: FS, texture: { sampleType: 'unfilterable-float' } },
    ],
  });
  const [traceMod, resolveMod, downMod, upMod, compMod] = await Promise.all([
    shader(TRACE, 'trace'), shader(RESOLVE, 'resolve'), shader(BLOOM_DOWN, 'bloom down'), shader(BLOOM_UP, 'bloom up'), shader(COMPOSITE, 'composite'),
  ]);
  const fsPipe = (module, targets, layout = 'auto', label) => device.createRenderPipelineAsync({
    label, layout, vertex: { module, entryPoint: 'vs' }, fragment: { module, entryPoint: 'fs', targets },
    primitive: { topology: 'triangle-list' },
  });
  const additive = { color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' }, alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' } };
  const [tracePipe, resolvePipe, downPipe, upPipe, compPipe] = await Promise.all([
    fsPipe(traceMod, [{ format: 'rgba16float' }], device.createPipelineLayout({ bindGroupLayouts: [traceBGL] }), 'trace'),
    fsPipe(resolveMod, [{ format: 'rgba32float' }, { format: 'rgba16float' }], device.createPipelineLayout({ bindGroupLayouts: [resolveBGL] }), 'resolve'),
    fsPipe(downMod, [{ format: 'rgba16float' }], 'auto', 'bloom down'),
    fsPipe(upMod, [{ format: 'rgba16float', blend: additive }], 'auto', 'bloom up'),
    fsPipe(compMod, [{ format: swapFormat }], 'auto', 'composite'),
  ]);

  // --------------------------------------------------------------------------------------
  // Static resources
  // --------------------------------------------------------------------------------------
  const FRAME_FLOATS = 80;
  const frameData = new Float32Array(FRAME_FLOATS);
  const frameUBO = device.createBuffer({ size: FRAME_FLOATS * 4, usage: GU.UNIFORM | GU.COPY_DST, label: 'frame' });
  const bbData = PH.blackbodyLUT();
  const bbBuf = device.createBuffer({ size: bbData.byteLength, usage: GU.STORAGE | GU.COPY_DST, label: 'blackbody' });
  device.queue.writeBuffer(bbBuf, 0, bbData);
  const T_LUT = 512;
  const tBuf = device.createBuffer({ size: T_LUT * 4, usage: GU.STORAGE | GU.COPY_DST, label: 'disk T' });
  const dispUBO = device.createBuffer({ size: 16, usage: GU.UNIFORM | GU.COPY_DST, label: 'display' });
  const dispData = new Float32Array(4);
  const linSamp = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
  const trilinear = device.createSampler({ magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear' });

  // Background galaxy: baked once into a cube map with a full mip chain.
  const skyMips = Math.log2(SKY_CUBE_SIZE) + 1;
  const skyTex = device.createTexture({
    label: 'sky cube', size: [SKY_CUBE_SIZE, SKY_CUBE_SIZE, 6], format: 'rgba16float', mipLevelCount: skyMips,
    usage: TU.TEXTURE_BINDING | TU.STORAGE_BINDING,
  });
  {
    const [bakePipe, mipPipe] = await Promise.all([
      shader(SKY_BAKE, 'sky bake').then((module) => device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } })),
      shader(CUBE_MIP, 'cube mip').then((module) => device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } })),
    ]);
    const level = (m) => skyTex.createView({ dimension: '2d-array', baseMipLevel: m, mipLevelCount: 1 });
    const enc = device.createCommandEncoder();
    const cp = enc.beginComputePass();
    cp.setPipeline(bakePipe);
    cp.setBindGroup(0, device.createBindGroup({ layout: bakePipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: level(0) }] }));
    cp.dispatchWorkgroups(SKY_CUBE_SIZE / 8, SKY_CUBE_SIZE / 8, 6);
    cp.setPipeline(mipPipe);
    for (let m = 1; m < skyMips; m++) {
      const n = SKY_CUBE_SIZE >> m;
      cp.setBindGroup(0, device.createBindGroup({ layout: mipPipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: level(m - 1) }, { binding: 1, resource: level(m) }] }));
      cp.dispatchWorkgroups(Math.ceil(n / 8), Math.ceil(n / 8), 6);
    }
    cp.end();
    device.queue.submit([enc.finish()]);
  }
  const skyView = skyTex.createView({ dimension: 'cube' });

  // Disk temperature LUT and derived radii depend on spin and disk size only.
  const bh = { key: '', a: 0, rH: 0, rCap: 0, rIn: 0, rOut: 0, rPh: [0, 0], xi: [0, 0] };
  function updateBlackHole() {
    const a = Math.max(-0.998, Math.min(0.998, P.spin));
    const rIn = PH.iscoRadius(a);
    const rOut = Math.max(P.diskOuter, rIn + 2);
    const key = a.toFixed(4) + '/' + rOut.toFixed(3);
    if (key === bh.key) return false;
    bh.key = key;
    Object.assign(bh, { a, rIn, rOut, rH: PH.horizonRadius(a), rCap: PH.captureRadius(a) });
    const [r1, r2] = PH.photonOrbitRadii(a);
    bh.rPh = [Math.min(r1, r2), Math.max(r1, r2)];
    bh.xi = Math.abs(a) < 1e-4 ? [0, 0] : [PH.shadowXi(bh.rPh[0], a), PH.shadowXi(bh.rPh[1], a)];
    const lut = PH.diskTemperatureLUT(a, rOut, T_LUT);
    bh.peak = lut.peak;
    device.queue.writeBuffer(tBuf, 0, lut.lut);
    return true;
  }
  updateBlackHole();

  // --------------------------------------------------------------------------------------
  // Size-dependent targets
  // --------------------------------------------------------------------------------------
  const fixedW = Number(qs.get('w')) || 0, fixedH = Number(qs.get('h')) || 0;
  let renderScale = Number(qs.get('scale')) || 0;   // 0 = automatic
  const autoScale = !renderScale;
  const T = { w: 0, h: 0 };
  let accumTex = [], accumView = [], curTex, curView, hdrTex, hdrView, bloomTex, bloomViews = [], traceBG, resolveBG = [], downBG = [], upBG = [], compBG;
  let bloomLevels = 0;
  const bloomUBOs = [];

  function displaySize() {
    if (fixedW && fixedH) return [fixedW, fixedH];
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    return [Math.max(1, Math.round(canvas.clientWidth * dpr)), Math.max(1, Math.round(canvas.clientHeight * dpr))];
  }

  function makeTargets(w, h) {
    for (const t of [...accumTex, curTex, hdrTex, bloomTex]) t?.destroy();
    T.w = w; T.h = h;
    accumTex = [0, 1].map((i) => device.createTexture({ label: 'accum' + i, size: [w, h], format: 'rgba32float', usage: TU.RENDER_ATTACHMENT | TU.TEXTURE_BINDING }));
    accumView = accumTex.map((t) => t.createView());
    curTex = device.createTexture({ label: 'current', size: [w, h], format: 'rgba16float', usage: TU.RENDER_ATTACHMENT | TU.TEXTURE_BINDING });
    curView = curTex.createView();
    hdrTex = device.createTexture({ label: 'hdr', size: [w, h], format: 'rgba16float', usage: TU.RENDER_ATTACHMENT | TU.TEXTURE_BINDING });
    hdrView = hdrTex.createView();
    const bw = Math.max(1, w >> 1), bhgt = Math.max(1, h >> 1);
    bloomLevels = Math.max(1, Math.min(7, Math.floor(Math.log2(Math.min(bw, bhgt))) - 2));
    bloomTex = device.createTexture({ label: 'bloom', size: [bw, bhgt], format: 'rgba16float', mipLevelCount: bloomLevels, usage: TU.RENDER_ATTACHMENT | TU.TEXTURE_BINDING });
    bloomViews = [];
    for (let m = 0; m < bloomLevels; m++) bloomViews.push(bloomTex.createView({ baseMipLevel: m, mipLevelCount: 1 }));
    traceBG = device.createBindGroup({
      layout: traceBGL,
      entries: [
        { binding: 0, resource: { buffer: frameUBO } },
        { binding: 1, resource: { buffer: bbBuf } },
        { binding: 2, resource: { buffer: tBuf } },
        { binding: 4, resource: skyView },
        { binding: 5, resource: trilinear },
      ],
    });
    resolveBG = [0, 1].map((i) => device.createBindGroup({
      layout: resolveBGL,
      entries: [
        { binding: 0, resource: { buffer: frameUBO } },
        { binding: 1, resource: curView },
        { binding: 2, resource: accumView[i] },
      ],
    }));
    while (bloomUBOs.length < 2 * bloomLevels) bloomUBOs.push(device.createBuffer({ size: 16, usage: GU.UNIFORM | GU.COPY_DST }));
    downBG = []; upBG = [];
    for (let m = 0; m < bloomLevels; m++) {
      const srcW = m === 0 ? w : Math.max(1, bw >> (m - 1)), srcH = m === 0 ? h : Math.max(1, bhgt >> (m - 1));
      device.queue.writeBuffer(bloomUBOs[m], 0, new Float32Array([1 / srcW, 1 / srcH, m === 0 ? 1 : 0, 0]));
      downBG.push(device.createBindGroup({ layout: downPipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: m === 0 ? hdrView : bloomViews[m - 1] },
        { binding: 1, resource: linSamp },
        { binding: 2, resource: { buffer: bloomUBOs[m] } },
      ] }));
    }
    for (let m = 1; m < bloomLevels; m++) {
      const u = bloomUBOs[bloomLevels + m];
      // each level is added onto the next finer one at 0.7x, so wide halos stay faint and
      // a bright disk does not flood the shadow with glare
      device.queue.writeBuffer(u, 0, new Float32Array([1 / Math.max(1, bw >> m), 1 / Math.max(1, bhgt >> m), 0.7, 0]));
      upBG[m] = device.createBindGroup({ layout: upPipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: bloomViews[m] },
        { binding: 1, resource: linSamp },
        { binding: 2, resource: { buffer: u } },
      ] });
    }
    compBG = device.createBindGroup({ layout: compPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: frameUBO } },
      { binding: 1, resource: hdrView },
      { binding: 2, resource: bloomViews[0] },
      { binding: 3, resource: linSamp },
      { binding: 4, resource: { buffer: dispUBO } },
    ] });
    resetAccum();
  }

  function fitTargets(force) {
    const [dw, dh] = displaySize();
    if (canvas.width !== dw || canvas.height !== dh) { canvas.width = dw; canvas.height = dh; force = true; }
    if (!renderScale) renderScale = Math.min(1, Math.sqrt(1.3e6 / (dw * dh)));
    const w = Math.max(16, Math.round(dw * renderScale)), h = Math.max(16, Math.round(dh * renderScale));
    if (force || w !== T.w || h !== T.h) makeTargets(w, h);
  }

  // --------------------------------------------------------------------------------------
  // GPU timing
  // --------------------------------------------------------------------------------------
  let querySet, queryResolve, queryRead = [], queryBusy = [false, false, false], gpuMs = 0;
  if (hasTS) {
    querySet = device.createQuerySet({ type: 'timestamp', count: 2 });
    queryResolve = device.createBuffer({ size: 16, usage: GU.QUERY_RESOLVE | GU.COPY_SRC });
    queryRead = [0, 1, 2].map(() => device.createBuffer({ size: 16, usage: GU.COPY_DST | GU.MAP_READ }));
  }

  // --------------------------------------------------------------------------------------
  // Camera
  // --------------------------------------------------------------------------------------
  const HOME = { dist: 30, incl: 86, az: -12 };
  const cam = { mode: 'orbit', ...HOME, pos: [0, 0, 0], yaw: 0, pitch: 0 };
  if (qs.get('cam')) {
    const v = qs.get('cam').split(',').map(Number);
    if (v.length >= 3) Object.assign(cam, { dist: v[0], incl: v[1], az: v[2] });
  }
  if (qs.get('fly')) {
    const v = qs.get('fly').split(',').map(Number);
    Object.assign(cam, { mode: 'fly', pos: v.slice(0, 3), yaw: v[3] || 0, pitch: v[4] || 0 });
  }
  function orbitPose() {
    const i = cam.incl * D2R, z = cam.az * D2R;
    const pos = [cam.dist * Math.sin(i) * Math.cos(z), cam.dist * Math.sin(i) * Math.sin(z), cam.dist * Math.cos(i)];
    const fwd = pos.map((v) => -v / cam.dist);
    return { pos, fwd, up: [0, 0, 1] };
  }
  function flyPose() {
    const y = cam.yaw * D2R, p = cam.pitch * D2R;
    return { pos: cam.pos, fwd: [Math.cos(p) * Math.cos(y), Math.cos(p) * Math.sin(y), Math.sin(p)], up: [0, 0, 1] };
  }
  function enterFly() {
    const o = orbitPose();
    cam.pos = o.pos.slice();
    cam.yaw = Math.atan2(o.fwd[1], o.fwd[0]) / D2R;
    cam.pitch = Math.asin(Math.max(-1, Math.min(1, o.fwd[2]))) / D2R;
    cam.mode = 'fly';
  }
  function minDist() { return bh.rH + 0.35; }

  const keys = new Set();
  let dragging = false, lastX = 0, lastY = 0, changed = true;
  canvas.addEventListener('mousedown', (e) => {
    if (cam.mode === 'fly') { canvas.requestPointerLock?.(); return; }
    dragging = true; lastX = e.clientX; lastY = e.clientY;
  });
  window.addEventListener('mouseup', () => { dragging = false; });
  window.addEventListener('mousemove', (e) => {
    if (cam.mode === 'fly' && document.pointerLockElement === canvas) {
      cam.yaw -= e.movementX * 0.1;
      cam.pitch = Math.max(-89, Math.min(89, cam.pitch - e.movementY * 0.1));
      changed = true;
    } else if (dragging) {
      cam.az -= (e.clientX - lastX) * 0.25;
      cam.incl = Math.max(0.5, Math.min(179.5, cam.incl - (e.clientY - lastY) * 0.25));
      lastX = e.clientX; lastY = e.clientY;
      changed = true;
    }
  });
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    if (cam.mode === 'orbit') cam.dist = Math.max(minDist(), Math.min(2000, cam.dist * Math.exp(e.deltaY * 0.001)));
    else P.fov = Math.max(10, Math.min(110, P.fov * Math.exp(e.deltaY * 0.001)));
    changed = true;
  }, { passive: false });
  // touch: one finger orbits, two fingers zoom
  let touches = new Map(), pinch0 = 0, dist0 = 0;
  canvas.addEventListener('pointerdown', (e) => { if (e.pointerType === 'touch') { touches.set(e.pointerId, [e.clientX, e.clientY]); if (touches.size === 2) { const [p, q] = [...touches.values()]; pinch0 = Math.hypot(p[0] - q[0], p[1] - q[1]); dist0 = cam.dist; } } });
  canvas.addEventListener('pointermove', (e) => {
    if (e.pointerType !== 'touch' || !touches.has(e.pointerId)) return;
    const prev = touches.get(e.pointerId);
    touches.set(e.pointerId, [e.clientX, e.clientY]);
    if (touches.size === 1 && cam.mode === 'orbit') {
      cam.az -= (e.clientX - prev[0]) * 0.3;
      cam.incl = Math.max(0.5, Math.min(179.5, cam.incl - (e.clientY - prev[1]) * 0.3));
    } else if (touches.size === 2) {
      const [p, q] = [...touches.values()];
      cam.dist = Math.max(minDist(), Math.min(2000, dist0 * pinch0 / Math.max(1, Math.hypot(p[0] - q[0], p[1] - q[1]))));
    }
    changed = true;
  });
  const endTouch = (e) => touches.delete(e.pointerId);
  canvas.addEventListener('pointerup', endTouch);
  canvas.addEventListener('pointercancel', endTouch);

  let paused = false, photo = false, autoOrbit = qs.get('orbit') === '1', timeM = Number(qs.get('t')) || 0;
  window.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    keys.add(e.code);
    if (e.code === 'KeyH') panel.classList.toggle('hidden');
    if (e.code === 'Space') { paused = !paused; e.preventDefault(); }
    if (e.code === 'KeyP') {
      photo = !photo; paused = photo;
      hud.style.display = photo ? 'none' : '';
      if (photo) { info.classList.add('hidden'); panel.classList.add('hidden'); }
    }
    if (e.code === 'KeyF') { if (cam.mode === 'orbit') enterFly(); else { cam.mode = 'orbit'; document.exitPointerLock?.(); } changed = true; }
    if (e.code === 'KeyR') { Object.assign(cam, { mode: 'orbit', ...HOME }); changed = true; }
    if (e.code === 'KeyO') { autoOrbit = !autoOrbit; }
    if (e.code === 'KeyI') setInfo(info.classList.contains('hidden'));
  });
  window.addEventListener('keyup', (e) => keys.delete(e.code));
  window.addEventListener('blur', () => keys.clear());

  function moveFly(dt) {
    if (cam.mode !== 'fly') return;
    const { fwd } = flyPose();
    const right = [fwd[1], -fwd[0], 0].map((v) => v / Math.max(1e-6, Math.hypot(fwd[0], fwd[1])));
    let m = [0, 0, 0];
    const add = (v, s) => { m = m.map((q, i) => q + v[i] * s); };
    if (keys.has('KeyW')) add(fwd, 1);
    if (keys.has('KeyS')) add(fwd, -1);
    if (keys.has('KeyD')) add(right, 1);
    if (keys.has('KeyA')) add(right, -1);
    if (keys.has('KeyE')) add([0, 0, 1], 1);
    if (keys.has('KeyQ') || keys.has('KeyC')) add([0, 0, 1], -1);
    if (!m.some((v) => v !== 0)) return;
    const r = PH.ksPoint(cam.pos[0], cam.pos[1], cam.pos[2], bh.a).r;
    const speed = 0.4 * Math.max(r - bh.rH, 0.3) * (keys.has('ShiftLeft') || keys.has('ShiftRight') ? 5 : 1);
    const next = cam.pos.map((q, i) => q + m[i] * speed * dt);
    if (PH.ksPoint(next[0], next[1], next[2], bh.a).r > minDist()) cam.pos = next;
    changed = true;
  }

  // --------------------------------------------------------------------------------------
  // UI
  // --------------------------------------------------------------------------------------
  const hud = document.getElementById('hud');
  const fpsEl = document.getElementById('fps');
  const panel = document.getElementById('panel');
  const info = document.getElementById('info');
  const setInfo = (show) => {
    info.classList.toggle('hidden', !show);
    try { localStorage.setItem('bh-info-hidden', show ? '0' : '1'); } catch { /* storage unavailable */ }
  };
  try { if (localStorage.getItem('bh-info-hidden') === '1') info.classList.add('hidden'); } catch { /* storage unavailable */ }
  document.getElementById('info-close').addEventListener('click', () => setInfo(false));
  if (qs.get('ui') === '0' || qs.get('info') === '0') info.classList.add('hidden');
  if (qs.get('ui') === '0') { hud.style.display = 'none'; }
  if (qs.get('panel') === '1') panel.classList.remove('hidden');
  const fmt = (v, step) => (step >= 1 ? String(Math.round(v)) : v.toFixed(step >= 0.1 ? 1 : step >= 0.01 ? 2 : 3));
  const controls = [];
  const refreshUI = () => controls.forEach((c) => c());
  {
    const row = document.createElement('label');
    row.classList.add('select', 'preset');
    row.append(Object.assign(document.createElement('span'), { textContent: 'Preset' }));
    const sel = document.createElement('select');
    sel.append(Object.assign(document.createElement('option'), { value: '', textContent: '—' }));
    for (const name of Object.keys(PRESETS)) sel.append(Object.assign(document.createElement('option'), { value: name, textContent: name }));
    sel.addEventListener('change', () => {
      if (!sel.value) return;
      Object.assign(P, DEFAULTS, PRESETS[sel.value]);
      refreshUI();
      changed = true;
    });
    row.append(sel, document.createElement('output'));
    panel.append(row);
  }
  for (const [key, label, a, b, step] of UI_SPEC) {
    const row = document.createElement('label');
    row.append(Object.assign(document.createElement('span'), { textContent: label }));
    if (Array.isArray(a)) {
      const sel = document.createElement('select');
      a.forEach((name, i) => sel.append(Object.assign(document.createElement('option'), { value: i, textContent: name })));
      sel.value = P[key];
      sel.addEventListener('change', () => { P[key] = Number(sel.value); changed = true; });
      controls.push(() => { sel.value = P[key]; });
      row.append(sel, document.createElement('output'));
      row.classList.add('select');
    } else {
      const inp = Object.assign(document.createElement('input'), { type: 'range', min: a, max: b, step, value: P[key] });
      const out = Object.assign(document.createElement('output'), { textContent: fmt(P[key], step) });
      inp.addEventListener('input', () => { P[key] = Number(inp.value); out.textContent = fmt(P[key], step); changed = true; });
      controls.push(() => { inp.value = P[key]; out.textContent = fmt(P[key], step); });
      row.append(inp, out);
    }
    panel.append(row);
  }
  // shareable link: camera plus every setting that differs from its default
  const linkBtn = Object.assign(document.createElement('button'), { textContent: 'Copy link to this view', className: 'link' });
  linkBtn.addEventListener('click', () => {
    const u = new URL(location.href);
    u.search = '';
    if (cam.mode === 'orbit') u.searchParams.set('cam', [cam.dist, cam.incl, cam.az].map((v) => +v.toFixed(2)).join(','));
    else u.searchParams.set('fly', [...cam.pos, cam.yaw, cam.pitch].map((v) => +v.toFixed(3)).join(','));
    for (const k of Object.keys(P)) if (P[k] !== DEFAULTS[k]) u.searchParams.set(k, +Number(P[k]).toFixed(4));
    u.searchParams.set('t', Math.round(timeM));
    const done = () => { linkBtn.textContent = 'Link copied'; setTimeout(() => { linkBtn.textContent = 'Copy link to this view'; }, 1500); };
    navigator.clipboard?.writeText(u.toString()).then(done, () => prompt('Link to this view:', u.toString()));
  });
  panel.append(linkBtn);
  const note = document.createElement('div');
  note.className = 'note';
  note.innerHTML = 'Drag to orbit · wheel to zoom · <b>O</b> auto-orbit · <b>F</b> free flight (WASD, E/Q, Shift) · <b>Space</b> pause time · <b>P</b> photo mode (still frame, refines) · <b>R</b> reset · <b>H</b> hide panel';
  panel.append(note);

  // --------------------------------------------------------------------------------------
  // Per-frame uniforms
  // --------------------------------------------------------------------------------------
  let accumN = 0, frameIndex = 0, frameDt = 0, movedThisFrame = false;
  const bloomCap = new Float32Array(4);
  function resetAccum() { accumN = 0; }
  const halton = (i, b) => { let f = 1, r = 0; while (i > 0) { f /= b; r += f * (i % b); i = Math.floor(i / b); } return r; };
  let lastParamKey = '';
  let observerUsed = 'static';

  function writeFrame() {
    const pose = cam.mode === 'orbit' ? orbitPose() : flyPose();
    if (cam.mode === 'orbit') cam.dist = Math.max(cam.dist, minDist());
    const fr = PH.observerFrame(pose.pos, pose.fwd, pose.up, bh.a, P.observer === 1 ? 'normal' : 'static');
    observerUsed = fr.observer;
    const d = frameData;
    d.set(fr.u, 0); d.set(fr.F, 4); d.set(fr.R, 8); d.set(fr.U, 12);
    const sinT = Math.hypot(pose.pos[0], pose.pos[1]) / Math.max(1e-9, Math.hypot(...pose.pos));
    d.set([...pose.pos, sinT > 1e-3 ? 1 : 0], 16);
    d.set([T.w, T.h, 1 / T.w, 1 / T.h], 20);
    // fov is the vertical field of view of a 16:9 frame; narrower frames (portrait phones)
    // keep that frame's horizontal field of view so the composition still fits
    const aspect = T.w / T.h;
    let tanY = Math.tan(P.fov * 0.5 * D2R);
    if (aspect < 16 / 9) tanY *= (16 / 9) / aspect;
    const tanX = tanY * aspect;
    const accumulating = accumN > 0;
    const hi = (accumN % 256) + 1;
    const jx = accumulating ? halton(hi, 2) - 0.5 : 0, jy = accumulating ? halton(hi, 3) - 0.5 : 0;
    // While time runs, history is blended at a fixed rate (jittered temporal anti-aliasing
    // with a little motion blur on the rotating gas); when paused it converges fully.
    const weight = paused ? 1 / (accumN + 1) : Math.max(1 / (accumN + 1), TAA_BLEND);
    d.set([tanX, tanY, jx, jy], 24);
    const rCam = PH.ksPoint(pose.pos[0], pose.pos[1], pose.pos[2], bh.a).r;
    const rEsc = PH.escapeRadius(bh.rOut, rCam);
    d.set([bh.a, bh.rH, bh.rCap, bh.rIn], 28);
    d.set([bh.rOut, rEsc, P.stepK, PH.STEP.kr], 32);
    d.set([PH.STEP.maxSteps, bh.rPh[0], bh.rPh[1], PH.STEP.kMax2], 36);
    d.set([bh.xi[0], bh.xi[1], timeM, weight], 40);
    d.set([P.tPeak, P.doppler, Math.pow(2, P.diskEV), P.opacity], 44);
    d.set([P.turbulence, P.gaps, FLOW_PERIOD, P.view], 48);
    d.set([P.stars, P.milkyWay, 2 * tanY / canvas.height, PH.farRadius(bh.rOut)], 52);
    // sky orientation: tilt the galactic plane about x, then turn it about z
    const ct = Math.cos(P.skyTilt * D2R), st = Math.sin(P.skyTilt * D2R), cy = Math.cos(P.skyYaw * D2R), sy = Math.sin(P.skyYaw * D2R);
    // columns of M = Rx(tilt) * Rz(yaw)
    d.set([cy, ct * sy, st * sy, 0], 56);
    d.set([-sy, ct * cy, st * cy, 0], 60);
    d.set([0, -st, ct, 0], 64);
    d.set([Math.pow(2, P.exposure), P.bloom, frameIndex % 4096, P.contrast], 68);
    d.set([P.thickness, P.atmosphere, paused ? 0 : 1, P.hueKeep], 72);
    // bloom input cap: 1.5x display white, in scene units before exposure
    bloomCap[0] = 1 / T.w; bloomCap[1] = 1 / T.h; bloomCap[2] = 1; bloomCap[3] = 1.5 / Math.pow(2, P.exposure);
    if (bloomUBOs[0]) device.queue.writeBuffer(bloomUBOs[0], 0, bloomCap);
    d.set([P.limb, frameDt, movedThisFrame ? 1 : 0, 0], 76);
    device.queue.writeBuffer(frameUBO, 0, d);
    dispData.set([canvas.width, canvas.height, srgbEncode, 0]);
    device.queue.writeBuffer(dispUBO, 0, dispData);
    return { rCam };
  }

  // --------------------------------------------------------------------------------------
  // Frame
  // --------------------------------------------------------------------------------------
  const shotFrames = Number(qs.get('shot')) || 0;
  const shotAccum = Number(qs.get('accum')) || 0;
  if (shotAccum) paused = true;
  let lastT = performance.now(), fpsAcc = 0, fpsN = 0, fps = 0, lastScaleChange = 0;
  window.addEventListener('resize', () => { changed = true; });
  fitTargets(true);

  function frame(now) {
    const dt = Math.min(0.1, (now - lastT) / 1000);
    lastT = now;
    fpsAcc += dt; fpsN++;
    if (fpsAcc > 0.5) { fps = fpsN / fpsAcc; fpsAcc = 0; fpsN = 0; }

    moveFly(dt);
    if (autoOrbit && cam.mode === 'orbit' && !dragging) { cam.az += dt * 4; changed = true; }
    frameDt = paused ? 0 : dt * P.timeSpeed * SPEED_M_PER_S;
    timeM += frameDt;
    if (updateBlackHole()) changed = true;
    const paramKey = JSON.stringify(P);
    if (paramKey !== lastParamKey) { lastParamKey = paramKey; changed = true; }

    // dynamic resolution: aim for ~12 ms of GPU time per frame
    if (autoScale && hasTS && gpuMs > 0 && !paused && now - lastScaleChange > 600) {
      const target = 12;
      let s = renderScale;
      if (gpuMs > target * 1.25) s *= Math.max(0.75, Math.sqrt(target / gpuMs));
      else if (gpuMs < target * 0.6) s *= 1.08;
      s = Math.max(0.25, Math.min(1, s));
      if (Math.abs(s - renderScale) / renderScale > 0.04) { renderScale = s; lastScaleChange = now; changed = true; }
    }
    fitTargets(false);

    // live: history survives changes (the resolve pass clamps it to the current frame);
    // paused: a converging still restarts on any change. New targets reset it anyway.
    if (changed && paused) resetAccum();
    movedThisFrame = changed;
    changed = false;
    const { rCam } = writeFrame();

    const enc = device.createCommandEncoder();
    const src = accumN % 2, dst = 1 - src;   // read accum[src], write accum[dst]
    const tp = enc.beginRenderPass({
      colorAttachments: [{ view: curView, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
      ...(hasTS ? { timestampWrites: { querySet, beginningOfPassWriteIndex: 0 } } : {}),
    });
    tp.setPipeline(tracePipe);
    tp.setBindGroup(0, traceBG);
    tp.draw(3);
    tp.end();
    const rp = enc.beginRenderPass({
      colorAttachments: [
        { view: accumView[dst], loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] },
        { view: hdrView, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] },
      ],
    });
    rp.setPipeline(resolvePipe);
    rp.setBindGroup(0, resolveBG[src]);
    rp.draw(3);
    rp.end();
    accumN++;

    for (let m = 0; m < bloomLevels; m++) {
      const p = enc.beginRenderPass({ colorAttachments: [{ view: bloomViews[m], loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }] });
      p.setPipeline(downPipe); p.setBindGroup(0, downBG[m]); p.draw(3); p.end();
    }
    for (let m = bloomLevels - 1; m >= 1; m--) {
      const p = enc.beginRenderPass({ colorAttachments: [{ view: bloomViews[m - 1], loadOp: 'load', storeOp: 'store' }] });
      p.setPipeline(upPipe); p.setBindGroup(0, upBG[m]); p.draw(3); p.end();
    }
    const cp = enc.beginRenderPass({
      colorAttachments: [{ view: ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
      ...(hasTS ? { timestampWrites: { querySet, endOfPassWriteIndex: 1 } } : {}),
    });
    cp.setPipeline(compPipe); cp.setBindGroup(0, compBG); cp.draw(3); cp.end();

    let qi = -1;
    if (hasTS) {
      qi = queryBusy.findIndex((b) => !b);
      enc.resolveQuerySet(querySet, 0, 2, queryResolve, 0);
      if (qi >= 0) enc.copyBufferToBuffer(queryResolve, 0, queryRead[qi], 0, 16);
    }
    device.queue.submit([enc.finish()]);
    if (qi >= 0) {
      queryBusy[qi] = true;
      const buf = queryRead[qi];
      buf.mapAsync(GPUMapMode.READ).then(() => {
        const t = new BigInt64Array(buf.getMappedRange());
        const ms = Number(t[1] - t[0]) / 1e6;
        if (ms > 0 && ms < 1000) gpuMs = gpuMs ? gpuMs * 0.85 + ms * 0.15 : ms;
        buf.unmap();
        queryBusy[qi] = false;
      }).catch(() => { queryBusy[qi] = false; });
    }
    frameIndex++;

    if (frameIndex % 10 === 0 || paused) {
      const obs = observerUsed === 'static' ? 'hovering' : 'free-falling';
      fpsEl.textContent = `${fps.toFixed(0)} fps` + (hasTS ? ` · GPU ${gpuMs.toFixed(1)} ms` : '') +
        ` · ${T.w}×${T.h}` + (paused ? ` · paused, ${accumN} samples` : '') +
        `\na = ${bh.a.toFixed(3)} · r = ${rCam.toFixed(1)} M · ISCO ${bh.rIn.toFixed(2)} M · horizon ${bh.rH.toFixed(2)} M · disk peak ${Math.round(P.tPeak * bh.peak / 100) * 100} K · ${obs} observer` +
        (cam.mode === 'fly' ? ' · free flight' : '');
    }
    window.__stats = { fps, gpuMs, w: T.w, h: T.h, samples: accumN, frame: frameIndex };
    if (shotFrames && frameIndex >= shotFrames && (!shotAccum || accumN >= shotAccum)) {
      device.queue.onSubmittedWorkDone().then(() => { window.__shotReady = true; });
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

main().catch((e) => { log('error: ' + (e.stack || e.message)); });
