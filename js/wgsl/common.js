// Shared WGSL: per-frame uniform block and hashing / noise helpers.

// Layout must match the FR_* float offsets in main.js (every member is a vec4f).
export const FRAME_WGSL = /* wgsl */ `
struct Frame {
  camU   : vec4f,  // observer 4-velocity, covariant (t, x, y, z)
  camF   : vec4f,  // forward leg of the observer tetrad, covariant
  camR   : vec4f,  // right leg
  camUp  : vec4f,  // up leg
  camPos : vec4f,  // xyz Kerr-Schild position, w = 1 when the analytic capture test is valid
  res    : vec4f,  // render width, height, 1/width, 1/height
  view   : vec4f,  // tan(fovX/2), tan(fovY/2), jitter x, jitter y (pixels)
  bh     : vec4f,  // spin a, horizon rH, capture radius, disk inner radius (ISCO)
  bh2    : vec4f,  // disk outer radius, escape radius, step k, radial step k
  bh3    : vec4f,  // max steps, min / max equatorial photon orbit radius, max |k|^2
  bh4    : vec4f,  // critical xi range lo / hi, time (M), accumulation weight
  disk   : vec4f,  // peak temperature (K), Doppler amount, brightness, opacity
  disk2  : vec4f,  // turbulence, gaps, flow period (fraction of an orbit), debug view
  sky    : vec4f,  // star brightness, Milky Way brightness, display pixel angle (rad), far radius
  skyR0  : vec4f,  // sky rotation matrix columns
  skyR1  : vec4f,
  skyR2  : vec4f,
  post   : vec4f,  // exposure (linear), bloom strength, frame index, contrast power
  vol    : vec4f,  // disk atmosphere scale height H/r, vertical optical depth, history clamp on, hue preservation
  extra  : vec4f,  // limb darkening amount, time step of this frame (M), camera moved (0/1), unused
};
`;

export const NOISE_WGSL = /* wgsl */ `
fn pcg3d(v0: vec3u) -> vec3u {
  var v = v0 * 1664525u + 1013904223u;
  v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
  v ^= v >> vec3u(16u);
  v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
  return v;
}
fn hash3i(p: vec3i) -> vec3f { return vec3f(pcg3d(bitcast<vec3u>(p))) * (1.0 / 4294967296.0); }

// Gradient noise, roughly in [-1, 1].
fn gnoise(p: vec3f) -> f32 {
  let i = vec3i(floor(p));
  let f = fract(p);
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  var n: array<f32, 8>;
  for (var c = 0; c < 8; c++) {
    let o = vec3i(c & 1, (c >> 1) & 1, (c >> 2) & 1);
    let g = hash3i(i + o) * 2.0 - 1.0;
    n[c] = dot(g, f - vec3f(o));
  }
  let x0 = mix(mix(n[0], n[1], u.x), mix(n[2], n[3], u.x), u.y);
  let x1 = mix(mix(n[4], n[5], u.x), mix(n[6], n[7], u.x), u.y);
  return mix(x0, x1, u.z) * 1.6;
}

// 2D gradient noise, periodic in x with an integer period (used around the disk).
fn gnoise2p(p: vec2f, period: i32, seed: i32) -> f32 {
  let i = vec2i(floor(p));
  let f = fract(p);
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  let x0 = ((i.x % period) + period) % period;
  let x1 = (x0 + 1) % period;
  let g00 = hash3i(vec3i(x0, i.y, seed)).xy * 2.0 - 1.0;
  let g10 = hash3i(vec3i(x1, i.y, seed)).xy * 2.0 - 1.0;
  let g01 = hash3i(vec3i(x0, i.y + 1, seed)).xy * 2.0 - 1.0;
  let g11 = hash3i(vec3i(x1, i.y + 1, seed)).xy * 2.0 - 1.0;
  let n0 = mix(dot(g00, f), dot(g10, f - vec2f(1.0, 0.0)), u.x);
  let n1 = mix(dot(g01, f - vec2f(0.0, 1.0)), dot(g11, f - vec2f(1.0, 1.0)), u.x);
  return mix(n0, n1, u.y) * 1.5;
}

// Footprint-filtered fbm of gnoise2p; octaves double in frequency so every octave stays
// periodic in x (its period doubles too).
fn fbm2p(p0: vec2f, period0: i32, fp: f32, octaves: i32, gain: f32, seed: i32) -> f32 {
  var p = p0;
  var per = period0;
  var amp = 0.5;
  var freq = 1.0;
  var sum = 0.0;
  for (var o = 0; o < octaves; o++) {
    let w = 1.0 - smoothstep(0.25, 0.6, fp * freq);
    if (w <= 0.0) { break; }
    sum += amp * w * gnoise2p(p, per, seed + o * 131);
    p = p * 2.0 + vec2f(0.0, 17.3);
    per *= 2;
    freq *= 2.0;
    amp *= gain;
  }
  return sum;
}

// Fractal noise whose octaves fade out once they get finer than the pixel footprint
// (fp, in noise-space units). Faded octaves contribute their mean (0), so filtering keeps
// the average brightness.
fn fbm(p0: vec3f, fp: f32, octaves: i32, gain: f32) -> f32 {
  var p = p0;
  var amp = 0.5;
  var freq = 1.0;
  var sum = 0.0;
  for (var o = 0; o < octaves; o++) {
    let w = 1.0 - smoothstep(0.25, 0.6, fp * freq);
    sum += amp * w * gnoise(p);
    p = p * 2.03 + vec3f(17.1, 3.7, 9.2);
    freq *= 2.03;
    amp *= gain;
  }
  return sum;
}
`;
