// Kerr geodesic ray tracer: one backward light ray per pixel, integrated with RK4 in
// Cartesian Kerr-Schild coordinates (see js/physics.js for the derivation; geodesicRHS here
// is the same arithmetic as rhs() there, which the node tests check against analytic
// results). The ray records up to three crossings of the thin disk, then the escaped
// direction picks up the lensed sky. Disk and sky are shaded after the loop, where screen
// derivatives of the hit positions give each pixel's footprint for texture filtering.
import { FRAME_WGSL, NOISE_WGSL } from './common.js';

export const FULLSCREEN_VS = /* wgsl */ `
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}
`;

const GEODESIC_WGSL = /* wgsl */ `
struct Deriv { dx: vec3f, dk: vec3f, r: f32, drdl: f32 };
struct St { p: vec3f, k: vec3f };

// Hamilton's equations for H = 1/2 g^mn k_m k_n with g^mn = eta^mn - f l^m l^n.
fn geodesicRHS(p: vec3f, k: vec3f, kt: f32) -> Deriv {
  let a = F.bh.x;
  let a2 = a * a;
  let w = dot(p, p) - a2;
  let D = max(sqrt(0.25 * w * w + a2 * p.z * p.z), 1e-12);
  let r2 = 0.5 * w + D;
  let r = sqrt(r2);
  let ir = 1.0 / r;
  let is = 1.0 / (r2 + a2);
  let l = vec3f((r * p.x + a * p.y) * is, (r * p.y - a * p.x) * is, p.z * ir);
  let iden = 1.0 / (r2 * r2 + a2 * p.z * p.z);
  let f = 2.0 * r2 * r * iden;
  let Q = -kt + dot(l, k);
  let gr = vec3f(p.x * r, p.y * r, p.z * (r2 + a2) * ir) * (0.5 / D);    // grad r
  let gf = f * (3.0 * ir - 4.0 * r2 * r * iden) * gr - vec3f(0.0, 0.0, 2.0 * f * a2 * p.z * iden);
  let A = (k.x * p.x + k.y * p.y - 2.0 * r * (k.x * l.x + k.y * l.y)) * is - k.z * p.z * ir * ir;
  let gQ = A * gr + vec3f((r * k.x - a * k.y) * is, (a * k.x + r * k.y) * is, k.z * ir);
  let fQ = f * Q;
  var o: Deriv;
  o.dx = k - fQ * l;
  o.dk = 0.5 * Q * Q * gf + fQ * gQ;
  o.r = r;
  o.drdl = dot(gr, o.dx);
  return o;
}

// One RK4 step of size h; d1 is the derivative already evaluated at (p, k).
fn rk4(p: vec3f, k: vec3f, kt: f32, d1: Deriv, h: f32) -> St {
  let d2 = geodesicRHS(p + 0.5 * h * d1.dx, k + 0.5 * h * d1.dk, kt);
  let d3 = geodesicRHS(p + 0.5 * h * d2.dx, k + 0.5 * h * d2.dk, kt);
  let d4 = geodesicRHS(p + h * d3.dx, k + h * d3.dk, kt);
  return St(p + (h / 6.0) * (d1.dx + 2.0 * (d2.dx + d3.dx) + d4.dx),
            k + (h / 6.0) * (d1.dk + 2.0 * (d2.dk + d3.dk) + d4.dk));
}

// Bardeen's critical curve and the analytic capture test (physics.js: isCaptured).
fn shadowXi(r: f32, a: f32) -> f32 { return -(r * r * r - 3.0 * r * r + a * a * r + a * a) / (a * (r - 1.0)); }
fn shadowEta(r: f32, a: f32) -> f32 {
  return -(r * r * r * (r * r * r - 6.0 * r * r + 9.0 * r - 4.0 * a * a)) / (a * a * (r - 1.0) * (r - 1.0));
}
fn isCaptured(xi: f32, eta: f32) -> bool {
  let a = F.bh.x;
  if (eta < 0.0) { return true; }
  if (abs(a) < 1e-4) { return eta < 27.0 - xi * xi; }
  let xLo = F.bh4.x;
  let xHi = F.bh4.y;
  if ((xi - xLo) * (xi - xHi) > 0.0) { return false; }
  var lo = F.bh3.y;
  var hi = F.bh3.z;
  for (var i = 0; i < 22; i++) {
    let mid = 0.5 * (lo + hi);
    if ((shadowXi(mid, a) - xi) * (xLo - xi) > 0.0) { lo = mid; } else { hi = mid; }
  }
  return eta < shadowEta(0.5 * (lo + hi), a);
}

// g = nu_obs / nu_emit for gas on a prograde Keplerian circular orbit at radius r.
fn diskRedshift(p: vec3f, k: vec3f, kt: f32, r: f32) -> f32 {
  let a = F.bh.x;
  let W = 1.0 / (pow(r, 1.5) + a);
  let gVV = -1.0 + W * W * (r * r + a * a) + (2.0 / r) * (1.0 - a * W) * (1.0 - a * W);
  let ut = inverseSqrt(max(-gVV, 1e-6));
  return 1.0 / (ut * (kt + W * (p.x * k.y - p.y * k.x)));
}
`;

const SHADE_WGSL = /* wgsl */ `
const TAU = 6.2831853;

fn blackbody(T: f32) -> vec3f {
  // LUT over log T in [1000 K, 100000 K], luminance 1 at 6500 K (physics.js: blackbodyLUT)
  let x = clamp(log(max(T, 1.0) / 1000.0) / log(100.0) * 255.0, 0.0, 255.0);
  let i = u32(x);
  let j = min(i + 1u, 255u);
  var c = mix(bbLUT[i].rgb, bbLUT[j].rgb, x - f32(i));
  if (T < 1000.0) { c *= pow(max(T, 1.0) / 1000.0, 10.0); }
  return c;
}

// Novikov-Thorne temperature profile, T / T_peak, sampled at r = rIn + (rOut - rIn) u^2.
fn diskProfile(r: f32) -> f32 {
  let u = sqrt(clamp((r - F.bh.w) / (F.bh2.x - F.bh.w), 0.0, 1.0));
  let x = u * 511.0;
  let i = u32(x);
  let j = min(i + 1u, 511u);
  return mix(tLUT[i], tLUT[j], x - f32(i));
}

// Opacity without small-scale structure. The in-loop early exit uses the lower bound
// (gaps fully open), so it never stops a ray that could still see through.
fn diskAlphaBase(r: f32) -> f32 {
  let rIn = F.bh.w;
  let rOut = F.bh2.x;
  let inner = smoothstep(rIn, rIn + 0.02, r);
  let outer = 1.0 - smoothstep(rOut * 0.92, rOut, r);
  return F.disk.w * inner * outer;
}
fn diskAlphaMin(r: f32) -> f32 {
  return diskAlphaBase(r) * (1.0 - F.disk2.y * smoothstep(F.bh2.x * 0.55, F.bh2.x * 0.9, r));
}

// Disk shading at a hit (x, y, r, g); fp = footprint of the pixel on the disk plane (M),
// mu = cosine of the emission angle from the disk normal in the gas frame.
// Returns premultiplied emission and opacity.
fn shadeDisk(hit: vec4f, fp: f32, mu: f32) -> vec4f {
  let r = hit.z;
  let g = hit.w;
  let a = F.bh.x;
  let phi = atan2(hit.y, hit.x);
  let lr = log(r);
  let W = 1.0 / (pow(r, 1.5) + a);
  let dW = 1.5 * sqrt(r) * W * W;   // |dOmega/dr|
  // restart period: a fixed fraction of the local orbital period, so every ring flows the
  // same fraction of an orbit before it is crossfaded (bounded winding at every radius)
  let period = F.disk2.z * TAU / W;
  let t = F.bh4.z;

  // Gas texture advected with the local Keplerian angular velocity. Differential rotation
  // shears it into spirals; two flow-map layers restarted half a period apart keep the
  // winding bounded, crossfaded with variance-preserving weights. A layer's cycle index
  // changes exactly where its weight is zero, so the radius-dependent period leaves no seam.
  var tex = 0.0;
  var gapN = 0.0;
  var clump = 0.0;
  var wsum2 = 0.0;
  let layers = select(2, 0, F.disk2.x <= 0.0 && F.disk2.y <= 0.0);
  for (var j = 0; j < layers; j++) {
    let ph = t / period + f32(j) * 0.5;
    let cyc = floor(ph);
    let tj = fract(ph) * period;
    let wj = 1.0 - abs(2.0 * fract(ph) - 1.0);
    let ang = phi - W * tj + f32(j) * 1.3;
    let N0 = 12;     // azimuthal periods of the lowest octave around the disk
    let kR = 8.0;    // radial noise frequency (per e-fold in r)
    let q = vec2f(ang / TAU * f32(N0), lr * kR);
    let fpN = (fp / r) * max(kR, f32(N0) / TAU * (1.0 + dW * tj * r));
    let seed = i32(cyc) * 977 + j * 31;
    tex += wj * fbm2p(q, N0, fpN, 9, 0.62, seed);   // fine octaves stop at the pixel footprint
    gapN += wj * fbm2p(q * vec2f(0.5, 0.5), N0 / 2, fpN * 0.5, 3, 0.5, seed + 7);
    clump += wj * fbm2p(q * vec2f(0.5, 1.2), N0 / 2, fpN * 1.2, 3, 0.5, seed + 13);   // ~10:1 along the orbit
    wsum2 += wj * wj;
  }
  let inv = inverseSqrt(max(wsum2, 1e-4));
  tex *= inv;
  gapN *= inv;
  clump *= inv;

  // The texture perturbs the local flux (filaments on top of larger knots); an optically
  // thick surface then emits a blackbody at T ~ F^(1/4), so structure shows as small
  // temperature changes, never as intensity above B(T).
  let turb = F.disk2.x;
  // exp2 of zero-mean noise raises the mean; divide by exp(sigma^2 ln^2 2 / 2), sigma ~ 0.6 turb
  let flux = exp2(turb * (1.6 * tex + 2.2 * clump) - 0.125 * turb * turb);
  let T = F.disk.x * diskProfile(r) * pow(flux, 0.25);
  let Tobs = T * pow(max(g, 1e-3), F.disk.y);
  // electron-scattering limb darkening (Chandrasekhar): I(mu) ~ 1 + 2.06 mu, flux-normalised
  let limb = mix(1.0, 0.4213 * (1.0 + 2.06 * clamp(mu, 0.0, 1.0)), F.extra.x);
  var emis = blackbody(Tobs) * (F.disk.z * limb);

  // gaps only open in the cooler outer disk; the bright inner disk stays optically thick
  let gapMask = smoothstep(-0.05, 0.3, gapN) * smoothstep(F.bh2.x * 0.55, F.bh2.x * 0.9, r);
  let alpha = diskAlphaBase(r) * (1.0 - F.disk2.y * gapMask);

  if (F.disk2.w > 0.5 && F.disk2.w < 1.5) {
    // debug: redshift map (blue = approaching, red = receding, white = g of 1)
    let s = clamp(log2(g) * 2.5, -1.0, 1.0);
    emis = select(mix(vec3f(1.0), vec3f(1.0, 0.15, 0.05), -s), mix(vec3f(1.0), vec3f(0.1, 0.3, 1.0), s), s > 0.0) * 0.6;
  }
  return vec4f(emis * alpha, alpha);
}

// Atmosphere above and below the disk: gas with a Gaussian vertical density profile of scale
// height H = (H/r) r and total vertical optical depth tau_v, in LTE at the local disk
// temperature. Along each straight step segment the column is integrated analytically (erf),
// giving tau = tau_v * column / sqrt(2 pi) / g (the gas-frame path is the coordinate path
// over g). The segment emits B(gT) (1 - e^-tau) and dims everything behind it by e^-tau, so
// a grazing view saturates at the blackbody limit instead of exceeding it.
fn erfApprox(x: f32) -> f32 {
  let x2 = x * x;
  let t = 0.147 * x2;
  return sign(x) * sqrt(1.0 - exp(-x2 * (1.2732395 + t) / (1.0 + t)));
}
fn atmosphere(p0: vec3f, p1: vec3f, kmid: vec3f, kt: f32) -> vec4f {
  let a = F.bh.x;
  let m = 0.5 * (p0 + p1);
  let rc = sqrt(max(dot(m.xy, m.xy) - a * a, 0.0));
  if (rc < F.bh.w || rc > F.bh2.x) { return vec4f(0.0, 0.0, 0.0, 1.0); }
  let H = F.vol.x * rc;
  let dz = p1.z - p0.z;
  let L = length(p1 - p0);
  var w = 0.0;   // integral of exp(-z^2 / 2H^2) along the segment, divided by H
  if (abs(dz) > 1e-3 * H) {
    w = L * 1.2533141 / abs(dz) * abs(erfApprox(p1.z * 0.70710678 / H) - erfApprox(p0.z * 0.70710678 / H));
  } else {
    w = L / H * exp(-0.5 * m.z * m.z / (H * H));
  }
  if (w < 1e-5) { return vec4f(0.0, 0.0, 0.0, 1.0); }
  let g = diskRedshift(vec3f(m.xy, 0.0), kmid, kt, rc);
  let T = F.disk.x * diskProfile(rc);
  let edge = 1.0 - smoothstep(F.bh2.x * 0.8, F.bh2.x, rc);
  let tau = F.vol.y * w * 0.39894228 / max(g, 0.05) * edge;
  let tr = exp(-tau);
  return vec4f(blackbody(T * pow(max(g, 1e-3), F.disk.y)) * (F.disk.z * (1.0 - tr)), tr);
}

// ---- sky --------------------------------------------------------------------------------
// Stars are point sources splatted with a Gaussian as wide as the pixel's footprint on the
// sky, so their flux is conserved under lensing: magnified regions show brighter, stretched
// stars, demagnified ones (near the shadow) a smooth glow of the layer's mean radiance.
fn faceDir(face: i32, uv: vec2f) -> vec3f {
  let s = select(-1.0, 1.0, (face & 1) == 0);
  if (face < 2) { return normalize(vec3f(s, uv.x, uv.y)); }
  if (face < 4) { return normalize(vec3f(uv.x, s, uv.y)); }
  return normalize(vec3f(uv.x, uv.y, s));
}

fn starLayer(dir: vec3f, fp: f32, cells: f32, prob0: f32, flux: f32, seed: i32, gs: f32) -> vec3f {
  let ad = abs(dir);
  var face = 0;
  var uv = vec2f(0.0);
  if (ad.x >= ad.y && ad.x >= ad.z) { face = select(1, 0, dir.x > 0.0); uv = dir.yz / ad.x; }
  else if (ad.y >= ad.z) { face = select(3, 2, dir.y > 0.0); uv = dir.xz / ad.y; }
  else { face = select(5, 4, dir.z > 0.0); uv = dir.xy / ad.z; }
  // star counts rise towards the galactic plane (z = 0 in galactic coordinates)
  let prob = prob0 * (0.25 + 1.2 * exp(-dir.z * dir.z / 0.06));
  let jac = pow(1.0 + dot(uv, uv), -0.75);         // sqrt of the cube-map solid-angle factor
  let cellAng = 2.0 / cells * jac;
  let sig2 = 0.25 * fp * fp + 1e-12;
  let mean = prob * flux * (1.0 / 9.0) / (cellAng * cellAng) * vec3f(1.0, 0.97, 0.94);
  let blend = smoothstep(0.25, 0.7, fp / cellAng);
  if (blend >= 1.0) { return mean; }
  let st = (uv * 0.5 + 0.5) * cells;
  let base = floor(st);
  var sum = vec3f(0.0);
  for (var j = -1; j <= 1; j++) {
    for (var i = -1; i <= 1; i++) {
      let c = base + vec2f(f32(i), f32(j));
      if (any(c < vec2f(0.0)) || any(c >= vec2f(cells))) { continue; }
      let h = hash3i(vec3i(vec2i(c), face * 7919 + seed));
      if (h.x > prob) { continue; }
      let h2 = hash3i(vec3i(vec2i(c), face * 7919 + seed + 101));
      let sdir = faceDir(face, (c + 0.15 + 0.7 * h.yz) / cells * 2.0 - 1.0);
      let d = dir - sdir;
      let w = exp(-dot(d, d) / (2.0 * sig2)) / (TAU * sig2);
      let mag = pow(h2.x, 8.0);                     // steep luminosity function: mostly faint
      let T = (3000.0 + 12000.0 * pow(h2.y, 2.5)) * gs;
      let bbc = blackbody(T);
      let tint = mix(bbc / max(dot(bbc, vec3f(0.2126, 0.7152, 0.0722)), 1e-6), vec3f(1.0), 0.5);
      sum += tint * (flux * mag * w);
    }
  }
  return mix(sum, mean, blend);
}

fn skyRadiance(dir0: vec3f, fp: f32, gs: f32) -> vec3f {
  let dir = mat3x3f(F.skyR0.xyz, F.skyR1.xyz, F.skyR2.xyz) * dir0;
  let px2 = F.sky.z * F.sky.z;                      // star flux in units of a display pixel's solid angle
  var c = vec3f(0.0);
  let boost = gs * gs * gs * gs;                    // bolometric I ~ g^4
  if (F.sky.x > 0.0) {
    var s = starLayer(dir, fp, 30.0, 0.22, 90.0 * px2, 11, gs);
    s += starLayer(dir, fp, 90.0, 0.14, 6.0 * px2, 23, gs);
    s += starLayer(dir, fp, 280.0, 0.1, 1.0 * px2, 37, gs);
    c += s * (F.sky.x * boost);
  }
  if (F.sky.y > 0.0) {
    let lod = log2(max(fp, 1e-6) * f32(textureDimensions(skyCube).x) * 0.5);
    c += textureSampleLevel(skyCube, skySamp, dir, lod).rgb * (F.sky.y * boost);
  }
  return c;
}
`;

export const TRACE = /* wgsl */ `diagnostic(off, derivative_uniformity);
` + FRAME_WGSL + NOISE_WGSL + /* wgsl */ `
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<storage, read> bbLUT: array<vec4f>;
@group(0) @binding(2) var<storage, read> tLUT: array<f32>;
@group(0) @binding(4) var skyCube: texture_cube<f32>;
@group(0) @binding(5) var skySamp: sampler;
` + GEODESIC_WGSL + SHADE_WGSL + FULLSCREEN_VS + /* wgsl */ `

fn footprint2(v: vec2f) -> f32 { return max(length(dpdx(v)), length(dpdy(v))); }

@fragment fn fs(@builtin(position) fc: vec4f) -> @location(0) vec4f {
  let a = F.bh.x;
  let rH = F.bh.y;
  let rCap = F.bh.z;
  let rIn = F.bh.w;
  let rOut = F.bh2.x;
  let rEsc = F.bh2.y;

  // ---- camera ray -----------------------------------------------------------------------
  let pix = fc.xy + F.view.zw;
  let ndc = vec2f(pix.x * F.res.z * 2.0 - 1.0, 1.0 - pix.y * F.res.w * 2.0);
  let px = ndc.x * F.view.x;
  let py = ndc.y * F.view.y;
  let k4 = -F.camU + (F.camF + px * F.camR + py * F.camUp) * inverseSqrt(1.0 + px * px + py * py);
  let kt = k4.x;
  var p = F.camPos.xyz;
  var k = k4.yzw;

  // conserved xi, eta of the ray -> analytic capture flag
  var doomed = false;
  if (F.camPos.w > 0.5) {
    let rc = geodesicRHS(p, k, kt).r;
    let cosT = p.z / rc;
    let sin2 = max(1.0 - cosT * cosT, 1e-8);
    let kphi = -p.y * k.x + p.x * k.y;
    let kth = (k.x * p.x + k.y * p.y) * cosT * inverseSqrt(sin2) - k.z * rc * sqrt(sin2);
    let xi = -kphi / kt;
    let eta = (kth * kth + cosT * cosT * (kphi * kphi / sin2 - a * a * kt * kt)) / (kt * kt);
    doomed = isCaptured(xi, eta);
  }

  // ---- integrate --------------------------------------------------------------------------
  var nh = 0u;
  var h0 = vec4f(0.0);
  var h1 = vec4f(0.0);
  var h2 = vec4f(0.0);
  var mu = vec3f(1.0);      // emission-angle cosines of the three hits
  var fate = 0u;            // 0 captured / lost, 1 escaped, 2 hidden behind the disk
  // atmosphere emission in front of the first disk crossing, and after each crossing
  var v0 = vec3f(0.0);
  var v1 = vec3f(0.0);
  var v2 = vec3f(0.0);
  var v3 = vec3f(0.0);
  var ta = 1.0;             // atmosphere transmittance so far, and at each disk crossing
  var ta3 = vec3f(1.0);
  let useAtm = F.vol.y > 0.0 && F.disk2.w < 0.5;
  var dir = vec3f(0.0, 0.0, 1.0);
  var trans = 1.0;
  var steps = 0.0;
  let maxSteps = i32(F.bh3.x);
  for (var i = 0; i < maxSteps; i++) {
    let d = geodesicRHS(p, k, kt);
    let radial = dot(p, d.dx);
    if ((d.r < rCap && radial < 0.0) || dot(k, k) > F.bh3.w || (doomed && radial < 0.0 && d.r < rIn)) { break; }
    if (d.r > rEsc && radial > 0.0) {
      fate = 1u;
      // remaining weak-field bending beyond rEsc: (2M / R) tan(psi / 2) towards the hole
      let R = length(p);
      let rh = p / R;
      var dd = normalize(d.dx);
      let c = dot(rh, dd);
      let perp = rh - c * dd;
      let pl = length(perp);
      if (pl > 1e-6) {
        let bend = (2.0 / R) * pl / max(1.0 + c, 1e-4);
        dd = normalize(dd * cos(bend) - (perp / pl) * sin(bend));
      }
      dir = dd;
      break;
    }
    let kStep = select(F.bh2.z, 0.3, d.r > F.sky.w);   // coarser far outside the disk
    var h = kStep * d.r / max(length(d.dx), 1e-9);
    if (d.drdl < 0.0) { h = min(h, F.bh2.w * (d.r - rH) / -d.drdl); }
    let n = rk4(p, k, kt, d, h);
    let nearPlane = useAtm && min(abs(p.z), abs(n.p.z)) < 4.0 * F.vol.x * d.r;
    var segStart = p;
    if (p.z * n.p.z < 0.0) {
      let c = rk4(p, k, kt, d, h * (p.z / (p.z - n.p.z)));
      let rc = sqrt(max(dot(c.p.xy, c.p.xy) - a * a, 0.0));
      if (rc >= rIn && rc <= rOut) {
        if (useAtm) {
          let e = atmosphere(p, c.p, 0.5 * (k + c.k), kt);
          if (nh == 0u) { v0 += ta * e.rgb; } else if (nh == 1u) { v1 += ta * e.rgb; } else { v2 += ta * e.rgb; }
          ta *= e.a;
          segStart = c.p;
        }
        let hit = vec4f(c.p.xy, rc, diskRedshift(c.p, c.k, kt, rc));
        // at the equator e_z = d/dz is a unit vector orthogonal to the gas velocity, and the
        // photon's energy in the gas frame is 1 / g, so cos(angle to the normal) = |k_z| g
        let m = min(abs(c.k.z) * hit.w, 1.0);
        if (nh == 0u) { h0 = hit; mu.x = m; ta3.x = ta; } else if (nh == 1u) { h1 = hit; mu.y = m; ta3.y = ta; } else { h2 = hit; mu.z = m; ta3.z = ta; }
        nh++;
        trans *= 1.0 - diskAlphaMin(rc);
        if (nh == 3u || trans * ta < 0.004) { fate = 2u; break; }
      }
    }
    if (nearPlane) {
      let e = atmosphere(segStart, n.p, 0.5 * (k + n.k), kt);
      if (nh == 0u) { v0 += ta * e.rgb; } else if (nh == 1u) { v1 += ta * e.rgb; } else if (nh == 2u) { v2 += ta * e.rgb; } else { v3 += ta * e.rgb; }
      ta *= e.a;
      if (ta * trans < 0.004) { fate = 2u; break; }
    }
    p = n.p;
    k = n.k;
    steps += 1.0;
  }

  // ---- shade (derivatives in uniform control flow, after the loop) -----------------------
  let fp0 = min(footprint2(h0.xy), 0.5 * h0.z + 0.01);
  let fp1 = min(footprint2(h1.xy), 0.5 * h1.z + 0.01);
  let fp2 = min(footprint2(h2.xy), 0.5 * h2.z + 0.01);
  let renderPix = 2.0 * F.view.y * F.res.w;         // angle of one render pixel
  let fpSky = clamp(max(length(dpdx(dir)), length(dpdy(dir))), renderPix * 0.7, 0.4);

  var col = v0;
  var T = 1.0;
  if (nh > 0u) { let s = shadeDisk(h0, fp0, mu.x); col += T * ta3.x * s.rgb; T *= 1.0 - s.a; col += T * v1; }
  if (nh > 1u) { let s = shadeDisk(h1, fp1, mu.y); col += T * ta3.y * s.rgb; T *= 1.0 - s.a; col += T * v2; }
  if (nh > 2u) { let s = shadeDisk(h2, fp2, mu.z); col += T * ta3.z * s.rgb; T *= 1.0 - s.a; col += T * v3; }
  if (fate == 1u) { col += T * ta * skyRadiance(dir, fpSky, 1.0 / max(kt, 0.05)); }

  if (F.disk2.w > 1.5) {
    // debug: integration steps (blue few -> red max)
    let s = steps / F.bh3.x;
    col = mix(vec3f(0.02, 0.05, 0.3), vec3f(1.0, 0.2, 0.05), sqrt(s)) * (0.3 + 0.7 * s);
  }
  col = max(col, vec3f(0.0));

  // Alpha carries the on-screen motion of the gas in the nearest disk image, in pixels per
  // frame (orbital speed times the frame's time step over the pixel's footprint on the
  // disk). The resolve pass drops history where the texture moves too fast to reuse.
  var motion = 0.0;
  if (nh > 0u) { motion = h0.z / (pow(h0.z, 1.5) + a) * F.extra.y / max(fp0, 1e-4); }
  return vec4f(min(col, vec3f(6e4)), motion);
}
`;
