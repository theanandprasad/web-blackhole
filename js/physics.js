// Kerr black hole physics shared by the renderer (CPU side) and the node tests.
// Units: G = c = M = 1. Spin a in (-1, 1); a > 0 rotates towards +phi (counter-clockwise
// about +z). The disk always orbits towards +phi, so a < 0 means a retrograde disk.
//
// Coordinates are Cartesian Kerr-Schild (Visser 2007, "The Kerr spacetime: a brief
// introduction"):
//   g_mn = eta_mn + f l_m l_n,   eta = diag(-1, 1, 1, 1)
//   f    = 2 r^3 / (r^4 + a^2 z^2)
//   l_m  = (1, (r x + a y)/(r^2 + a^2), (r y - a x)/(r^2 + a^2), z / r)
// with r the Boyer-Lindquist radius, r^4 - (x^2 + y^2 + z^2 - a^2) r^2 - a^2 z^2 = 0.
// They have no coordinate singularity on the horizon or on the spin axis, which keeps a
// single integrator valid everywhere a camera can be.
//
// Light rays are traced backwards from the camera with the Hamiltonian form of the
// geodesic equation, H = 1/2 g^mn k_m k_n = 1/2 (|k|^2 - k_t^2 - f Q^2), Q = l^m k_m,
// which needs the inverse metric only: g^mn = eta^mn - f l^m l^n, l^m = (-1, lx, ly, lz).
// k_t is conserved (stationary metric), so the state is position + spatial covector.

export const horizonRadius = (a) => 1 + Math.sqrt(Math.max(0, 1 - a * a));

// Innermost stable circular orbit for a disk orbiting towards +phi (Bardeen, Press &
// Teukolsky 1972). Signed a: a > 0 prograde, a < 0 retrograde.
export function iscoRadius(a) {
  const z1 = 1 + Math.cbrt(1 - a * a) * (Math.cbrt(1 + a) + Math.cbrt(1 - a));
  const z2 = Math.sqrt(3 * a * a + z1 * z1);
  return 3 + z2 - Math.sign(a) * Math.sqrt((3 - z1) * (3 + z1 + 2 * z2));
}

// Circular equatorial photon orbits: [co-rotating with +phi, counter-rotating].
export function photonOrbitRadii(a) {
  return [2 * (1 + Math.cos((2 / 3) * Math.acos(-a))), 2 * (1 + Math.cos((2 / 3) * Math.acos(a)))];
}

// Kerr-Schild scalars at a point. Returns r, f, l (spatial part) and the gradient of r.
export function ksPoint(x, y, z, a) {
  const a2 = a * a;
  const w = x * x + y * y + z * z - a2;
  const D = Math.sqrt(0.25 * w * w + a2 * z * z);
  const r2 = 0.5 * w + D;
  const r = Math.sqrt(r2);
  const s = r2 + a2;
  return {
    r, D, s,
    f: (2 * r2 * r) / (r2 * r2 + a2 * z * z),
    lx: (r * x + a * y) / s, ly: (r * y - a * x) / s, lz: z / r,
  };
}

// Covariant metric g_mn at (x, y, z) as a 16-element row-major array (t, x, y, z).
export function metric(x, y, z, a) {
  const p = ksPoint(x, y, z, a);
  const l = [1, p.lx, p.ly, p.lz];
  const g = new Float64Array(16);
  for (let m = 0; m < 4; m++) {
    for (let n = 0; n < 4; n++) g[m * 4 + n] = (m === n ? (m === 0 ? -1 : 1) : 0) + p.f * l[m] * l[n];
  }
  return g;
}

// Right-hand side of Hamilton's equations. s = [x, y, z, kx, ky, kz]; writes the 6 rates
// to out, plus dr/dlambda in out[6] when out has room for it.
// Kept line-for-line equivalent to geodesicRHS in js/wgsl/trace.js.
export function rhs(s, kt, a, out) {
  const x = s[0], y = s[1], z = s[2], kx = s[3], ky = s[4], kz = s[5];
  const a2 = a * a;
  const w = x * x + y * y + z * z - a2;
  const D = Math.max(Math.sqrt(0.25 * w * w + a2 * z * z), 1e-12);
  const r2 = 0.5 * w + D;
  const r = Math.sqrt(r2);
  const sR = r2 + a2;
  const lx = (r * x + a * y) / sR, ly = (r * y - a * x) / sR, lz = z / r;
  const den = r2 * r2 + a2 * z * z;
  const f = (2 * r2 * r) / den;
  const Q = -kt + lx * kx + ly * ky + lz * kz;
  // gradient of r (from implicit differentiation of the r(x, y, z) quartic)
  const gx = (x * r) / (2 * D), gy = (y * r) / (2 * D), gz = (z * sR) / (2 * r * D);
  // gradient of f
  const cf = f * (3 / r - (4 * r2 * r) / den);
  const fx = cf * gx, fy = cf * gy, fz = cf * gz - (f * 2 * a2 * z) / den;
  // gradient of Q at fixed k
  const A = (kx * x + ky * y - 2 * r * (kx * lx + ky * ly)) / sR - (kz * z) / r2;
  const Qx = A * gx + (r * kx - a * ky) / sR;
  const Qy = A * gy + (a * kx + r * ky) / sR;
  const Qz = A * gz + kz / r;
  const fQ = f * Q;
  out[0] = kx - fQ * lx;
  out[1] = ky - fQ * ly;
  out[2] = kz - fQ * lz;
  const hq = 0.5 * Q * Q;
  out[3] = hq * fx + fQ * Qx;
  out[4] = hq * fy + fQ * Qy;
  out[5] = hq * fz + fQ * Qz;
  out[6] = gx * out[0] + gy * out[1] + gz * out[2]; // dr / dlambda
  return r;
}

// Hamiltonian (should stay at 0 for a light ray).
export function hamiltonian(s, kt, a) {
  const p = ksPoint(s[0], s[1], s[2], a);
  const Q = -kt + p.lx * s[3] + p.ly * s[4] + p.lz * s[5];
  return 0.5 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5] - kt * kt - p.f * Q * Q);
}

// Conserved quantities of a ray from its state: E, Lz and Carter's Q. For a backward
// ray (k = -p) the ratios xi = Lz / E and eta = Q / E^2 are those of the real photon.
export function conserved(s, kt, a) {
  const [x, y, z, kx, ky, kz] = s;
  const r = ksPoint(x, y, z, a).r;
  const cosT = z / r;
  const sin2 = Math.max(1 - cosT * cosT, 1e-12);
  const sinT = Math.sqrt(sin2);
  const kphi = -y * kx + x * ky;
  const kth = (kx * x + ky * y) * (cosT / sinT) - kz * r * sinT;
  const E = kt, Lz = -kphi;
  const Qc = kth * kth + cosT * cosT * ((kphi * kphi) / sin2 - a * a * kt * kt);
  return { E, Lz, Q: Qc, xi: Lz / E, eta: Qc / (E * E) };
}

// Critical curve of a Kerr black hole (Bardeen 1973), parametrised by photon orbit radius.
export const shadowXi = (r, a) => -(r * r * r - 3 * r * r + a * a * r + a * a) / (a * (r - 1));
export const shadowEta = (r, a) => -(r * r * r * (r * r * r - 6 * r * r + 9 * r - 4 * a * a)) / (a * a * (r - 1) * (r - 1));

// Analytic capture test from the conserved (xi, eta) of a ray. Photons with eta < 0
// (vortical orbits) have no radial turning point outside the horizon; otherwise the ray is
// captured when it lies inside the critical curve, eta < eta_c(xi). Used to stop rays early
// once they are falling inside the disk's inner edge and can no longer hit anything.
export function criticalEta(xi, a) {
  if (Math.abs(a) < 1e-4) return 27 - xi * xi;
  const [r1, r2] = photonOrbitRadii(a);
  let lo = Math.min(r1, r2), hi = Math.max(r1, r2);
  const xLo = shadowXi(lo, a), xHi = shadowXi(hi, a);
  if ((xi - xLo) * (xi - xHi) > 0) return -Infinity; // outside the curve's xi range
  for (let i = 0; i < 24; i++) {
    const mid = 0.5 * (lo + hi);
    if ((shadowXi(mid, a) - xi) * (xLo - xi) > 0) lo = mid; else hi = mid;
  }
  return shadowEta(0.5 * (lo + hi), a);
}
export const isCaptured = (xi, eta, a) => eta < 0 || eta < criticalEta(xi, a);

// ---------------------------------------------------------------------------------------
// Observer frame. Builds the covectors the shader needs so that a pixel's backward ray is
//   k_m = -u_m + (F_m + px R_m + py U_m) / sqrt(1 + px^2 + py^2)
// which is a null covector with k.u = 1, i.e. unit energy in the observer's frame.
// Observers: 'static' hovers at fixed (r, theta, phi) (only outside the ergosphere);
// otherwise the Kerr-Schild normal observer is used, which exists everywhere.
// fwd, up are Euclidean 3-vectors in Kerr-Schild Cartesian coordinates.
// ---------------------------------------------------------------------------------------
export function observerFrame(pos, fwd, up, a, mode = 'static') {
  const [x, y, z] = pos;
  const g = metric(x, y, z, a);
  const p = ksPoint(x, y, z, a);
  const dot = (A, B) => {
    let s = 0;
    for (let m = 0; m < 4; m++) for (let n = 0; n < 4; n++) s += g[m * 4 + n] * A[m] * B[n];
    return s;
  };
  const lower = (A) => [0, 1, 2, 3].map((m) => g[m * 4] * A[0] + g[m * 4 + 1] * A[1] + g[m * 4 + 2] * A[2] + g[m * 4 + 3] * A[3]);
  let u;
  let used = mode;
  if (mode === 'static' && g[0] < -1e-4) {
    const n = Math.sqrt(-g[0]);
    u = [1 / n, 0, 0, 0];
  } else {
    // normal observer: u_m = (-alpha, 0, 0, 0); u^m = g^mn u_n with g^tt = -1 - f, g^ti = f l_i
    const alpha = 1 / Math.sqrt(1 + p.f);
    u = [alpha * (1 + p.f), -alpha * p.f * p.lx, -alpha * p.f * p.ly, -alpha * p.f * p.lz];
    used = 'normal';
  }
  const project = (v) => {
    const V = [0, v[0], v[1], v[2]];
    const c = dot(V, u);
    return V.map((q, i) => q + c * u[i]);
  };
  const normalize = (V) => {
    const n = Math.sqrt(dot(V, V));
    return V.map((q) => q / n);
  };
  const sub = (V, W) => {
    const c = dot(V, W);
    return V.map((q, i) => q - c * W[i]);
  };
  const right0 = [fwd[1] * up[2] - fwd[2] * up[1], fwd[2] * up[0] - fwd[0] * up[2], fwd[0] * up[1] - fwd[1] * up[0]];
  const F = normalize(project(fwd));
  const U = normalize(sub(project(up), F));
  const R = normalize(sub(sub(project(right0), F), U));
  return { u: lower(u), F: lower(F), R: lower(R), U: lower(U), uUp: u, observer: used };
}

// Backward ray for screen offsets (px, py) = tangent-plane coordinates.
export function cameraRay(frame, px, py) {
  const n = 1 / Math.sqrt(1 + px * px + py * py);
  const k = [0, 1, 2, 3].map((m) => -frame.u[m] + (frame.F[m] + px * frame.R[m] + py * frame.U[m]) * n);
  return { kt: k[0], k: [k[1], k[2], k[3]] };
}

// ---------------------------------------------------------------------------------------
// Ray integration: RK4 with a step proportional to distance from the horizon. The same
// stepping rule and termination tests are used by the shader.
// ---------------------------------------------------------------------------------------
// Step rule: each step moves the ray about k * r in space, and moves it inwards by at most
// kr * (r - rH). The second limit matters for rays falling in: traced backwards they head
// for the past horizon, where (in these ingoing coordinates) the covector diverges like
// 1 / (r - rH), so inward progress has to stay a fraction of the remaining distance.
// Rays skimming the horizon of a fast spinner are not slowed down by it. A diverging
// covector, or reaching just above the horizon while moving in, counts as capture.
export const STEP = { k: 0.15, kFar: 0.3, kr: 0.12, maxSteps: 600, kMax2: 1e6, capture: 0.9 };

// Beyond rFar (well outside the disk) steps grow to kFar * r, and rays are followed out to
// 8 rFar before the remaining weak-field bending is added analytically.
export const farRadius = (rOut) => Math.max(1.3 * rOut, 20);
export const escapeRadius = (rOut, rCam) => Math.max(8 * farRadius(rOut), 1.1 * rCam);

// rH + capture * (r_photon_prograde - rH): always below every photon turning point.
export const captureRadius = (a) => {
  const rH = horizonRadius(a);
  return rH + STEP.capture * (photonOrbitRadii(Math.abs(a))[0] - rH);
};

export function stepSize(r, rH, dxLen, drdl, k = STEP.k, kr = STEP.kr) {
  const h = (k * r) / Math.max(dxLen, 1e-9);
  return drdl < 0 ? Math.min(h, (kr * (r - rH)) / -drdl) : h;
}

export function rk4(s, kt, a, h, tmp) {
  const { k1, k2, k3, k4, y } = tmp;
  rhs(s, kt, a, k1);
  for (let i = 0; i < 6; i++) y[i] = s[i] + 0.5 * h * k1[i];
  rhs(y, kt, a, k2);
  for (let i = 0; i < 6; i++) y[i] = s[i] + 0.5 * h * k2[i];
  rhs(y, kt, a, k3);
  for (let i = 0; i < 6; i++) y[i] = s[i] + h * k3[i];
  rhs(y, kt, a, k4);
  const out = new Float64Array(6);
  for (let i = 0; i < 6; i++) out[i] = s[i] + (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
  return out;
}

export function makeTmp() {
  return { k1: new Float64Array(7), k2: new Float64Array(7), k3: new Float64Array(7), k4: new Float64Array(7), y: new Float64Array(7) };
}

// Traces one backward ray. Returns { fate: 'captured' | 'escaped' | 'lost', steps,
// state, diskHits: [{ x, y, r, g }] } where diskHits are crossings of the equatorial
// plane between rIn and rOut, in order along the backward ray.
export function traceRay(pos, kt, k, a, opts = {}) {
  const rH = horizonRadius(a);
  const rEsc = opts.rEscape ?? 60;
  const maxSteps = opts.maxSteps ?? STEP.maxSteps;
  const stepK = opts.stepK ?? STEP.k;
  const kFar = opts.kFar ?? STEP.kFar, rFar = opts.rFar ?? Infinity;
  const rCap = captureRadius(a);
  const rIn = opts.rIn ?? iscoRadius(a), rOut = opts.rOut ?? 0;
  const c0 = conserved(Float64Array.from([pos[0], pos[1], pos[2], k[0], k[1], k[2]]), kt, a);
  const doomed = opts.analyticCapture !== false && isCaptured(c0.xi, c0.eta, a);
  const tmp = makeTmp();
  let s = Float64Array.from([pos[0], pos[1], pos[2], k[0], k[1], k[2]]);
  const d = new Float64Array(7);
  const hits = [];
  for (let i = 0; i < maxSteps; i++) {
    const r = rhs(s, kt, a, d);
    const radial = s[0] * d[0] + s[1] * d[1] + s[2] * d[2];
    const kk = s[3] * s[3] + s[4] * s[4] + s[5] * s[5];
    if ((r < rCap && radial < 0) || kk > STEP.kMax2 || (doomed && radial < 0 && r < rIn)) return { fate: 'captured', steps: i, state: s, diskHits: hits };
    if (r > rEsc && radial > 0) return { fate: 'escaped', steps: i, state: s, dir: [d[0], d[1], d[2]], diskHits: hits };
    const h = stepSize(r, rH, Math.hypot(d[0], d[1], d[2]), d[6], r > rFar ? kFar : stepK);
    const n = rk4(s, kt, a, h, tmp);
    if (rOut > 0 && s[2] * n[2] <= 0 && s[2] !== n[2]) {
      const c = rk4(s, kt, a, h * (s[2] / (s[2] - n[2])), tmp);
      const rc = Math.sqrt(Math.max(c[0] * c[0] + c[1] * c[1] - a * a, 0));
      if (rc >= rIn && rc <= rOut) hits.push({ x: c[0], y: c[1], r: rc, g: diskRedshift(c, kt, a, rc) });
    }
    s = n;
  }
  return { fate: 'lost', steps: maxSteps, state: s, diskHits: hits };
}

// ---------------------------------------------------------------------------------------
// Thin disk (Novikov & Thorne 1973; Page & Thorne 1974)
// ---------------------------------------------------------------------------------------
export const keplerOmega = (r, a) => 1 / (Math.pow(r, 1.5) + a);

// Contravariant time component of a circular equatorial orbit, in KS Cartesian form
// u = u^t (1, -Omega y, Omega x, 0); uses x^2 + y^2 = r^2 + a^2 on the equator.
export function diskUt(r, a) {
  const W = keplerOmega(r, a);
  const gVV = -1 + W * W * (r * r + a * a) + (2 / r) * (1 - a * W) * (1 - a * W);
  return gVV < 0 ? 1 / Math.sqrt(-gVV) : NaN;
}

// Redshift factor g = nu_observed / nu_emitted for a backward ray crossing the disk.
export function diskRedshift(s, kt, a, r) {
  const W = keplerOmega(r, a);
  const ut = diskUt(r, a);
  return 1 / (ut * (kt + W * (s[0] * s[4] - s[1] * s[3])));
}

// Novikov-Thorne flux F(r) for r >= r_isco, by direct integration of the Page-Thorne
// formula F = -Omega' / (r (E - Omega L)^2) * Integral[(E - Omega L) L' dr].
// Returns { r: Float64Array, T: Float64Array } with T = F^(1/4) normalised to max 1.
export function diskTemperatureProfile(a, rOut, n = 2048) {
  const rIn = iscoRadius(a);
  const sq = (r) => Math.sqrt(r);
  const den = (r) => Math.pow(r, 0.75) * Math.sqrt(r * sq(r) - 3 * sq(r) + 2 * a);
  const E = (r) => (r * sq(r) - 2 * sq(r) + a) / den(r);
  const L = (r) => (r * r - 2 * a * sq(r) + a * a) / den(r);
  const W = (r) => keplerOmega(r, a);
  const d = (fn, r) => { const h = 1e-5 * r; return (fn(r + h) - fn(r - h)) / (2 * h); };
  const rs = new Float64Array(n), T = new Float64Array(n);
  // sample densely near the inner edge: r = rIn + (rOut - rIn) * u^2
  let integral = 0, prevR = rIn, prevI = 0;
  let tMax = 0;
  for (let i = 0; i < n; i++) {
    const u = i / (n - 1);
    const r = rIn + (rOut - rIn) * u * u;
    const integrand = (E(r) - W(r) * L(r)) * d(L, Math.max(r, rIn * (1 + 1e-6)));
    if (i > 0) integral += 0.5 * (prevI + integrand) * (r - prevR);
    prevR = r; prevI = integrand;
    const EmWL = E(r) - W(r) * L(r);
    const F = i === 0 ? 0 : Math.max(0, (-d(W, r) / (r * EmWL * EmWL)) * integral);
    rs[i] = r; T[i] = Math.pow(F, 0.25);
    tMax = Math.max(tMax, T[i]);
  }
  for (let i = 0; i < n; i++) T[i] /= tMax;
  return { r: rs, T, rIn, tMax };
}

// Peak F^(1/4) of a non-spinning hole's disk; the renderer's temperature scale refers to it,
// so at a fixed accretion rate faster spins get hotter inner disks while the outer disk,
// where the flux approaches the Newtonian 3 M Mdot / (8 pi r^3), stays the same.
let tMaxSchwarzschild = 0;
export function schwarzschildPeak() {
  if (!tMaxSchwarzschild) tMaxSchwarzschild = diskTemperatureProfile(0, 60, 4096).tMax;
  return tMaxSchwarzschild;
}

// Resamples the profile on u in [0, 1] with r = rIn + (rOut - rIn) u^2 (the shader uses
// the same mapping to read the lookup table). Values are T / T_peak(a = 0) at the same
// accretion rate.
export function diskTemperatureLUT(a, rOut, size = 512) {
  const p = diskTemperatureProfile(a, rOut, 4096);
  const scale = p.tMax / schwarzschildPeak();
  const out = new Float32Array(size);
  let j = 0;
  for (let i = 0; i < size; i++) {
    const u = i / (size - 1);
    const r = p.rIn + (rOut - p.rIn) * u * u;
    while (j < p.r.length - 2 && p.r[j + 1] < r) j++;
    const t = Math.min(1, Math.max(0, (r - p.r[j]) / (p.r[j + 1] - p.r[j])));
    out[i] = (p.T[j] * (1 - t) + p.T[j + 1] * t) * scale;
  }
  return { lut: out, rIn: p.rIn, peak: scale };
}

// ---------------------------------------------------------------------------------------
// Blackbody colour: Planck spectrum integrated against the CIE 1931 2-degree observer
// (multi-lobe Gaussian fit of Wyman, Sloan & Shirley 2013), converted to linear Rec.709.
// ---------------------------------------------------------------------------------------
const lobe = (l, mu, s1, s2) => { const t = (l - mu) / (l < mu ? s1 : s2); return Math.exp(-0.5 * t * t); };
export function cie1931(l) {
  return [
    1.056 * lobe(l, 599.8, 37.9, 31.0) + 0.362 * lobe(l, 442.0, 16.0, 26.7) - 0.065 * lobe(l, 501.1, 20.4, 26.2),
    0.821 * lobe(l, 568.8, 46.9, 40.5) + 0.286 * lobe(l, 530.9, 16.3, 31.1),
    1.217 * lobe(l, 437.0, 11.8, 36.0) + 0.681 * lobe(l, 459.0, 26.0, 13.8),
  ];
}
export function blackbodyRGB(T) {
  const c2 = 1.4387769e7; // hc/k in nm K
  let X = 0, Y = 0, Z = 0;
  for (let l = 380; l <= 780; l += 2) {
    const B = 1 / (Math.pow(l / 1000, 5) * (Math.exp(c2 / (l * T)) - 1)); // arbitrary units
    const [xb, yb, zb] = cie1931(l);
    X += B * xb; Y += B * yb; Z += B * zb;
  }
  return [
    3.2406 * X - 1.5372 * Y - 0.4986 * Z,
    -0.9689 * X + 1.8758 * Y + 0.0415 * Z,
    0.0557 * X - 0.2040 * Y + 1.0570 * Z,
    Y,
  ];
}

// LUT over log temperature, normalised so a 6500 K blackbody has luminance 1.
export const BB = { tMin: 1000, tMax: 100000, size: 256 };
export function blackbodyLUT() {
  const ref = blackbodyRGB(6500)[3];
  const out = new Float32Array(BB.size * 4);
  for (let i = 0; i < BB.size; i++) {
    const T = BB.tMin * Math.pow(BB.tMax / BB.tMin, i / (BB.size - 1));
    const [r, g, b, Y] = blackbodyRGB(T);
    out.set([Math.max(r, 0) / ref, Math.max(g, 0) / ref, Math.max(b, 0) / ref, Y / ref], i * 4);
  }
  return out;
}
