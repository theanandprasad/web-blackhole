// Physics checks for the Kerr ray tracer.  node test/physics.test.mjs
// Compares the integrator against closed-form results: the Schwarzschild critical impact
// parameter, Bardeen's Kerr shadow curve, circular-orbit redshifts, ISCO radii, and the
// Novikov-Thorne temperature peak.
import * as P from '../js/physics.js';

let failures = 0;
function check(name, ok, detail) {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`);
  if (!ok) failures++;
}
const close = (x, y, tol) => Math.abs(x - y) <= tol * Math.max(1, Math.abs(y));

// --- 1. analytic Hamilton equations vs finite differences of H ------------------------
{
  let worst = 0;
  const d = new Float64Array(6);
  for (let n = 0; n < 200; n++) {
    const a = Math.random() * 1.98 - 0.99;
    const s = Float64Array.from([
      (Math.random() - 0.5) * 20, (Math.random() - 0.5) * 20, (Math.random() - 0.5) * 20,
      Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5]);
    if (P.ksPoint(s[0], s[1], s[2], a).r < 2.5) continue;
    const kt = Math.random() + 0.2;
    P.rhs(s, kt, a, d);
    for (let i = 0; i < 6; i++) {
      const h = 1e-6;
      const sp = Float64Array.from(s), sm = Float64Array.from(s);
      sp[i] += h; sm[i] -= h;
      const dH = (P.hamiltonian(sp, kt, a) - P.hamiltonian(sm, kt, a)) / (2 * h);
      // dx/dl = dH/dk, dk/dl = -dH/dx
      const expect = i < 3 ? -dH : dH;
      const got = i < 3 ? d[i + 3] : d[i - 3];
      worst = Math.max(worst, Math.abs(got - expect) / Math.max(1e-3, Math.abs(expect)));
    }
  }
  check('Hamilton equations match finite differences of H', worst < 1e-5, `worst rel err ${worst.toExponential(2)}`);
}

// --- 2. camera rays are null with unit observed energy --------------------------------
{
  let worstH = 0, worstE = 0;
  for (const a of [0, 0.5, 0.9, -0.7, 0.998]) {
    for (const pos of [[30, 0, 0], [0, 12, 5], [-6, -4, 3], [20, 20, -15]]) {
      const len = Math.hypot(...pos);
      const fwd = pos.map((v) => -v / len);
      const fr = P.observerFrame(pos, fwd, [0, 0, 1], a, 'static');
      for (const [px, py] of [[0, 0], [0.3, -0.2], [-0.8, 0.5]]) {
        const ray = P.cameraRay(fr, px, py);
        const s = Float64Array.from([...pos, ...ray.k]);
        worstH = Math.max(worstH, Math.abs(P.hamiltonian(s, ray.kt, a)));
        // k.u in the observer frame must be 1 (k_m u^m)
        const ku = ray.kt * fr.uUp[0] + ray.k[0] * fr.uUp[1] + ray.k[1] * fr.uUp[2] + ray.k[2] * fr.uUp[3];
        worstE = Math.max(worstE, Math.abs(ku - 1));
      }
    }
  }
  check('camera rays are null', worstH < 1e-12, `max |H| ${worstH.toExponential(2)}`);
  check('camera rays have unit energy in observer frame', worstE < 1e-12, `max |k.u - 1| ${worstE.toExponential(2)}`);
}

// --- 3. H stays ~0 along a strongly bent ray ------------------------------------------
{
  const a = 0.95;
  const pos = [25, 0, 3];
  const fr = P.observerFrame(pos, [-1, 0, -0.12], [0, 0, 1], a);
  const ray = P.cameraRay(fr, 0.28, 0.02);
  const res = P.traceRay(pos, ray.kt, ray.k, a, { maxSteps: 4000, rEscape: 60 });
  const bend = Math.acos(Math.min(1, -(res.dir[0] * -1 + res.dir[2] * -0.12) / Math.hypot(...res.dir) / Math.hypot(1, 0.12)));
  const H = P.hamiltonian(res.state, ray.kt, a);
  const c0 = P.conserved(Float64Array.from([...pos, ...ray.k]), ray.kt, a);
  const c1 = P.conserved(res.state, ray.kt, a);
  check('null constraint conserved along a strongly bent ray', res.fate === 'escaped' && Math.abs(H) < 1e-4,
    `${res.fate} after ${res.steps} steps, deflected ${(bend * 180 / Math.PI).toFixed(0)}°, |H| ${Math.abs(H).toExponential(2)}`);
  check('Lz and Carter Q conserved', close(c0.Lz, c1.Lz, 1e-4) && close(c0.Q, c1.Q, 1e-3),
    `Lz ${c0.Lz.toFixed(5)}→${c1.Lz.toFixed(5)}, Q ${c0.Q.toFixed(4)}→${c1.Q.toFixed(4)}`);
}

// --- 4. shadow edge vs analytic critical curve ----------------------------------------
// Bisect along a screen line for the capture threshold, then compare the conserved
// (xi, eta) of the threshold ray with the analytic photon-orbit values.
function threshold(a, pos, fwd, py, side, maxSteps) {
  const fr = P.observerFrame(pos, fwd, [0, 0, 1], a);
  const rObs = Math.hypot(...pos);
  const captured = (px) => {
    const ray = P.cameraRay(fr, px, py);
    return P.traceRay(pos, ray.kt, ray.k, a, { maxSteps, rEscape: rObs * 1.1 }).fate !== 'escaped';
  };
  let lo = 0, hi = side * 12 / rObs; // lo captured, hi escapes
  for (let i = 0; i < 40; i++) {
    const mid = 0.5 * (lo + hi);
    if (captured(mid)) lo = mid; else hi = mid;
  }
  const ray = P.cameraRay(fr, lo, py);
  return P.conserved(Float64Array.from([...pos, ...ray.k]), ray.kt, a);
}

{
  const pos = [1000, 0, 0], fwd = [-1, 0, 0];
  const c = threshold(0, pos, fwd, 0, 1, 20000);
  const b = Math.sqrt(c.xi * c.xi + c.eta);
  check('Schwarzschild critical impact parameter = 3√3', close(b, 3 * Math.sqrt(3), 1e-3), `b = ${b.toFixed(5)} (expect 5.19615)`);
  const cFast = threshold(0, pos, fwd, 0, 1, P.STEP.maxSteps);
  const bFast = Math.sqrt(cFast.xi * cFast.xi + cFast.eta);
  check('…also with the shader step budget', close(bFast, 3 * Math.sqrt(3), 5e-3), `b = ${bFast.toFixed(5)}`);
}

for (const a of [0.9, 0.998, -0.6]) {
  const [rPro, rRet] = P.photonOrbitRadii(a);
  const pos = [1000, 0, 0], fwd = [-1, 0, 0];
  // camera on +x looking at -x with up +z: screen right is +y... the disk's +phi motion
  // at the near side (x>0) is towards +y, so the prograde edge is on the right.
  const cR = threshold(a, pos, fwd, 0, 1, 20000);
  const cL = threshold(a, pos, fwd, 0, -1, 20000);
  const xiPro = P.shadowXi(rPro, a), xiRet = P.shadowXi(rRet, a);
  const got = [cR.xi, cL.xi].sort((p, q) => p - q), want = [xiPro, xiRet].sort((p, q) => p - q);
  check(`Kerr a=${a} equatorial shadow edges`, close(got[0], want[0], 2e-3) && close(got[1], want[1], 2e-3),
    `xi ${got.map((v) => v.toFixed(4)).join(', ')} expect ${want.map((v) => v.toFixed(4)).join(', ')}`);

  // off-equator, inclined observer
  for (const inc of [90, 55]) {
    const th = (inc * Math.PI) / 180;
    const p2 = [1000 * Math.sin(th), 0, 1000 * Math.cos(th)];
    const f2 = p2.map((v) => -v / 1000);
    let worst = 0, detail = '';
    for (const [py, side] of [[0.003, 1], [0.003, -1], [-0.002, 1], [0.0045, -1]]) {
      const c = threshold(a, p2, f2, py, side, 20000);
      // find photon orbit radius with xi(r) = c.xi
      let lo = Math.min(rPro, rRet), hi = Math.max(rPro, rRet);
      const sLo = Math.sign(P.shadowXi(lo, a) - c.xi);
      for (let i = 0; i < 80; i++) {
        const mid = 0.5 * (lo + hi);
        if (Math.sign(P.shadowXi(mid, a) - c.xi) === sLo) lo = mid; else hi = mid;
      }
      const etaWant = P.shadowEta(0.5 * (lo + hi), a);
      const err = Math.abs(c.eta - etaWant) / Math.max(1, Math.abs(etaWant));
      if (err > worst) { worst = err; detail = `eta ${c.eta.toFixed(3)} vs ${etaWant.toFixed(3)} at xi ${c.xi.toFixed(3)}`; }
    }
    check(`Kerr a=${a} incl ${inc}° shadow matches Bardeen curve`, worst < 3e-3, detail);
  }
}

// --- 4b. analytic capture flag agrees with integration --------------------------------
{
  let agree = 0, total = 0, worst = '';
  for (const a of [0, 0.5, 0.9, 0.998, -0.8]) {
    for (let n = 0; n < 60; n++) {
      const th = Math.acos(Math.random() * 2 - 1), ph = Math.random() * 2 * Math.PI, d = 15 + Math.random() * 30;
      const pos = [d * Math.sin(th) * Math.cos(ph), d * Math.sin(th) * Math.sin(ph), d * Math.cos(th)];
      const fwd = pos.map((v) => -v / d);
      const up = Math.abs(fwd[2]) > 0.95 ? [1, 0, 0] : [0, 0, 1];
      const fr = P.observerFrame(pos, fwd, up, a);
      const ray = P.cameraRay(fr, (Math.random() - 0.5) * 0.8, (Math.random() - 0.5) * 0.8);
      const c = P.conserved(Float64Array.from([...pos, ...ray.k]), ray.kt, a);
      const crit = P.criticalEta(c.xi, a);
      if (Math.abs(c.eta - crit) < 0.05 * Math.max(1, Math.abs(crit))) continue; // too close to the boundary to judge
      const res = P.traceRay(pos, ray.kt, ray.k, a, { maxSteps: 20000, rEscape: d * 1.2, analyticCapture: false });
      total++;
      if ((res.fate !== 'escaped') === P.isCaptured(c.xi, c.eta, a)) agree++;
      else worst = `a=${a} xi=${c.xi.toFixed(3)} eta=${c.eta.toFixed(3)} crit=${crit.toFixed(3)} fate=${res.fate}`;
    }
  }
  check('analytic capture test agrees with traced fate', agree === total, `${agree}/${total} ${worst}`);
}

// --- 5. disk redshift ------------------------------------------------------------------
{
  // zero-angular-momentum photon from the Schwarzschild ISCO: g = sqrt(1 - 3/r)
  const r = 6, a = 0;
  const s = [r, 0, 0, 0, 0, 0];
  const g = P.diskRedshift(s, 1, a, r);
  check('Schwarzschild ISCO transverse redshift = sqrt(1/2)', close(g, Math.sqrt(0.5), 1e-9), `g = ${g.toFixed(6)}`);
  // approaching side blueshifted: disk at (r, 0) moves towards +y; a backward ray whose
  // spatial covector points along -y sees the gas coming at it.
  const gApp = P.diskRedshift([0, 0, 0, 0, -0.9, 0].map((v, i) => (i === 0 ? r : v)), 1, a, r);
  const gRec = P.diskRedshift([0, 0, 0, 0, 0.9, 0].map((v, i) => (i === 0 ? r : v)), 1, a, r);
  check('approaching gas is blueshifted, receding gas redshifted', gApp > 1 && gRec < 0.6, `g ${gApp.toFixed(3)} / ${gRec.toFixed(3)}`);
}

// --- 6. ISCO, horizon, temperature profile -------------------------------------------
check('ISCO a=0 is 6M', close(P.iscoRadius(0), 6, 1e-9), P.iscoRadius(0).toFixed(6));
check('ISCO a=0.998 is 1.237M', close(P.iscoRadius(0.998), 1.2369, 1e-3), P.iscoRadius(0.998).toFixed(4));
check('ISCO a=-1 (retrograde) is 9M', close(P.iscoRadius(-0.999999), 9, 1e-3), P.iscoRadius(-0.999999).toFixed(4));
check('circular orbit at ISCO is timelike', Number.isFinite(P.diskUt(P.iscoRadius(0.9) * 1.0001, 0.9)));
{
  const p = P.diskTemperatureProfile(0, 60, 4096);
  let iMax = 0;
  for (let i = 0; i < p.T.length; i++) if (p.T[i] > p.T[iMax]) iMax = i;
  check('Novikov-Thorne temperature peaks near 9.5M for a=0', p.r[iMax] > 8.5 && p.r[iMax] < 10.5, `peak at r = ${p.r[iMax].toFixed(2)}`);
  check('flux vanishes at the ISCO', p.T[0] === 0);
}
{
  const [r, g, b] = P.blackbodyRGB(6500);
  const m = (r + g + b) / 3;
  check('6500 K blackbody is near white', Math.abs(r / m - 1) < 0.08 && Math.abs(g / m - 1) < 0.08 && Math.abs(b / m - 1) < 0.08,
    `rgb ratios ${(r / m).toFixed(3)} ${(g / m).toFixed(3)} ${(b / m).toFixed(3)}`);
  const [r3, , b3] = P.blackbodyRGB(3000);
  const [r2, , b2] = P.blackbodyRGB(20000);
  check('3000 K is red, 20000 K is blue', r3 > b3 * 3 && b2 > r2);
}

console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
process.exit(failures ? 1 : 0);
