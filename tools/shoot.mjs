// Headless screenshot runner for automated visual checks (no dependencies).
//   node tools/shoot.mjs <url-or-query> <out.png> [--wait=ms] [--w=1600 --h=900] [--eval=js]
// Starts headless Chrome with WebGPU, loads the page, waits until the page sets
// window.__shotReady (or the timeout passes), then captures the viewport. Console output
// from the page is echoed so shader errors show up in the terminal.
//
// A url starting with "?" is resolved against http://localhost:8090/.
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const args = process.argv.slice(2);
const opt = Object.fromEntries(args.filter((a) => a.startsWith('--')).map((a) => {
  const [k, v] = a.slice(2).split('=');
  return [k, v ?? '1'];
}));
const pos = args.filter((a) => !a.startsWith('--'));
let url = pos[0] || '?';
if (url.startsWith('?')) url = 'http://localhost:8090/' + url;
const out = pos[1] || 'shot.png';
const W = Number(opt.w || 1600), H = Number(opt.h || 900);
const timeout = Number(opt.wait || 20000);

const CHROME = process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const port = 9300 + Math.floor(Math.random() * 500);
const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'bh-chrome-'));
const chrome = spawn(CHROME, [
  '--headless=new', `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`,
  '--enable-unsafe-webgpu', '--enable-gpu', '--ignore-gpu-blocklist', '--use-angle=metal',
  '--no-first-run', '--no-default-browser-check', '--hide-scrollbars', '--mute-audio',
  `--window-size=${W},${H}`, 'about:blank',
], { stdio: ['ignore', 'ignore', 'pipe'] });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function getJSON(p) {
  for (let i = 0; i < 100; i++) {
    try { return await (await fetch(`http://127.0.0.1:${port}${p}`)).json(); } catch { await sleep(100); }
  }
  throw new Error('chrome did not start');
}

let ws, nextId = 1;
const pending = new Map();
const send = (method, params = {}) => new Promise((resolve, reject) => {
  const id = nextId++;
  pending.set(id, { resolve, reject });
  ws.send(JSON.stringify({ id, method, params }));
});

function finish(code) {
  try { ws?.close(); } catch { /* closing */ }
  chrome.kill('SIGKILL');
  setTimeout(() => { try { fs.rmSync(profile, { recursive: true, force: true }); } catch { /* busy */ } process.exit(code); }, 200);
}

try {
  const targets = await getJSON('/json/list');
  const page = targets.find((t) => t.type === 'page');
  ws = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((r) => ws.addEventListener('open', r, { once: true }));
  ws.addEventListener('message', (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.id && pending.has(msg.id)) {
      const p = pending.get(msg.id); pending.delete(msg.id);
      msg.error ? p.reject(new Error(msg.error.message)) : p.resolve(msg.result);
    } else if (msg.method === 'Runtime.consoleAPICalled') {
      console.log('[page]', msg.params.args.map((a) => a.value ?? a.description).join(' '));
    } else if (msg.method === 'Runtime.exceptionThrown') {
      console.log('[page exception]', msg.params.exceptionDetails.exception?.description || msg.params.exceptionDetails.text);
    }
  });
  await send('Runtime.enable');
  await send('Page.enable');
  await send('Emulation.setDeviceMetricsOverride', { width: W, height: H, deviceScaleFactor: 1, mobile: false });
  await send('Page.navigate', { url });
  const t0 = Date.now();
  let ready = false;
  while (Date.now() - t0 < timeout) {
    await sleep(250);
    const r = await send('Runtime.evaluate', { expression: 'window.__shotReady === true', returnByValue: true });
    if (r.result.value) { ready = true; break; }
  }
  if (!ready) console.log('warning: page never set __shotReady; capturing anyway');
  if (opt.eval) {
    const r = await send('Runtime.evaluate', { expression: opt.eval, returnByValue: true, awaitPromise: true });
    console.log('eval:', JSON.stringify(r.result.value));
  }
  const shot = await send('Page.captureScreenshot', { format: 'png' });
  fs.mkdirSync(path.dirname(path.resolve(out)), { recursive: true });
  fs.writeFileSync(out, Buffer.from(shot.data, 'base64'));
  const stats = await send('Runtime.evaluate', { expression: 'JSON.stringify(window.__stats || null)', returnByValue: true });
  console.log('saved', out, stats.result.value);
  finish(0);
} catch (e) {
  console.error('shoot failed:', e.message);
  finish(1);
}
