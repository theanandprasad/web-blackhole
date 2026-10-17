// Bundles index.html + js modules into self-contained pages that open straight from disk
// (file://) without a server:
//   blackhole.html      readable bundle
//   blackhole.min.html  minified bundle (WGSL comments/whitespace stripped; JS minified with
//                       esbuild when it can be found, otherwise comments/indentation stripped)
//   node build.mjs
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const root = path.dirname(fileURLToPath(import.meta.url));
const order = ['js/physics.js', 'js/wgsl/common.js', 'js/wgsl/trace.js', 'js/wgsl/sky.js', 'js/wgsl/post.js', 'js/main.js'];
const wgslModules = order.filter((f) => f.includes('/wgsl/'));
const read = (f) => fs.readFileSync(path.join(root, f), 'utf8');

// physics.js is imported as a namespace (import * as PH); recreate that object.
const physicsExports = [...read('js/physics.js').matchAll(/^export\s+(?:const|function|let|class)\s+([A-Za-z_$][\w$]*)/gm)].map((m) => m[1]);
const stripModuleSyntax = (src) => src
  .replace(/^import\s+\*\s+as\s+(\w+)\s+from\s+['"]\.\/physics\.js['"];\s*$/gm, (_, ns) => `const ${ns} = { ${physicsExports.join(', ')} };`)
  .replace(/^import[\s\S]*?from\s+['"][^'"]+['"];\s*$/gm, '')
  .replace(/^export\s+(const|function|async function|let|class)\s/gm, '$1 ');

const page = read('index.html');
const SCRIPT_TAG = '<script type="module" src="js/main.js"></script>';
const write = (name, html) => {
  fs.writeFileSync(path.join(root, name), html);
  console.log(`wrote ${name}`, (html.length / 1024).toFixed(0) + ' KB');
};

// ---- readable bundle --------------------------------------------------------------------
let code = '';
for (const f of order) code += `// ---- ${f} ----\n${stripModuleSyntax(read(f))}\n`;
write('blackhole.html', page.replace(SCRIPT_TAG, () => `<script type="module">\n${code}\n</script>`));

// ---- minified bundle --------------------------------------------------------------------
// WGSL has no string literals, so comments and whitespace can be removed safely. Spaces are
// kept around + - / so that e.g. "a - -b" never fuses into other tokens.
function minifyWGSL(src) {
  return src
    .replace(/\/\*[\s\S]*?\*\//g, ' ')
    .replace(/\/\/[^\n]*/g, ' ')
    .replace(/\s+/g, ' ')
    .replace(/ ?([{}()[\],;:=<>*&|^!%@.]) ?/g, '$1')
    .trim();
}
// Minify every WGSL template literal in the shader modules. ${...} interpolations are
// swapped for identifier-like placeholders first so tokens never fuse across them.
function minifyTemplates(src) {
  return src.replace(/`([^`]*)`/g, (_, body) => {
    const parts = [];
    const masked = body.replace(/\$\{[^}]*\}/g, (m) => `__P${parts.push(m) - 1}__`);
    const out = minifyWGSL(masked).replace(/__P(\d+)__/g, (_, i) => parts[Number(i)]);
    return '`' + out + '`';
  });
}
let minJs = '';
for (const f of order) {
  const src = read(f);
  minJs += stripModuleSyntax(wgslModules.includes(f) ? minifyTemplates(src) : src) + '\n';
}

function findEsbuild() {
  const req = createRequire(import.meta.url);
  try { return req('esbuild'); } catch { /* not installed in the project */ }
  // fall back to an esbuild left in the npx cache by other tools
  const caches = [path.join(os.homedir(), '.npm', '_npx'), path.join(os.homedir(), 'AppData', 'Local', 'npm-cache', '_npx')];
  for (const cache of caches) {
    for (const dir of fs.existsSync(cache) ? fs.readdirSync(cache) : []) {
      const pkg = path.join(cache, dir, 'node_modules', 'esbuild');
      if (fs.existsSync(pkg)) { try { return req(pkg); } catch { /* try next */ } }
    }
  }
  return null;
}

const esbuild = findEsbuild();
if (esbuild) {
  minJs = esbuild.transformSync(minJs, { minify: true, format: 'esm', target: 'es2022', legalComments: 'none' }).code;
} else {
  console.log('esbuild not found: JS gets comment/indentation stripping only');
  minJs = minJs.split('\n').map((l) => l.trim()).filter((l) => l && !l.startsWith('//')).join('\n');
}

const minPage = page
  .replace(SCRIPT_TAG, () => `<script type="module">${minJs}</script>`)
  .replace(/<style>([\s\S]*?)<\/style>/, (_, css) => `<style>${css.replace(/\s+/g, ' ').replace(/ ?([{}:;,>]) ?/g, '$1').trim()}</style>`)
  .replace(/>\s+</g, '><');
write('blackhole.min.html', minPage);
