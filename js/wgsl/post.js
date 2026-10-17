// Bloom (downsample / upsample chain) and final composite with AgX tone mapping.
import { FRAME_WGSL } from './common.js';
import { FULLSCREEN_VS } from './trace.js';

// Temporal accumulation. Paused: a plain running average of jittered frames (converges to
// a supersampled image). Live: exponential history with the history clamped to the range
// of the current frame's 3x3 neighbourhood (in a luminance-compressed YCoCg space), which
// anti-aliases static thin features such as the photon ring without smearing the gas.
export const RESOLVE = FRAME_WGSL + FULLSCREEN_VS + /* wgsl */ `
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var cur: texture_2d<f32>;
@group(0) @binding(2) var hist: texture_2d<f32>;
struct Out { @location(0) accum: vec4f, @location(1) hdr: vec4f };

fn compress(c: vec3f) -> vec3f { return c / (1.0 + max(c.r, max(c.g, c.b))); }
fn expand(c: vec3f) -> vec3f { return c / max(1.0 - max(c.r, max(c.g, c.b)), 1e-4); }
fn toYCoCg(c: vec3f) -> vec3f { return vec3f(0.25 * c.r + 0.5 * c.g + 0.25 * c.b, 0.5 * c.r - 0.5 * c.b, -0.25 * c.r + 0.5 * c.g - 0.25 * c.b); }
fn fromYCoCg(c: vec3f) -> vec3f { return vec3f(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z); }

@fragment fn fs(@builtin(position) fc: vec4f) -> Out {
  let p = vec2i(fc.xy);
  let hi = vec2i(textureDimensions(cur)) - 1;
  let cur4 = textureLoad(cur, p, 0);
  let c = cur4.rgb;
  var h = textureLoad(hist, p, 0).rgb;
  var w = F.bh4.w;
  if (F.vol.z > 0.5) {
    // no motion vectors through the lens: lean on the current frame where gas moves
    // faster than about a pixel per frame, and while the camera moves
    w = max(w, mix(w, 0.9, smoothstep(0.25, 1.5, cur4.a)));
    w = max(w, 0.5 * F.extra.z);
    var mn = vec3f(1e9);
    var mx = vec3f(-1e9);
    for (var j = -1; j <= 1; j++) {
      for (var i = -1; i <= 1; i++) {
        let v = toYCoCg(compress(textureLoad(cur, clamp(p + vec2i(i, j), vec2i(0), hi), 0).rgb));
        mn = min(mn, v);
        mx = max(mx, v);
      }
    }
    h = expand(fromYCoCg(clamp(toYCoCg(compress(h)), mn, mx)));
  }
  let acc = max(mix(h, c, w), vec3f(0.0));
  var o: Out;
  o.accum = vec4f(acc, 1.0);
  o.hdr = vec4f(acc, 1.0);
  return o;
}
`;

// 13-tap downsample (Jimenez 2014, "Next generation post processing in Call of Duty").
// The first level uses a Karis average so single bright pixels (stars, the photon ring)
// do not flicker in the bloom.
export const BLOOM_DOWN = FULLSCREEN_VS + /* wgsl */ `
struct P { texel: vec2f, first: f32, cap: f32 };
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> U: P;

fn luma(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }
// The first level caps each sample's luminance, so an extremely bright inner disk (high
// spin) cannot throw glare deep into the shadow.
fn tap(uv: vec2f, o: vec2f) -> vec3f {
  let c = textureSampleLevel(src, samp, uv + o * U.texel, 0.0).rgb;
  if (U.first > 0.5) { return c * min(1.0, U.cap / max(luma(c), 1e-6)); }
  return c;
}

@fragment fn fs(@builtin(position) fc: vec4f) -> @location(0) vec4f {
  let dims = vec2f(textureDimensions(src));
  let uv = fc.xy * 2.0 / dims;
  let a = tap(uv, vec2f(-2.0, -2.0)); let b = tap(uv, vec2f(0.0, -2.0)); let c = tap(uv, vec2f(2.0, -2.0));
  let d = tap(uv, vec2f(-1.0, -1.0)); let e = tap(uv, vec2f(1.0, -1.0));
  let f = tap(uv, vec2f(-2.0, 0.0)); let g = tap(uv, vec2f(0.0, 0.0)); let h = tap(uv, vec2f(2.0, 0.0));
  let i = tap(uv, vec2f(-1.0, 1.0)); let j = tap(uv, vec2f(1.0, 1.0));
  let k = tap(uv, vec2f(-2.0, 2.0)); let l = tap(uv, vec2f(0.0, 2.0)); let m = tap(uv, vec2f(2.0, 2.0));
  if (U.first > 0.5) {
    let g0 = (d + e + i + j) * 0.25; let g1 = (a + b + f + g) * 0.25; let g2 = (b + c + g + h) * 0.25;
    let g3 = (f + g + k + l) * 0.25; let g4 = (g + h + l + m) * 0.25;
    let w0 = 1.0 / (1.0 + luma(g0)); let w1 = 1.0 / (1.0 + luma(g1)); let w2 = 1.0 / (1.0 + luma(g2));
    let w3 = 1.0 / (1.0 + luma(g3)); let w4 = 1.0 / (1.0 + luma(g4));
    let s = g0 * w0 * 0.5 + (g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4) * 0.125;
    return vec4f(s / (w0 * 0.5 + (w1 + w2 + w3 + w4) * 0.125), 1.0);
  }
  let s = (d + e + i + j) * 0.125 + (a + c + k + m) * 0.03125 + (b + f + h + l) * 0.0625 + g * 0.125;
  return vec4f(s, 1.0);
}
`;

// 3x3 tent upsample, added onto the next larger level (additive blend in the pipeline).
export const BLOOM_UP = FULLSCREEN_VS + /* wgsl */ `
struct P { texel: vec2f, weight: f32, pad: f32 };
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> U: P;
fn tap(uv: vec2f, o: vec2f) -> vec3f { return textureSampleLevel(src, samp, uv + o * U.texel, 0.0).rgb; }

@fragment fn fs(@builtin(position) fc: vec4f) -> @location(0) vec4f {
  let uv = fc.xy / (vec2f(textureDimensions(src)) * 2.0);
  var s = tap(uv, vec2f(0.0)) * 4.0;
  s += (tap(uv, vec2f(-1.0, 0.0)) + tap(uv, vec2f(1.0, 0.0)) + tap(uv, vec2f(0.0, -1.0)) + tap(uv, vec2f(0.0, 1.0))) * 2.0;
  s += tap(uv, vec2f(-1.0, -1.0)) + tap(uv, vec2f(1.0, -1.0)) + tap(uv, vec2f(-1.0, 1.0)) + tap(uv, vec2f(1.0, 1.0));
  return vec4f(s * (U.weight / 16.0), 1.0);
}
`;

// HDR (render resolution) + bloom -> display. Catmull-Rom upscale, AgX tone mapping
// (Troy Sobotka's AgX, polynomial fit of the default contrast curve), sRGB encode, dither.
export const COMPOSITE = FRAME_WGSL + FULLSCREEN_VS + /* wgsl */ `
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var hdr: texture_2d<f32>;
@group(0) @binding(2) var bloom: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;
@group(0) @binding(4) var<uniform> D: vec4f;    // display width, height, srgb-encode flag, unused

fn catmullRom(uv: vec2f) -> vec3f {
  let size = vec2f(textureDimensions(hdr));
  let sp = uv * size - 0.5;
  let tc = floor(sp) + 0.5;
  let f = sp - floor(sp);
  let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
  let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
  let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
  let w3 = f * f * (-0.5 + 0.5 * f);
  let w12 = w1 + w2;
  let o12 = w2 / w12;
  let t0 = (tc - 1.0) / size;
  let t3 = (tc + 2.0) / size;
  let t12 = (tc + o12) / size;
  var c = vec3f(0.0);
  c += textureSampleLevel(hdr, samp, vec2f(t0.x, t0.y), 0.0).rgb * w0.x * w0.y;
  c += textureSampleLevel(hdr, samp, vec2f(t12.x, t0.y), 0.0).rgb * w12.x * w0.y;
  c += textureSampleLevel(hdr, samp, vec2f(t3.x, t0.y), 0.0).rgb * w3.x * w0.y;
  c += textureSampleLevel(hdr, samp, vec2f(t0.x, t12.y), 0.0).rgb * w0.x * w12.y;
  c += textureSampleLevel(hdr, samp, vec2f(t12.x, t12.y), 0.0).rgb * w12.x * w12.y;
  c += textureSampleLevel(hdr, samp, vec2f(t3.x, t12.y), 0.0).rgb * w3.x * w12.y;
  c += textureSampleLevel(hdr, samp, vec2f(t0.x, t3.y), 0.0).rgb * w0.x * w3.y;
  c += textureSampleLevel(hdr, samp, vec2f(t12.x, t3.y), 0.0).rgb * w12.x * w3.y;
  c += textureSampleLevel(hdr, samp, vec2f(t3.x, t3.y), 0.0).rgb * w3.x * w3.y;
  return max(c, vec3f(0.0));
}

fn agx(c0: vec3f) -> vec3f {
  let m = mat3x3f(vec3f(0.842479062253094, 0.0423282422610123, 0.0423756549057051),
                  vec3f(0.0784335999999992, 0.878468636469772, 0.0784336),
                  vec3f(0.0792237451477643, 0.0791661274605434, 0.879142973793104));
  let mi = mat3x3f(vec3f(1.19687900512017, -0.0528968517574562, -0.0529716355144438),
                   vec3f(-0.0980208811401368, 1.15190312990417, -0.0980434501171241),
                   vec3f(-0.0990297440797205, -0.0989611768448433, 1.15107367264116));
  let lo = -12.47393;
  let hi = 4.026069;
  var v = m * max(c0, vec3f(1e-10));
  v = (clamp(log2(v), vec3f(lo), vec3f(hi)) - lo) / (hi - lo);
  let x2 = v * v;
  let x4 = x2 * x2;
  v = 15.5 * x4 * x2 - 40.14 * x4 * v + 31.96 * x4 - 6.868 * x2 * v + 0.4298 * x2 + 0.1191 * v - 0.00232;
  // look: a little extra contrast and saturation (between AgX "base" and "punchy")
  v = pow(max(v, vec3f(0.0)), vec3f(F.post.w));
  let lum = dot(v, vec3f(0.2126, 0.7152, 0.0722));
  v = lum + (v - lum) * 1.3;
  v = mi * v;
  return pow(clamp(v, vec3f(0.0), vec3f(1.0)), vec3f(2.2));   // back to linear
}

fn srgb(c: vec3f) -> vec3f {
  return select(1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055, c * 12.92, c <= vec3f(0.0031308));
}

@fragment fn fs(@builtin(position) fc: vec4f) -> @location(0) vec4f {
  let uv = fc.xy / D.xy;
  var c = catmullRom(uv);
  c += textureSampleLevel(bloom, samp, uv, 0.0).rgb * F.post.y;
  c *= F.post.x;
  // AgX desaturates bright colours channel by channel, which would turn the blueshifted
  // side of the disk white. Blend with a hue-preserving version that tone-maps the largest
  // channel and keeps the ratios between channels.
  let m = max(c.r, max(c.g, c.b));
  let hp = c * (agx(vec3f(m)).g / max(m, 1e-6));
  c = mix(agx(c), hp, F.vol.w);
  if (D.z > 0.5) { c = srgb(c); }
  // triangular dither, 1 LSB
  let n = fract(sin(dot(fc.xy + F.post.z * 0.618, vec2f(12.9898, 78.233))) * 43758.5453);
  let n2 = fract(sin(dot(fc.xy + F.post.z * 0.381 + 3.1, vec2f(39.346, 11.135))) * 24634.6345);
  c += (n + n2 - 1.0) / 255.0;
  return vec4f(c, 1.0);
}
`;
