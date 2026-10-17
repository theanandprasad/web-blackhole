// Background galaxy, baked once into a mip-mapped cube map. The tracer samples it along the
// lensed escape direction with a mip level chosen from the pixel's footprint on the sky.
// Stars are not baked: they are point sources and are splatted analytically in trace.js.
import { NOISE_WGSL } from './common.js';

export const SKY_CUBE_SIZE = 1024;

// Direction for texel (s, t) in [-1, 1] of a cube face, WebGPU face order +X -X +Y -Y +Z -Z.
const CUBE_DIR_WGSL = /* wgsl */ `
fn cubeDir(face: u32, s: f32, t: f32) -> vec3f {
  switch face {
    case 0u: { return normalize(vec3f(1.0, -t, -s)); }
    case 1u: { return normalize(vec3f(-1.0, -t, s)); }
    case 2u: { return normalize(vec3f(s, 1.0, t)); }
    case 3u: { return normalize(vec3f(s, -1.0, -t)); }
    case 4u: { return normalize(vec3f(s, -t, 1.0)); }
    default: { return normalize(vec3f(-s, -t, -1.0)); }
  }
}
`;

// Milky Way seen from inside the disk of a spiral galaxy, in galactic coordinates (the
// galactic plane is z = 0, the centre towards +x): a thin band of unresolved stars with
// dark dust lanes, a warm central bulge, patchy star clouds and faint emission nebulae.
export const SKY_BAKE = NOISE_WGSL + CUBE_DIR_WGSL + /* wgsl */ `
@group(0) @binding(0) var dst: texture_storage_2d_array<rgba16float, write>;

fn galaxy(dir: vec3f, fp: f32) -> vec3f {
  let b = asin(clamp(dir.z, -1.0, 1.0));            // latitude
  let l = atan2(dir.y, dir.x);                      // longitude
  let q = dir * 2.4;
  let clouds = fbm(q, fp * 2.4, 8, 0.58);
  let warp = vec3f(fbm(q * 1.3 + 3.0, fp * 3.1, 4, 0.5), fbm(q * 1.3 + 9.0, fp * 3.1, 4, 0.5), 0.0) * 0.08;
  let bw = b + warp.x * 0.4;
  let width = 0.11 + 0.035 * clouds + 0.05 * exp(-l * l / 0.6);
  let band = exp(-bw * bw / (width * width));
  let halo = exp(-abs(bw) / 0.45) * 0.12;
  let bulge = exp(-l * l / 0.22 - bw * bw / 0.03);
  // dust: filamentary (ridged noise), concentrated slightly off the mid-plane
  let dq = (dir + warp) * 5.0;
  let ridge = 1.0 - abs(fbm(dq, fp * 5.0, 8, 0.6));
  let fil = pow(clamp(ridge, 0.0, 1.0), 5.0);
  let patches = smoothstep(-0.2, 0.3, fbm(dq * 0.5 + 4.0, fp * 2.5, 5, 0.55));
  let lane = exp(-(bw + 0.02) * (bw + 0.02) / 0.0035) + 0.5 * exp(-(bw - 0.05) * (bw - 0.05) / 0.0015);
  let ext = clamp((fil * 0.9 + patches * 0.35) * (0.15 + 1.1 * lane) * patches, 0.0, 0.95);
  let starry = clamp(0.55 + 0.9 * clouds, 0.05, 1.6);
  let glow = band * starry + 1.8 * bulge * (0.7 + 0.5 * clouds) + halo;
  let tint = mix(vec3f(0.66, 0.76, 1.0), vec3f(1.0, 0.86, 0.7), clamp(bulge * 1.4, 0.0, 1.0));
  // dust reddens what it does not block
  let reddening = mix(vec3f(1.0), vec3f(1.0, 0.75, 0.55), ext);
  var c = tint * glow * (1.0 - ext) * reddening;
  // H-alpha emission nebulae hugging the plane
  let neb = smoothstep(0.25, 0.65, fbm(q * 3.0 + vec3f(5.0, 2.0, 7.0), fp * 7.2, 6, 0.55)) * exp(-bw * bw / 0.006);
  c += vec3f(1.0, 0.28, 0.32) * neb * 0.35 * (1.0 - ext);
  return c * 0.018;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(dst).x;
  if (id.x >= size || id.y >= size) { return; }
  let st = (vec2f(id.xy) + 0.5) / f32(size) * 2.0 - 1.0;
  let dir = cubeDir(id.z, st.x, st.y);
  textureStore(dst, vec2i(id.xy), i32(id.z), vec4f(galaxy(dir, 2.0 / f32(size)), 1.0));
}
`;

export const CUBE_MIP = /* wgsl */ `
@group(0) @binding(0) var src: texture_2d_array<f32>;
@group(0) @binding(1) var dst: texture_storage_2d_array<rgba16float, write>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(dst);
  if (id.x >= size.x || id.y >= size.y) { return; }
  let p = vec2i(id.xy) * 2;
  let l = i32(id.z);
  let c = textureLoad(src, p, l, 0) + textureLoad(src, p + vec2i(1, 0), l, 0) +
          textureLoad(src, p + vec2i(0, 1), l, 0) + textureLoad(src, p + vec2i(1, 1), l, 0);
  textureStore(dst, vec2i(id.xy), l, c * 0.25);
}
`;
