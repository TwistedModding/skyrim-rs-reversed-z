//! The immediate context's calls, mirrored while a world depth target is bound: each vertex shader
//! swapped for its twin (clip z = w - z, [`crate::twin`]), viewport depth ranges [a, b] made
//! [1 - b, 1 - a], depth tests LESS <-> GREATER, depth biases negated, depth clears 1 - value. Every
//! depth the world targets hold is then 1 - the vanilla value. Their standard copies
//! ([`crate::copy`]) are refreshed when a written world view is unbound (readers cannot sample a
//! depth target bound for writing) and after the engine copies the scene depth into another target.

use core::ffi::c_void;
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};

use sre::raw::globals;
use windows::Win32::Graphics::Direct3D11::{
    D3D11_COMPARISON_FUNC, D3D11_COMPARISON_GREATER, D3D11_COMPARISON_GREATER_EQUAL,
    D3D11_COMPARISON_LESS, D3D11_COMPARISON_LESS_EQUAL, D3D11_DEPTH_STENCIL_DESC,
    D3D11_DEPTH_STENCIL_VIEW_DESC, D3D11_DEPTH_WRITE_MASK_ALL, D3D11_RASTERIZER_DESC,
    D3D11_VIEWPORT, ID3D11DepthStencilState, ID3D11DepthStencilView, ID3D11Device,
    ID3D11DeviceContext, ID3D11RasterizerState, ID3D11RenderTargetView, ID3D11VertexShader,
};
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
use windows::core::Interface;

type SetTargetsFn = unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void, *mut c_void);
type SetTargetsUavFn = unsafe extern "system" fn(
    *mut c_void,
    u32,
    *const *mut c_void,
    *mut c_void,
    u32,
    u32,
    *const *mut c_void,
    *const u32,
);
type SetDepthStateFn = unsafe extern "system" fn(*mut c_void, *mut c_void, u32);
type SetRasterFn = unsafe extern "system" fn(*mut c_void, *mut c_void);
type SetViewportsFn = unsafe extern "system" fn(*mut c_void, u32, *const D3D11_VIEWPORT);
type ClearDepthFn = unsafe extern "system" fn(*mut c_void, *mut c_void, u32, f32, u8);
type SetShaderFn = unsafe extern "system" fn(*mut c_void, *mut c_void, *const *mut c_void, u32);
type CopyFn = unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void);
type SetResourcesFn = unsafe extern "system" fn(*mut c_void, u32, u32, *const *mut c_void);
type DrawFn = unsafe extern "system" fn(*mut c_void, u32, u32);
type DrawIndexedFn = unsafe extern "system" fn(*mut c_void, u32, u32, i32);
type DrawInstancedFn = unsafe extern "system" fn(*mut c_void, u32, u32, u32, u32);
type DrawIndexedInstancedFn = unsafe extern "system" fn(*mut c_void, u32, u32, u32, i32, u32);

/// `Renderer` (`g_Renderer`) per depth target (0x98 bytes each): the texture +0x2018, the
/// depth view of slice 0 +0x2020, its read-only view +0x2060, the shader view +0x20a0.
const TARGET_STRIDE: usize = 0x98;
pub(crate) const TARGET_TEXTURE: usize = 0x2018;
const TARGET_VIEW: usize = 0x2020;
const TARGET_READ_ONLY_VIEW: usize = 0x2060;
pub(crate) const TARGET_SHADER_VIEW: usize = 0x20a0;
/// The depth targets the world cameras draw into: the scene depth and the first-person copy.
const WORLD_TARGETS: [usize; 2] = [0, 7];
/// Address Library id of the engine's current vertex shader record (`SetCurrentVertexShader`
/// stores it before binding): the D3D shader +0x8, the bytecode size +0x10, the bytecode +0x68.
const CURRENT_VERTEX_SHADER: u64 = 388_906;
const RECORD_SHADER: usize = 0x8;
const RECORD_SIZE: usize = 0x10;
const RECORD_BYTECODE: usize = 0x68;
const D3D11_CLEAR_DEPTH: u32 = 1;
const KEEP_TARGETS: u32 = 0xffff_ffff;

macro_rules! hooks {
    ($($name:ident: $ty:ty = $field:ident => $detour:ident,)*) => {
        $(static $name: Mutex<Option<hook::Hook<$ty>>> = Mutex::new(None);)*

        fn install_hooks(table: &windows::Win32::Graphics::Direct3D11::ID3D11DeviceContext_Vtbl) -> Result<(), String> {
            $(
                let hook = unsafe { hook::Hook::<$ty>::create(table.$field as usize, $detour as $ty) }.map_err(|e| format!("{}: {e}", stringify!($field)))?;
                let mut slot = $name.lock().unwrap();
                // enabled with the slot filled: a call on another thread waits for it instead of finding it empty
                slot.insert(hook).enable().map_err(|e| format!("{}: {e}", stringify!($field)))?;
            )*
            Ok(())
        }

        pub(crate) fn remove() {
            $(
                if let Some(h) = $name.lock().unwrap().take() {
                    let _ = unsafe { h.remove() };
                }
            )*
        }
    };
}

hooks! {
    SET_TARGETS: SetTargetsFn = OMSetRenderTargets => set_targets,
    SET_TARGETS_UAV: SetTargetsUavFn = OMSetRenderTargetsAndUnorderedAccessViews => set_targets_uav,
    SET_DEPTH_STATE: SetDepthStateFn = OMSetDepthStencilState => set_depth_state,
    SET_RASTER: SetRasterFn = RSSetState => set_raster,
    SET_VIEWPORTS: SetViewportsFn = RSSetViewports => set_viewports,
    CLEAR_DEPTH: ClearDepthFn = ClearDepthStencilView => clear_depth,
    SET_VERTEX_SHADER: SetShaderFn = VSSetShader => set_vertex_shader,
    SET_PIXEL_RESOURCES: SetResourcesFn = PSSetShaderResources => set_pixel_resources,
    SET_COMPUTE_RESOURCES: SetResourcesFn = CSSetShaderResources => set_compute_resources,
    COPY: CopyFn = CopyResource => copy_resource,
    DRAW: DrawFn = Draw => draw,
    DRAW_INDEXED: DrawIndexedFn = DrawIndexed => draw_indexed,
    DRAW_INSTANCED: DrawInstancedFn = DrawInstanced => draw_instanced,
    DRAW_INDEXED_INSTANCED: DrawIndexedInstancedFn = DrawIndexedInstanced => draw_indexed_instanced,
}

static WORLD_VIEWS: [AtomicUsize; 4] = [const { AtomicUsize::new(0) }; 4];
static KNOWN_VIEWS: Mutex<Option<HashMap<usize, Option<usize>>>> = Mutex::new(None);
/// A world depth view is bound and the calls sent to the context are mirrored.
static BOUND: AtomicBool = AtomicBool::new(false);
static LAST_DEPTH_STATE: AtomicUsize = AtomicUsize::new(0);
static LAST_STENCIL_REF: AtomicU32 = AtomicU32::new(0);
static LAST_RASTER: AtomicUsize = AtomicUsize::new(0);
static LAST_VIEWPORTS: Mutex<Vec<D3D11_VIEWPORT>> = Mutex::new(Vec::new());
static LAST_VERTEX_SHADER: AtomicUsize = AtomicUsize::new(0);
static LAST_DEPTH_VIEW: AtomicUsize = AtomicUsize::new(0);
/// The engine's current depth state writes depth.
static WRITES: AtomicBool = AtomicBool::new(true);
/// The bound world target.
static BOUND_TARGET: AtomicUsize = AtomicUsize::new(0);
/// Per world target: written since its copy was last refreshed.
static WRITTEN: [AtomicBool; 2] = [const { AtomicBool::new(true) }; 2];
static MIRRORED_DEPTH: Mutex<Option<HashMap<usize, usize>>> = Mutex::new(None);
static MIRRORED_RASTER: Mutex<Option<HashMap<usize, usize>>> = Mutex::new(None);
static DEPTH_WRITES: Mutex<Option<HashMap<usize, bool>>> = Mutex::new(None);
static VERTEX_TWINS: Mutex<Option<HashMap<usize, Option<ID3D11VertexShader>>>> = Mutex::new(None);
static TWINS_FAILED: AtomicU64 = AtomicU64::new(0);
static CLEARS: AtomicU64 = AtomicU64::new(0);

fn original<F: Copy>(slot: &Mutex<Option<hook::Hook<F>>>) -> F {
    slot.lock()
        .unwrap()
        .as_ref()
        .expect("hooks are removed only after the plugin is off")
        .original()
}

pub(crate) fn device() -> Option<ID3D11Device> {
    let p = unsafe { *(crate::address(globals::g_D3D11Device)? as *const *mut c_void) };
    unsafe { ID3D11Device::from_raw_borrowed(&p) }.cloned()
}

pub(crate) fn immediate() -> Option<ID3D11DeviceContext> {
    let p = unsafe { *(crate::address(globals::g_D3D11Context)? as *const *mut c_void) };
    unsafe { ID3D11DeviceContext::from_raw_borrowed(&p) }.cloned()
}

/// The field at `offset` of depth target `target` in `g_Renderer`.
pub(crate) fn target_field(target: usize, offset: usize) -> Option<*mut usize> {
    Some((crate::address(globals::g_Renderer)? + target * TARGET_STRIDE + offset) as *mut usize)
}

pub(crate) fn install() -> Result<(), String> {
    let context = immediate().ok_or("no immediate context")?;
    for (i, (target, offset)) in WORLD_TARGETS
        .iter()
        .flat_map(|t| [(*t, TARGET_VIEW), (*t, TARGET_READ_ONLY_VIEW)])
        .enumerate()
    {
        let view = unsafe { *target_field(target, offset).ok_or("no renderer")? };
        WORLD_VIEWS[i].store(view, Ordering::Relaxed);
    }
    let view = WORLD_VIEWS[0].load(Ordering::Relaxed) as *mut c_void;
    let view = unsafe { ID3D11DepthStencilView::from_raw_borrowed(&view) }
        .ok_or("the world depth target has no view")?;
    let mut desc = D3D11_DEPTH_STENCIL_VIEW_DESC::default();
    unsafe { view.GetDesc(&raw mut desc) };
    if desc.Format != DXGI_FORMAT_D32_FLOAT_S8X24_UINT {
        return Err(format!(
            "the world depth target is format {}, created before the format patches",
            desc.Format.0
        ));
    }
    install_hooks(context.vtable())
}

/// The world target (scene depth or first-person) a depth view is of, whichever view object it is.
fn world_target(view: *mut c_void) -> Option<usize> {
    if view.is_null() {
        return None;
    }
    if let Some(i) = WORLD_VIEWS
        .iter()
        .position(|v| v.load(Ordering::Relaxed) == view as usize)
    {
        return Some(WORLD_TARGETS[i / 2]);
    }
    let mut known = KNOWN_VIEWS.lock().unwrap();
    *known
        .get_or_insert_with(HashMap::new)
        .entry(view as usize)
        .or_insert_with(|| {
            let texture = unsafe { ID3D11DepthStencilView::from_raw_borrowed(&view) }
                .and_then(|v| unsafe { v.GetResource() }.ok())
                .map(|r| r.as_raw());
            WORLD_TARGETS.into_iter().find(|&i| {
                texture.is_some_and(|t| {
                    target_field(i, TARGET_TEXTURE).is_some_and(|f| unsafe { *f } == t as usize)
                })
            })
        })
}

fn written(target: usize) -> Option<&'static AtomicBool> {
    WORLD_TARGETS
        .iter()
        .position(|&t| t == target)
        .map(|k| &WRITTEN[k])
}

/// Refreshes a world target's copy when a written view of it was unbound, and sends the last states
/// and vertex shader again once the bound depth view switched between world and other.
fn bind(context: *mut c_void, view: *mut c_void) {
    let previous = LAST_DEPTH_VIEW.swap(view as usize, Ordering::Relaxed) as *mut c_void;
    if previous != view
        && let Some(t) = world_target(previous)
        && written(t).is_some_and(|w| w.load(Ordering::Relaxed))
    {
        crate::copy::run(t);
    }
    let target = world_target(view).filter(|_| crate::active());
    if let Some(t) = target {
        BOUND_TARGET.store(t, Ordering::Relaxed);
    }
    let bound = target.is_some();
    if BOUND.swap(bound, Ordering::Relaxed) == bound {
        return;
    }
    unsafe {
        original(&SET_DEPTH_STATE)(
            context,
            depth_state(LAST_DEPTH_STATE.load(Ordering::Relaxed) as *mut c_void),
            LAST_STENCIL_REF.load(Ordering::Relaxed),
        );
        original(&SET_RASTER)(
            context,
            raster(LAST_RASTER.load(Ordering::Relaxed) as *mut c_void),
        );
        let viewports = viewports(&LAST_VIEWPORTS.lock().unwrap());
        original(&SET_VIEWPORTS)(
            context,
            u32::try_from(viewports.len()).unwrap_or(0),
            viewports.as_ptr(),
        );
        original(&SET_VERTEX_SHADER)(
            context,
            vertex_shader(LAST_VERTEX_SHADER.load(Ordering::Relaxed) as *mut c_void),
            core::ptr::null(),
            0,
        );
    }
}

/// Runs `f` with no output targets bound, past the bind bookkeeping, and binds them again after.
pub(crate) fn without_targets(context: &ID3D11DeviceContext, f: impl FnOnce()) {
    let mut views: [Option<ID3D11RenderTargetView>; 8] = Default::default();
    let mut depth: Option<ID3D11DepthStencilView> = None;
    unsafe { context.OMGetRenderTargets(Some(&mut views), Some(&raw mut depth)) };
    let set = original(&SET_TARGETS);
    unsafe {
        set(
            context.as_raw(),
            0,
            core::ptr::null(),
            core::ptr::null_mut(),
        );
    };
    f();
    let raw: Vec<*mut c_void> = views
        .iter()
        .map(|v| v.as_ref().map_or(core::ptr::null_mut(), Interface::as_raw))
        .collect();
    let count = raw.iter().rposition(|v| !v.is_null()).map_or(0, |i| i + 1);
    unsafe {
        set(
            context.as_raw(),
            u32::try_from(count).unwrap_or(0),
            raw.as_ptr(),
            depth
                .as_ref()
                .map_or(core::ptr::null_mut(), Interface::as_raw),
        );
    };
}

pub(crate) fn mark_copied(target: usize) {
    if let Some(w) = written(target) {
        w.store(false, Ordering::Relaxed);
    }
}

unsafe extern "system" fn set_targets(
    context: *mut c_void,
    count: u32,
    views: *const *mut c_void,
    depth: *mut c_void,
) {
    unsafe { original(&SET_TARGETS)(context, count, views, depth) };
    bind(context, depth);
}

unsafe extern "system" fn set_targets_uav(
    context: *mut c_void,
    count: u32,
    views: *const *mut c_void,
    depth: *mut c_void,
    uav_start: u32,
    uav_count: u32,
    uavs: *const *mut c_void,
    counts: *const u32,
) {
    unsafe {
        original(&SET_TARGETS_UAV)(
            context, count, views, depth, uav_start, uav_count, uavs, counts,
        );
    };
    if count != KEEP_TARGETS {
        bind(context, depth);
    }
}

unsafe extern "system" fn set_depth_state(
    context: *mut c_void,
    state: *mut c_void,
    stencil_ref: u32,
) {
    LAST_DEPTH_STATE.store(state as usize, Ordering::Relaxed);
    LAST_STENCIL_REF.store(stencil_ref, Ordering::Relaxed);
    WRITES.store(writes_depth(state), Ordering::Relaxed);
    unsafe { original(&SET_DEPTH_STATE)(context, depth_state(state), stencil_ref) };
}

unsafe extern "system" fn set_raster(context: *mut c_void, state: *mut c_void) {
    LAST_RASTER.store(state as usize, Ordering::Relaxed);
    unsafe { original(&SET_RASTER)(context, raster(state)) };
}

unsafe extern "system" fn set_viewports(
    context: *mut c_void,
    count: u32,
    list: *const D3D11_VIEWPORT,
) {
    let given = if list.is_null() {
        &[][..]
    } else {
        unsafe { core::slice::from_raw_parts(list, count as usize) }
    };
    let mut stored = LAST_VIEWPORTS.lock().unwrap();
    stored.clear();
    stored.extend_from_slice(given);
    let sent = viewports(&stored);
    drop(stored);
    unsafe {
        original(&SET_VIEWPORTS)(
            context,
            count,
            if list.is_null() { list } else { sent.as_ptr() },
        );
    };
}

unsafe extern "system" fn clear_depth(
    context: *mut c_void,
    view: *mut c_void,
    flags: u32,
    depth: f32,
    stencil: u8,
) {
    let target = world_target(view).filter(|_| crate::active() && flags & D3D11_CLEAR_DEPTH != 0);
    let mirror = target.is_some();
    if let Some(w) = target.and_then(written) {
        CLEARS.fetch_add(1, Ordering::Relaxed);
        w.store(true, Ordering::Relaxed);
    }
    unsafe {
        original(&CLEAR_DEPTH)(
            context,
            view,
            flags,
            if mirror { 1.0 - depth } else { depth },
            stencil,
        );
    };
}

unsafe extern "system" fn set_vertex_shader(
    context: *mut c_void,
    shader: *mut c_void,
    instances: *const *mut c_void,
    count: u32,
) {
    LAST_VERTEX_SHADER.store(shader as usize, Ordering::Relaxed);
    let sent = if count == 0 {
        vertex_shader(shader)
    } else {
        shader
    };
    unsafe { original(&SET_VERTEX_SHADER)(context, sent, instances, count) };
}

unsafe extern "system" fn set_pixel_resources(
    context: *mut c_void,
    start: u32,
    count: u32,
    views: *const *mut c_void,
) {
    let swapped = substituted(count, views);
    unsafe {
        original(&SET_PIXEL_RESOURCES)(
            context,
            start,
            count,
            swapped.as_ref().map_or(views, Vec::as_ptr),
        );
    };
}

unsafe extern "system" fn set_compute_resources(
    context: *mut c_void,
    start: u32,
    count: u32,
    views: *const *mut c_void,
) {
    let swapped = substituted(count, views);
    unsafe {
        original(&SET_COMPUTE_RESOURCES)(
            context,
            start,
            count,
            swapped.as_ref().map_or(views, Vec::as_ptr),
        );
    };
}

/// Shader views with a reversed depth target's own view replaced by its copy's.
fn substituted(count: u32, views: *const *mut c_void) -> Option<Vec<*mut c_void>> {
    if views.is_null() {
        return None;
    }
    let given = unsafe { core::slice::from_raw_parts(views, count as usize) };
    given
        .iter()
        .any(|&v| crate::copy::stand_in(v).is_some())
        .then(|| {
            given
                .iter()
                .map(|&v| crate::copy::stand_in(v).unwrap_or(v))
                .collect()
        })
}

/// Binds a compute shader view as given.
pub(crate) fn set_compute_resource_as_is(context: &ID3D11DeviceContext, view: *mut c_void) {
    unsafe { original(&SET_COMPUTE_RESOURCES)(context.as_raw(), 0, 1, &raw const view) };
}

fn texture_of(target: usize) -> usize {
    target_field(target, TARGET_TEXTURE).map_or(0, |f| unsafe { *f })
}

/// After the engine copies the scene depth into another reversed target, that target's copy is
/// refreshed.
unsafe extern "system" fn copy_resource(context: *mut c_void, dst: *mut c_void, src: *mut c_void) {
    unsafe { original(&COPY)(context, dst, src) };
    if crate::active()
        && src as usize == texture_of(0)
        && let Some(&t) = crate::copy::REVERSED_TARGETS
            .iter()
            .find(|&&t| t != 0 && texture_of(t) == dst as usize)
    {
        crate::copy::run(t);
    }
}

fn drawn() {
    if BOUND.load(Ordering::Relaxed)
        && WRITES.load(Ordering::Relaxed)
        && let Some(w) = written(BOUND_TARGET.load(Ordering::Relaxed))
    {
        w.store(true, Ordering::Relaxed);
    }
}

unsafe extern "system" fn draw(context: *mut c_void, vertices: u32, start: u32) {
    drawn();
    unsafe { original(&DRAW)(context, vertices, start) };
}

unsafe extern "system" fn draw_indexed(context: *mut c_void, indices: u32, start: u32, base: i32) {
    drawn();
    unsafe { original(&DRAW_INDEXED)(context, indices, start, base) };
}

unsafe extern "system" fn draw_instanced(
    context: *mut c_void,
    vertices: u32,
    instances: u32,
    start: u32,
    first: u32,
) {
    drawn();
    unsafe { original(&DRAW_INSTANCED)(context, vertices, instances, start, first) };
}

unsafe extern "system" fn draw_indexed_instanced(
    context: *mut c_void,
    indices: u32,
    instances: u32,
    start: u32,
    base: i32,
    first: u32,
) {
    drawn();
    unsafe { original(&DRAW_INDEXED_INSTANCED)(context, indices, instances, start, base, first) };
}

fn viewports(given: &[D3D11_VIEWPORT]) -> Vec<D3D11_VIEWPORT> {
    let bound = BOUND.load(Ordering::Relaxed);
    given
        .iter()
        .map(|v| {
            if bound {
                D3D11_VIEWPORT {
                    MinDepth: 1.0 - v.MaxDepth,
                    MaxDepth: 1.0 - v.MinDepth,
                    ..*v
                }
            } else {
                *v
            }
        })
        .collect()
}

/// The shader to bind: the engine's, or its twin while a world view is bound. A twin is made from
/// the bytecode of the engine's current record the first time its shader is bound under a world view.
fn vertex_shader(shader: *mut c_void) -> *mut c_void {
    if !BOUND.load(Ordering::Relaxed) || shader.is_null() {
        return shader;
    }
    let mut twins = VERTEX_TWINS.lock().unwrap();
    let twins = twins.get_or_insert_with(HashMap::new);
    if let Some(twin) = twins.get(&(shader as usize)) {
        return twin.as_ref().map_or(shader, Interface::as_raw);
    }
    let Some(record) = crate::address(CURRENT_VERTEX_SHADER)
        .map(|a| unsafe { *(a as *const usize) })
        .filter(|&r| r != 0)
    else {
        return shader;
    };
    if unsafe { *((record + RECORD_SHADER) as *const usize) } != shader as usize {
        return shader;
    }
    let bytecode = unsafe {
        core::slice::from_raw_parts(
            (record + RECORD_BYTECODE) as *const u8,
            *((record + RECORD_SIZE) as *const u32) as usize,
        )
    };
    let twin = crate::twin::mirrored(bytecode)
        .zip(device())
        .and_then(|(code, device)| {
            let mut out = None;
            unsafe { device.CreateVertexShader(&code, None, Some(&raw mut out)) }.ok()?;
            out
        });
    if twin.is_none() {
        TWINS_FAILED.fetch_add(1, Ordering::Relaxed);
    }
    let sent = twin.as_ref().map_or(shader, Interface::as_raw);
    twins.insert(shader as usize, twin);
    sent
}

fn writes_depth(state: *mut c_void) -> bool {
    let mut cache = DEPTH_WRITES.lock().unwrap();
    *cache
        .get_or_insert_with(HashMap::new)
        .entry(state as usize)
        .or_insert_with(|| {
            let Some(s) = (unsafe { ID3D11DepthStencilState::from_raw_borrowed(&state) }) else {
                return true;
            };
            let mut desc = D3D11_DEPTH_STENCIL_DESC::default();
            unsafe { s.GetDesc(&raw mut desc) };
            desc.DepthEnable.as_bool() && desc.DepthWriteMask == D3D11_DEPTH_WRITE_MASK_ALL
        })
}

fn mirrored_func(f: D3D11_COMPARISON_FUNC) -> D3D11_COMPARISON_FUNC {
    match f {
        D3D11_COMPARISON_LESS => D3D11_COMPARISON_GREATER,
        D3D11_COMPARISON_GREATER => D3D11_COMPARISON_LESS,
        D3D11_COMPARISON_LESS_EQUAL => D3D11_COMPARISON_GREATER_EQUAL,
        D3D11_COMPARISON_GREATER_EQUAL => D3D11_COMPARISON_LESS_EQUAL,
        other => other,
    }
}

/// The state to send: the engine's, or its mirrored twin while a world view is bound.
fn depth_state(state: *mut c_void) -> *mut c_void {
    if !BOUND.load(Ordering::Relaxed) {
        return state;
    }
    twin(&MIRRORED_DEPTH, state, |device| {
        let mut desc = D3D11_DEPTH_STENCIL_DESC {
            DepthEnable: true.into(),
            DepthWriteMask: D3D11_DEPTH_WRITE_MASK_ALL,
            DepthFunc: D3D11_COMPARISON_LESS,
            ..Default::default()
        };
        if let Some(s) = unsafe { ID3D11DepthStencilState::from_raw_borrowed(&state) } {
            unsafe { s.GetDesc(&raw mut desc) };
        }
        desc.DepthFunc = mirrored_func(desc.DepthFunc);
        let mut out = None;
        unsafe { device.CreateDepthStencilState(&raw const desc, Some(&raw mut out)) }.ok()?;
        out.map(Interface::into_raw)
    })
}

fn raster(state: *mut c_void) -> *mut c_void {
    let Some(s) = unsafe { ID3D11RasterizerState::from_raw_borrowed(&state) }
        .filter(|_| BOUND.load(Ordering::Relaxed))
    else {
        return state;
    };
    let s = s.clone();
    twin(&MIRRORED_RASTER, state, move |device| {
        let mut desc = D3D11_RASTERIZER_DESC::default();
        unsafe { s.GetDesc(&raw mut desc) };
        if desc.DepthBias == 0 && desc.SlopeScaledDepthBias == 0.0 {
            return Some(s.as_raw());
        }
        desc.DepthBias = -desc.DepthBias;
        desc.SlopeScaledDepthBias = -desc.SlopeScaledDepthBias;
        desc.DepthBiasClamp = -desc.DepthBiasClamp;
        let mut out = None;
        unsafe { device.CreateRasterizerState(&raw const desc, Some(&raw mut out)) }.ok()?;
        out.map(Interface::into_raw)
    })
}

/// The cached twin of `state`, made once by `build`; the state itself when it cannot be made.
fn twin(
    cache: &Mutex<Option<HashMap<usize, usize>>>,
    state: *mut c_void,
    build: impl FnOnce(&ID3D11Device) -> Option<*mut c_void>,
) -> *mut c_void {
    let mut cache = cache.lock().unwrap();
    let cache = cache.get_or_insert_with(HashMap::new);
    if let Some(t) = cache.get(&(state as usize)) {
        return *t as *mut c_void;
    }
    let made = device().and_then(|d| build(&d)).unwrap_or(state);
    cache.insert(state as usize, made as usize);
    made
}

pub(crate) fn counters() -> String {
    let twins = |c: &Mutex<Option<HashMap<usize, usize>>>| {
        c.lock().unwrap().as_ref().map_or(0, HashMap::len)
    };
    let shaders = VERTEX_TWINS
        .lock()
        .unwrap()
        .as_ref()
        .map_or(0, |t| t.values().filter(|v| v.is_some()).count());
    format!(
        "vertex shader twins {shaders} (not made {}), depth state twins {}, raster twins {}, world depth clears {}, copies {}",
        TWINS_FAILED.load(Ordering::Relaxed),
        twins(&MIRRORED_DEPTH),
        twins(&MIRRORED_RASTER),
        CLEARS.load(Ordering::Relaxed),
        crate::copy::count()
    )
}
