// Copyright (c) 2026 TwistedModding. SPDX-License-Identifier: MPL-2.0-no-copyleft-exception
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not
// distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.
//! The immediate context's calls, mirrored while a world depth target is bound: each vertex shader
//! swapped for its twin (clip z = w - z, [`crate::twin`]), viewport depth ranges [a, b] made
//! [1 - b, 1 - a], depth tests LESS <-> GREATER, depth biases negated, depth clears 1 - value. Every
//! depth the world targets hold is then 1 - the vanilla value. Their standard copies
//! ([`crate::copy`]) are refreshed when a written world view is unbound (readers cannot sample a
//! depth target bound for writing) and after the engine copies the scene depth into another target.
//! Each precise twin receives private constants for the bound camera during its draw.
//!
//! Below a D3D11 wrapper (ENB) the hooks sit on the system's context instead of the game's, so the
//! wrapper's own draws, depth tests and depth reads are mirrored as the game's are.

use core::cell::Cell;
use core::ffi::c_void;
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};

use sre::raw::globals;
use windows::Win32::Foundation::HMODULE;
use windows::Win32::Graphics::Direct3D::{D3D_DRIVER_TYPE, D3D_DRIVER_TYPE_WARP};
use windows::Win32::Graphics::Direct3D11::{
    D3D11_BIND_CONSTANT_BUFFER, D3D11_BUFFER_DESC, D3D11_COMPARISON_FUNC, D3D11_COMPARISON_GREATER,
    D3D11_COMPARISON_GREATER_EQUAL, D3D11_COMPARISON_LESS, D3D11_COMPARISON_LESS_EQUAL,
    D3D11_DEPTH_STENCIL_DESC, D3D11_DEPTH_STENCIL_VIEW_DESC, D3D11_DEPTH_WRITE_MASK_ALL,
    D3D11_RASTERIZER_DESC, D3D11_SDK_VERSION, D3D11_USAGE_DEFAULT, D3D11_VIEWPORT, ID3D11Buffer,
    ID3D11DepthStencilState, ID3D11DepthStencilView, ID3D11Device, ID3D11DeviceContext,
    ID3D11DeviceContext_Vtbl, ID3D11PixelShader, ID3D11RasterizerState, ID3D11RenderTargetView,
    ID3D11Resource, ID3D11UnorderedAccessView, ID3D11VertexShader,
};
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
use windows::Win32::System::LibraryLoader::GetProcAddress;
use windows::core::{HRESULT, Interface, s};

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
type GetShaderFn =
    unsafe extern "system" fn(*mut c_void, *mut *mut c_void, *mut *mut c_void, *mut u32);
type CopyFn = unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void);
type CopyRegionFn = unsafe extern "system" fn(
    *mut c_void,
    *mut c_void,
    u32,
    u32,
    u32,
    u32,
    *mut c_void,
    u32,
    *const windows::Win32::Graphics::Direct3D11::D3D11_BOX,
);
type SetResourcesFn = unsafe extern "system" fn(*mut c_void, u32, u32, *const *mut c_void);
type DrawFn = unsafe extern "system" fn(*mut c_void, u32, u32);
type DrawIndexedFn = unsafe extern "system" fn(*mut c_void, u32, u32, i32);
type DrawInstancedFn = unsafe extern "system" fn(*mut c_void, u32, u32, u32, u32);
type DrawIndexedInstancedFn = unsafe extern "system" fn(*mut c_void, u32, u32, u32, i32, u32);
type DrawIndirectFn = unsafe extern "system" fn(*mut c_void, *mut c_void, u32);
type DrawAutoFn = unsafe extern "system" fn(*mut c_void);

const D3D11_CLEAR_DEPTH: u32 = 1;
/// Each precise twin receives [near / 15, A - 1, A near, orthographic]; A - 1 as A near / far,
/// the derivation Drawcall Limit Fix's twins share, so the two paths agree bit for bit.
/// The near plane the engine's far push was made for.
const VANILLA_NEAR: f32 = 15.0;
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
            remove_game_hooks();
        }
    };
}

#[cfg(test)]
pub(crate) fn install_test_hooks(context: &ID3D11DeviceContext) {
    IMMEDIATE.store(context.as_raw() as usize, Ordering::Relaxed);
    install_hooks(context.vtable()).unwrap();
}

hooks! {
    SET_TARGETS: SetTargetsFn = OMSetRenderTargets => set_targets,
    SET_TARGETS_UAV: SetTargetsUavFn = OMSetRenderTargetsAndUnorderedAccessViews => set_targets_uav,
    SET_DEPTH_STATE: SetDepthStateFn = OMSetDepthStencilState => set_depth_state,
    SET_RASTER: SetRasterFn = RSSetState => set_raster,
    SET_VIEWPORTS: SetViewportsFn = RSSetViewports => set_viewports,
    CLEAR_DEPTH: ClearDepthFn = ClearDepthStencilView => clear_depth,
    SET_VERTEX_SHADER: SetShaderFn = VSSetShader => set_vertex_shader,
    SET_PIXEL_SHADER: SetShaderFn = PSSetShader => set_pixel_shader,
    GET_VERTEX_SHADER: GetShaderFn = VSGetShader => get_vertex_shader,
    GET_PIXEL_SHADER: GetShaderFn = PSGetShader => get_pixel_shader,
    SET_PIXEL_RESOURCES: SetResourcesFn = PSSetShaderResources => set_pixel_resources,
    SET_COMPUTE_RESOURCES: SetResourcesFn = CSSetShaderResources => set_compute_resources,
    SET_VERTEX_RESOURCES: SetResourcesFn = VSSetShaderResources => set_vertex_resources,
    SET_GEOMETRY_RESOURCES: SetResourcesFn = GSSetShaderResources => set_geometry_resources,
    SET_HULL_RESOURCES: SetResourcesFn = HSSetShaderResources => set_hull_resources,
    SET_DOMAIN_RESOURCES: SetResourcesFn = DSSetShaderResources => set_domain_resources,
    COPY: CopyFn = CopyResource => copy_resource,
    COPY_REGION: CopyRegionFn = CopySubresourceRegion => copy_region,
    CLEAR_STATE: DrawAutoFn = ClearState => clear_state,
    DRAW: DrawFn = Draw => draw,
    DRAW_INDEXED: DrawIndexedFn = DrawIndexed => draw_indexed,
    DRAW_INSTANCED: DrawInstancedFn = DrawInstanced => draw_instanced,
    DRAW_INDEXED_INSTANCED: DrawIndexedInstancedFn = DrawIndexedInstanced => draw_indexed_instanced,
    DRAW_INDIRECT: DrawIndirectFn = DrawInstancedIndirect => draw_indirect,
    DRAW_INDEXED_INDIRECT: DrawIndirectFn = DrawIndexedInstancedIndirect => draw_indexed_indirect,
    DRAW_AUTO: DrawAutoFn = DrawAuto => draw_auto,
}

static IMMEDIATE: AtomicUsize = AtomicUsize::new(0);
/// A world depth view is bound and the calls sent to the context are mirrored.
static BOUND: AtomicBool = AtomicBool::new(false);
static LAST_DEPTH_STATE: AtomicUsize = AtomicUsize::new(0);
static LAST_STENCIL_REF: AtomicU32 = AtomicU32::new(0);
static LAST_RASTER: AtomicUsize = AtomicUsize::new(0);
static LAST_VIEWPORTS: Mutex<Vec<D3D11_VIEWPORT>> = Mutex::new(Vec::new());
static LAST_VERTEX_SHADER: Mutex<Option<ID3D11VertexShader>> = Mutex::new(None);
static LAST_PIXEL_SHADER: Mutex<Option<ID3D11PixelShader>> = Mutex::new(None);
static VERTEX_INSTANCES: AtomicU32 = AtomicU32::new(0);
static PIXEL_INSTANCES: AtomicU32 = AtomicU32::new(0);
static DRAWING: AtomicBool = AtomicBool::new(false);
static DEPTH_TESTS: AtomicBool = AtomicBool::new(true);
static VERTEX_CONSTANT_SLOT: AtomicU32 = AtomicU32::new(u32::MAX);
static LAST_DEPTH_VIEW: Mutex<Option<ID3D11DepthStencilView>> = Mutex::new(None);
/// The engine's current depth state writes depth.
static WRITES: AtomicBool = AtomicBool::new(true);
/// The bound world target.
static BOUND_RESOURCE: AtomicUsize = AtomicUsize::new(0);
static MIRRORED_DEPTH: Mutex<Option<HashMap<usize, usize>>> = Mutex::new(None);
static MIRRORED_RASTER: Mutex<Option<HashMap<usize, usize>>> = Mutex::new(None);
static DEPTH_MODES: Mutex<Option<HashMap<usize, (bool, bool)>>> = Mutex::new(None);
struct VertexConstants {
    buffer: ID3D11Buffer,
    sent: [u32; 4],
}
static VERTEX_CONSTANTS: Mutex<Option<VertexConstants>> = Mutex::new(None);
static CAMERA_STATE: AtomicUsize = AtomicUsize::new(0);
static CLEARS: AtomicU64 = AtomicU64::new(0);

pub(crate) fn original<F: Copy>(slot: &Mutex<Option<hook::Hook<F>>>) -> F {
    slot.lock()
        .unwrap()
        .as_ref()
        .expect("hooks are removed only after the plugin is off")
        .original()
}

pub(crate) fn device() -> Option<ID3D11Device> {
    let context = IMMEDIATE.load(Ordering::Relaxed) as *mut c_void;
    if let Some(context) = unsafe { ID3D11DeviceContext::from_raw_borrowed(&context) } {
        return unsafe { context.GetDevice() }.ok();
    }
    let p = unsafe { *(crate::address(sre::id!(globals::g_D3D11Device))? as *const *mut c_void) };
    unsafe { ID3D11Device::from_raw_borrowed(&p) }.cloned()
}

pub(crate) fn immediate() -> Option<ID3D11DeviceContext> {
    let p = unsafe { *(crate::address(sre::id!(globals::g_D3D11Context))? as *const *mut c_void) };
    unsafe { ID3D11DeviceContext::from_raw_borrowed(&p) }.cloned()
}

/// The record of depth target `target` in `g_Renderer`.
fn target_record(target: usize) -> Option<usize> {
    Some(
        crate::address(sre::id!(globals::g_Renderer))?
            + sre::member_offset!("Renderer.depthStencilTargets")
            + target * sre::class_size!("DepthStencilRecord", sre::DepthStencilRecord),
    )
}

/// The texture pointer of depth target `target`.
pub(crate) fn target_texture(target: usize) -> Option<*mut usize> {
    Some((target_record(target)? + sre::member_offset!("DepthStencilRecord.texture")) as *mut usize)
}

pub(crate) fn install() -> Result<(), String> {
    let context = immediate().ok_or("no immediate context")?;
    CAMERA_STATE.store(
        crate::address(sre::id!(globals::g_CurrentCameraState)).ok_or("no camera state")?,
        Ordering::Relaxed,
    );
    let record = target_record(0).ok_or("no renderer")?;
    let view =
        unsafe { *((record + sre::member_offset!("DepthStencilRecord.dsv")) as *const *mut c_void) };
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
    match install_below_wrapper(&context) {
        Ok(true) => {
            crate::log("the game's D3D11 context is a wrapper's; reversed depth runs on the system's below it");
            return Ok(());
        }
        Ok(false) => {}
        Err(e) => {
            remove();
            return Err(format!("system context below the game's: {e}"));
        }
    }
    IMMEDIATE.store(context.as_raw() as usize, Ordering::Relaxed);
    let device = device().ok_or("no device")?;
    create_vertex_constants(&device)?;
    crate::copy::initialize(&device)?;
    crate::copy::sync_sources();
    if let Err(e) = crate::shader::install_device(&device) {
        crate::log(&format!("replacement shader capture: {e}"));
    }
    if let Err(e) = install_hooks(context.vtable()) {
        remove();
        return Err(e);
    }
    Ok(())
}

type CreateDeviceFn = unsafe extern "system" fn(
    *mut c_void,
    D3D_DRIVER_TYPE,
    HMODULE,
    u32,
    *const c_void,
    u32,
    u32,
    *mut *mut c_void,
    *mut c_void,
    *mut *mut c_void,
) -> HRESULT;

/// The hooks are on the system's context below a wrapper's.
static BELOW_WRAPPER: AtomicBool = AtomicBool::new(false);
static GAME_TARGETS: Mutex<Option<hook::Hook<SetTargetsFn>>> = Mutex::new(None);
static GAME_TARGETS_UAV: Mutex<Option<hook::Hook<SetTargetsUavFn>>> = Mutex::new(None);
/// Each of the game's depth textures with the system's texture behind it.
static SYSTEM_TEXTURES: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());
static SYSTEM_SETUP_FAILED: AtomicBool = AtomicBool::new(false);
thread_local! {
    /// The depth view a bind of the game's is passing down through the wrapper.
    static GAME_VIEW: Cell<usize> = const { Cell::new(0) };
}

/// Runs `f` on the system D3D11's immediate context table, read from a WARP device of its own that
/// lives as long as the call (the table belongs to the device).
fn with_system_table<R>(f: impl FnOnce(&ID3D11DeviceContext_Vtbl) -> R) -> Option<R> {
    let module = crate::shader::system_d3d11()?;
    let create: CreateDeviceFn =
        unsafe { core::mem::transmute(GetProcAddress(module, s!("D3D11CreateDevice"))?) };
    let (mut device, mut context) = (core::ptr::null_mut(), core::ptr::null_mut());
    unsafe {
        create(
            core::ptr::null_mut(),
            D3D_DRIVER_TYPE_WARP,
            HMODULE::default(),
            0,
            core::ptr::null(),
            0,
            D3D11_SDK_VERSION,
            &raw mut device,
            core::ptr::null_mut(),
            &raw mut context,
        )
    }
    .ok()
    .ok()?;
    let _device = (!device.is_null()).then(|| unsafe { ID3D11Device::from_raw(device) });
    let context =
        (!context.is_null()).then(|| unsafe { ID3D11DeviceContext::from_raw(context) })?;
    Some(f(context.vtable()))
}

/// Hooks the system's context when the game's is a wrapper's (ENB): the wrapper draws with
/// shaders and depth tests of its own there, past the game's table. The game's table keeps only
/// its target binds, which name the system's immediate context and textures on their way down.
fn install_below_wrapper(game: &ID3D11DeviceContext) -> Result<bool, String> {
    if crate::shader::dxvk().is_some() {
        return Ok(false);
    }
    let game = game.vtable();
    with_system_table(|table| {
        if table.OMSetDepthStencilState as usize == game.OMSetDepthStencilState as usize {
            return Ok(false);
        }
        install_hooks(table)?;
        install_game_hook(&GAME_TARGETS, game.OMSetRenderTargets as usize, game_set_targets as SetTargetsFn)?;
        install_game_hook(
            &GAME_TARGETS_UAV,
            game.OMSetRenderTargetsAndUnorderedAccessViews as usize,
            game_set_targets_uav as SetTargetsUavFn,
        )?;
        BELOW_WRAPPER.store(true, Ordering::Relaxed);
        Ok(true)
    })
    .unwrap_or(Ok(false))
}

fn install_game_hook<F: Copy>(
    slot: &Mutex<Option<hook::Hook<F>>>,
    target: usize,
    detour: F,
) -> Result<(), String> {
    let hook = unsafe { hook::Hook::create(target, detour) }.map_err(|e| e.to_string())?;
    slot.lock().unwrap().insert(hook).enable().map_err(|e| e.to_string())
}

fn remove_game_hooks() {
    if let Some(h) = GAME_TARGETS.lock().unwrap().take() {
        let _ = unsafe { h.remove() };
    }
    if let Some(h) = GAME_TARGETS_UAV.lock().unwrap().take() {
        let _ = unsafe { h.remove() };
    }
}

unsafe extern "system" fn game_set_targets(
    context: *mut c_void,
    count: u32,
    views: *const *mut c_void,
    depth: *mut c_void,
) {
    GAME_VIEW.set(depth as usize);
    unsafe { original(&GAME_TARGETS)(context, count, views, depth) };
    GAME_VIEW.set(0);
}

unsafe extern "system" fn game_set_targets_uav(
    context: *mut c_void,
    count: u32,
    views: *const *mut c_void,
    depth: *mut c_void,
    uav_start: u32,
    uav_count: u32,
    uavs: *const *mut c_void,
    counts: *const u32,
) {
    GAME_VIEW.set(if count == KEEP_TARGETS { 0 } else { depth as usize });
    unsafe {
        original(&GAME_TARGETS_UAV)(
            context, count, views, depth, uav_start, uav_count, uavs, counts,
        );
    };
    GAME_VIEW.set(0);
}

/// Below a wrapper, a bind of the game's reaching the system's context: the first names the
/// system's immediate context, which the objects of this module are then made on; each pairs the
/// game's depth texture with the system's.
fn learn(context: *mut c_void, view: *mut c_void) {
    let game = GAME_VIEW.replace(0);
    if game == 0 {
        return;
    }
    if IMMEDIATE.load(Ordering::Relaxed) == 0 {
        if SYSTEM_SETUP_FAILED.load(Ordering::Relaxed) {
            return;
        }
        let device = unsafe { ID3D11DeviceContext::from_raw_borrowed(&context) }
            .and_then(|c| unsafe { c.GetDevice() }.ok());
        let made = device.ok_or_else(|| "no device".to_owned()).and_then(|device| {
            create_vertex_constants(&device)?;
            crate::copy::initialize(&device)
        });
        if let Err(e) = made {
            SYSTEM_SETUP_FAILED.store(true, Ordering::Relaxed);
            crate::log(&format!("not engaged on the system's context: {e}"));
            return;
        }
        IMMEDIATE.store(context as usize, Ordering::Relaxed);
    }
    if context as usize != IMMEDIATE.load(Ordering::Relaxed) {
        return;
    }
    if let (Some(game), Some(system)) = (depth_resource(game as *mut c_void), depth_resource(view)) {
        let pair = (game.as_raw() as usize, system.as_raw() as usize);
        let mut textures = SYSTEM_TEXTURES.lock().unwrap();
        if !textures.contains(&pair) {
            textures.retain(|&(game, _)| game != pair.0);
            textures.push(pair);
        }
    }
}

/// The texture of depth target `target` as the hooks see it: the engine's, or below a wrapper the
/// system's behind it once a bind has named it.
pub(crate) fn world_texture(target: usize) -> usize {
    let game = target_texture(target).map_or(0, |field| unsafe { *field });
    if !BELOW_WRAPPER.load(Ordering::Relaxed) {
        return game;
    }
    SYSTEM_TEXTURES
        .lock()
        .unwrap()
        .iter()
        .find(|&&(engine, _)| engine == game)
        .map_or(0, |&(_, system)| system)
}

fn depth_resource(view: *mut c_void) -> Option<ID3D11Resource> {
    let view = unsafe { ID3D11DepthStencilView::from_raw_borrowed(&view) }?;
    unsafe { view.GetResource() }.ok()
}

/// Refreshes a world target's copy when a written view of it was unbound, and sends the last states
/// and vertex shader again once the bound depth view switched between world and other.
fn bind(context: *mut c_void, view: *mut c_void) {
    crate::copy::sync_sources();
    let next = unsafe { ID3D11DepthStencilView::from_raw_borrowed(&view) }.cloned();
    let previous = core::mem::replace(&mut *LAST_DEPTH_VIEW.lock().unwrap(), next);
    let previous_raw = previous
        .as_ref()
        .map_or(core::ptr::null_mut(), Interface::as_raw);
    if previous_raw != view
        && let Some(resource) = depth_resource(previous_raw)
    {
        let context = unsafe { ID3D11DeviceContext::from_raw_borrowed(&context) }.unwrap();
        crate::copy::run(context, &resource);
    }
    let resource = depth_resource(view);
    BOUND_RESOURCE.store(
        resource
            .as_ref()
            .map_or(0, |resource| resource.as_raw() as usize),
        Ordering::Relaxed,
    );
    let bound = resource.as_ref().is_some_and(crate::copy::reversed);
    if BOUND.swap(bound, Ordering::Relaxed) == bound {
        return;
    }
    if let (true, Some(resource)) = (bound, &resource) {
        let context = unsafe { ID3D11DeviceContext::from_raw_borrowed(&context) }.unwrap();
        crate::copy::ensure(context, resource);
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
    }
}

fn with_shaders(context: *mut c_void, draw: impl FnOnce()) {
    if !BOUND.load(Ordering::Relaxed) || !DEPTH_TESTS.load(Ordering::Relaxed) {
        draw();
        return;
    }
    let vertex = LAST_VERTEX_SHADER.lock().unwrap().clone();
    let pixel = LAST_PIXEL_SHADER.lock().unwrap().clone();
    let sources = [
        vertex
            .as_ref()
            .map_or(core::ptr::null_mut(), Interface::as_raw),
        pixel
            .as_ref()
            .map_or(core::ptr::null_mut(), Interface::as_raw),
    ];
    let stages = [
        (
            &SET_VERTEX_SHADER,
            &VERTEX_INSTANCES,
            vertex_shader as fn(*mut c_void) -> *mut c_void,
        ),
        (
            &SET_PIXEL_SHADER,
            &PIXEL_INSTANCES,
            pixel_shader as fn(*mut c_void) -> *mut c_void,
        ),
    ];
    let previous = DRAWING.swap(true, Ordering::Relaxed);
    let mut changed = [false; 2];
    VERTEX_CONSTANT_SLOT.store(u32::MAX, Ordering::Relaxed);
    for (i, (hook, instances, rewrite)) in stages.iter().enumerate() {
        if instances.load(Ordering::Relaxed) != 0 {
            crate::copy::report("dynamic shader linkage is unsupported for reversed depth");
        } else if !sources[i].is_null() {
            unsafe { original(hook)(context, rewrite(sources[i]), core::ptr::null(), 0) };
            changed[i] = true;
        }
    }
    with_vertex_constants(context, draw);
    DRAWING.store(previous, Ordering::Relaxed);
    for (i, (hook, _, _)) in stages.iter().enumerate() {
        if changed[i] {
            let source = if previous {
                stages[i].2(sources[i])
            } else {
                sources[i]
            };
            unsafe { original(hook)(context, source, core::ptr::null(), 0) };
        }
    }
}

fn with_vertex_constants(context: *mut c_void, draw: impl FnOnce()) {
    let slot = VERTEX_CONSTANT_SLOT.load(Ordering::Relaxed);
    if slot == u32::MAX || !BOUND.load(Ordering::Relaxed) {
        draw();
        return;
    }
    let context = unsafe { ID3D11DeviceContext::from_raw_borrowed(&context) }.unwrap();
    let state = CAMERA_STATE.load(Ordering::Relaxed);
    let at = sre::member_offset!("CurrentCameraState.projection");
    let projection = unsafe { ((state + at) as *const [f32; 16]).read_unaligned() };
    let near = skyrim::render::depth_shift_near();
    let far = skyrim::render::depth_shift_far();
    let scale = if near > 0.0 { near / VANILLA_NEAR } else { 1.0 };
    let a_near = -projection[14];
    let a_less_one = if far > 0.0 { a_near / far } else { projection[10] - 1.0 };
    let data = [scale, a_less_one, a_near, projection[15]];
    draw_with_constants(context, slot, data, draw);
}

pub(crate) fn create_vertex_constants(device: &ID3D11Device) -> Result<(), String> {
    let desc = D3D11_BUFFER_DESC {
        ByteWidth: 16,
        Usage: D3D11_USAGE_DEFAULT,
        BindFlags: D3D11_BIND_CONSTANT_BUFFER.0 as u32,
        ..Default::default()
    };
    let mut buffer = None;
    unsafe { device.CreateBuffer(&raw const desc, None, Some(&raw mut buffer)) }
        .map_err(|e| format!("vertex depth constants: {e}"))?;
    *VERTEX_CONSTANTS.lock().unwrap() = Some(VertexConstants {
        buffer: buffer.ok_or("no vertex depth constants")?,
        sent: [0; 4],
    });
    Ok(())
}

pub(crate) fn draw_with_constants(
    context: &ID3D11DeviceContext,
    slot: u32,
    data: [f32; 4],
    draw: impl FnOnce(),
) {
    let bits = data.map(f32::to_bits);
    let mut constants = VERTEX_CONSTANTS.lock().unwrap();
    let constants = constants.as_mut().unwrap();
    if constants.sent != bits {
        unsafe {
            context.UpdateSubresource(&constants.buffer, 0, None, data.as_ptr().cast(), 0, 0);
        };
        constants.sent = bits;
    }
    let mut previous: [Option<ID3D11Buffer>; 1] = [None];
    unsafe {
        context.VSGetConstantBuffers(slot, Some(&mut previous));
        context.VSSetConstantBuffers(slot, Some(&[Some(constants.buffer.clone())]));
    }
    draw();
    unsafe { context.VSSetConstantBuffers(slot, Some(&previous)) };
}

macro_rules! immediate_only {
    ($original:ident, $context:ident $(, $argument:ident)*) => {
        if $context as usize != IMMEDIATE.load(Ordering::Relaxed) || crate::copy::internal() {
            unsafe { original(&$original)($context $(, $argument)*) };
            return;
        }
    };
}

pub(crate) fn release_push_scale() {
    VERTEX_CONSTANTS.lock().unwrap().take();
    LAST_VERTEX_SHADER.lock().unwrap().take();
    LAST_PIXEL_SHADER.lock().unwrap().take();
    LAST_DEPTH_VIEW.lock().unwrap().take();
}

/// Runs `f` with no output targets bound, past the bind bookkeeping, and binds them again after.
pub(crate) fn without_targets(context: &ID3D11DeviceContext, f: impl FnOnce()) {
    let mut views: [Option<ID3D11RenderTargetView>; 8] = Default::default();
    let mut depth: Option<ID3D11DepthStencilView> = None;
    unsafe { context.OMGetRenderTargets(Some(&mut views), Some(&raw mut depth)) };
    let count = views.iter().rposition(Option::is_some).map_or(0, |i| i + 1);
    let mut unordered: [Option<ID3D11UnorderedAccessView>; 8] = Default::default();
    if count < 8 {
        unsafe {
            context.OMGetRenderTargetsAndUnorderedAccessViews(
                None,
                None,
                u32::try_from(count).unwrap(),
                Some(&mut unordered[count..]),
            );
        };
    }
    let set = original(&SET_TARGETS_UAV);
    let empty = [core::ptr::null_mut(); 8];
    unsafe {
        set(
            context.as_raw(),
            0,
            core::ptr::null(),
            core::ptr::null_mut(),
            0,
            8,
            empty.as_ptr(),
            core::ptr::null(),
        );
    };
    f();
    let raw = views
        .each_ref()
        .map(|v| v.as_ref().map_or(core::ptr::null_mut(), Interface::as_raw));
    let uavs = unordered
        .each_ref()
        .map(|v| v.as_ref().map_or(core::ptr::null_mut(), Interface::as_raw));
    let counters = [u32::MAX; 8];
    unsafe {
        set(
            context.as_raw(),
            u32::try_from(count).unwrap_or(0),
            raw.as_ptr(),
            depth
                .as_ref()
                .map_or(core::ptr::null_mut(), Interface::as_raw),
            u32::try_from(count).unwrap(),
            u32::try_from(8 - count).unwrap(),
            uavs[count..].as_ptr(),
            counters.as_ptr(),
        );
    };
}

unsafe extern "system" fn set_targets(
    context: *mut c_void,
    count: u32,
    views: *const *mut c_void,
    depth: *mut c_void,
) {
    learn(context, depth);
    immediate_only!(SET_TARGETS, context, count, views, depth);
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
    if count != KEEP_TARGETS {
        learn(context, depth);
    }
    immediate_only!(
        SET_TARGETS_UAV,
        context,
        count,
        views,
        depth,
        uav_start,
        uav_count,
        uavs,
        counts
    );
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
    immediate_only!(SET_DEPTH_STATE, context, state, stencil_ref);
    LAST_DEPTH_STATE.store(state as usize, Ordering::Relaxed);
    LAST_STENCIL_REF.store(stencil_ref, Ordering::Relaxed);
    let (tests, writes) = depth_mode(state);
    let changed = DEPTH_TESTS.swap(tests, Ordering::Relaxed) != tests;
    WRITES.store(writes, Ordering::Relaxed);
    unsafe { original(&SET_DEPTH_STATE)(context, depth_state(state), stencil_ref) };
    if changed && BOUND.load(Ordering::Relaxed) {
        unsafe {
            original(&SET_RASTER)(
                context,
                raster(LAST_RASTER.load(Ordering::Relaxed) as *mut c_void),
            );
        };
        let viewports = viewports(&LAST_VIEWPORTS.lock().unwrap());
        unsafe {
            original(&SET_VIEWPORTS)(
                context,
                u32::try_from(viewports.len()).unwrap(),
                viewports.as_ptr(),
            );
        };
    }
}

unsafe extern "system" fn set_raster(context: *mut c_void, state: *mut c_void) {
    immediate_only!(SET_RASTER, context, state);
    LAST_RASTER.store(state as usize, Ordering::Relaxed);
    unsafe { original(&SET_RASTER)(context, raster(state)) };
}

unsafe extern "system" fn set_viewports(
    context: *mut c_void,
    count: u32,
    list: *const D3D11_VIEWPORT,
) {
    immediate_only!(SET_VIEWPORTS, context, count, list);
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
    immediate_only!(CLEAR_DEPTH, context, view, flags, depth, stencil);
    let resource = depth_resource(view);
    let mirror = crate::active()
        && resource.as_ref().is_some_and(|resource| {
            crate::copy::seeded_world(resource) || crate::copy::reversed(resource)
        });
    unsafe {
        original(&CLEAR_DEPTH)(
            context,
            view,
            flags,
            if mirror { 1.0 - depth } else { depth },
            stencil,
        );
    };
    if flags & D3D11_CLEAR_DEPTH != 0
        && let Some(resource) = resource
    {
        crate::copy::mark(&resource, mirror);
        if mirror {
            CLEARS.fetch_add(1, Ordering::Relaxed);
        }
        if resource.as_raw() as usize == BOUND_RESOURCE.load(Ordering::Relaxed) {
            bind(context, view);
        }
    }
}

unsafe extern "system" fn set_vertex_shader(
    context: *mut c_void,
    shader: *mut c_void,
    instances: *const *mut c_void,
    count: u32,
) {
    immediate_only!(SET_VERTEX_SHADER, context, shader, instances, count);
    *LAST_VERTEX_SHADER.lock().unwrap() =
        unsafe { ID3D11VertexShader::from_raw_borrowed(&shader) }.cloned();
    VERTEX_INSTANCES.store(count, Ordering::Relaxed);
    let sent = if count == 0 && DRAWING.load(Ordering::Relaxed) {
        vertex_shader(shader)
    } else {
        VERTEX_CONSTANT_SLOT.store(u32::MAX, Ordering::Relaxed);
        shader
    };
    unsafe { original(&SET_VERTEX_SHADER)(context, sent, instances, count) };
}

fn pixel_shader(shader: *mut c_void) -> *mut c_void {
    if !BOUND.load(Ordering::Relaxed) || shader.is_null() {
        return shader;
    }
    let original = unsafe { ID3D11PixelShader::from_raw_borrowed(&shader) }.unwrap();
    crate::shader::pixel(original).map_or(shader, |twin| twin.as_raw())
}

unsafe extern "system" fn set_pixel_shader(
    context: *mut c_void,
    shader: *mut c_void,
    instances: *const *mut c_void,
    count: u32,
) {
    immediate_only!(SET_PIXEL_SHADER, context, shader, instances, count);
    *LAST_PIXEL_SHADER.lock().unwrap() =
        unsafe { ID3D11PixelShader::from_raw_borrowed(&shader) }.cloned();
    PIXEL_INSTANCES.store(count, Ordering::Relaxed);
    let sent = if count == 0 && DRAWING.load(Ordering::Relaxed) {
        pixel_shader(shader)
    } else {
        shader
    };
    unsafe { original(&SET_PIXEL_SHADER)(context, sent, instances, count) };
}

macro_rules! shader_getter {
    ($name:ident, $hook:ident, $last:ident, $shader:ty) => {
        unsafe extern "system" fn $name(
            context: *mut c_void,
            shader: *mut *mut c_void,
            instances: *mut *mut c_void,
            count: *mut u32,
        ) {
            immediate_only!($hook, context, shader, instances, count);
            unsafe { original(&$hook)(context, shader, instances, count) };
            if DRAWING.load(Ordering::Relaxed) && !shader.is_null() {
                let logical = $last.lock().unwrap().clone();
                if !unsafe { *shader }.is_null() {
                    drop(unsafe { <$shader>::from_raw(*shader) });
                }
                unsafe { *shader = logical.map_or(core::ptr::null_mut(), Interface::into_raw) };
            }
        }
    };
}
shader_getter!(
    get_vertex_shader,
    GET_VERTEX_SHADER,
    LAST_VERTEX_SHADER,
    ID3D11VertexShader
);
shader_getter!(
    get_pixel_shader,
    GET_PIXEL_SHADER,
    LAST_PIXEL_SHADER,
    ID3D11PixelShader
);

#[cfg(test)]
pub(crate) fn test_shader_readback(
    device: &ID3D11Device,
    context: &ID3D11DeviceContext,
    depth: &ID3D11DepthStencilView,
) {
    let code = crate::twin::tests::compile(crate::twin::tests::CONSTANT);
    let mut vertex = None;
    unsafe { device.CreateVertexShader(&code, None, Some(&raw mut vertex)) }.unwrap();
    let vertex = vertex.unwrap();
    let code = crate::twin::tests::compile_stage(
        "float4 main(float4 p : SV_Position) : SV_Target { return p.zzzz; }",
        windows::core::s!("ps_5_0"),
    );
    let mut pixel = None;
    unsafe { device.CreatePixelShader(&code, None, Some(&raw mut pixel)) }.unwrap();
    let pixel = pixel.unwrap();
    crate::copy::mark(&unsafe { depth.GetResource() }.unwrap(), true);
    unsafe {
        context.VSSetShader(&vertex, None);
        context.PSSetShader(&pixel, None);
        context.OMSetRenderTargets(None, depth);
    }
    macro_rules! check {
        ($get:ident, $set:ident, $hook:ident, $expected:ident, $shader:ty) => {
            let mut restored = None;
            unsafe { context.$get(&raw mut restored, None, None) };
            assert_eq!(
                restored.as_ref().map(Interface::as_raw),
                Some($expected.as_raw())
            );
            unsafe { context.$set(restored.as_ref(), None) };
            let mut native = core::ptr::null_mut();
            unsafe {
                original(&$hook)(
                    context.as_raw(),
                    &raw mut native,
                    core::ptr::null_mut(),
                    core::ptr::null_mut(),
                )
            };
            let native = unsafe { <$shader>::from_raw(native) };
            assert_ne!(native.as_raw(), $expected.as_raw());
        };
    }
    with_shaders(context.as_raw(), || {
        with_shaders(context.as_raw(), || {
            check!(
                VSGetShader,
                VSSetShader,
                GET_VERTEX_SHADER,
                vertex,
                ID3D11VertexShader
            );
            check!(
                PSGetShader,
                PSSetShader,
                GET_PIXEL_SHADER,
                pixel,
                ID3D11PixelShader
            );
        });
        check!(
            VSGetShader,
            VSSetShader,
            GET_VERTEX_SHADER,
            vertex,
            ID3D11VertexShader
        );
        check!(
            PSGetShader,
            PSSetShader,
            GET_PIXEL_SHADER,
            pixel,
            ID3D11PixelShader
        );
    });
    let mut restored_vertex = None;
    let mut restored_pixel = None;
    unsafe {
        context.VSGetShader(&raw mut restored_vertex, None, None);
        context.PSGetShader(&raw mut restored_pixel, None, None);
    }
    assert_eq!(
        restored_vertex.as_ref().map(Interface::as_raw),
        Some(vertex.as_raw())
    );
    assert_eq!(
        restored_pixel.as_ref().map(Interface::as_raw),
        Some(pixel.as_raw())
    );
    unsafe { context.ClearState() };
}

macro_rules! resource_hook {
    ($name:ident, $slot:ident) => {
        unsafe extern "system" fn $name(
            context: *mut c_void,
            start: u32,
            count: u32,
            views: *const *mut c_void,
        ) {
            immediate_only!($slot, context, start, count, views);
            let swapped = substituted(context, count, views);
            unsafe {
                original(&$slot)(
                    context,
                    start,
                    count,
                    swapped.as_ref().map_or(views, Vec::as_ptr),
                )
            };
        }
    };
}
resource_hook!(set_pixel_resources, SET_PIXEL_RESOURCES);
resource_hook!(set_compute_resources, SET_COMPUTE_RESOURCES);
resource_hook!(set_vertex_resources, SET_VERTEX_RESOURCES);
resource_hook!(set_geometry_resources, SET_GEOMETRY_RESOURCES);
resource_hook!(set_hull_resources, SET_HULL_RESOURCES);
resource_hook!(set_domain_resources, SET_DOMAIN_RESOURCES);

/// Shader views with a reversed depth target's own view replaced by its copy's.
fn substituted(
    context: *mut c_void,
    count: u32,
    views: *const *mut c_void,
) -> Option<Vec<*mut c_void>> {
    if views.is_null() {
        return None;
    }
    let given = unsafe { core::slice::from_raw_parts(views, count as usize) };
    let context = unsafe { ID3D11DeviceContext::from_raw_borrowed(&context) }.unwrap();
    given
        .iter()
        .any(|&v| crate::copy::stand_in(context, v).is_some())
        .then(|| {
            given
                .iter()
                .map(|&v| crate::copy::stand_in(context, v).unwrap_or(v))
                .collect()
        })
}

/// Binds a compute shader view as given.
pub(crate) fn set_compute_resource_as_is(context: &ID3D11DeviceContext, view: *mut c_void) {
    unsafe { original(&SET_COMPUTE_RESOURCES)(context.as_raw(), 0, 1, &raw const view) };
}

unsafe extern "system" fn copy_resource(context: *mut c_void, dst: *mut c_void, src: *mut c_void) {
    immediate_only!(COPY, context, dst, src);
    unsafe { original(&COPY)(context, dst, src) };
    if let (Some(destination), Some(source), Some(context)) = unsafe {
        (
            ID3D11Resource::from_raw_borrowed(&dst),
            ID3D11Resource::from_raw_borrowed(&src),
            ID3D11DeviceContext::from_raw_borrowed(&context),
        )
    } {
        crate::copy::copied(context, destination, source);
        refresh_target_binding(context.as_raw(), destination.as_raw());
    }
}

fn refresh_target_binding(context: *mut c_void, destination: *mut c_void) {
    if destination as usize == BOUND_RESOURCE.load(Ordering::Relaxed) {
        let view = LAST_DEPTH_VIEW.lock().unwrap().clone();
        bind(
            context,
            view.as_ref()
                .map_or(core::ptr::null_mut(), Interface::as_raw),
        );
    }
}

#[allow(clippy::too_many_arguments)]
unsafe extern "system" fn copy_region(
    context: *mut c_void,
    destination: *mut c_void,
    destination_subresource: u32,
    x: u32,
    y: u32,
    z: u32,
    source: *mut c_void,
    source_subresource: u32,
    region: *const windows::Win32::Graphics::Direct3D11::D3D11_BOX,
) {
    immediate_only!(
        COPY_REGION,
        context,
        destination,
        destination_subresource,
        x,
        y,
        z,
        source,
        source_subresource,
        region
    );
    let borrowed = unsafe {
        (
            ID3D11DeviceContext::from_raw_borrowed(&context),
            ID3D11Resource::from_raw_borrowed(&destination),
            ID3D11Resource::from_raw_borrowed(&source),
        )
    };
    let empty = unsafe { region.as_ref() }.is_some_and(|region| {
        region.left >= region.right || region.top >= region.bottom || region.front >= region.back
    });
    let mut track = !empty;
    if !empty
        && let (Some(context), Some(destination), Some(source)) = borrowed
        && let Err(error) = crate::copy::prepare_region(
            context,
            destination,
            source,
            destination_subresource,
            [x, y, z],
            unsafe { region.as_ref() },
        )
    {
        crate::copy::report(&error);
        track = false;
    }
    unsafe {
        original(&COPY_REGION)(
            context,
            destination,
            destination_subresource,
            x,
            y,
            z,
            source,
            source_subresource,
            region,
        );
    };
    if track && let (Some(context), Some(destination), Some(source)) = borrowed {
        crate::copy::copied_region(context, destination, source);
        refresh_target_binding(context.as_raw(), destination.as_raw());
    }
}

unsafe extern "system" fn clear_state(context: *mut c_void) {
    immediate_only!(CLEAR_STATE, context);
    let previous = LAST_DEPTH_VIEW.lock().unwrap().take();
    unsafe { original(&CLEAR_STATE)(context) };
    BOUND.store(false, Ordering::Relaxed);
    BOUND_RESOURCE.store(0, Ordering::Relaxed);
    LAST_VERTEX_SHADER.lock().unwrap().take();
    LAST_PIXEL_SHADER.lock().unwrap().take();
    LAST_DEPTH_STATE.store(0, Ordering::Relaxed);
    LAST_STENCIL_REF.store(0, Ordering::Relaxed);
    LAST_RASTER.store(0, Ordering::Relaxed);
    LAST_VIEWPORTS.lock().unwrap().clear();
    VERTEX_CONSTANT_SLOT.store(u32::MAX, Ordering::Relaxed);
    WRITES.store(true, Ordering::Relaxed);
    DEPTH_TESTS.store(true, Ordering::Relaxed);
    VERTEX_INSTANCES.store(0, Ordering::Relaxed);
    PIXEL_INSTANCES.store(0, Ordering::Relaxed);
    if let Some(view) = previous
        && let Ok(resource) = unsafe { view.GetResource() }
    {
        crate::copy::run(
            unsafe { ID3D11DeviceContext::from_raw_borrowed(&context) }.unwrap(),
            &resource,
        );
    }
}

fn drawn() {
    let resource = BOUND_RESOURCE.load(Ordering::Relaxed) as *mut c_void;
    if WRITES.load(Ordering::Relaxed)
        && let Some(resource) = unsafe { ID3D11Resource::from_raw_borrowed(&resource) }
    {
        crate::copy::mark(resource, BOUND.load(Ordering::Relaxed));
    }
}

unsafe extern "system" fn draw(context: *mut c_void, vertices: u32, start: u32) {
    immediate_only!(DRAW, context, vertices, start);
    drawn();
    with_shaders(context, || unsafe {
        original(&DRAW)(context, vertices, start);
    });
}

unsafe extern "system" fn draw_indexed(context: *mut c_void, indices: u32, start: u32, base: i32) {
    immediate_only!(DRAW_INDEXED, context, indices, start, base);
    drawn();
    with_shaders(context, || unsafe {
        original(&DRAW_INDEXED)(context, indices, start, base);
    });
}

unsafe extern "system" fn draw_instanced(
    context: *mut c_void,
    vertices: u32,
    instances: u32,
    start: u32,
    first: u32,
) {
    immediate_only!(DRAW_INSTANCED, context, vertices, instances, start, first);
    drawn();
    with_shaders(context, || unsafe {
        original(&DRAW_INSTANCED)(context, vertices, instances, start, first);
    });
}

unsafe extern "system" fn draw_indexed_instanced(
    context: *mut c_void,
    indices: u32,
    instances: u32,
    start: u32,
    base: i32,
    first: u32,
) {
    immediate_only!(
        DRAW_INDEXED_INSTANCED,
        context,
        indices,
        instances,
        start,
        base,
        first
    );
    drawn();
    with_shaders(context, || unsafe {
        original(&DRAW_INDEXED_INSTANCED)(context, indices, instances, start, base, first);
    });
}

unsafe extern "system" fn draw_indirect(context: *mut c_void, arguments: *mut c_void, offset: u32) {
    immediate_only!(DRAW_INDIRECT, context, arguments, offset);
    drawn();
    with_shaders(context, || unsafe {
        original(&DRAW_INDIRECT)(context, arguments, offset);
    });
}

unsafe extern "system" fn draw_indexed_indirect(
    context: *mut c_void,
    arguments: *mut c_void,
    offset: u32,
) {
    immediate_only!(DRAW_INDEXED_INDIRECT, context, arguments, offset);
    drawn();
    with_shaders(context, || unsafe {
        original(&DRAW_INDEXED_INDIRECT)(context, arguments, offset);
    });
}

unsafe extern "system" fn draw_auto(context: *mut c_void) {
    immediate_only!(DRAW_AUTO, context);
    drawn();
    with_shaders(context, || unsafe { original(&DRAW_AUTO)(context) });
}

fn viewports(given: &[D3D11_VIEWPORT]) -> Vec<D3D11_VIEWPORT> {
    let bound = BOUND.load(Ordering::Relaxed) && DEPTH_TESTS.load(Ordering::Relaxed);
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
    VERTEX_CONSTANT_SLOT.store(u32::MAX, Ordering::Relaxed);
    if !BOUND.load(Ordering::Relaxed) || shader.is_null() {
        return shader;
    }
    let original = unsafe { ID3D11VertexShader::from_raw_borrowed(&shader) }.unwrap();
    let twin = crate::shader::mirrored(original, || {
        let record = crate::address(sre::id!(globals::g_CurrentVertexShaderRecord))
            .map(|a| unsafe { *(a as *const usize) })
            .filter(|&r| r != 0)?;
        let at = sre::member_offset!("VertexShaderRecord.shader");
        if unsafe { *((record + at) as *const usize) } != shader as usize {
            return None;
        }
        Some(
            unsafe {
                core::slice::from_raw_parts(
                    (record + sre::member_offset!("VertexShaderRecord.bytecode")) as *const u8,
                    *((record + sre::member_offset!("VertexShaderRecord.bytecodeSize")) as *const u32)
                        as usize,
                )
            }
            .to_vec(),
        )
    });
    twin.map_or(shader, |(twin, slot)| {
        VERTEX_CONSTANT_SLOT.store(slot.unwrap_or(u32::MAX), Ordering::Relaxed);
        twin.as_raw()
    })
}

fn depth_mode(state: *mut c_void) -> (bool, bool) {
    let mut cache = DEPTH_MODES.lock().unwrap();
    *cache
        .get_or_insert_with(HashMap::new)
        .entry(state as usize)
        .or_insert_with(|| {
            let Some(s) = (unsafe { ID3D11DepthStencilState::from_raw_borrowed(&state) }) else {
                return (true, true);
            };
            let mut desc = D3D11_DEPTH_STENCIL_DESC::default();
            unsafe { s.GetDesc(&raw mut desc) };
            (
                desc.DepthEnable.as_bool(),
                desc.DepthEnable.as_bool() && desc.DepthWriteMask == D3D11_DEPTH_WRITE_MASK_ALL,
            )
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
    if !BOUND.load(Ordering::Relaxed) || !DEPTH_TESTS.load(Ordering::Relaxed) {
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
        .filter(|_| BOUND.load(Ordering::Relaxed) && DEPTH_TESTS.load(Ordering::Relaxed))
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
    format!(
        "{}{}, depth state twins {}, raster twins {}, world depth clears {}, {}",
        if BELOW_WRAPPER.load(Ordering::Relaxed) {
            format!("below a wrapper ({} depth textures named), ", SYSTEM_TEXTURES.lock().unwrap().len())
        } else {
            String::new()
        },
        crate::shader::counters(),
        twins(&MIRRORED_DEPTH),
        twins(&MIRRORED_RASTER),
        CLEARS.load(Ordering::Relaxed),
        crate::copy::counters()
    )
}
