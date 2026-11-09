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
//! Twins are bound when the engine binds a shader or the state changes, not per draw, and each
//! precise twin's private camera constants stay bound between its draws.
//!
//! Below a D3D11 wrapper (ENB) the hooks sit on the system's context instead of the game's, so the
//! wrapper's own draws, depth tests and depth reads are mirrored as the game's are.

use core::ffi::c_void;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::ops::Deref;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};

use sre::raw::globals;
use windows::Win32::Graphics::Direct3D11::{
    D3D11_BIND_CONSTANT_BUFFER, D3D11_BUFFER_DESC, D3D11_COMPARISON_FUNC, D3D11_COMPARISON_GREATER,
    D3D11_COMPARISON_GREATER_EQUAL, D3D11_COMPARISON_LESS, D3D11_COMPARISON_LESS_EQUAL,
    D3D11_DEPTH_STENCIL_DESC, D3D11_DEPTH_STENCIL_VIEW_DESC, D3D11_DEVICE_CONTEXT_IMMEDIATE, D3D11_DEPTH_WRITE_MASK_ALL,
    D3D11_RASTERIZER_DESC, D3D11_USAGE_DEFAULT, D3D11_VIEWPORT,
    D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE, ID3D11Buffer,
    ID3D11DepthStencilState, ID3D11DepthStencilView, ID3D11Device, ID3D11DeviceContext,
    ID3D11DomainShader, ID3D11PixelShader, ID3D11RasterizerState, ID3D11RenderTargetView,
    ID3D11Resource, ID3D11UnorderedAccessView, ID3D11VertexShader,
};
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
use windows::core::{IUnknown, Interface};

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
type GetConstantsFn = unsafe extern "system" fn(*mut c_void, u32, u32, *mut *mut c_void);
type GetRasterFn = unsafe extern "system" fn(*mut c_void, *mut *mut c_void);
type GetDepthStateFn = unsafe extern "system" fn(*mut c_void, *mut *mut c_void, *mut u32);
type GetViewportsFn = unsafe extern "system" fn(*mut c_void, *mut u32, *mut D3D11_VIEWPORT);
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
const VIEWPORTS: usize = D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE as usize;

macro_rules! hooks {
    ($($name:ident: $ty:ty = $field:ident => $detour:ident,)*) => {
        $(static $name: Mutex<Option<hook::Hook<$ty>>> = Mutex::new(None);)*

        /// Creates every hook without enabling one; on a failure the created ones are discarded, which
        /// is safe because nothing runs through them yet.
        fn create_hooks(table: &windows::Win32::Graphics::Direct3D11::ID3D11DeviceContext_Vtbl) -> Result<(), String> {
            $(
                match unsafe { hook::Hook::<$ty>::create(table.$field as usize, $detour as $ty) } {
                    Ok(hook) => *$name.lock().unwrap() = Some(hook),
                    Err(e) => {
                        remove();
                        return Err(format!("{}: {e}", stringify!($field)));
                    }
                }
            )*
            Ok(())
        }

        /// Enables the created hooks; each slot is filled first, so a call on another thread finds it.
        fn enable_hooks() -> Result<(), String> {
            $(
                $name
                    .lock()
                    .unwrap()
                    .as_ref()
                    .ok_or(concat!(stringify!($field), ": not created"))?
                    .enable()
                    .map_err(|e| format!("{}: {e}", stringify!($field)))?;
            )*
            Ok(())
        }

        fn install_hooks(table: &windows::Win32::Graphics::Direct3D11::ID3D11DeviceContext_Vtbl) -> Result<(), String> {
            create_hooks(table)?;
            enable_hooks()
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
    GET_DEPTH_STATE: GetDepthStateFn = OMGetDepthStencilState => get_depth_state,
    GET_RASTER: GetRasterFn = RSGetState => get_raster,
    GET_VIEWPORTS: GetViewportsFn = RSGetViewports => get_viewports,
    CLEAR_DEPTH: ClearDepthFn = ClearDepthStencilView => clear_depth,
    SET_VERTEX_SHADER: SetShaderFn = VSSetShader => set_vertex_shader,
    SET_PIXEL_SHADER: SetShaderFn = PSSetShader => set_pixel_shader,
    SET_DOMAIN_SHADER: SetShaderFn = DSSetShader => set_domain_shader,
    GET_VERTEX_SHADER: GetShaderFn = VSGetShader => get_vertex_shader,
    GET_PIXEL_SHADER: GetShaderFn = PSGetShader => get_pixel_shader,
    GET_DOMAIN_SHADER: GetShaderFn = DSGetShader => get_domain_shader,
    GET_VERTEX_CONSTANTS: GetConstantsFn = VSGetConstantBuffers => get_vertex_constants,
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
static LAST_DOMAIN_SHADER: Mutex<Option<ID3D11DomainShader>> = Mutex::new(None);
static VERTEX_INSTANCES: AtomicU32 = AtomicU32::new(0);
static PIXEL_INSTANCES: AtomicU32 = AtomicU32::new(0);
static DOMAIN_INSTANCES: AtomicU32 = AtomicU32::new(0);
const VERTEX: u8 = 1;
const DOMAIN: u8 = 2;
const PIXEL: u8 = 4;
/// `copy::count() + 1` when the bound target was last marked as drawn into, 0 when not since bind.
static MARKED: AtomicU64 = AtomicU64::new(0);
static DEPTH_TESTS: AtomicBool = AtomicBool::new(true);
static VERTEX_CONSTANT_SLOT: AtomicU32 = AtomicU32::new(u32::MAX);
static LAST_DEPTH_VIEW: Mutex<Option<ID3D11DepthStencilView>> = Mutex::new(None);
/// The engine's current depth state writes depth.
static WRITES: AtomicBool = AtomicBool::new(true);
/// The bound world target.
static BOUND_RESOURCE: AtomicUsize = AtomicUsize::new(0);
/// Hashes the state caches' pointer keys with a single multiply.
#[derive(Default)]
struct PointerHasher(u64);

impl Hasher for PointerHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 = (self.0.rotate_left(8) ^ u64::from(byte)).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        }
    }

    fn write_usize(&mut self, n: usize) {
        self.0 = (n as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
}

type PointerMap<V> = HashMap<usize, V, BuildHasherDefault<PointerHasher>>;

static MIRRORED_DEPTH: Mutex<Option<PointerMap<usize>>> = Mutex::new(None);
static MIRRORED_RASTER: Mutex<Option<PointerMap<usize>>> = Mutex::new(None);
static DEPTH_MODES: Mutex<Option<PointerMap<(bool, bool)>>> = Mutex::new(None);
struct VertexConstants {
    buffer: ID3D11Buffer,
    sent: [u32; 4],
    /// The slot the buffer stays bound at between draws, and the buffer it displaced there.
    placed: Option<(u32, Option<ID3D11Buffer>)>,
}
static VERTEX_CONSTANTS: Mutex<Option<VertexConstants>> = Mutex::new(None);
static CONSTANTS_PLACED: AtomicBool = AtomicBool::new(false);
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
        Err(e) => return Err(format!("system context below the game's: {e}")),
    }
    IMMEDIATE.store(context.as_raw() as usize, Ordering::Relaxed);
    let device = device().ok_or("no device")?;
    create_vertex_constants(&device)?;
    crate::copy::initialize(&device)?;
    crate::copy::sync_sources();
    if let Err(e) = crate::shader::install_device(&device) {
        crate::log(&format!("replacement shader capture: {e}"));
    }
    install_hooks(context.vtable())
}

/// The hooks are on the system's context below a wrapper's.
static BELOW_WRAPPER: AtomicBool = AtomicBool::new(false);
static SYSTEM_SETUP_FAILED: AtomicBool = AtomicBool::new(false);

/// Hooks the system's context when the game's is a wrapper's (ENB): the wrapper draws with
/// shaders and depth tests of its own there, past the game's table. The wrapper hands the game
/// the system's own textures, so the world targets are the same objects on both sides.
fn install_below_wrapper(game: &ID3D11DeviceContext) -> Result<bool, String> {
    if crate::shader::dxvk().is_some() {
        return Ok(false);
    }
    let Some((_device, system)) = crate::shader::system_device() else {
        return Ok(false);
    };
    let table = system.vtable();
    if table.OMSetDepthStencilState as usize == game.vtable().OMSetDepthStencilState as usize {
        return Ok(false);
    }
    if let Err(e) = create_hooks(table) {
        crate::log(&format!(
            "the system's context below the game's could not be hooked ({e}); the game's is hooked instead"
        ));
        return Ok(false);
    }
    BELOW_WRAPPER.store(true, Ordering::Relaxed);
    enable_hooks()?;
    Ok(true)
}

/// Enables a hook created into `slot`.
pub(crate) fn enable_created<F: Copy>(slot: &Mutex<Option<hook::Hook<F>>>) -> Result<(), String> {
    slot.lock().unwrap().as_ref().ok_or("hook not created")?.enable().map_err(|e| e.to_string())
}

/// Below a wrapper, the first bind of a world depth target on an immediate context names the
/// system's context the game draws on; the objects of this module are then made on its device.
fn learn(context: *mut c_void, view: *mut c_void) {
    if IMMEDIATE.load(Ordering::Relaxed) != 0
        || !BELOW_WRAPPER.load(Ordering::Relaxed)
        || SYSTEM_SETUP_FAILED.load(Ordering::Relaxed)
    {
        return;
    }
    let Some(resource) = depth_resource(view) else { return };
    let world = crate::copy::REVERSED_TARGETS
        .into_iter()
        .any(|t| target_texture(t).is_some_and(|field| unsafe { *field } == resource.as_raw() as usize));
    let Some(c) = (unsafe { ID3D11DeviceContext::from_raw_borrowed(&context) }) else { return };
    if !world || unsafe { c.GetType() } != D3D11_DEVICE_CONTEXT_IMMEDIATE {
        return;
    }
    let made = unsafe { c.GetDevice() }.map_err(|e| e.to_string()).and_then(|device| {
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
    MARKED.store(0, Ordering::Relaxed);
    if BOUND.swap(bound, Ordering::Relaxed) == bound {
        return;
    }
    resend(context, VERTEX | DOMAIN | PIXEL);
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

/// Sends `stages` the engine's shader, or its twin while a depth-tested world view is bound. Done
/// when the shader or the state changes, not per draw: the runtime switches the context's draw entry
/// points with some states, so a draw may pass the draw hooks.
fn resend(context: *mut c_void, stages: u8) {
    let mirrored = mirrored();
    if stages & VERTEX != 0 {
        // with tessellation the domain stage writes the position; the vertex stage only feeds patches
        let tessellated = LAST_DOMAIN_SHADER.lock().unwrap().is_some();
        VERTEX_CONSTANT_SLOT.store(u32::MAX, Ordering::Relaxed);
        send(
            context,
            &SET_VERTEX_SHADER,
            &VERTEX_INSTANCES,
            raw(&LAST_VERTEX_SHADER),
            mirrored && !tessellated,
            vertex_shader,
        );
    }
    if stages & DOMAIN != 0 {
        send(
            context,
            &SET_DOMAIN_SHADER,
            &DOMAIN_INSTANCES,
            raw(&LAST_DOMAIN_SHADER),
            mirrored,
            domain_shader,
        );
    }
    if stages & PIXEL != 0 {
        send(
            context,
            &SET_PIXEL_SHADER,
            &PIXEL_INSTANCES,
            raw(&LAST_PIXEL_SHADER),
            mirrored,
            pixel_shader,
        );
    }
}

fn mirrored() -> bool {
    BOUND.load(Ordering::Relaxed) && DEPTH_TESTS.load(Ordering::Relaxed)
}

/// The engine's shader for a stage, kept alive by the stage's record.
fn raw<I: Interface>(last: &Mutex<Option<I>>) -> *mut c_void {
    last.lock()
        .unwrap()
        .as_ref()
        .map_or(core::ptr::null_mut(), Interface::as_raw)
}

fn send(
    context: *mut c_void,
    hook: &Mutex<Option<hook::Hook<SetShaderFn>>>,
    instances: &AtomicU32,
    shader: *mut c_void,
    mirror: bool,
    twin: fn(*mut c_void) -> *mut c_void,
) {
    // a shader with class instances is sent as given
    if instances.load(Ordering::Relaxed) != 0 {
        if mirror {
            crate::copy::report("dynamic shader linkage is unsupported for reversed depth");
        }
        return;
    }
    let sent = if mirror { twin(shader) } else { shader };
    unsafe { original(hook)(context, sent, core::ptr::null(), 0) };
}

fn set_stage(
    context: *mut c_void,
    hook: &Mutex<Option<hook::Hook<SetShaderFn>>>,
    stage: u8,
    shader: *mut c_void,
    instances: (*const *mut c_void, u32, &AtomicU32),
) {
    let (list, count, stored) = instances;
    stored.store(count, Ordering::Relaxed);
    if count == 0 {
        resend(context, stage);
    } else {
        unsafe { original(hook)(context, shader, list, count) };
    }
}

fn with_vertex_constants(context: *mut c_void) {
    let slot = VERTEX_CONSTANT_SLOT.load(Ordering::Relaxed);
    let wanted = slot != u32::MAX && BOUND.load(Ordering::Relaxed);
    if wanted || CONSTANTS_PLACED.load(Ordering::Relaxed) {
        let context = unsafe { ID3D11DeviceContext::from_raw_borrowed(&context) }.unwrap();
        place_constants(context, wanted.then(|| (slot, camera_constants())));
    }
}

fn camera_constants() -> [f32; 4] {
    let state = CAMERA_STATE.load(Ordering::Relaxed);
    let at = sre::member_offset!("CurrentCameraState.projection");
    let projection = unsafe { ((state + at) as *const [f32; 16]).read_unaligned() };
    let near = skyrim::render::depth_shift_near();
    let far = skyrim::render::depth_shift_far();
    let scale = if near > 0.0 { near / VANILLA_NEAR } else { 1.0 };
    let a_near = -projection[14];
    let a_less_one = if far > 0.0 {
        a_near / far
    } else {
        projection[10] - 1.0
    };
    [scale, a_less_one, a_near, projection[15]]
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
        placed: None,
    });
    CONSTANTS_PLACED.store(false, Ordering::Relaxed);
    Ok(())
}

/// The vertex constant buffer bound at `slot`, read past the hook.
fn vertex_constants_at(context: &ID3D11DeviceContext, slot: u32) -> Option<ID3D11Buffer> {
    let get = GET_VERTEX_CONSTANTS
        .lock()
        .unwrap()
        .as_ref()
        .map_or(context.vtable().VSGetConstantBuffers, hook::Hook::original);
    let mut buffer: *mut c_void = core::ptr::null_mut();
    unsafe { get(context.as_raw(), slot, 1, (&raw mut buffer).cast()) };
    (!buffer.is_null()).then(|| unsafe { ID3D11Buffer::from_raw(buffer) })
}

/// Binds the private constants `data` at `slot` for the next draws, or with `None` gives the slot
/// back the buffer they displaced. The binding stays across draws until another slot or none is
/// wanted; a buffer the engine bound over it in between is displaced again.
pub(crate) fn place_constants(context: &ID3D11DeviceContext, wanted: Option<(u32, [f32; 4])>) {
    let mut constants = VERTEX_CONSTANTS.lock().unwrap();
    let Some(constants) = constants.as_mut() else {
        return;
    };
    if let Some((at, _)) = constants.placed
        && wanted.is_none_or(|(slot, _)| slot != at)
    {
        let (at, displaced) = constants.placed.take().unwrap();
        if vertex_constants_at(context, at).as_ref() == Some(&constants.buffer) {
            unsafe { context.VSSetConstantBuffers(at, Some(&[displaced])) };
        }
    }
    let Some((slot, data)) = wanted else {
        CONSTANTS_PLACED.store(false, Ordering::Relaxed);
        return;
    };
    let bits = data.map(f32::to_bits);
    if constants.sent != bits {
        unsafe {
            context.UpdateSubresource(&constants.buffer, 0, None, data.as_ptr().cast(), 0, 0);
        };
        constants.sent = bits;
    }
    let current = vertex_constants_at(context, slot);
    if current.as_ref() != Some(&constants.buffer) {
        unsafe { context.VSSetConstantBuffers(slot, Some(&[Some(constants.buffer.clone())])) };
        constants.placed = Some((slot, current));
    }
    CONSTANTS_PLACED.store(true, Ordering::Relaxed);
}

macro_rules! immediate_only {
    ($original:ident, $context:ident $(, $argument:ident)*) => {
        if $context as usize != IMMEDIATE.load(Ordering::Relaxed) || crate::copy::internal() {
            unsafe { original(&$original)($context $(, $argument)*) };
            return;
        }
    };
}

/// The engine's buffers, with the one the private constants displaced in place of theirs.
unsafe extern "system" fn get_vertex_constants(
    context: *mut c_void,
    start: u32,
    count: u32,
    buffers: *mut *mut c_void,
) {
    immediate_only!(GET_VERTEX_CONSTANTS, context, start, count, buffers);
    unsafe { original(&GET_VERTEX_CONSTANTS)(context, start, count, buffers) };
    if buffers.is_null() || !CONSTANTS_PLACED.load(Ordering::Relaxed) {
        return;
    }
    let constants = VERTEX_CONSTANTS.lock().unwrap();
    let Some(VertexConstants {
        buffer,
        placed: Some((slot, displaced)),
        ..
    }) = constants.as_ref()
    else {
        return;
    };
    if !(start..start.saturating_add(count)).contains(slot) {
        return;
    }
    let entry = unsafe { buffers.add((slot - start) as usize) };
    if unsafe { *entry } == buffer.as_raw() {
        drop(unsafe { ID3D11Buffer::from_raw(*entry) });
        let substitute = displaced
            .clone()
            .map_or(core::ptr::null_mut(), Interface::into_raw);
        unsafe { *entry = substitute };
    }
}

pub(crate) fn release_push_scale() {
    VERTEX_CONSTANTS.lock().unwrap().take();
    LAST_VERTEX_SHADER.lock().unwrap().take();
    LAST_PIXEL_SHADER.lock().unwrap().take();
    LAST_DOMAIN_SHADER.lock().unwrap().take();
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
        resend(context, VERTEX | DOMAIN | PIXEL);
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
    let (count, list) = if list.is_null() {
        (count, list)
    } else {
        (u32::try_from(sent.len()).unwrap(), sent.as_ptr())
    };
    unsafe { original(&SET_VIEWPORTS)(context, count, list) };
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
    VERTEX_CONSTANT_SLOT.store(u32::MAX, Ordering::Relaxed);
    set_stage(
        context,
        &SET_VERTEX_SHADER,
        VERTEX,
        shader,
        (instances, count, &VERTEX_INSTANCES),
    );
}

fn domain_shader(shader: *mut c_void) -> *mut c_void {
    if !BOUND.load(Ordering::Relaxed) || shader.is_null() {
        return shader;
    }
    let original = unsafe { ID3D11DomainShader::from_raw_borrowed(&shader) }.unwrap();
    crate::shader::domain(original).map_or(shader, |twin| twin.as_raw())
}

unsafe extern "system" fn set_domain_shader(
    context: *mut c_void,
    shader: *mut c_void,
    instances: *const *mut c_void,
    count: u32,
) {
    immediate_only!(SET_DOMAIN_SHADER, context, shader, instances, count);
    *LAST_DOMAIN_SHADER.lock().unwrap() =
        unsafe { ID3D11DomainShader::from_raw_borrowed(&shader) }.cloned();
    set_stage(
        context,
        &SET_DOMAIN_SHADER,
        DOMAIN,
        shader,
        (instances, count, &DOMAIN_INSTANCES),
    );
    // the vertex stage's twin depends on whether a domain shader is bound
    if mirrored() {
        resend(context, VERTEX);
    }
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
    set_stage(
        context,
        &SET_PIXEL_SHADER,
        PIXEL,
        shader,
        (instances, count, &PIXEL_INSTANCES),
    );
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
            if !shader.is_null() {
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
shader_getter!(
    get_domain_shader,
    GET_DOMAIN_SHADER,
    LAST_DOMAIN_SHADER,
    ID3D11DomainShader
);

/// Hands a state getter's caller the state it set when the bound one is that state's twin, so a
/// caller that compares or restores what it read sees its own state.
fn logical_state(
    cache: &Mutex<Option<PointerMap<usize>>>,
    last: &AtomicUsize,
    out: *mut *mut c_void,
) {
    if out.is_null() {
        return;
    }
    let bound = unsafe { *out };
    let logical = last.load(Ordering::Relaxed);
    let sent = cache
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|c| c.get(&logical).copied());
    if bound.is_null() || bound as usize == logical || sent != Some(bound as usize) {
        return;
    }
    let logical = logical as *mut c_void;
    unsafe {
        drop(IUnknown::from_raw(bound));
        *out = IUnknown::from_raw_borrowed(&logical)
            .map_or(core::ptr::null_mut(), |state| state.clone().into_raw());
    }
}

unsafe extern "system" fn get_depth_state(
    context: *mut c_void,
    state: *mut *mut c_void,
    stencil_ref: *mut u32,
) {
    immediate_only!(GET_DEPTH_STATE, context, state, stencil_ref);
    unsafe { original(&GET_DEPTH_STATE)(context, state, stencil_ref) };
    logical_state(&MIRRORED_DEPTH, &LAST_DEPTH_STATE, state);
}

unsafe extern "system" fn get_raster(context: *mut c_void, state: *mut *mut c_void) {
    immediate_only!(GET_RASTER, context, state);
    unsafe { original(&GET_RASTER)(context, state) };
    logical_state(&MIRRORED_RASTER, &LAST_RASTER, state);
}

unsafe extern "system" fn get_viewports(
    context: *mut c_void,
    count: *mut u32,
    list: *mut D3D11_VIEWPORT,
) {
    immediate_only!(GET_VIEWPORTS, context, count, list);
    let capacity = if count.is_null() { 0 } else { unsafe { *count } };
    unsafe { original(&GET_VIEWPORTS)(context, count, list) };
    if count.is_null() || list.is_null() {
        return;
    }
    let filled = unsafe { *count }.min(capacity) as usize;
    let given = unsafe { core::slice::from_raw_parts_mut(list, filled) };
    let stored = LAST_VIEWPORTS.lock().unwrap();
    if let Some(stored) = stored.get(..filled)
        && *viewports(stored) == *given
    {
        given.copy_from_slice(stored);
    }
}

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
    // the engine's shaders read back while their twins stay bound, also after a state save and
    // restore, and the engine's own bound again once the world view is unbound
    macro_rules! check {
        ($get:ident, $set:ident, $hook:ident, $expected:ident, $shader:ty, $twinned:expr) => {
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
            assert_eq!(native.as_raw() != $expected.as_raw(), $twinned);
        };
    }
    for twinned in [true, true, false] {
        if !twinned {
            unsafe { context.OMSetRenderTargets(None, None) };
        }
        check!(
            VSGetShader,
            VSSetShader,
            GET_VERTEX_SHADER,
            vertex,
            ID3D11VertexShader,
            twinned
        );
        check!(
            PSGetShader,
            PSSetShader,
            GET_PIXEL_SHADER,
            pixel,
            ID3D11PixelShader,
            twinned
        );
    }
    unsafe { context.ClearState() };
}

#[cfg(test)]
pub(crate) fn test_state_readback(
    device: &ID3D11Device,
    context: &ID3D11DeviceContext,
    depth: &ID3D11DepthStencilView,
) {
    let mut depth_state = None;
    let desc = D3D11_DEPTH_STENCIL_DESC {
        DepthEnable: true.into(),
        DepthWriteMask: D3D11_DEPTH_WRITE_MASK_ALL,
        DepthFunc: D3D11_COMPARISON_LESS,
        ..Default::default()
    };
    unsafe { device.CreateDepthStencilState(&raw const desc, Some(&raw mut depth_state)) }.unwrap();
    let depth_state = depth_state.unwrap();
    let mut raster_state = None;
    let desc = D3D11_RASTERIZER_DESC {
        FillMode: windows::Win32::Graphics::Direct3D11::D3D11_FILL_SOLID,
        CullMode: windows::Win32::Graphics::Direct3D11::D3D11_CULL_BACK,
        DepthBias: 256,
        ..Default::default()
    };
    unsafe { device.CreateRasterizerState(&raw const desc, Some(&raw mut raster_state)) }.unwrap();
    let raster_state = raster_state.unwrap();
    let viewport = D3D11_VIEWPORT {
        Width: 13.0,
        Height: 11.0,
        MinDepth: 0.0,
        MaxDepth: 0.75,
        ..Default::default()
    };
    crate::copy::mark(&unsafe { depth.GetResource() }.unwrap(), true);
    let native = || {
        let (mut depth_state, mut stencil_ref, mut raster) =
            (core::ptr::null_mut(), 0, core::ptr::null_mut());
        let mut count = 1;
        let mut viewport = D3D11_VIEWPORT::default();
        unsafe {
            original(&GET_DEPTH_STATE)(context.as_raw(), &raw mut depth_state, &raw mut stencil_ref);
            original(&GET_RASTER)(context.as_raw(), &raw mut raster);
            original(&GET_VIEWPORTS)(context.as_raw(), &raw mut count, &raw mut viewport);
            drop(ID3D11DepthStencilState::from_raw(depth_state));
            drop(ID3D11RasterizerState::from_raw(raster));
        }
        (depth_state, raster, viewport)
    };
    unsafe {
        context.OMSetRenderTargets(None, depth);
        context.OMSetDepthStencilState(&depth_state, 3);
        context.RSSetState(&raster_state);
        context.RSSetViewports(Some(&[viewport]));
    }
    let mirrored = native();
    assert_ne!(mirrored.0, depth_state.as_raw());
    assert_ne!(mirrored.1, raster_state.as_raw());
    assert_eq!((mirrored.2.MinDepth, mirrored.2.MaxDepth), (0.25, 1.0));
    for _ in 0..2 {
        let (mut read_depth, mut read_ref) = (None, 0);
        let mut count = 1;
        let mut read_viewport = [D3D11_VIEWPORT::default()];
        unsafe {
            context.OMGetDepthStencilState(Some(&raw mut read_depth), Some(&raw mut read_ref));
            context.RSGetViewports(&raw mut count, Some(read_viewport.as_mut_ptr()));
        }
        assert_eq!(
            read_depth.as_ref().map(Interface::as_raw),
            Some(depth_state.as_raw())
        );
        assert_eq!(read_ref, 3);
        let read_raster = unsafe { context.RSGetState() }.ok();
        assert_eq!(
            read_raster.as_ref().map(Interface::as_raw),
            Some(raster_state.as_raw())
        );
        assert_eq!(read_viewport[0], viewport);
        unsafe {
            context.OMSetDepthStencilState(read_depth.as_ref(), read_ref);
            context.RSSetState(read_raster.as_ref());
            context.RSSetViewports(Some(&read_viewport));
        }
        assert_eq!(native(), mirrored, "restoring what was read must not mirror it twice");
    }
    let (mut read_depth, mut read_ref) = (None, 0);
    unsafe {
        context.OMSetDepthStencilState(None, 0);
        context.RSSetState(None);
        context.OMGetDepthStencilState(Some(&raw mut read_depth), Some(&raw mut read_ref));
    }
    assert!(read_depth.is_none(), "a caller that set the default depth state reads none");
    assert!(unsafe { context.RSGetState() }.is_err(), "a caller that set the default rasterizer reads none");
    unsafe { context.ClearState() };
}

macro_rules! resource_hook {
    ($name:ident, $slot:ident, $stage:ident) => {
        unsafe extern "system" fn $name(
            context: *mut c_void,
            start: u32,
            count: u32,
            views: *const *mut c_void,
        ) {
            immediate_only!($slot, context, start, count, views);
            let swapped = substituted(context, count, views);
            let sent = swapped.as_ref().map_or(views, Vec::as_ptr);
            crate::copy::bind_views(
                crate::copy::$stage,
                start,
                count,
                (!sent.is_null())
                    .then(|| unsafe { core::slice::from_raw_parts(sent, count as usize) }),
            );
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
resource_hook!(set_pixel_resources, SET_PIXEL_RESOURCES, PIXEL);
resource_hook!(set_compute_resources, SET_COMPUTE_RESOURCES, COMPUTE);
resource_hook!(set_vertex_resources, SET_VERTEX_RESOURCES, VERTEX);
resource_hook!(set_geometry_resources, SET_GEOMETRY_RESOURCES, GEOMETRY);
resource_hook!(set_hull_resources, SET_HULL_RESOURCES, HULL);
resource_hook!(set_domain_resources, SET_DOMAIN_RESOURCES, DOMAIN);

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
    crate::copy::unbind_views();
    BOUND.store(false, Ordering::Relaxed);
    BOUND_RESOURCE.store(0, Ordering::Relaxed);
    LAST_VERTEX_SHADER.lock().unwrap().take();
    LAST_PIXEL_SHADER.lock().unwrap().take();
    LAST_DOMAIN_SHADER.lock().unwrap().take();
    LAST_DEPTH_STATE.store(0, Ordering::Relaxed);
    LAST_STENCIL_REF.store(0, Ordering::Relaxed);
    LAST_RASTER.store(0, Ordering::Relaxed);
    LAST_VIEWPORTS.lock().unwrap().clear();
    VERTEX_CONSTANT_SLOT.store(u32::MAX, Ordering::Relaxed);
    if let Some(constants) = VERTEX_CONSTANTS.lock().unwrap().as_mut() {
        constants.placed = None;
    }
    CONSTANTS_PLACED.store(false, Ordering::Relaxed);
    WRITES.store(true, Ordering::Relaxed);
    DEPTH_TESTS.store(true, Ordering::Relaxed);
    VERTEX_INSTANCES.store(0, Ordering::Relaxed);
    PIXEL_INSTANCES.store(0, Ordering::Relaxed);
    DOMAIN_INSTANCES.store(0, Ordering::Relaxed);
    MARKED.store(0, Ordering::Relaxed);
    if let Some(view) = previous
        && let Ok(resource) = unsafe { view.GetResource() }
    {
        crate::copy::run(
            unsafe { ID3D11DeviceContext::from_raw_borrowed(&context) }.unwrap(),
            &resource,
        );
    }
}

/// Marks the bound target as drawn into, once per bind and per copy refresh, and binds the
/// constants the draw's twin needs.
fn drawn(context: *mut c_void) {
    let resource = BOUND_RESOURCE.load(Ordering::Relaxed) as *mut c_void;
    let generation = crate::copy::count() + 1;
    if WRITES.load(Ordering::Relaxed)
        && MARKED.swap(generation, Ordering::Relaxed) != generation
        && let Some(resource) = unsafe { ID3D11Resource::from_raw_borrowed(&resource) }
    {
        crate::copy::mark(resource, BOUND.load(Ordering::Relaxed));
    }
    with_vertex_constants(context);
}

unsafe extern "system" fn draw(context: *mut c_void, vertices: u32, start: u32) {
    immediate_only!(DRAW, context, vertices, start);
    drawn(context);
    unsafe {
        original(&DRAW)(context, vertices, start);
    };
}

unsafe extern "system" fn draw_indexed(context: *mut c_void, indices: u32, start: u32, base: i32) {
    immediate_only!(DRAW_INDEXED, context, indices, start, base);
    drawn(context);
    unsafe {
        original(&DRAW_INDEXED)(context, indices, start, base);
    };
}

unsafe extern "system" fn draw_instanced(
    context: *mut c_void,
    vertices: u32,
    instances: u32,
    start: u32,
    first: u32,
) {
    immediate_only!(DRAW_INSTANCED, context, vertices, instances, start, first);
    drawn(context);
    unsafe {
        original(&DRAW_INSTANCED)(context, vertices, instances, start, first);
    };
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
    drawn(context);
    unsafe {
        original(&DRAW_INDEXED_INSTANCED)(context, indices, instances, start, base, first);
    };
}

unsafe extern "system" fn draw_indirect(context: *mut c_void, arguments: *mut c_void, offset: u32) {
    immediate_only!(DRAW_INDIRECT, context, arguments, offset);
    drawn(context);
    unsafe {
        original(&DRAW_INDIRECT)(context, arguments, offset);
    };
}

unsafe extern "system" fn draw_indexed_indirect(
    context: *mut c_void,
    arguments: *mut c_void,
    offset: u32,
) {
    immediate_only!(DRAW_INDEXED_INDIRECT, context, arguments, offset);
    drawn(context);
    unsafe {
        original(&DRAW_INDEXED_INDIRECT)(context, arguments, offset);
    };
}

unsafe extern "system" fn draw_auto(context: *mut c_void) {
    immediate_only!(DRAW_AUTO, context);
    drawn(context);
    unsafe { original(&DRAW_AUTO)(context) };
}

/// Viewports as sent to the context.
struct Viewports {
    list: [D3D11_VIEWPORT; VIEWPORTS],
    len: usize,
}

impl Deref for Viewports {
    type Target = [D3D11_VIEWPORT];

    fn deref(&self) -> &[D3D11_VIEWPORT] {
        &self.list[..self.len]
    }
}

/// `given` with depth ranges mirrored while a depth-tested world view is bound.
fn viewports(given: &[D3D11_VIEWPORT]) -> Viewports {
    let bound = BOUND.load(Ordering::Relaxed) && DEPTH_TESTS.load(Ordering::Relaxed);
    let mut sent = Viewports {
        list: [D3D11_VIEWPORT::default(); VIEWPORTS],
        len: given.len().min(VIEWPORTS),
    };
    for (out, v) in sent.list.iter_mut().zip(given) {
        *out = if bound {
            D3D11_VIEWPORT {
                MinDepth: 1.0 - v.MaxDepth,
                MaxDepth: 1.0 - v.MinDepth,
                ..*v
            }
        } else {
            *v
        };
    }
    sent
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
        .get_or_insert_with(PointerMap::default)
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
    twin(&MIRRORED_RASTER, state, |device| {
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
    cache: &Mutex<Option<PointerMap<usize>>>,
    state: *mut c_void,
    build: impl FnOnce(&ID3D11Device) -> Option<*mut c_void>,
) -> *mut c_void {
    let mut cache = cache.lock().unwrap();
    let cache = cache.get_or_insert_with(PointerMap::default);
    if let Some(t) = cache.get(&(state as usize)) {
        return *t as *mut c_void;
    }
    let made = device().and_then(|d| build(&d)).unwrap_or(state);
    cache.insert(state as usize, made as usize);
    made
}

pub(crate) fn counters() -> String {
    let twins =
        |c: &Mutex<Option<PointerMap<usize>>>| c.lock().unwrap().as_ref().map_or(0, HashMap::len);
    format!(
        "{}{}, depth state twins {}, raster twins {}, world depth clears {}, {}",
        if BELOW_WRAPPER.load(Ordering::Relaxed) { "below a wrapper, " } else { "" },
        crate::shader::counters(),
        twins(&MIRRORED_DEPTH),
        twins(&MIRRORED_RASTER),
        CLEARS.load(Ordering::Relaxed),
        crate::copy::counters()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::twin::tests::{compile, compile_stage};
    use windows::Win32::Foundation::HMODULE;
    use windows::Win32::Graphics::Direct3D::{
        D3D_DRIVER_TYPE_HARDWARE, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, D3D_SRV_DIMENSION_TEXTURE2D,
    };
    use windows::Win32::Graphics::Direct3D11::*;
    use windows::Win32::Graphics::Dxgi::Common::{
        DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS,
        DXGI_FORMAT_R32G8X24_TYPELESS, DXGI_SAMPLE_DESC,
    };
    use windows::core::s;

    /// CPU time per draw sent through the hooks with a world target bound, against the same draws
    /// with no hooks, and of a depth copy refreshed at a reader's bind. `cargo test --release -p reversed-z per_draw_cost -- --ignored --nocapture`
    #[test]
    #[ignore = "benchmark on the hardware device"]
    #[allow(clippy::too_many_lines)]
    fn per_draw_cost() {
        crate::shader::install().unwrap();
        let (mut device, mut context) = (None, None);
        unsafe {
            D3D11CreateDevice(
                None,
                D3D_DRIVER_TYPE_HARDWARE,
                HMODULE::default(),
                D3D11_CREATE_DEVICE_FLAG(0),
                None,
                D3D11_SDK_VERSION,
                Some(&raw mut device),
                None,
                Some(&raw mut context),
            )
        }
        .unwrap();
        let (device, context) = (device.unwrap(), context.unwrap());
        let vertex: Vec<ID3D11VertexShader> = [0.25, 0.5]
            .map(|z| {
                let code = compile(&format!(
                    "float4 main(uint id : SV_VertexID) : SV_Position {{ return float4(0, 0, {z}, 1); }}"
                ));
                let mut shader = None;
                unsafe { device.CreateVertexShader(&code, None, Some(&raw mut shader)) }.unwrap();
                shader.unwrap()
            })
            .into();
        let pixel: Vec<ID3D11PixelShader> = ["p.zzzz", "p.zzzz * 0.5"]
            .map(|expression| {
                let code = compile_stage(
                    &format!(
                        "float4 main(float4 p : SV_Position) : SV_Target {{ return {expression}; }}"
                    ),
                    s!("ps_5_0"),
                );
                let mut shader = None;
                unsafe { device.CreatePixelShader(&code, None, Some(&raw mut shader)) }.unwrap();
                shader.unwrap()
            })
            .into();
        let desc = D3D11_TEXTURE2D_DESC {
            Width: 64,
            Height: 64,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_R32G8X24_TYPELESS,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: (D3D11_BIND_DEPTH_STENCIL.0 | D3D11_BIND_SHADER_RESOURCE.0).cast_unsigned(),
            ..Default::default()
        };
        let mut depth = None;
        unsafe { device.CreateTexture2D(&raw const desc, None, Some(&raw mut depth)) }.unwrap();
        let depth = depth.unwrap();
        let view_desc = D3D11_DEPTH_STENCIL_VIEW_DESC {
            Format: DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
            ViewDimension: D3D11_DSV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        let mut view = None;
        unsafe {
            device.CreateDepthStencilView(&depth, Some(&raw const view_desc), Some(&raw mut view))
        }
        .unwrap();
        let view = view.unwrap();
        let reader_desc = D3D11_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS,
            ViewDimension: D3D_SRV_DIMENSION_TEXTURE2D,
            Anonymous: D3D11_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D11_TEX2D_SRV {
                    MostDetailedMip: 0,
                    MipLevels: 1,
                },
            },
        };
        let mut reader = None;
        unsafe {
            device.CreateShaderResourceView(
                &depth,
                Some(&raw const reader_desc),
                Some(&raw mut reader),
            )
        }
        .unwrap();
        let reader = reader.unwrap();
        let textures: Vec<Option<ID3D11ShaderResourceView>> = (0..8)
            .map(|_| {
                let desc = D3D11_TEXTURE2D_DESC {
                    Width: 4,
                    Height: 4,
                    MipLevels: 1,
                    ArraySize: 1,
                    Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    Usage: D3D11_USAGE_DEFAULT,
                    BindFlags: D3D11_BIND_SHADER_RESOURCE.0.cast_unsigned(),
                    ..Default::default()
                };
                let mut texture = None;
                unsafe { device.CreateTexture2D(&raw const desc, None, Some(&raw mut texture)) }
                    .unwrap();
                let mut view = None;
                unsafe {
                    device.CreateShaderResourceView(&texture.unwrap(), None, Some(&raw mut view))
                }
                .unwrap();
                view
            })
            .collect();
        let state_desc = D3D11_DEPTH_STENCIL_DESC {
            DepthEnable: true.into(),
            DepthWriteMask: D3D11_DEPTH_WRITE_MASK_ALL,
            DepthFunc: D3D11_COMPARISON_LESS,
            ..Default::default()
        };
        let mut state = None;
        unsafe { device.CreateDepthStencilState(&raw const state_desc, Some(&raw mut state)) }
            .unwrap();
        let states: Vec<ID3D11DepthStencilState> =
            [D3D11_COMPARISON_LESS, D3D11_COMPARISON_LESS_EQUAL]
                .map(|func| {
                    let desc = D3D11_DEPTH_STENCIL_DESC {
                        DepthFunc: func,
                        ..state_desc
                    };
                    let mut state = None;
                    unsafe {
                        device.CreateDepthStencilState(&raw const desc, Some(&raw mut state))
                    }
                    .unwrap();
                    state.unwrap()
                })
                .into();
        let rasters: Vec<ID3D11RasterizerState> = [0, 16]
            .map(|bias| {
                let desc = D3D11_RASTERIZER_DESC {
                    FillMode: D3D11_FILL_SOLID,
                    CullMode: D3D11_CULL_BACK,
                    DepthBias: bias,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                };
                let mut raster = None;
                unsafe { device.CreateRasterizerState(&raw const desc, Some(&raw mut raster)) }
                    .unwrap();
                raster.unwrap()
            })
            .into();
        let viewport = D3D11_VIEWPORT {
            Width: 64.0,
            Height: 64.0,
            MaxDepth: 1.0,
            ..Default::default()
        };

        // the fastest of several short runs, so other work on the machine and the GPU drops out
        let best = |draw: &dyn Fn(u32)| {
            const DRAWS: u32 = 20_000;
            (0..9)
                .map(|_| {
                    unsafe { context.Flush() };
                    let start = std::time::Instant::now();
                    for i in 0..DRAWS {
                        draw(i);
                    }
                    start.elapsed().as_nanos() / u128::from(DRAWS)
                })
                .min()
                .unwrap()
        };
        let run = |label: &str| {
            unsafe {
                context.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                context.OMSetDepthStencilState(state.as_ref(), 0);
                context.RSSetViewports(Some(&[viewport]));
                context.OMSetRenderTargets(None, &view);
                context.VSSetShader(&vertex[0], None);
                context.PSSetShader(&pixel[0], None);
            }
            let same = best(&|_| unsafe { context.Draw(3, 0) });
            let changing = best(&|i| unsafe {
                context.VSSetShader(&vertex[(i & 1) as usize], None);
                context.PSSetShader(&pixel[(i & 1) as usize], None);
                context.Draw(3, 0);
            });
            println!("{label}: {same} ns per draw, {changing} with shaders changing every draw");
            let textures = best(&|i| unsafe {
                context.PSSetShaderResources((i & 1) * 8, Some(&textures));
            });
            println!("{label}: {textures} ns per bind of 8 textures");
            let state_sets = best(&|i| unsafe {
                let pick = (i & 1) as usize;
                context.OMSetDepthStencilState(&states[pick], 0);
                context.RSSetState(&rasters[pick]);
                context.RSSetViewports(Some(&[viewport]));
            });
            println!("{label}: {state_sets} ns per depth state, raster state and viewport set");
            unsafe { context.ClearState() };
        };
        run("warm-up");
        crate::copy::initialize(&device).unwrap();
        create_vertex_constants(&device).unwrap();
        for _ in 0..2 {
            run("no hooks");
            install_test_hooks(&context);
            crate::copy::mark(&depth.cast().unwrap(), true);
            run("hooks, world bound");
            unsafe {
                context.OMSetRenderTargets(None, &view);
                context.VSSetShader(&vertex[0], None);
            }
            let plain = best(&|_| unsafe { context.Draw(3, 0) });
            let constants = best(&|_| {
                place_constants(&context, Some((13, [1.0, 2.0, 3.0, 4.0])));
                unsafe { context.Draw(3, 0) };
            });
            place_constants(&context, None);
            println!("hooks, private constants: {constants} ns per draw against {plain}");
            let bound_copy = best(&|_| unsafe {
                context.PSSetShaderResources(0, Some(&[Some(reader.clone())]));
            });
            let refreshed_copy = best(&|_| unsafe {
                crate::copy::mark(&depth.cast().unwrap(), true);
                context.PSSetShaderResources(0, Some(&[Some(reader.clone())]));
            });
            println!(
                "hooks, depth copy refreshed at a reader's bind: {refreshed_copy} ns against {bound_copy} for the bind alone"
            );
            unsafe { context.ClearState() };
            remove();
        }
        crate::copy::release();
        crate::shader::remove();
    }
}
