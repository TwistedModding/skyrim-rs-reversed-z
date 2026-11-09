// Copyright (c) 2026 TwistedModding. SPDX-License-Identifier: MPL-2.0-no-copyleft-exception
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not
// distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.
//! Conventional-depth views owned by each reversed-depth texture.

use core::ffi::c_void;
use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::{D3D_SRV_DIMENSION_TEXTURE2D, ID3DBlob};
use windows::Win32::Graphics::Direct3D11::{
    D3D11_BIND_DEPTH_STENCIL, D3D11_BIND_SHADER_RESOURCE, D3D11_BIND_UNORDERED_ACCESS, D3D11_BOX,
    D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT, D3D11_SHADER_MAX_INTERFACES,
    D3D11_SHADER_RESOURCE_VIEW_DESC, D3D11_SHADER_RESOURCE_VIEW_DESC_0, D3D11_TEX2D_SRV,
    D3D11_TEXTURE2D_DESC, D3D11_USAGE_DEFAULT, ID3D11ClassInstance, ID3D11ComputeShader,
    ID3D11Device, ID3D11DeviceContext, ID3D11Resource, ID3D11ShaderResourceView, ID3D11Texture2D,
    ID3D11UnorderedAccessView,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT_D24_UNORM_S8_UINT, DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
    DXGI_FORMAT_R24_UNORM_X8_TYPELESS, DXGI_FORMAT_R24G8_TYPELESS, DXGI_FORMAT_R32_FLOAT,
    DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS, DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_R32G8X24_TYPELESS,
    DXGI_SAMPLE_DESC,
};
use windows::core::{GUID, Interface, s};

pub(crate) const REVERSED_TARGETS: [usize; 3] = [
    sre::DepthStencilTargetId::scene as usize,
    sre::DepthStencilTargetId::second as usize,
    sre::DepthStencilTargetId::screen as usize,
];
const STATE: GUID = GUID::from_u128(0xa232cb36_429b_4650_b548_0c28ca56d252);
const VIEW: GUID = GUID::from_u128(0x9a2b2bf0_1066_4371_8c77_1fbc938e8d43);
const TARGET: GUID = GUID::from_u128(0x542f353a_264f_4dfe_8199_74ae282df10f);
const STANDARD: u32 = 0;
const DIRTY: u32 = 1;
const CLEAN: u32 = 2;
const STANDARD_CLEAN: u32 = 3;
const GROUP: u32 = 8;
const SHADER: &str = "Texture2D<float> depth : register(t0);\nRWTexture2D<float> standard : register(u0);\n[numthreads(8, 8, 1)]\nvoid main(uint3 id : SV_DispatchThreadID) { uint width, height; standard.GetDimensions(width, height); if (id.x < width && id.y < height) standard[id.xy] = 1.0 - depth[id.xy]; }\n";

struct Copy {
    standard: ID3D11Texture2D,
    view: ID3D11ShaderResourceView,
    target: ID3D11UnorderedAccessView,
    engine_view: ID3D11ShaderResourceView,
    width: u32,
    height: u32,
}

const SLOTS: usize = D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT as usize;
/// Shader stages, as indices into the tracked resource slots.
pub(crate) const PIXEL: usize = 0;
pub(crate) const COMPUTE: usize = 1;
pub(crate) const VERTEX: usize = 2;
pub(crate) const GEOMETRY: usize = 3;
pub(crate) const HULL: usize = 4;
pub(crate) const DOMAIN: usize = 5;

/// The copy views bound in each shader stage's resource slots, so a copy whose target is unbound
/// or copied into is refreshed then only while a reader still has it bound; otherwise the reader
/// that binds it next refreshes it.
struct Bindings {
    views: HashSet<usize>,
    slots: [[usize; SLOTS]; 6],
    bound: HashMap<usize, u32>,
}

static BINDINGS: Mutex<Option<Bindings>> = Mutex::new(None);
static SHADER_OBJECT: Mutex<Option<ID3D11ComputeShader>> = Mutex::new(None);
static STANDARD_SHADER_OBJECT: Mutex<Option<ID3D11ComputeShader>> = Mutex::new(None);
static SEEDS: [AtomicUsize; 3] = [const { AtomicUsize::new(0) }; 3];
static COUNT: AtomicU64 = AtomicU64::new(0);
static COPIED: AtomicU64 = AtomicU64::new(0);
static UNSUPPORTED: AtomicU64 = AtomicU64::new(0);
thread_local! { static COPYING: Cell<bool> = const { Cell::new(false) }; }

pub(crate) fn internal() -> bool {
    COPYING.get()
}

struct Internal(bool);
impl Internal {
    fn enter() -> Self {
        Self(COPYING.replace(true))
    }
}
impl Drop for Internal {
    fn drop(&mut self) {
        COPYING.set(self.0);
    }
}

fn bytes(blob: &ID3DBlob) -> &[u8] {
    unsafe {
        core::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
    }
}

fn normalize(
    context: &ID3D11DeviceContext,
    destination: &ID3D11Resource,
    inverted: bool,
) -> Result<(), String> {
    let standard = refresh(context, destination, None)?;
    let texture = destination
        .cast::<ID3D11Texture2D>()
        .map_err(|e| e.to_string())?;
    let device = unsafe { texture.GetDevice() }.map_err(|e| e.to_string())?;
    let mut desc = D3D11_TEXTURE2D_DESC::default();
    unsafe { texture.GetDesc(&raw mut desc) };
    if matches!(
        desc.Format,
        DXGI_FORMAT_R32_TYPELESS | DXGI_FORMAT_R32_FLOAT | DXGI_FORMAT_D32_FLOAT
    ) {
        let _internal = Internal::enter();
        if inverted {
            let copy = create_copy(&device, &texture, standard.as_raw())?;
            let shader = SHADER_OBJECT
                .lock()
                .unwrap()
                .clone()
                .ok_or("no depth copy shader")?;
            convert(context, &shader, &copy);
            unsafe { context.CopyResource(destination, &copy.standard) };
        } else {
            let source = unsafe { standard.GetResource() }.map_err(|e| e.to_string())?;
            unsafe { context.CopyResource(destination, &source) };
        }
    } else {
        return Err("mixed partial copy requires a single-channel float texture".into());
    }
    mark(destination, inverted);
    Ok(())
}

pub(crate) fn prepare_region(
    context: &ID3D11DeviceContext,
    destination: &ID3D11Resource,
    source: &ID3D11Resource,
    destination_subresource: u32,
    offset: [u32; 3],
    source_box: Option<&D3D11_BOX>,
) -> Result<(), String> {
    if !reversed(source) && !reversed(destination) {
        return Ok(());
    }
    let (destination_desc, source_desc) = compatible(destination, source)?;
    let size = source_box.map_or([source_desc.Width, source_desc.Height, 1], |region| {
        [
            region.right.saturating_sub(region.left),
            region.bottom.saturating_sub(region.top),
            region.back.saturating_sub(region.front),
        ]
    });
    let complete = destination_subresource == 0
        && offset == [0; 3]
        && size == [destination_desc.Width, destination_desc.Height, 1];
    if (destination_desc.BindFlags | source_desc.BindFlags)
        & D3D11_BIND_DEPTH_STENCIL.0.cast_unsigned()
        != 0
        && (!complete || source_box.is_some())
    {
        return Err("depth-stencil copies require a whole subresource".into());
    }
    if !complete && reversed(source) != reversed(destination) {
        normalize(context, destination, reversed(source))?;
    }
    Ok(())
}

pub(crate) fn count() -> u64 {
    COUNT.load(Ordering::Relaxed)
}

pub(crate) fn initialize(device: &ID3D11Device) -> Result<(), String> {
    for (slot, invert) in [(&SHADER_OBJECT, true), (&STANDARD_SHADER_OBJECT, false)] {
        let code = compile(invert)?;
        let mut shader = None;
        unsafe { device.CreateComputeShader(bytes(&code), None, Some(&raw mut shader)) }
            .map_err(|e| format!("depth copy shader: {e}"))?;
        *slot.lock().unwrap() = Some(shader.ok_or("no depth copy shader")?);
    }
    Ok(())
}

pub(crate) fn sync_sources() {
    for (slot, target) in SEEDS.iter().zip(REVERSED_TARGETS) {
        slot.store(
            crate::context::target_texture(target)
                .map_or(0, |field| unsafe { *field }),
            Ordering::Relaxed,
        );
    }
}

fn state(resource: &ID3D11Resource) -> Option<u32> {
    let mut value = 0;
    let mut size = 4;
    unsafe { resource.GetPrivateData(&STATE, &raw mut size, Some((&raw mut value).cast())) }
        .ok()?;
    Some(value)
}

pub(crate) fn reversed(resource: &ID3D11Resource) -> bool {
    state(resource).map_or_else(
        || {
            crate::active()
                && SEEDS
                    .iter()
                    .any(|seed| seed.load(Ordering::Relaxed) == resource.as_raw() as usize)
        },
        |value| matches!(value, DIRTY | CLEAN),
    )
}

pub(crate) fn mark(resource: &ID3D11Resource, reversed: bool) {
    let value = if reversed { DIRTY } else { STANDARD };
    let _ = unsafe { resource.SetPrivateData(&STATE, 4, Some((&raw const value).cast())) };
}

pub(crate) fn seeded_world(resource: &ID3D11Resource) -> bool {
    [0, 2]
        .into_iter()
        .any(|i| SEEDS[i].load(Ordering::Relaxed) == resource.as_raw() as usize)
}

fn private_interface<T: Interface>(resource: &ID3D11Resource, key: &GUID) -> Option<T> {
    let mut pointer: *mut c_void = core::ptr::null_mut();
    let mut size = 8;
    unsafe { resource.GetPrivateData(key, &raw mut size, Some((&raw mut pointer).cast())) }.ok()?;
    (!pointer.is_null()).then(|| unsafe { T::from_raw(pointer) })
}

fn store_interface(
    resource: &ID3D11Resource,
    key: &GUID,
    value: &impl Interface,
) -> Result<(), String> {
    unsafe {
        (resource.vtable().base__.SetPrivateDataInterface)(resource.as_raw(), key, value.as_raw())
    }
    .ok()
    .map_err(|e| e.to_string())
}

fn source_view(texture: &ID3D11Texture2D) -> Result<ID3D11ShaderResourceView, String> {
    let mut desc = D3D11_TEXTURE2D_DESC::default();
    unsafe { texture.GetDesc(&raw mut desc) };
    if desc.SampleDesc.Count != 1 || desc.ArraySize != 1 || desc.MipLevels != 1 {
        return Err("depth arrays, mip chains or multisampling are unsupported".into());
    }
    let format = match desc.Format {
        DXGI_FORMAT_R32G8X24_TYPELESS | DXGI_FORMAT_D32_FLOAT_S8X24_UINT => {
            DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS
        }
        DXGI_FORMAT_R32_TYPELESS | DXGI_FORMAT_R32_FLOAT | DXGI_FORMAT_D32_FLOAT => {
            DXGI_FORMAT_R32_FLOAT
        }
        DXGI_FORMAT_R24G8_TYPELESS | DXGI_FORMAT_D24_UNORM_S8_UINT => {
            DXGI_FORMAT_R24_UNORM_X8_TYPELESS
        }
        _ => {
            return Err(format!(
                "unsupported depth resource format {}",
                desc.Format.0
            ));
        }
    };
    let view_desc = D3D11_SHADER_RESOURCE_VIEW_DESC {
        Format: format,
        ViewDimension: D3D_SRV_DIMENSION_TEXTURE2D,
        Anonymous: D3D11_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D11_TEX2D_SRV {
                MostDetailedMip: 0,
                MipLevels: 1,
            },
        },
    };
    let device = unsafe { texture.GetDevice() }.map_err(|e| e.to_string())?;
    let mut view = None;
    unsafe {
        device.CreateShaderResourceView(texture, Some(&raw const view_desc), Some(&raw mut view))
    }
    .map_err(|e| e.to_string())?;
    view.ok_or_else(|| "no depth source view".into())
}

fn cached_copy(
    resource: &ID3D11Resource,
    input: &ID3D11ShaderResourceView,
) -> Result<Copy, String> {
    let texture = resource
        .cast::<ID3D11Texture2D>()
        .map_err(|e| e.to_string())?;
    let mut desc = D3D11_TEXTURE2D_DESC::default();
    unsafe { texture.GetDesc(&raw mut desc) };
    if let (Some(view), Some(target)) = (
        private_interface::<ID3D11ShaderResourceView>(resource, &VIEW),
        private_interface::<ID3D11UnorderedAccessView>(resource, &TARGET),
    ) {
        let standard = unsafe { view.GetResource() }
            .and_then(|resource| resource.cast::<ID3D11Texture2D>())
            .map_err(|e| e.to_string())?;
        return Ok(Copy {
            standard,
            view,
            target,
            engine_view: input.clone(),
            width: desc.Width,
            height: desc.Height,
        });
    }
    let device = unsafe { texture.GetDevice() }.map_err(|e| e.to_string())?;
    let copy = create_copy(&device, &texture, input.as_raw())?;
    store_interface(resource, &VIEW, &copy.view)?;
    store_interface(resource, &TARGET, &copy.target)?;
    Ok(copy)
}

fn refresh(
    context: &ID3D11DeviceContext,
    resource: &ID3D11Resource,
    input: Option<&ID3D11ShaderResourceView>,
) -> Result<ID3D11ShaderResourceView, String> {
    if matches!(state(resource), Some(CLEAN | STANDARD_CLEAN))
        && let Some(view) = private_interface(resource, &VIEW)
    {
        return Ok(view);
    }
    let input = match input {
        Some(input) => input.clone(),
        None => source_view(
            &resource
                .cast::<ID3D11Texture2D>()
                .map_err(|e| e.to_string())?,
        )?,
    };
    let copy = cached_copy(resource, &input)?;
    let inverted = reversed(resource);
    let shader = (if inverted {
        &SHADER_OBJECT
    } else {
        &STANDARD_SHADER_OBJECT
    })
    .lock()
    .unwrap()
    .clone()
    .ok_or("no depth copy shader")?;
    let value = if inverted { CLEAN } else { STANDARD_CLEAN };
    unsafe { resource.SetPrivateData(&STATE, 4, Some((&raw const value).cast())) }
        .map_err(|e| e.to_string())?;
    convert(context, &shader, &copy);
    COUNT.fetch_add(1, Ordering::Relaxed);
    Ok(copy.view)
}

/// The conventional copy's texture, once a reader has asked for one.
pub(crate) fn standard_texture(resource: &ID3D11Resource) -> Option<ID3D11Texture2D> {
    let view: ID3D11ShaderResourceView = private_interface(resource, &VIEW)?;
    unsafe { view.GetResource() }.ok()?.cast().ok()
}

/// Makes the copy of a world target that has none yet, for readers past the context.
pub(crate) fn ensure(context: &ID3D11DeviceContext, resource: &ID3D11Resource) {
    if private_interface::<ID3D11ShaderResourceView>(resource, &VIEW).is_none()
        && let Err(error) = refresh(context, resource, None)
    {
        report(&error);
    }
}

pub(crate) fn report(message: &str) {
    if UNSUPPORTED.fetch_add(1, Ordering::Relaxed) == 0 {
        crate::log(&format!("depth compatibility gap: {message}"));
    }
}

/// Refreshes `resource`'s stale copy while a reader can sample it without binding it again: one
/// that kept it bound, or one past the context.
pub(crate) fn run(context: &ID3D11DeviceContext, resource: &ID3D11Resource) {
    if COPYING.get() || matches!(state(resource), Some(CLEAN | STANDARD_CLEAN)) {
        return;
    }
    let Some(view) = private_interface::<ID3D11ShaderResourceView>(resource, &VIEW) else {
        return;
    };
    if !crate::vulkan::engaged() && !bound(view.as_raw() as usize) {
        return;
    }
    if let Err(error) = refresh(context, resource, None) {
        report(&error);
    }
}

fn bound(view: usize) -> bool {
    BINDINGS
        .lock()
        .unwrap()
        .as_ref()
        .is_some_and(|b| b.bound.contains_key(&view))
}

/// Records the views sent to `count` resource slots of `stage` from `start`; `None` unbinds them.
pub(crate) fn bind_views(stage: usize, start: u32, count: u32, views: Option<&[*mut c_void]>) {
    let mut bindings = BINDINGS.lock().unwrap();
    let Some(b) = bindings.as_mut() else {
        return;
    };
    for i in 0..count as usize {
        let Some(slot) = b.slots[stage].get_mut(start as usize + i) else {
            break;
        };
        let view = views.map_or(0, |v| v[i] as usize);
        let copy = if b.views.contains(&view) { view } else { 0 };
        if *slot == copy {
            continue;
        }
        if *slot != 0
            && let Some(n) = b.bound.get_mut(slot)
        {
            *n -= 1;
            if *n == 0 {
                b.bound.remove(slot);
            }
        }
        if copy != 0 {
            *b.bound.entry(copy).or_default() += 1;
        }
        *slot = copy;
    }
}

/// The stages and slots that hold `view`.
fn held(view: usize) -> Vec<(usize, u32)> {
    let bindings = BINDINGS.lock().unwrap();
    let Some(b) = bindings.as_ref().filter(|b| b.bound.contains_key(&view)) else {
        return Vec::new();
    };
    b.slots
        .iter()
        .enumerate()
        .flat_map(|(stage, slots)| {
            slots
                .iter()
                .enumerate()
                .filter(move |&(_, &bound)| bound == view)
                .map(move |(slot, _)| (stage, u32::try_from(slot).unwrap()))
        })
        .collect()
}

/// Forgets every bound view after the context's state is cleared.
pub(crate) fn unbind_views() {
    if let Some(b) = BINDINGS.lock().unwrap().as_mut() {
        b.slots = [[0; SLOTS]; 6];
        b.bound.clear();
    }
}

pub(crate) fn stand_in(context: &ID3D11DeviceContext, view: *mut c_void) -> Option<*mut c_void> {
    if COPYING.get() {
        return None;
    }
    let input = unsafe { ID3D11ShaderResourceView::from_raw_borrowed(&view) }?;
    let mut desc = D3D11_SHADER_RESOURCE_VIEW_DESC::default();
    unsafe { input.GetDesc(&raw mut desc) };
    if !matches!(
        desc.Format,
        DXGI_FORMAT_R32_FLOAT
            | DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS
            | DXGI_FORMAT_R24_UNORM_X8_TYPELESS
    ) {
        return None;
    }
    let resource = unsafe { input.GetResource() }.ok()?;
    if !reversed(&resource)
        && private_interface::<ID3D11ShaderResourceView>(&resource, &VIEW).is_none()
    {
        return None;
    }
    match refresh(context, &resource, Some(input)) {
        Ok(copy) => Some(copy.as_raw()),
        Err(error) => {
            report(&error);
            None
        }
    }
}

pub(crate) fn copied(
    context: &ID3D11DeviceContext,
    destination: &ID3D11Resource,
    source: &ID3D11Resource,
) {
    if (reversed(source) || reversed(destination))
        && let Err(error) = compatible(destination, source).and_then(|(destination, source)| {
            if destination.Width == source.Width && destination.Height == source.Height {
                Ok(())
            } else {
                Err("full depth copy dimensions differ".into())
            }
        })
    {
        report(&error);
        return;
    }
    copied_region(context, destination, source);
}

fn compatible(
    destination: &ID3D11Resource,
    source: &ID3D11Resource,
) -> Result<(D3D11_TEXTURE2D_DESC, D3D11_TEXTURE2D_DESC), String> {
    let mut descriptions = Vec::new();
    for resource in [destination, source] {
        let texture = resource
            .cast::<ID3D11Texture2D>()
            .map_err(|e| e.to_string())?;
        let mut desc = D3D11_TEXTURE2D_DESC::default();
        unsafe { texture.GetDesc(&raw mut desc) };
        if desc.SampleDesc.Count != 1 || desc.ArraySize != 1 || desc.MipLevels != 1 {
            return Err("depth arrays, mip chains or multisampling are unsupported".into());
        }
        descriptions.push(desc);
    }
    let group = |format| match format {
        DXGI_FORMAT_R32G8X24_TYPELESS | DXGI_FORMAT_D32_FLOAT_S8X24_UINT => 1,
        DXGI_FORMAT_R32_TYPELESS | DXGI_FORMAT_R32_FLOAT | DXGI_FORMAT_D32_FLOAT => 2,
        DXGI_FORMAT_R24G8_TYPELESS | DXGI_FORMAT_D24_UNORM_S8_UINT => 3,
        _ => 0,
    };
    if group(descriptions[0].Format) == 0
        || group(descriptions[0].Format) != group(descriptions[1].Format)
    {
        return Err(format!(
            "incompatible depth copy formats {} -> {}",
            descriptions[1].Format.0, descriptions[0].Format.0
        ));
    }
    Ok((descriptions[0], descriptions[1]))
}

pub(crate) fn copied_region(
    context: &ID3D11DeviceContext,
    destination: &ID3D11Resource,
    source: &ID3D11Resource,
) {
    if COPYING.get() {
        return;
    }
    if reversed(source)
        || state(destination).is_some()
        || SEEDS
            .iter()
            .any(|seed| seed.load(Ordering::Relaxed) == destination.as_raw() as usize)
    {
        let inverted = reversed(source);
        mark(destination, inverted);
        COPIED.fetch_add(1, Ordering::Relaxed);
        if inverted {
            ensure(context, destination);
            crate::vulkan::track(destination);
        }
        run(context, destination);
    }
}

pub(crate) fn release() {
    SHADER_OBJECT.lock().unwrap().take();
    STANDARD_SHADER_OBJECT.lock().unwrap().take();
    for seed in &SEEDS {
        seed.store(0, Ordering::Relaxed);
    }
}

pub(crate) fn counters() -> String {
    format!(
        "depth copies {} (tracked transfers {}, unsupported {})",
        count(),
        COPIED.load(Ordering::Relaxed),
        UNSUPPORTED.load(Ordering::Relaxed)
    )
}

fn create_copy(
    device: &ID3D11Device,
    texture: &ID3D11Texture2D,
    engine_view: *mut c_void,
) -> Result<Copy, String> {
    let mut depth = D3D11_TEXTURE2D_DESC::default();
    unsafe { texture.GetDesc(&raw mut depth) };
    if depth.SampleDesc.Count != 1 || depth.ArraySize != 1 || depth.MipLevels != 1 {
        return Err("depth arrays, mip chains or multisampling are unsupported".into());
    }
    let desc = D3D11_TEXTURE2D_DESC {
        Width: depth.Width,
        Height: depth.Height,
        MipLevels: 1,
        ArraySize: 1,
        Format: DXGI_FORMAT_R32_FLOAT,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Usage: D3D11_USAGE_DEFAULT,
        BindFlags: (D3D11_BIND_SHADER_RESOURCE.0 | D3D11_BIND_UNORDERED_ACCESS.0) as u32,
        ..Default::default()
    };
    let mut standard = None;
    unsafe { device.CreateTexture2D(&raw const desc, None, Some(&raw mut standard)) }
        .map_err(|e| format!("copy texture: {e}"))?;
    let standard = standard.ok_or("copy texture")?;
    let mut view = None;
    unsafe { device.CreateShaderResourceView(&standard, None, Some(&raw mut view)) }
        .map_err(|e| format!("copy view: {e}"))?;
    let mut unordered = None;
    unsafe { device.CreateUnorderedAccessView(&standard, None, Some(&raw mut unordered)) }
        .map_err(|e| format!("copy target: {e}"))?;
    let view = view.ok_or("copy view")?;
    BINDINGS
        .lock()
        .unwrap()
        .get_or_insert_with(|| Bindings {
            views: HashSet::new(),
            slots: [[0; SLOTS]; 6],
            bound: HashMap::new(),
        })
        .views
        .insert(view.as_raw() as usize);
    Ok(Copy {
        standard,
        view,
        target: unordered.ok_or("copy target")?,
        engine_view: unsafe { ID3D11ShaderResourceView::from_raw_borrowed(&engine_view) }
            .cloned()
            .ok_or("no source depth view")?,
        width: depth.Width,
        height: depth.Height,
    })
}

fn compile(invert: bool) -> Result<ID3DBlob, String> {
    let source = if invert {
        SHADER.to_owned()
    } else {
        SHADER.replace("1.0 - depth[id.xy]", "depth[id.xy]")
    };
    let mut code = None;
    let mut errors = None;
    unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            s!("reversed-z"),
            None,
            None,
            s!("main"),
            s!("cs_5_0"),
            0,
            0,
            &raw mut code,
            Some(&raw mut errors),
        )
    }
    .map_err(|e| {
        let detail = errors.map(|b: ID3DBlob| unsafe {
            String::from_utf8_lossy(core::slice::from_raw_parts(
                b.GetBufferPointer().cast::<u8>(),
                b.GetBufferSize(),
            ))
            .into_owned()
        });
        format!("compile: {e} {}", detail.unwrap_or_default())
    })?;
    code.ok_or_else(|| "compile: no code".into())
}

/// Runs `run` and gives back the shader resource slots a copy disturbs: compute slot 0, which it
/// binds to the source, and the slots holding `copy`, which binding the copy for writing empties.
fn preserve_resources(
    context: &ID3D11DeviceContext,
    copy: &ID3D11ShaderResourceView,
    run: impl FnOnce(),
) {
    let held = held(copy.as_raw() as usize);
    let mut compute = [None];
    unsafe { context.CSGetShaderResources(0, Some(&mut compute)) };
    run();
    unsafe { context.CSSetShaderResources(0, Some(&compute)) };
    let views = [Some(copy.clone())];
    for (stage, slot) in held {
        let views = Some(&views[..]);
        unsafe {
            match stage {
                PIXEL => context.PSSetShaderResources(slot, views),
                COMPUTE => context.CSSetShaderResources(slot, views),
                VERTEX => context.VSSetShaderResources(slot, views),
                GEOMETRY => context.GSSetShaderResources(slot, views),
                HULL => context.HSSetShaderResources(slot, views),
                _ => context.DSSetShaderResources(slot, views),
            }
        }
    }
}

fn convert(context: &ID3D11DeviceContext, shader: &ID3D11ComputeShader, c: &Copy) {
    let _internal = Internal::enter();
    preserve_resources(context, &c.view, || {
        crate::context::without_targets(context, || unsafe {
            let mut previous_shader = None;
            let mut instances: [Option<ID3D11ClassInstance>; D3D11_SHADER_MAX_INTERFACES as usize] =
                core::array::from_fn(|_| None);
            let mut instance_count = D3D11_SHADER_MAX_INTERFACES;
            context.CSGetShader(
                &raw mut previous_shader,
                Some(instances.as_mut_ptr()),
                Some(&raw mut instance_count),
            );
            // ENB leaves pointers it does not hand over in the slots past the returned count.
            let instance_count = instance_count.min(D3D11_SHADER_MAX_INTERFACES) as usize;
            instances[instance_count..]
                .iter_mut()
                .for_each(|slot| core::mem::forget(slot.take()));
            let mut outputs: [Option<ID3D11UnorderedAccessView>; 1] = Default::default();
            context.CSGetUnorderedAccessViews(0, Some(&mut outputs));

            context.CSSetShader(shader, None);
            crate::context::set_compute_resource_as_is(context, c.engine_view.as_raw());
            let unordered = [Some(c.target.clone())];
            context.CSSetUnorderedAccessViews(0, 1, Some(unordered.as_ptr()), None);
            context.Dispatch(c.width.div_ceil(GROUP), c.height.div_ceil(GROUP), 1);

            context.CSSetUnorderedAccessViews(0, 1, Some(outputs.as_ptr()), None);
            context.CSSetShader(
                previous_shader.as_ref(),
                Some(&instances[..instance_count]),
            );
        });
    });
}

#[cfg(test)]
fn alias_index(view: *mut c_void, textures: &[usize]) -> Option<usize> {
    let view = unsafe { ID3D11ShaderResourceView::from_raw_borrowed(&view) }?;
    let mut desc = D3D11_SHADER_RESOURCE_VIEW_DESC::default();
    unsafe { view.GetDesc(&raw mut desc) };
    if !matches!(
        desc.Format,
        DXGI_FORMAT_R32_FLOAT
            | DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS
            | DXGI_FORMAT_R24_UNORM_X8_TYPELESS
    ) {
        return None;
    }
    let resource = unsafe { view.GetResource() }.ok()?;
    textures
        .iter()
        .position(|&texture| texture == resource.as_raw() as usize)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use windows::Win32::Graphics::Direct3D::D3D_SRV_DIMENSION_TEXTURE2D;
    use windows::Win32::Graphics::Direct3D11::*;
    use windows::Win32::Graphics::Dxgi::Common::{
        DXGI_FORMAT_R32G8X24_TYPELESS, DXGI_FORMAT_X32_TYPELESS_G8X24_UINT,
    };

    #[allow(clippy::too_many_lines)]
    pub(crate) fn exercise(device: &ID3D11Device, context: &ID3D11DeviceContext) {
        crate::context::install_test_hooks(context);
        let depth_desc = D3D11_TEXTURE2D_DESC {
            Width: 13,
            Height: 11,
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
        unsafe { device.CreateTexture2D(&raw const depth_desc, None, Some(&raw mut depth)) }
            .unwrap();
        let depth = depth.unwrap();
        let make_view = |format| {
            let desc = D3D11_SHADER_RESOURCE_VIEW_DESC {
                Format: format,
                ViewDimension: D3D_SRV_DIMENSION_TEXTURE2D,
                Anonymous: D3D11_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2D: D3D11_TEX2D_SRV {
                        MostDetailedMip: 0,
                        MipLevels: 1,
                    },
                },
            };
            let mut view = None;
            unsafe {
                device.CreateShaderResourceView(&depth, Some(&raw const desc), Some(&raw mut view))
            }
            .unwrap();
            view.unwrap()
        };
        let original = make_view(DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS);
        let alias = make_view(DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS);
        let stencil = make_view(DXGI_FORMAT_X32_TYPELESS_G8X24_UINT);
        let textures = [depth.as_raw() as usize];
        assert_ne!(original.as_raw(), alias.as_raw());
        assert_eq!(alias_index(original.as_raw(), &textures), Some(0));
        assert_eq!(alias_index(alias.as_raw(), &textures), Some(0));
        assert_eq!(alias_index(stencil.as_raw(), &textures), None);
        assert_eq!(alias_index(alias.as_raw(), &[0]), None);

        let data: Vec<f32> = (0..143)
            .map(|i| f32::from(u16::try_from(i).unwrap()) / 143.0)
            .collect();
        let source_desc = D3D11_TEXTURE2D_DESC {
            Format: DXGI_FORMAT_R32_FLOAT,
            BindFlags: D3D11_BIND_SHADER_RESOURCE.0.cast_unsigned(),
            ..depth_desc
        };
        let initial = D3D11_SUBRESOURCE_DATA {
            pSysMem: data.as_ptr().cast(),
            SysMemPitch: 13 * 4,
            ..Default::default()
        };
        let mut source = None;
        unsafe {
            device.CreateTexture2D(
                &raw const source_desc,
                Some(&raw const initial),
                Some(&raw mut source),
            )
        }
        .unwrap();
        let source = source.unwrap();
        let mut input = None;
        unsafe { device.CreateShaderResourceView(&source, None, Some(&raw mut input)) }.unwrap();
        let input = input.unwrap();
        let copy = create_copy(device, &source, input.as_raw()).unwrap();
        let om_copy = create_copy(device, &source, input.as_raw()).unwrap();
        let cs_copy = create_copy(device, &source, input.as_raw()).unwrap();
        let code = compile(true).unwrap();
        let code = unsafe {
            core::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
        };
        let mut shader = None;
        unsafe { device.CreateComputeShader(code, None, Some(&raw mut shader)) }.unwrap();
        let shader = shader.unwrap();
        let mut depth_view = None;
        let dsv_desc = D3D11_DEPTH_STENCIL_VIEW_DESC {
            Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
            ViewDimension: D3D11_DSV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        unsafe {
            device.CreateDepthStencilView(
                &depth,
                Some(&raw const dsv_desc),
                Some(&raw mut depth_view),
            )
        }
        .unwrap();
        let depth_view = depth_view.unwrap();
        unsafe {
            context.VSSetShaderResources(31, Some(&[Some(copy.view.clone())]));
            context.HSSetShaderResources(32, Some(&[Some(copy.view.clone())]));
            context.DSSetShaderResources(33, Some(&[Some(copy.view.clone())]));
            context.GSSetShaderResources(34, Some(&[Some(copy.view.clone())]));
            context.PSSetShaderResources(35, Some(&[Some(copy.view.clone())]));
            context.CSSetShaderResources(36, Some(&[Some(copy.view.clone())]));
            context.CSSetUnorderedAccessViews(
                0,
                1,
                Some([Some(cs_copy.target.clone())].as_ptr()),
                None,
            );
            context.OMSetRenderTargetsAndUnorderedAccessViews(
                None,
                &depth_view,
                3,
                1,
                Some([Some(om_copy.target.clone())].as_ptr()),
                None,
            );
        }
        convert(context, &shader, &copy);
        macro_rules! restored {
            ($($get:ident($slot:expr)),*) => { $(
                let mut views = [None];
                unsafe { context.$get($slot, Some(&mut views)) };
                assert_eq!(views[0].as_ref().map(Interface::as_raw), Some(copy.view.as_raw()));
            )* };
        }
        restored!(
            VSGetShaderResources(31),
            HSGetShaderResources(32),
            DSGetShaderResources(33),
            GSGetShaderResources(34),
            PSGetShaderResources(35),
            CSGetShaderResources(36)
        );
        let mut om_uav = [None];
        let mut cs_uav = [None];
        let mut restored_depth = None;
        unsafe {
            context.OMGetRenderTargetsAndUnorderedAccessViews(
                None,
                Some(&raw mut restored_depth),
                3,
                Some(&mut om_uav),
            );
            context.CSGetUnorderedAccessViews(0, Some(&mut cs_uav));
        }
        assert_eq!(
            om_uav[0].as_ref().map(Interface::as_raw),
            Some(om_copy.target.as_raw())
        );
        assert_eq!(
            cs_uav[0].as_ref().map(Interface::as_raw),
            Some(cs_copy.target.as_raw())
        );
        assert_eq!(
            restored_depth.as_ref().map(Interface::as_raw),
            Some(depth_view.as_raw())
        );
        let read_desc = D3D11_TEXTURE2D_DESC {
            Usage: D3D11_USAGE_STAGING,
            BindFlags: 0,
            CPUAccessFlags: D3D11_CPU_ACCESS_READ.0.cast_unsigned(),
            ..source_desc
        };
        let mut read = None;
        unsafe { device.CreateTexture2D(&raw const read_desc, None, Some(&raw mut read)) }.unwrap();
        let read = read.unwrap();
        unsafe {
            let standard = copy.view.GetResource().unwrap();
            context.CopyResource(&read, &standard);
            let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
            context
                .Map(&read, 0, D3D11_MAP_READ, 0, Some(&raw mut mapped))
                .unwrap();
            for y in 0..11 {
                let row = mapped.pData.cast::<u8>().add(y * mapped.RowPitch as usize);
                for x in 0..13 {
                    let bytes = row.add(x * 4).cast::<[u8; 4]>().read_unaligned();
                    assert_eq!(f32::from_ne_bytes(bytes), 1.0 - data[y * 13 + x]);
                }
            }
            context.Unmap(&read, 0);
            context.ClearState();
        }
        initialize(device).unwrap();
        let resource = source.cast::<ID3D11Resource>().unwrap();
        mark(&resource, true);
        let mut destination = None;
        unsafe { device.CreateTexture2D(&raw const source_desc, None, Some(&raw mut destination)) }
            .unwrap();
        let destination = destination.unwrap();
        let destination_resource = destination.cast::<ID3D11Resource>().unwrap();
        unsafe { context.CopyResource(&destination, &source) };
        copied(context, &destination_resource, &resource);
        assert!(reversed(&destination_resource));
        let destination_view = source_view(&destination).unwrap();
        let stand_in = stand_in(context, destination_view.as_raw()).unwrap();
        assert_ne!(stand_in, destination_view.as_raw());
        let stand_in = unsafe { ID3D11ShaderResourceView::from_raw_borrowed(&stand_in) }
            .unwrap()
            .clone();
        let read_first = |resource: &ID3D11Resource| unsafe {
            context.CopyResource(&read, resource);
            let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
            context
                .Map(&read, 0, D3D11_MAP_READ, 0, Some(&raw mut mapped))
                .unwrap();
            let first = mapped.pData.cast::<f32>().read_unaligned();
            context.Unmap(&read, 0);
            first
        };
        let standard_resource = unsafe { stand_in.GetResource() }.unwrap();
        assert_eq!(read_first(&standard_resource), 1.0);
        let mut chained = None;
        unsafe { device.CreateTexture2D(&raw const source_desc, None, Some(&raw mut chained)) }
            .unwrap();
        let chained = chained.unwrap();
        let chained_resource = chained.cast::<ID3D11Resource>().unwrap();
        unsafe { context.CopyResource(&chained, &destination) };
        copied(context, &chained_resource, &destination_resource);
        assert!(reversed(&chained_resource));
        let chained_view = source_view(&chained).unwrap();
        assert!(super::stand_in(context, chained_view.as_raw()).is_some());
        mark(&resource, false);
        unsafe { context.CopyResource(&destination, &source) };
        copied(context, &destination_resource, &resource);
        assert!(!reversed(&destination_resource));
        assert_eq!(
            super::stand_in(context, destination_view.as_raw()).unwrap(),
            stand_in.as_raw()
        );
        assert_eq!(
            read_first(&standard_resource),
            0.0,
            "an existing standard view must refresh when the source changes convention"
        );
        partial_copy(device, context, depth_desc, &input);
        crate::context::test_shader_readback(device, context, &depth_view);
        crate::context::test_state_readback(device, context, &depth_view);
        deferred_refresh(context, &depth.cast().unwrap(), &depth_view, &original);
        release();
        crate::context::remove();
    }

    /// A copy nobody reads is left stale when its target is unbound and refreshed by the reader
    /// that binds it; one a reader still has bound is refreshed at the unbind.
    fn deferred_refresh(
        context: &ID3D11DeviceContext,
        depth: &ID3D11Resource,
        depth_view: &ID3D11DepthStencilView,
        engine_view: &ID3D11ShaderResourceView,
    ) {
        let drawn_and_unbound = || unsafe {
            context.OMSetRenderTargets(None, depth_view);
            mark(depth, true);
            context.OMSetRenderTargets(None, None);
        };
        mark(depth, true);
        drawn_and_unbound();
        assert_eq!(
            state(depth),
            Some(DIRTY),
            "an unread copy waits for its reader"
        );
        unsafe { context.PSSetShaderResources(0, Some(&[Some(engine_view.clone())])) };
        assert_eq!(
            state(depth),
            Some(CLEAN),
            "the reader's bind refreshes the copy"
        );
        drawn_and_unbound();
        assert_eq!(
            state(depth),
            Some(CLEAN),
            "a copy still bound is refreshed at the unbind"
        );
        unsafe { context.PSSetShaderResources(0, Some(&[None])) };
        drawn_and_unbound();
        assert_eq!(state(depth), Some(DIRTY), "an unbound copy waits again");
        unsafe { context.ClearState() };
    }

    fn partial_copy(
        device: &ID3D11Device,
        context: &ID3D11DeviceContext,
        desc: D3D11_TEXTURE2D_DESC,
        bound_input: &ID3D11ShaderResourceView,
    ) {
        let desc = D3D11_TEXTURE2D_DESC {
            Format: DXGI_FORMAT_R32_FLOAT,
            BindFlags: D3D11_BIND_SHADER_RESOURCE.0.cast_unsigned(),
            ..desc
        };
        let mut textures = Vec::new();
        for value in [0.2f32, 0.6] {
            let data = [value; 143];
            let initial = D3D11_SUBRESOURCE_DATA {
                pSysMem: data.as_ptr().cast(),
                SysMemPitch: 13 * 4,
                ..Default::default()
            };
            let mut texture = None;
            unsafe {
                device.CreateTexture2D(
                    &raw const desc,
                    Some(&raw const initial),
                    Some(&raw mut texture),
                )
            }
            .unwrap();
            textures.push(texture.unwrap());
        }
        let source = textures[0].cast::<ID3D11Resource>().unwrap();
        let destination = textures[1].cast::<ID3D11Resource>().unwrap();
        mark(&source, true);
        mark(&destination, false);
        unsafe { context.PSSetShaderResources(7, Some(&[Some(bound_input.clone())])) };
        let region = D3D11_BOX {
            left: 0,
            top: 0,
            front: 0,
            right: 4,
            bottom: 4,
            back: 1,
        };
        prepare_region(context, &destination, &source, 0, [0; 3], Some(&region)).unwrap();
        let mut restored = [None];
        unsafe { context.PSGetShaderResources(7, Some(&mut restored)) };
        assert_eq!(
            restored[0].as_ref().map(Interface::as_raw),
            Some(bound_input.as_raw())
        );
        unsafe {
            context.CopySubresourceRegion(
                &destination,
                0,
                0,
                0,
                0,
                &source,
                0,
                Some(&raw const region),
            );
        }
        copied_region(context, &destination, &source);
        assert!(reversed(&destination));
        let desc = D3D11_TEXTURE2D_DESC {
            Usage: D3D11_USAGE_STAGING,
            BindFlags: 0,
            CPUAccessFlags: D3D11_CPU_ACCESS_READ.0.cast_unsigned(),
            ..desc
        };
        let mut read = None;
        unsafe { device.CreateTexture2D(&raw const desc, None, Some(&raw mut read)) }.unwrap();
        let read = read.unwrap();
        unsafe {
            context.CopyResource(&read, &destination);
            let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
            context
                .Map(&read, 0, D3D11_MAP_READ, 0, Some(&raw mut mapped))
                .unwrap();
            let first = mapped.pData.cast::<u8>();
            let last = first.add(10 * mapped.RowPitch as usize + 12 * 4);
            let depth =
                |pointer: *const u8| f32::from_ne_bytes(pointer.cast::<[u8; 4]>().read_unaligned());
            assert!((depth(first) - 0.2).abs() < 0.000_001);
            assert!(
                (depth(last) - 0.4).abs() < 0.000_001,
                "untouched depth must be normalized"
            );
            context.Unmap(&read, 0);
            context.ClearState();
        }
    }
}
