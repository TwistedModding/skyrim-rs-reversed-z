//! Drawcall Limit Fix draws the world through Community Shaders' own Vulkan pipelines on DXVK's
//! device, past the D3D11 context. It takes its entry points from `vkGetDeviceProcAddr` and
//! `vkGetInstanceProcAddr`, so the resolutions made from CommunityShaders.dll are answered with
//! wrappers while DXVK's own keep the real pointers. A rendering whose depth attachment is a
//! reversed world image (found through DXVK's interop interface) gets its pipelines' compare and
//! biases mirrored (made dynamic at creation, set at each bind), its viewport ranges [a, b] made
//! [1 - a, 1 - b] for the unchanged clip z, and its clears 1 - value.

use core::ffi::{CStr, c_char, c_void};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{LazyLock, Mutex, OnceLock};

use ash::vk;
use ash::vk::Handle;
use windows::Win32::Foundation::HMODULE;
use windows::Win32::Graphics::Direct3D11::ID3D11Resource;
use windows::Win32::System::Diagnostics::Debug::RtlCaptureStackBackTrace;
use windows::Win32::System::LibraryLoader::{
    GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
    GetModuleHandleExW, GetModuleHandleW, GetProcAddress, LoadLibraryW,
};
use windows::core::{GUID, HRESULT, Interface, PCWSTR, s, w};

use crate::context::original;

type GpaFn = unsafe extern "system" fn(usize, *const c_char) -> Option<unsafe extern "system" fn()>;
type CreatePipelinesFn = unsafe extern "system" fn(
    vk::Device,
    vk::PipelineCache,
    u32,
    *const vk::GraphicsPipelineCreateInfo<'static>,
    *const vk::AllocationCallbacks<'static>,
    *mut vk::Pipeline,
) -> vk::Result;
type DestroyPipelineFn =
    unsafe extern "system" fn(vk::Device, vk::Pipeline, *const vk::AllocationCallbacks<'static>);
type BindPipelineFn = unsafe extern "system" fn(vk::CommandBuffer, vk::PipelineBindPoint, vk::Pipeline);
type SetViewportFn = unsafe extern "system" fn(vk::CommandBuffer, u32, u32, *const vk::Viewport);
type SetViewportCountFn = unsafe extern "system" fn(vk::CommandBuffer, u32, *const vk::Viewport);
type SetBiasFn = unsafe extern "system" fn(vk::CommandBuffer, f32, f32, f32);
type SetCompareFn = unsafe extern "system" fn(vk::CommandBuffer, vk::CompareOp);
type BeginRenderingFn =
    unsafe extern "system" fn(vk::CommandBuffer, *const vk::RenderingInfo<'static>);
type EndRenderingFn = unsafe extern "system" fn(vk::CommandBuffer);
type ClearImageFn = unsafe extern "system" fn(
    vk::CommandBuffer,
    vk::Image,
    vk::ImageLayout,
    *const vk::ClearDepthStencilValue,
    u32,
    *const vk::ImageSubresourceRange,
);
type ClearAttachmentsFn = unsafe extern "system" fn(
    vk::CommandBuffer,
    u32,
    *const vk::ClearAttachment,
    u32,
    *const vk::ClearRect,
);
type CreateViewFn = unsafe extern "system" fn(
    vk::Device,
    *const vk::ImageViewCreateInfo<'static>,
    *const vk::AllocationCallbacks<'static>,
    *mut vk::ImageView,
) -> vk::Result;
type DestroyViewFn =
    unsafe extern "system" fn(vk::Device, vk::ImageView, *const vk::AllocationCallbacks<'static>);
type BeginBufferFn = unsafe extern "system" fn(
    vk::CommandBuffer,
    *const vk::CommandBufferBeginInfo<'static>,
) -> vk::Result;
type GetHandlesFn =
    unsafe extern "system" fn(*mut c_void, *mut vk::Instance, *mut vk::PhysicalDevice, *mut vk::Device);
type GetImageInfoFn = unsafe extern "system" fn(
    *mut c_void,
    *mut vk::Image,
    *mut vk::ImageLayout,
    *mut vk::ImageCreateInfo<'static>,
) -> HRESULT;
type ReleaseFn = unsafe extern "system" fn(*mut c_void) -> u32;

/// DXVK's `IDXGIVkInteropDevice` and `IDXGIVkInteropSurface` (zlib, `dxgi_interfaces.h`).
const INTEROP_DEVICE: GUID = GUID::from_u128(0xe2ef5fa5_dc21_4af7_90c4_f67ef6a09323);
const INTEROP_SURFACE: GUID = GUID::from_u128(0x5546cf8c_77e7_4341_b05d_8d4d5000e77d);

/// The exported `vkGetInstanceProcAddr` and `vkGetDeviceProcAddr` of vulkan-1.dll and of
/// Streamline's interposer, which DXVK may load in its place.
static GPA_HOOKS: [Mutex<Option<hook::Hook<GpaFn>>>; 4] = [const { Mutex::new(None) }; 4];
static LOADER_DEVICE_GPA: AtomicUsize = AtomicUsize::new(0);
static HANDED_DEVICE_GPA: AtomicUsize = AtomicUsize::new(0);
static DEVICE: AtomicUsize = AtomicUsize::new(0);
static SET_COMPARE: AtomicUsize = AtomicUsize::new(0);
static SET_BIAS: AtomicUsize = AtomicUsize::new(0);
static OWN_RANGE: OnceLock<(usize, usize)> = OnceLock::new();
static CS_RANGE: OnceLock<(usize, usize)> = OnceLock::new();
static RESOLVED: AtomicU64 = AtomicU64::new(0);
static WORLD_PIPELINES: AtomicU64 = AtomicU64::new(0);
static WORLD_RENDERINGS: AtomicU64 = AtomicU64::new(0);
static WORLD_VIEWS: AtomicU64 = AtomicU64::new(0);
static WORLD_CLEARS: AtomicU64 = AtomicU64::new(0);
static NOTES: AtomicU64 = AtomicU64::new(0);

struct World {
    resource: usize,
    format: vk::Format,
}
struct PipelineDepth {
    compare: vk::CompareOp,
    /// The static bias factors made dynamic, sent at each bind.
    bias: Option<[f32; 3]>,
}
#[derive(Default)]
struct Buffer {
    reversed: bool,
    viewports: Option<(u32, Vec<vk::Viewport>, bool)>,
    bias: Option<[f32; 3]>,
    compare: Option<vk::CompareOp>,
    pipeline: u64,
}

static IMAGES: LazyLock<Mutex<HashMap<u64, World>>> = LazyLock::new(Mutex::default);
static VIEWS: LazyLock<Mutex<HashMap<u64, u64>>> = LazyLock::new(Mutex::default);
static PIPELINES: LazyLock<Mutex<HashMap<u64, PipelineDepth>>> = LazyLock::new(Mutex::default);
static BUFFERS: LazyLock<Mutex<HashMap<u64, Buffer>>> = LazyLock::new(Mutex::default);

fn note(message: &str) {
    if NOTES.fetch_add(1, Ordering::Relaxed) < 24 {
        crate::log(&format!("vulkan: {message}"));
    }
}

fn pointer<F: Copy>(slot: &AtomicUsize) -> Option<F> {
    const { assert!(size_of::<F>() == size_of::<usize>()) };
    let value = slot.load(Ordering::Relaxed);
    (value != 0).then(|| unsafe { core::mem::transmute_copy(&value) })
}

fn module_range(module: HMODULE) -> (usize, usize) {
    let base = module.0 as usize;
    let headers = base + unsafe { *((base + 0x3c) as *const u32) } as usize;
    let size = unsafe { *((headers + 0x50) as *const u32) } as usize;
    (base, base + size)
}

fn own_range() -> (usize, usize) {
    *OWN_RANGE.get_or_init(|| {
        let mut module = HMODULE::default();
        unsafe {
            GetModuleHandleExW(
                GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                PCWSTR(install as *const u16),
                &raw mut module,
            )
        }
        .map_or((0, 0), |()| module_range(module))
    })
}

fn community_shaders_range() -> Option<(usize, usize)> {
    CS_RANGE
        .get()
        .copied()
        .or_else(|| {
            let module = unsafe { GetModuleHandleW(w!("CommunityShaders.dll")) }.ok()?;
            let range = module_range(module);
            let _ = CS_RANGE.set(range);
            Some(range)
        })
}

/// The nearest caller outside this module is CommunityShaders.dll.
fn from_community_shaders() -> bool {
    let Some((start, end)) = community_shaders_range() else {
        return false;
    };
    let (own_start, own_end) = own_range();
    let mut frames = [core::ptr::null_mut(); 8];
    let count = unsafe { RtlCaptureStackBackTrace(0, &mut frames, None) } as usize;
    frames[..count]
        .iter()
        .map(|&f| f as usize)
        .find(|&f| f < own_start || f >= own_end)
        .is_some_and(|f| f >= start && f < end)
}

macro_rules! wrapped {
    ($($name:literal => $real:ident: $ty:ty = $wrapper:ident,)*) => {
        $(static $real: AtomicUsize = AtomicUsize::new(0);)*

        fn wrapper_for(name: &CStr, found: usize) -> Option<usize> {
            match name.to_bytes() {
                $($name => {
                    let wrapper = $wrapper as $ty as usize;
                    if found != wrapper {
                        let _ = $real.compare_exchange(0, found, Ordering::Relaxed, Ordering::Relaxed);
                    }
                    Some(wrapper)
                })*
                _ => None,
            }
        }
    };
}

wrapped! {
    b"vkCreateGraphicsPipelines" => CREATE_PIPELINES: CreatePipelinesFn = create_graphics_pipelines,
    b"vkDestroyPipeline" => DESTROY_PIPELINE: DestroyPipelineFn = destroy_pipeline,
    b"vkCmdBindPipeline" => BIND_PIPELINE: BindPipelineFn = bind_pipeline,
    b"vkCmdSetViewport" => SET_VIEWPORT: SetViewportFn = set_viewport,
    b"vkCmdSetViewportWithCount" => SET_VIEWPORT_COUNT: SetViewportCountFn = set_viewport_with_count,
    b"vkCmdSetViewportWithCountEXT" => SET_VIEWPORT_COUNT_EXT: SetViewportCountFn = set_viewport_with_count_ext,
    b"vkCmdSetDepthBias" => SET_BIAS_CS: SetBiasFn = set_depth_bias,
    b"vkCmdSetDepthCompareOp" => SET_COMPARE_CS: SetCompareFn = set_depth_compare,
    b"vkCmdSetDepthCompareOpEXT" => SET_COMPARE_CS_EXT: SetCompareFn = set_depth_compare_ext,
    b"vkCmdBeginRendering" => BEGIN_RENDERING: BeginRenderingFn = begin_rendering,
    b"vkCmdBeginRenderingKHR" => BEGIN_RENDERING_KHR: BeginRenderingFn = begin_rendering_khr,
    b"vkCmdEndRendering" => END_RENDERING: EndRenderingFn = end_rendering,
    b"vkCmdEndRenderingKHR" => END_RENDERING_KHR: EndRenderingFn = end_rendering_khr,
    b"vkCmdClearDepthStencilImage" => CLEAR_IMAGE: ClearImageFn = clear_depth_image,
    b"vkCmdClearAttachments" => CLEAR_ATTACHMENTS: ClearAttachmentsFn = clear_attachments,
    b"vkCreateImageView" => CREATE_VIEW: CreateViewFn = create_image_view,
    b"vkDestroyImageView" => DESTROY_VIEW: DestroyViewFn = destroy_image_view,
    b"vkBeginCommandBuffer" => BEGIN_BUFFER: BeginBufferFn = begin_command_buffer,
}

macro_rules! gpa_detour {
    ($name:ident, $index:expr) => {
        unsafe extern "system" fn $name(
            handle: usize,
            name: *const c_char,
        ) -> Option<unsafe extern "system" fn()> {
            let found = unsafe { original(&GPA_HOOKS[$index])(handle, name) };
            resolved(handle, name, found, $index % 2 == 1)
        }
    };
}
gpa_detour!(loader_instance_gpa, 0);
gpa_detour!(loader_device_gpa, 1);
gpa_detour!(interposer_instance_gpa, 2);
gpa_detour!(interposer_device_gpa, 3);

/// The `vkGetDeviceProcAddr` handed to CommunityShaders.dll when it asks for one by name.
unsafe extern "system" fn handed_device_gpa(
    device: usize,
    name: *const c_char,
) -> Option<unsafe extern "system" fn()> {
    let real: GpaFn = pointer(&HANDED_DEVICE_GPA)?;
    let found = unsafe { real(device, name) };
    resolved(device, name, found, true)
}

fn resolved(
    handle: usize,
    name: *const c_char,
    found: Option<unsafe extern "system" fn()>,
    device: bool,
) -> Option<unsafe extern "system" fn()> {
    let real = found?;
    if name.is_null() || !from_community_shaders() {
        return Some(real);
    }
    let name = unsafe { CStr::from_ptr(name) };
    if device {
        DEVICE.store(handle, Ordering::Relaxed);
    }
    let wrapper = if name.to_bytes() == b"vkGetDeviceProcAddr" {
        let _ = HANDED_DEVICE_GPA.compare_exchange(0, real as usize, Ordering::Relaxed, Ordering::Relaxed);
        handed_device_gpa as GpaFn as usize
    } else if let Some(wrapper) = wrapper_for(name, real as usize) {
        wrapper
    } else {
        return Some(real);
    };
    RESOLVED.fetch_add(1, Ordering::Relaxed);
    Some(unsafe { core::mem::transmute::<usize, unsafe extern "system" fn()>(wrapper) })
}

/// Hooks the loaders' resolution exports; before Community Shaders adopts DXVK's device.
pub(crate) fn install() -> Result<(), String> {
    let loader = unsafe { LoadLibraryW(w!("vulkan-1.dll")) }.map_err(|e| format!("vulkan-1.dll: {e}"))?;
    let device_gpa = unsafe { GetProcAddress(loader, s!("vkGetDeviceProcAddr")) }
        .ok_or("vulkan-1.dll has no vkGetDeviceProcAddr")?;
    LOADER_DEVICE_GPA.store(device_gpa as usize, Ordering::Relaxed);
    let detours: [GpaFn; 4] = [
        loader_instance_gpa,
        loader_device_gpa,
        interposer_instance_gpa,
        interposer_device_gpa,
    ];
    let modules = [
        Some(loader),
        unsafe { GetModuleHandleW(w!("sl.interposer.dll")) }.ok(),
    ];
    for (i, detour) in detours.into_iter().enumerate() {
        let Some(module) = modules[i / 2] else { continue };
        let export = if i % 2 == 0 { s!("vkGetInstanceProcAddr") } else { s!("vkGetDeviceProcAddr") };
        let Some(target) = (unsafe { GetProcAddress(module, export) }) else { continue };
        let hook = unsafe { hook::Hook::<GpaFn>::create(target as usize, detour) }
            .map_err(|e| format!("{i}: {e}"))?;
        let mut slot = GPA_HOOKS[i].lock().unwrap();
        slot.insert(hook).enable().map_err(|e| format!("{i}: {e}"))?;
    }
    Ok(())
}

pub(crate) fn remove() {
    for slot in &GPA_HOOKS {
        if let Some(hook) = slot.lock().unwrap().take() {
            let _ = unsafe { hook.remove() };
        }
    }
}

unsafe fn query(unknown: &impl Interface, iid: &GUID) -> Option<*mut c_void> {
    let mut pointer = core::ptr::null_mut();
    unsafe { unknown.query(iid, &raw mut pointer) }.ok().ok()?;
    (!pointer.is_null()).then_some(pointer)
}

unsafe fn method<F: Copy>(object: *mut c_void, index: usize) -> F {
    const { assert!(size_of::<F>() == size_of::<usize>()) };
    let table = unsafe { *(object as *const *const usize) };
    unsafe { core::mem::transmute_copy(&*table.add(index)) }
}

unsafe fn release(object: *mut c_void) {
    let release: ReleaseFn = unsafe { method(object, 2) };
    unsafe { release(object) };
}

/// Finds the world depth images on DXVK's device and the dynamic-state setters.
pub(crate) fn engage() -> Result<String, String> {
    let d3d = crate::context::device().ok_or("no device")?;
    let interop = unsafe { query(&d3d, &INTEROP_DEVICE) }
        .ok_or("the D3D11 device is not DXVK's; nothing draws past the context")?;
    let mut instance = vk::Instance::null();
    let mut physical = vk::PhysicalDevice::null();
    let mut device = vk::Device::null();
    unsafe {
        let handles: GetHandlesFn = method(interop, 3);
        handles(interop, &raw mut instance, &raw mut physical, &raw mut device);
        release(interop);
    }
    if device.is_null() {
        return Err("DXVK gave no Vulkan device".into());
    }
    let handle = usize::try_from(device.as_raw()).unwrap();
    DEVICE.store(handle, Ordering::Relaxed);
    let gpa: GpaFn = pointer(&LOADER_DEVICE_GPA).ok_or("no loader")?;
    let resolve = |names: &[&CStr]| {
        names
            .iter()
            .find_map(|n| unsafe { gpa(handle, n.as_ptr()) })
            .map_or(0, |f| f as usize)
    };
    SET_COMPARE.store(
        resolve(&[c"vkCmdSetDepthCompareOp", c"vkCmdSetDepthCompareOpEXT"]),
        Ordering::Relaxed,
    );
    SET_BIAS.store(resolve(&[c"vkCmdSetDepthBias"]), Ordering::Relaxed);
    if SET_COMPARE.load(Ordering::Relaxed) == 0 {
        return Err("the device has no dynamic depth compare".into());
    }
    sync_images();
    let images = IMAGES.lock().unwrap();
    let formats: Vec<i32> = images.values().map(|w| w.format.as_raw()).collect();
    Ok(format!(
        "{} world images on DXVK's device, formats {formats:?}, {} resolutions wrapped",
        images.len(),
        RESOLVED.load(Ordering::Relaxed)
    ))
}

/// Maps the world depth textures to their DXVK images.
pub(crate) fn sync_images() {
    let mut images = IMAGES.lock().unwrap();
    images.clear();
    for target in crate::copy::REVERSED_TARGETS {
        let Some(field) = crate::context::target_field(target, crate::context::TARGET_TEXTURE) else {
            continue;
        };
        let texture = unsafe { *field } as *mut c_void;
        let Some(resource) = (unsafe { ID3D11Resource::from_raw_borrowed(&texture) }) else {
            continue;
        };
        let Some(surface) = (unsafe { query(resource, &INTEROP_SURFACE) }) else {
            continue;
        };
        let mut image = vk::Image::null();
        let mut layout = vk::ImageLayout::UNDEFINED;
        let mut info = vk::ImageCreateInfo::default();
        let result = unsafe {
            let get: GetImageInfoFn = method(surface, 4);
            let r = get(surface, &raw mut image, &raw mut layout, &raw mut info);
            release(surface);
            r
        };
        if result.is_ok() && !image.is_null() {
            images.insert(
                image.as_raw(),
                World {
                    resource: texture as usize,
                    format: info.format,
                },
            );
        }
    }
}

fn reversed(image: vk::Image) -> bool {
    let images = IMAGES.lock().unwrap();
    let Some(world) = images.get(&image.as_raw()) else {
        return false;
    };
    let resource = world.resource as *mut c_void;
    unsafe { ID3D11Resource::from_raw_borrowed(&resource) }.is_some_and(crate::copy::reversed)
}

fn world_format(format: vk::Format) -> bool {
    IMAGES.lock().unwrap().values().any(|w| w.format == format)
}

fn mirrored(op: vk::CompareOp) -> vk::CompareOp {
    match op {
        vk::CompareOp::LESS => vk::CompareOp::GREATER,
        vk::CompareOp::GREATER => vk::CompareOp::LESS,
        vk::CompareOp::LESS_OR_EQUAL => vk::CompareOp::GREATER_OR_EQUAL,
        vk::CompareOp::GREATER_OR_EQUAL => vk::CompareOp::LESS_OR_EQUAL,
        other => other,
    }
}

fn flipped(viewports: &[vk::Viewport]) -> Vec<vk::Viewport> {
    viewports
        .iter()
        .map(|v| vk::Viewport {
            min_depth: 1.0 - v.min_depth,
            max_depth: 1.0 - v.max_depth,
            ..*v
        })
        .collect()
}

fn negated(bias: [f32; 3]) -> [f32; 3] {
    bias.map(|b| -b)
}

fn depth_format(info: &vk::GraphicsPipelineCreateInfo<'_>) -> vk::Format {
    let mut next = info.p_next.cast::<vk::BaseInStructure>();
    while !next.is_null() {
        let base = unsafe { *next };
        if base.s_type == vk::StructureType::PIPELINE_RENDERING_CREATE_INFO {
            return unsafe { *next.cast::<vk::PipelineRenderingCreateInfo>() }.depth_attachment_format;
        }
        next = base.p_next;
    }
    vk::Format::UNDEFINED
}

struct DynamicStates {
    states: Vec<vk::DynamicState>,
    info: vk::PipelineDynamicStateCreateInfo<'static>,
}

unsafe extern "system" fn create_graphics_pipelines(
    device: vk::Device,
    cache: vk::PipelineCache,
    count: u32,
    infos: *const vk::GraphicsPipelineCreateInfo<'static>,
    allocator: *const vk::AllocationCallbacks<'static>,
    pipelines: *mut vk::Pipeline,
) -> vk::Result {
    let real: CreatePipelinesFn = pointer(&CREATE_PIPELINES).unwrap();
    if infos.is_null() || count == 0 {
        return unsafe { real(device, cache, count, infos, allocator, pipelines) };
    }
    let given = unsafe { core::slice::from_raw_parts(infos, count as usize) };
    let mut patched = given.to_vec();
    let mut dynamics: Vec<Box<DynamicStates>> = Vec::new();
    let mut records: Vec<Option<PipelineDepth>> = Vec::with_capacity(given.len());
    for (i, info) in given.iter().enumerate() {
        records.push(None);
        if !world_format(depth_format(info)) || info.p_depth_stencil_state.is_null() {
            continue;
        }
        let depth = unsafe { *info.p_depth_stencil_state };
        if depth.depth_test_enable == vk::FALSE {
            continue;
        }
        let mut states = if info.p_dynamic_state.is_null() {
            Vec::new()
        } else {
            let existing = unsafe { *info.p_dynamic_state };
            unsafe {
                core::slice::from_raw_parts(existing.p_dynamic_states, existing.dynamic_state_count as usize)
            }
            .to_vec()
        };
        let mut added = false;
        if !states.contains(&vk::DynamicState::DEPTH_COMPARE_OP) {
            states.push(vk::DynamicState::DEPTH_COMPARE_OP);
            added = true;
        }
        let raster = (!info.p_rasterization_state.is_null()).then(|| unsafe { *info.p_rasterization_state });
        let bias = raster
            .filter(|r| r.depth_bias_enable != vk::FALSE && !states.contains(&vk::DynamicState::DEPTH_BIAS))
            .map(|r| {
                states.push(vk::DynamicState::DEPTH_BIAS);
                added = true;
                [r.depth_bias_constant_factor, r.depth_bias_clamp, r.depth_bias_slope_factor]
            });
        if added {
            let previous = (!info.p_dynamic_state.is_null()).then(|| unsafe { *info.p_dynamic_state });
            let mut boxed = Box::new(DynamicStates {
                states,
                info: vk::PipelineDynamicStateCreateInfo::default(),
            });
            boxed.info = vk::PipelineDynamicStateCreateInfo {
                p_next: previous.map_or(core::ptr::null(), |p| p.p_next),
                flags: previous.map_or(vk::PipelineDynamicStateCreateFlags::default(), |p| p.flags),
                dynamic_state_count: u32::try_from(boxed.states.len()).unwrap(),
                p_dynamic_states: boxed.states.as_ptr(),
                ..Default::default()
            };
            patched[i].p_dynamic_state = &raw const boxed.info;
            dynamics.push(boxed);
        }
        records[i] = Some(PipelineDepth {
            compare: depth.depth_compare_op,
            bias,
        });
    }
    let result = unsafe { real(device, cache, count, patched.as_ptr(), allocator, pipelines) };
    drop(dynamics);
    if result == vk::Result::SUCCESS && !pipelines.is_null() {
        let made = unsafe { core::slice::from_raw_parts(pipelines, count as usize) };
        let mut table = PIPELINES.lock().unwrap();
        for (pipeline, record) in made.iter().zip(records) {
            if let Some(record) = record {
                if WORLD_PIPELINES.fetch_add(1, Ordering::Relaxed) == 0 {
                    note(&format!(
                        "first world pipeline: compare {}, static bias {:?}",
                        record.compare.as_raw(),
                        record.bias
                    ));
                }
                table.insert(pipeline.as_raw(), record);
            }
        }
    }
    result
}

unsafe extern "system" fn destroy_pipeline(
    device: vk::Device,
    pipeline: vk::Pipeline,
    allocator: *const vk::AllocationCallbacks<'static>,
) {
    PIPELINES.lock().unwrap().remove(&pipeline.as_raw());
    let real: DestroyPipelineFn = pointer(&DESTROY_PIPELINE).unwrap();
    unsafe { real(device, pipeline, allocator) };
}

/// Sends the bound pipeline's compare and bias for the buffer's current rendering.
fn apply_pipeline(buffer: vk::CommandBuffer, state: &Buffer) {
    let table = PIPELINES.lock().unwrap();
    let Some(record) = table.get(&state.pipeline) else {
        return;
    };
    if let Some(set_compare) = pointer::<SetCompareFn>(&SET_COMPARE) {
        let op = state.compare.unwrap_or(record.compare);
        unsafe { set_compare(buffer, if state.reversed { mirrored(op) } else { op }) };
    }
    if let (Some(bias), Some(set_bias)) = (record.bias, pointer::<SetBiasFn>(&SET_BIAS)) {
        let bias = state.bias.unwrap_or(bias);
        let [constant, clamp, slope] = if state.reversed { negated(bias) } else { bias };
        unsafe { set_bias(buffer, constant, clamp, slope) };
    }
}

unsafe extern "system" fn bind_pipeline(
    buffer: vk::CommandBuffer,
    point: vk::PipelineBindPoint,
    pipeline: vk::Pipeline,
) {
    let real: BindPipelineFn = pointer(&BIND_PIPELINE).unwrap();
    unsafe { real(buffer, point, pipeline) };
    if point == vk::PipelineBindPoint::GRAPHICS {
        let mut buffers = BUFFERS.lock().unwrap();
        let state = buffers.entry(buffer.as_raw()).or_default();
        state.pipeline = pipeline.as_raw();
        apply_pipeline(buffer, state);
    }
}

fn send_viewports(buffer: vk::CommandBuffer, state: &Buffer) {
    let Some((first, viewports, with_count)) = &state.viewports else {
        return;
    };
    let sent = if state.reversed { flipped(viewports) } else { viewports.clone() };
    let count = u32::try_from(sent.len()).unwrap();
    if *with_count {
        let real: SetViewportCountFn = pointer(&SET_VIEWPORT_COUNT)
            .or_else(|| pointer(&SET_VIEWPORT_COUNT_EXT))
            .unwrap();
        unsafe { real(buffer, count, sent.as_ptr()) };
    } else {
        let real: SetViewportFn = pointer(&SET_VIEWPORT).unwrap();
        unsafe { real(buffer, *first, count, sent.as_ptr()) };
    }
}

fn store_viewports(buffer: vk::CommandBuffer, first: u32, count: u32, list: *const vk::Viewport, with_count: bool) {
    if list.is_null() {
        return;
    }
    let given = unsafe { core::slice::from_raw_parts(list, count as usize) };
    let mut buffers = BUFFERS.lock().unwrap();
    let state = buffers.entry(buffer.as_raw()).or_default();
    state.viewports = Some((first, given.to_vec(), with_count));
    send_viewports(buffer, state);
}

unsafe extern "system" fn set_viewport(
    buffer: vk::CommandBuffer,
    first: u32,
    count: u32,
    list: *const vk::Viewport,
) {
    store_viewports(buffer, first, count, list, false);
}

unsafe extern "system" fn set_viewport_with_count(
    buffer: vk::CommandBuffer,
    count: u32,
    list: *const vk::Viewport,
) {
    store_viewports(buffer, 0, count, list, true);
}

unsafe extern "system" fn set_viewport_with_count_ext(
    buffer: vk::CommandBuffer,
    count: u32,
    list: *const vk::Viewport,
) {
    store_viewports(buffer, 0, count, list, true);
}

fn send_bias(buffer: vk::CommandBuffer, state: &Buffer) {
    if let (Some(bias), Some(real)) = (state.bias, pointer::<SetBiasFn>(&SET_BIAS_CS)) {
        let [constant, clamp, slope] = if state.reversed { negated(bias) } else { bias };
        unsafe { real(buffer, constant, clamp, slope) };
    }
}

unsafe extern "system" fn set_depth_bias(buffer: vk::CommandBuffer, constant: f32, clamp: f32, slope: f32) {
    let mut buffers = BUFFERS.lock().unwrap();
    let state = buffers.entry(buffer.as_raw()).or_default();
    state.bias = Some([constant, clamp, slope]);
    send_bias(buffer, state);
}

fn send_compare(buffer: vk::CommandBuffer, state: &Buffer) {
    if let (Some(op), Some(real)) = (
        state.compare,
        pointer::<SetCompareFn>(&SET_COMPARE_CS).or_else(|| pointer(&SET_COMPARE_CS_EXT)),
    ) {
        unsafe { real(buffer, if state.reversed { mirrored(op) } else { op }) };
    }
}

unsafe extern "system" fn set_depth_compare(buffer: vk::CommandBuffer, op: vk::CompareOp) {
    let mut buffers = BUFFERS.lock().unwrap();
    let state = buffers.entry(buffer.as_raw()).or_default();
    state.compare = Some(op);
    send_compare(buffer, state);
}

unsafe extern "system" fn set_depth_compare_ext(buffer: vk::CommandBuffer, op: vk::CompareOp) {
    unsafe { set_depth_compare(buffer, op) };
}

fn begin(buffer: vk::CommandBuffer, info: *const vk::RenderingInfo<'static>, real: BeginRenderingFn) {
    if info.is_null() {
        unsafe { real(buffer, info) };
        return;
    }
    let given = unsafe { *info };
    let attachment = (!given.p_depth_attachment.is_null()).then(|| unsafe { *given.p_depth_attachment });
    let image = attachment.and_then(|a| VIEWS.lock().unwrap().get(&a.image_view.as_raw()).copied());
    let reversed = image.is_some_and(|i| reversed(vk::Image::from_raw(i)));
    let mut patched = given;
    let mut cleared;
    if reversed && let Some(a) = attachment && a.load_op == vk::AttachmentLoadOp::CLEAR {
        cleared = a;
        cleared.clear_value.depth_stencil.depth = 1.0 - unsafe { a.clear_value.depth_stencil.depth };
        patched.p_depth_attachment = &raw const cleared;
        WORLD_CLEARS.fetch_add(1, Ordering::Relaxed);
    }
    unsafe { real(buffer, &raw const patched) };
    if reversed {
        WORLD_RENDERINGS.fetch_add(1, Ordering::Relaxed);
    }
    let mut buffers = BUFFERS.lock().unwrap();
    let state = buffers.entry(buffer.as_raw()).or_default();
    if state.reversed != reversed || reversed {
        state.reversed = reversed;
        send_viewports(buffer, state);
        send_bias(buffer, state);
        send_compare(buffer, state);
        apply_pipeline(buffer, state);
    }
}

unsafe extern "system" fn begin_rendering(buffer: vk::CommandBuffer, info: *const vk::RenderingInfo<'static>) {
    begin(buffer, info, pointer(&BEGIN_RENDERING).unwrap());
}

unsafe extern "system" fn begin_rendering_khr(buffer: vk::CommandBuffer, info: *const vk::RenderingInfo<'static>) {
    begin(buffer, info, pointer(&BEGIN_RENDERING_KHR).unwrap());
}

unsafe extern "system" fn end_rendering(buffer: vk::CommandBuffer) {
    let real: EndRenderingFn = pointer(&END_RENDERING).unwrap();
    unsafe { real(buffer) };
}

unsafe extern "system" fn end_rendering_khr(buffer: vk::CommandBuffer) {
    let real: EndRenderingFn = pointer(&END_RENDERING_KHR).unwrap();
    unsafe { real(buffer) };
}

unsafe extern "system" fn clear_depth_image(
    buffer: vk::CommandBuffer,
    image: vk::Image,
    layout: vk::ImageLayout,
    value: *const vk::ClearDepthStencilValue,
    count: u32,
    ranges: *const vk::ImageSubresourceRange,
) {
    let real: ClearImageFn = pointer(&CLEAR_IMAGE).unwrap();
    let mut mirrored_value;
    let sent = if !value.is_null() && reversed(image) {
        mirrored_value = unsafe { *value };
        mirrored_value.depth = 1.0 - mirrored_value.depth;
        WORLD_CLEARS.fetch_add(1, Ordering::Relaxed);
        &raw const mirrored_value
    } else {
        value
    };
    unsafe { real(buffer, image, layout, sent, count, ranges) };
}

unsafe extern "system" fn clear_attachments(
    buffer: vk::CommandBuffer,
    count: u32,
    attachments: *const vk::ClearAttachment,
    rect_count: u32,
    rects: *const vk::ClearRect,
) {
    let real: ClearAttachmentsFn = pointer(&CLEAR_ATTACHMENTS).unwrap();
    let reversed = BUFFERS
        .lock()
        .unwrap()
        .get(&buffer.as_raw())
        .is_some_and(|s| s.reversed);
    if !reversed || attachments.is_null() {
        unsafe { real(buffer, count, attachments, rect_count, rects) };
        return;
    }
    let mut patched = unsafe { core::slice::from_raw_parts(attachments, count as usize) }.to_vec();
    for attachment in &mut patched {
        if attachment.aspect_mask.contains(vk::ImageAspectFlags::DEPTH) {
            attachment.clear_value.depth_stencil.depth =
                1.0 - unsafe { attachment.clear_value.depth_stencil.depth };
            WORLD_CLEARS.fetch_add(1, Ordering::Relaxed);
        }
    }
    unsafe { real(buffer, count, patched.as_ptr(), rect_count, rects) };
}

unsafe extern "system" fn create_image_view(
    device: vk::Device,
    info: *const vk::ImageViewCreateInfo<'static>,
    allocator: *const vk::AllocationCallbacks<'static>,
    view: *mut vk::ImageView,
) -> vk::Result {
    let real: CreateViewFn = pointer(&CREATE_VIEW).unwrap();
    let result = unsafe { real(device, info, allocator, view) };
    if result == vk::Result::SUCCESS && !info.is_null() && !view.is_null() {
        let info = unsafe { *info };
        if IMAGES.lock().unwrap().contains_key(&info.image.as_raw()) {
            VIEWS
                .lock()
                .unwrap()
                .insert(unsafe { *view }.as_raw(), info.image.as_raw());
            if WORLD_VIEWS.fetch_add(1, Ordering::Relaxed) == 0 {
                note(&format!(
                    "first view on a world image: format {}, aspect {:#x}",
                    info.format.as_raw(),
                    info.subresource_range.aspect_mask.as_raw()
                ));
            }
        }
    }
    result
}

unsafe extern "system" fn destroy_image_view(
    device: vk::Device,
    view: vk::ImageView,
    allocator: *const vk::AllocationCallbacks<'static>,
) {
    VIEWS.lock().unwrap().remove(&view.as_raw());
    let real: DestroyViewFn = pointer(&DESTROY_VIEW).unwrap();
    unsafe { real(device, view, allocator) };
}

unsafe extern "system" fn begin_command_buffer(
    buffer: vk::CommandBuffer,
    info: *const vk::CommandBufferBeginInfo<'static>,
) -> vk::Result {
    BUFFERS.lock().unwrap().remove(&buffer.as_raw());
    let real: BeginBufferFn = pointer(&BEGIN_BUFFER).unwrap();
    unsafe { real(buffer, info) }
}

pub(crate) fn counters() -> String {
    format!(
        "vulkan: {} resolutions wrapped, world pipelines {}, world renderings {}, world views {}, world clears {}",
        RESOLVED.load(Ordering::Relaxed),
        WORLD_PIPELINES.load(Ordering::Relaxed),
        WORLD_RENDERINGS.load(Ordering::Relaxed),
        WORLD_VIEWS.load(Ordering::Relaxed),
        WORLD_CLEARS.load(Ordering::Relaxed),
    )
}
