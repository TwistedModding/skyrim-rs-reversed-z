//! Drawcall Limit Fix draws the world through Community Shaders' own Vulkan pipelines on DXVK's
//! device, past the D3D11 context. It takes its entry points from `vkGetDeviceProcAddr` and
//! `vkGetInstanceProcAddr`, so the resolutions made from CommunityShaders.dll are answered with
//! wrappers while DXVK's own keep the real pointers. A rendering whose depth attachment is a
//! reversed world image (found through DXVK's interop interface) gets its pipelines' compare and
//! biases mirrored (made dynamic at creation, set at each bind), its viewport ranges [a, b] made
//! and its clears 1 - value. Its vertex shaders are swapped for SPIR-V twins ([`crate::spirv`],
//! clip z = w - z) and the viewport ranges mirrored like the context's; a pipeline whose vertex
//! shader could not be twinned keeps its clip z and gets [a, b] flipped to [1 - a, 1 - b] instead.
//! Descriptors sampling a reversed world image are pointed at its conventional copy
//! ([`crate::copy`]).

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
type GetDescriptorFn = unsafe extern "system" fn(
    vk::Device,
    *const vk::DescriptorGetInfoEXT<'static>,
    usize,
    *mut c_void,
);
type CreateModuleFn = unsafe extern "system" fn(
    vk::Device,
    *const vk::ShaderModuleCreateInfo<'static>,
    *const vk::AllocationCallbacks<'static>,
    *mut vk::ShaderModule,
) -> vk::Result;
type DestroyModuleFn =
    unsafe extern "system" fn(vk::Device, vk::ShaderModule, *const vk::AllocationCallbacks<'static>);
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
static WORLD_READS: AtomicU64 = AtomicU64::new(0);
static TWINS_MADE: AtomicU64 = AtomicU64::new(0);
static TWINS_REFUSED: AtomicU64 = AtomicU64::new(0);
static NOTES: AtomicU64 = AtomicU64::new(0);

struct World {
    resource: usize,
    format: vk::Format,
    target: usize,
}
struct PipelineDepth {
    compare: vk::CompareOp,
    /// The static bias factors made dynamic, sent at each bind.
    bias: Option<[f32; 3]>,
    /// The vertex shader is a twin writing w - z.
    twinned: bool,
}
#[derive(Default)]
struct Buffer {
    reversed: bool,
    image: u64,
    layout: vk::ImageLayout,
    viewports: Option<(u32, Vec<vk::Viewport>, bool)>,
    bias: Option<[f32; 3]>,
    compare: Option<vk::CompareOp>,
    pipeline: u64,
}

static IMAGES: LazyLock<Mutex<HashMap<u64, World>>> = LazyLock::new(Mutex::default);
static VIEWS: LazyLock<Mutex<HashMap<u64, u64>>> = LazyLock::new(Mutex::default);
static PIPELINES: LazyLock<Mutex<HashMap<u64, PipelineDepth>>> = LazyLock::new(Mutex::default);
static BUFFERS: LazyLock<Mutex<HashMap<u64, Buffer>>> = LazyLock::new(Mutex::default);
/// D3D textures already looked up on DXVK's device, by resource pointer.
static TRACKED: LazyLock<Mutex<std::collections::HashSet<usize>>> = LazyLock::new(Mutex::default);
/// The code of every vertex shader module Community Shaders made, and each one's twin.
static MODULES: LazyLock<Mutex<HashMap<u64, Vec<u32>>>> = LazyLock::new(Mutex::default);
static MODULE_TWINS: LazyLock<Mutex<HashMap<u64, Option<vk::ShaderModule>>>> =
    LazyLock::new(Mutex::default);
/// Each world image's view of its conventional copy.
static COPIES: LazyLock<Mutex<HashMap<u64, StandardCopy>>> = LazyLock::new(Mutex::default);

/// A world image's conventional copy on DXVK's device.
#[derive(Clone, Copy)]
struct StandardCopy {
    view: vk::ImageView,
    image: vk::Image,
    width: u32,
    height: u32,
}

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
    b"vkGetDescriptorEXT" => GET_DESCRIPTOR: GetDescriptorFn = get_descriptor,
    b"vkCreateShaderModule" => CREATE_MODULE: CreateModuleFn = create_shader_module,
    b"vkDestroyShaderModule" => DESTROY_MODULE: DestroyModuleFn = destroy_shader_module,
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
    crate::stream::setup(device, gpa)?;
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
    TRACKED.lock().unwrap().clear();
    for target in crate::copy::REVERSED_TARGETS {
        let Some(field) = crate::context::target_field(target, crate::context::TARGET_TEXTURE) else {
            continue;
        };
        let texture = unsafe { *field } as *mut c_void;
        let Some(resource) = (unsafe { ID3D11Resource::from_raw_borrowed(&texture) }) else {
            continue;
        };
        if let Some((image, _, info)) = vulkan_image(resource) {
            images.insert(
                image.as_raw(),
                World {
                    resource: texture as usize,
                    format: info.format,
                    target,
                },
            );
        }
    }
}

/// Adds a texture that holds depth copied from a world target, so descriptors sampling it are
/// pointed at its conventional copy as well.
pub(crate) fn track(resource: &ID3D11Resource) {
    if !TRACKED.lock().unwrap().insert(resource.as_raw() as usize) {
        return;
    }
    if let Some((image, _, info)) = vulkan_image(resource) {
        IMAGES.lock().unwrap().entry(image.as_raw()).or_insert(World {
            resource: resource.as_raw() as usize,
            format: info.format,
            target: usize::MAX,
        });
    }
}

/// A DXVK texture's image, current layout and creation info.
fn vulkan_image(
    resource: &ID3D11Resource,
) -> Option<(vk::Image, vk::ImageLayout, vk::ImageCreateInfo<'static>)> {
    let surface = unsafe { query(resource, &INTEROP_SURFACE) }?;
    let mut image = vk::Image::null();
    let mut layout = vk::ImageLayout::UNDEFINED;
    let mut info = vk::ImageCreateInfo::default();
    let result = unsafe {
        let get: GetImageInfoFn = method(surface, 4);
        let r = get(surface, &raw mut image, &raw mut layout, &raw mut info);
        release(surface);
        r
    };
    (result.is_ok() && !image.is_null()).then_some((image, layout, info))
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
    IMAGES
        .lock()
        .unwrap()
        .values()
        .any(|w| w.target != usize::MAX && w.format == format)
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

/// Mirrored (as the context's) for a twin's w - z into a reversed target; flipped when only
/// one of the two is reversed, which with the engine's [0, 1] ranges stores the other convention.
fn sent_viewports(viewports: &[vk::Viewport], reversed: bool, twinned: bool) -> Vec<vk::Viewport> {
    viewports
        .iter()
        .map(|v| match (reversed, twinned) {
            (true, true) => vk::Viewport {
                min_depth: 1.0 - v.max_depth,
                max_depth: 1.0 - v.min_depth,
                ..*v
            },
            (false, false) => *v,
            _ => vk::Viewport {
                min_depth: 1.0 - v.min_depth,
                max_depth: 1.0 - v.max_depth,
                ..*v
            },
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

/// The far plane the precise endings are made for; twins are not remade when it changes.
fn far() -> Option<f32> {
    let far = skyrim::render::depth_shift_far();
    (far > 0.0).then_some(far)
}

/// The twin module of a vertex module Community Shaders made, built once.
fn twin_module(device: vk::Device, module: vk::ShaderModule) -> Option<vk::ShaderModule> {
    let mut twins = MODULE_TWINS.lock().unwrap();
    if let Some(twin) = twins.get(&module.as_raw()) {
        return *twin;
    }
    let made = (|| {
        let code = MODULES.lock().unwrap().get(&module.as_raw())?.clone();
        let twin = crate::spirv::mirrored(&code, far())?;
        create_module(device, &twin)
    })();
    (if made.is_some() { &TWINS_MADE } else { &TWINS_REFUSED }).fetch_add(1, Ordering::Relaxed);
    if made.is_none() && TWINS_REFUSED.load(Ordering::Relaxed) == 1 {
        note("a world vertex shader could not be twinned; its pipelines keep conventional clip z");
    }
    twins.insert(module.as_raw(), made);
    made
}

fn create_module(device: vk::Device, code: &[u32]) -> Option<vk::ShaderModule> {
    let create: CreateModuleFn = pointer(&CREATE_MODULE)?;
    let info = vk::ShaderModuleCreateInfo {
        code_size: code.len() * 4,
        p_code: code.as_ptr(),
        ..Default::default()
    };
    let mut module = vk::ShaderModule::null();
    let result = unsafe { create(device, &raw const info, core::ptr::null(), &raw mut module) };
    (result == vk::Result::SUCCESS && !module.is_null()).then_some(module)
}

/// The vertex stage's twin: a twin module for a module stage, rewritten code for an inline one.
struct TwinStage {
    stages: Vec<vk::PipelineShaderStageCreateInfo<'static>>,
    _inline: Option<(Vec<u32>, Box<vk::ShaderModuleCreateInfo<'static>>)>,
}

fn twin_stages(device: vk::Device, info: &vk::GraphicsPipelineCreateInfo<'_>) -> Option<TwinStage> {
    if info.p_stages.is_null() {
        return None;
    }
    let given = unsafe { core::slice::from_raw_parts(info.p_stages, info.stage_count as usize) };
    let vertex = given.iter().position(|s| s.stage == vk::ShaderStageFlags::VERTEX)?;
    let mut stages: Vec<vk::PipelineShaderStageCreateInfo<'static>> =
        given.iter().map(|s| unsafe { core::mem::transmute_copy(s) }).collect();
    let stage = &mut stages[vertex];
    if !stage.module.is_null() {
        stage.module = twin_module(device, stage.module)?;
        return Some(TwinStage { stages, _inline: None });
    }
    let mut next = stage.p_next.cast::<vk::BaseInStructure>();
    while !next.is_null() {
        let base = unsafe { *next };
        if base.s_type == vk::StructureType::SHADER_MODULE_CREATE_INFO {
            let inline = unsafe { *next.cast::<vk::ShaderModuleCreateInfo>() };
            let code = unsafe { core::slice::from_raw_parts(inline.p_code, inline.code_size / 4) };
            let Some(twin) = crate::spirv::mirrored(code, far()) else {
                TWINS_REFUSED.fetch_add(1, Ordering::Relaxed);
                return None;
            };
            let patched = Box::new(vk::ShaderModuleCreateInfo {
                p_next: inline.p_next,
                flags: inline.flags,
                code_size: twin.len() * 4,
                p_code: twin.as_ptr(),
                ..Default::default()
            });
            stage.p_next = (&raw const *patched).cast();
            TWINS_MADE.fetch_add(1, Ordering::Relaxed);
            return Some(TwinStage {
                stages,
                _inline: Some((twin, patched)),
            });
        }
        next = base.p_next;
    }
    None
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
    let mut twins: Vec<TwinStage> = Vec::new();
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
        let twinned = twin_stages(device, info).is_some_and(|twin| {
            patched[i].p_stages = twin.stages.as_ptr();
            twins.push(twin);
            true
        });
        records[i] = Some(PipelineDepth {
            compare: depth.depth_compare_op,
            bias,
            twinned,
        });
    }
    let mut result = unsafe { real(device, cache, count, patched.as_ptr(), allocator, pipelines) };
    if result != vk::Result::SUCCESS && !twins.is_empty() {
        note(&format!("twinned pipeline creation failed with {}; retrying without twins", result.as_raw()));
        for (i, info) in given.iter().enumerate() {
            patched[i].p_stages = info.p_stages;
            if let Some(record) = records[i].as_mut() {
                record.twinned = false;
            }
        }
        result = unsafe { real(device, cache, count, patched.as_ptr(), allocator, pipelines) };
    }
    drop(dynamics);
    drop(twins);
    if result == vk::Result::SUCCESS && !pipelines.is_null() {
        let made = unsafe { core::slice::from_raw_parts(pipelines, count as usize) };
        let mut table = PIPELINES.lock().unwrap();
        for (pipeline, record) in made.iter().zip(records) {
            if let Some(record) = record {
                if WORLD_PIPELINES.fetch_add(1, Ordering::Relaxed) == 0 {
                    note(&format!(
                        "first world pipeline: compare {}, static bias {:?}, twinned {}",
                        record.compare.as_raw(),
                        record.bias,
                        record.twinned
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
        send_viewports(buffer, state);
    }
}

fn send_viewports(buffer: vk::CommandBuffer, state: &Buffer) {
    let Some((first, viewports, with_count)) = &state.viewports else {
        return;
    };
    let twinned = PIPELINES
        .lock()
        .unwrap()
        .get(&state.pipeline)
        .is_some_and(|p| p.twinned);
    let sent = sent_viewports(viewports, state.reversed, twinned);
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
    state.image = if reversed { image.unwrap_or(0) } else { 0 };
    state.layout = attachment.map_or(vk::ImageLayout::UNDEFINED, |a| a.image_layout);
    {
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

/// After a rendering into a world image, its copy is refreshed in the same command stream
/// ([`crate::stream`]), before the buffer's next read of it.
fn ended(buffer: vk::CommandBuffer) {
    let (image, layout) = BUFFERS
        .lock()
        .unwrap()
        .get(&buffer.as_raw())
        .map_or((0, vk::ImageLayout::UNDEFINED), |s| (s.image, s.layout));
    if image == 0 {
        return;
    }
    let world = vk::Image::from_raw(image);
    if !crate::stream::is_prepared(world) {
        let device = vk::Device::from_raw(DEVICE.load(Ordering::Relaxed) as u64);
        let format = IMAGES.lock().unwrap().get(&image).map(|w| w.format);
        if copy_view(device, image).is_none() {
            return;
        }
        let copy = COPIES.lock().unwrap().get(&image).copied();
        if let (Some(format), Some(copy)) = (format, copy)
            && let Err(e) = crate::stream::prepare(world, format, copy.image, copy.view, copy.width, copy.height)
        {
            note(&format!("in-stream copy unavailable: {e}"));
            return;
        }
    }
    crate::stream::record(buffer, world, layout);
}

unsafe extern "system" fn end_rendering(buffer: vk::CommandBuffer) {
    let real: EndRenderingFn = pointer(&END_RENDERING).unwrap();
    unsafe { real(buffer) };
    ended(buffer);
}

unsafe extern "system" fn end_rendering_khr(buffer: vk::CommandBuffer) {
    let real: EndRenderingFn = pointer(&END_RENDERING_KHR).unwrap();
    unsafe { real(buffer) };
    ended(buffer);
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

/// A view of the world image's conventional copy, made on first use.
fn copy_view(device: vk::Device, world: u64) -> Option<vk::ImageView> {
    let mut copies = COPIES.lock().unwrap();
    if let Some(copy) = copies.get(&world) {
        return Some(copy.view);
    }
    let resource = IMAGES.lock().unwrap().get(&world)?.resource as *mut c_void;
    let resource = unsafe { ID3D11Resource::from_raw_borrowed(&resource) }?;
    let standard: ID3D11Resource = crate::copy::standard_texture(resource)?.cast().ok()?;
    let (image, layout, info) = vulkan_image(&standard)?;
    let create: CreateViewFn = pointer(&CREATE_VIEW)?;
    let desc = vk::ImageViewCreateInfo {
        image,
        view_type: vk::ImageViewType::TYPE_2D,
        format: info.format,
        subresource_range: vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            level_count: 1,
            layer_count: 1,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut view = vk::ImageView::null();
    let result = unsafe { create(device, &raw const desc, core::ptr::null(), &raw mut view) };
    if result != vk::Result::SUCCESS || view.is_null() {
        note(&format!("copy view refused: {}", result.as_raw()));
        return None;
    }
    if copies.is_empty() {
        note(&format!(
            "copy image format {}, layout {}",
            info.format.as_raw(),
            layout.as_raw()
        ));
    }
    copies.insert(
        world,
        StandardCopy {
            view,
            image,
            width: info.extent.width,
            height: info.extent.height,
        },
    );
    Some(view)
}

unsafe extern "system" fn get_descriptor(
    device: vk::Device,
    info: *const vk::DescriptorGetInfoEXT<'static>,
    size: usize,
    out: *mut c_void,
) {
    let real: GetDescriptorFn = pointer(&GET_DESCRIPTOR).unwrap();
    let given = (!info.is_null()).then(|| unsafe { *info });
    let image = given
        .filter(|g| {
            matches!(
                g.ty,
                vk::DescriptorType::SAMPLED_IMAGE | vk::DescriptorType::COMBINED_IMAGE_SAMPLER
            )
        })
        .map(|g| unsafe { g.data.p_sampled_image })
        .filter(|p| !p.is_null())
        .map(|p| unsafe { *p });
    let world = image.and_then(|i| VIEWS.lock().unwrap().get(&i.image_view.as_raw()).copied());
    let Some((image, world)) = image.zip(world).filter(|&(_, w)| reversed(vk::Image::from_raw(w)))
    else {
        unsafe { real(device, info, size, out) };
        return;
    };
    let Some(view) = copy_view(device, world) else {
        unsafe { real(device, info, size, out) };
        return;
    };
    let substitute = vk::DescriptorImageInfo {
        image_view: view,
        image_layout: vk::ImageLayout::GENERAL,
        ..image
    };
    let mut patched = given.unwrap();
    patched.data = vk::DescriptorDataEXT {
        p_sampled_image: &raw const substitute,
    };
    WORLD_READS.fetch_add(1, Ordering::Relaxed);
    unsafe { real(device, &raw const patched, size, out) };
}

unsafe extern "system" fn create_shader_module(
    device: vk::Device,
    info: *const vk::ShaderModuleCreateInfo<'static>,
    allocator: *const vk::AllocationCallbacks<'static>,
    module: *mut vk::ShaderModule,
) -> vk::Result {
    let real: CreateModuleFn = pointer(&CREATE_MODULE).unwrap();
    let result = unsafe { real(device, info, allocator, module) };
    if result == vk::Result::SUCCESS
        && let Some(info) = unsafe { info.as_ref() }
        && !info.p_code.is_null()
        && !module.is_null()
    {
        let code = unsafe { core::slice::from_raw_parts(info.p_code, info.code_size / 4) };
        if crate::spirv::is_vertex(code) {
            MODULES
                .lock()
                .unwrap()
                .insert(unsafe { *module }.as_raw(), code.to_vec());
        }
    }
    result
}

unsafe extern "system" fn destroy_shader_module(
    device: vk::Device,
    module: vk::ShaderModule,
    allocator: *const vk::AllocationCallbacks<'static>,
) {
    MODULES.lock().unwrap().remove(&module.as_raw());
    let real: DestroyModuleFn = pointer(&DESTROY_MODULE).unwrap();
    if let Some(Some(twin)) = MODULE_TWINS.lock().unwrap().remove(&module.as_raw()) {
        unsafe { real(device, twin, allocator) };
    }
    unsafe { real(device, module, allocator) };
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
        "vulkan: {} resolutions wrapped, world pipelines {}, world renderings {}, world views {}, world clears {}, world reads {}, in-stream copies {}, tracked images {}, vertex twins {} (refused {})",
        RESOLVED.load(Ordering::Relaxed),
        WORLD_PIPELINES.load(Ordering::Relaxed),
        WORLD_RENDERINGS.load(Ordering::Relaxed),
        WORLD_VIEWS.load(Ordering::Relaxed),
        WORLD_CLEARS.load(Ordering::Relaxed),
        WORLD_READS.load(Ordering::Relaxed),
        crate::stream::recorded(),
        IMAGES.lock().unwrap().len(),
        TWINS_MADE.load(Ordering::Relaxed),
        TWINS_REFUSED.load(Ordering::Relaxed),
    )
}
