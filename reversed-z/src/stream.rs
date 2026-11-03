//! The conventional copy of a world depth image refreshed inside Community Shaders' own command
//! stream: a compute dispatch writing 1 - depth into the copy, recorded into the command buffer a
//! rendering into the image ended in, before that buffer's next dispatch reads the copy (the
//! hierarchical depth buffer is built right after the depth draws). A refresh through the D3D
//! context would land in DXVK's stream after that read.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex};

use ash::vk;
use ash::vk::Handle;

const GROUP: u32 = 8;

type GpaFn = unsafe extern "system" fn(usize, *const core::ffi::c_char) -> Option<unsafe extern "system" fn()>;

struct Functions {
    create_layout: vk::PFN_vkCreateDescriptorSetLayout,
    create_pipeline_layout: vk::PFN_vkCreatePipelineLayout,
    create_module: vk::PFN_vkCreateShaderModule,
    create_pipelines: vk::PFN_vkCreateComputePipelines,
    create_pool: vk::PFN_vkCreateDescriptorPool,
    allocate_sets: vk::PFN_vkAllocateDescriptorSets,
    update_sets: vk::PFN_vkUpdateDescriptorSets,
    create_view: vk::PFN_vkCreateImageView,
    bind_pipeline: vk::PFN_vkCmdBindPipeline,
    bind_sets: vk::PFN_vkCmdBindDescriptorSets,
    barrier: vk::PFN_vkCmdPipelineBarrier,
    dispatch: vk::PFN_vkCmdDispatch,
}

struct Converter {
    device: vk::Device,
    functions: Functions,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    pool: vk::DescriptorPool,
}

/// A world image's descriptor set: its depth view at binding 0, its copy's storage view at 1.
pub(crate) struct Prepared {
    set: vk::DescriptorSet,
    copy: vk::Image,
    width: u32,
    height: u32,
}

static CONVERTER: Mutex<Option<Converter>> = Mutex::new(None);
static PREPARED: LazyLock<Mutex<HashMap<u64, Prepared>>> = LazyLock::new(Mutex::default);
static RECORDED: AtomicU64 = AtomicU64::new(0);

pub(crate) fn recorded() -> u64 {
    RECORDED.load(Ordering::Relaxed)
}

macro_rules! resolve {
    ($gpa:expr, $device:expr, $name:literal) => {{
        let f = unsafe { $gpa($device, $name.as_ptr()) }.ok_or(concat!("no ", stringify!($name)))?;
        unsafe { core::mem::transmute::<unsafe extern "system" fn(), _>(f) }
    }};
}

/// Makes the compute pipeline on DXVK's device.
#[allow(clippy::too_many_lines, clippy::missing_transmute_annotations)]
pub(crate) fn setup(device: vk::Device, gpa: GpaFn) -> Result<(), String> {
    let handle = usize::try_from(device.as_raw()).map_err(|e| e.to_string())?;
    let functions = Functions {
        create_layout: resolve!(gpa, handle, c"vkCreateDescriptorSetLayout"),
        create_pipeline_layout: resolve!(gpa, handle, c"vkCreatePipelineLayout"),
        create_module: resolve!(gpa, handle, c"vkCreateShaderModule"),
        create_pipelines: resolve!(gpa, handle, c"vkCreateComputePipelines"),
        create_pool: resolve!(gpa, handle, c"vkCreateDescriptorPool"),
        allocate_sets: resolve!(gpa, handle, c"vkAllocateDescriptorSets"),
        update_sets: resolve!(gpa, handle, c"vkUpdateDescriptorSets"),
        create_view: resolve!(gpa, handle, c"vkCreateImageView"),
        bind_pipeline: resolve!(gpa, handle, c"vkCmdBindPipeline"),
        bind_sets: resolve!(gpa, handle, c"vkCmdBindDescriptorSets"),
        barrier: resolve!(gpa, handle, c"vkCmdPipelineBarrier"),
        dispatch: resolve!(gpa, handle, c"vkCmdDispatch"),
    };
    let bindings = [
        vk::DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            ..Default::default()
        },
        vk::DescriptorSetLayoutBinding {
            binding: 1,
            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            ..Default::default()
        },
    ];
    let layout_info = vk::DescriptorSetLayoutCreateInfo {
        binding_count: 2,
        p_bindings: bindings.as_ptr(),
        ..Default::default()
    };
    let mut set_layout = vk::DescriptorSetLayout::null();
    check(
        unsafe { (functions.create_layout)(device, &raw const layout_info, core::ptr::null(), &raw mut set_layout) },
        "descriptor set layout",
    )?;
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo {
        set_layout_count: 1,
        p_set_layouts: &raw const set_layout,
        ..Default::default()
    };
    let mut layout = vk::PipelineLayout::null();
    check(
        unsafe {
            (functions.create_pipeline_layout)(device, &raw const pipeline_layout_info, core::ptr::null(), &raw mut layout)
        },
        "pipeline layout",
    )?;
    let code = shader();
    let module_info = vk::ShaderModuleCreateInfo {
        code_size: code.len() * 4,
        p_code: code.as_ptr(),
        ..Default::default()
    };
    let mut module = vk::ShaderModule::null();
    check(
        unsafe { (functions.create_module)(device, &raw const module_info, core::ptr::null(), &raw mut module) },
        "shader module",
    )?;
    let pipeline_info = vk::ComputePipelineCreateInfo {
        stage: vk::PipelineShaderStageCreateInfo {
            stage: vk::ShaderStageFlags::COMPUTE,
            module,
            p_name: c"main".as_ptr(),
            ..Default::default()
        },
        layout,
        ..Default::default()
    };
    let mut pipeline = vk::Pipeline::null();
    check(
        unsafe {
            (functions.create_pipelines)(
                device,
                vk::PipelineCache::null(),
                1,
                &raw const pipeline_info,
                core::ptr::null(),
                &raw mut pipeline,
            )
        },
        "compute pipeline",
    )?;
    let sizes = [
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::SAMPLED_IMAGE,
            descriptor_count: 16,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_IMAGE,
            descriptor_count: 16,
        },
    ];
    let pool_info = vk::DescriptorPoolCreateInfo {
        max_sets: 16,
        pool_size_count: 2,
        p_pool_sizes: sizes.as_ptr(),
        ..Default::default()
    };
    let mut pool = vk::DescriptorPool::null();
    check(
        unsafe { (functions.create_pool)(device, &raw const pool_info, core::ptr::null(), &raw mut pool) },
        "descriptor pool",
    )?;
    *CONVERTER.lock().unwrap() = Some(Converter {
        device,
        functions,
        set_layout,
        layout,
        pipeline,
        pool,
    });
    Ok(())
}

fn check(result: vk::Result, what: &str) -> Result<(), String> {
    if result == vk::Result::SUCCESS {
        Ok(())
    } else {
        Err(format!("{what}: {}", result.as_raw()))
    }
}

/// Gives a world image its descriptor set, once: a depth view of it and the storage view of
/// its copy.
pub(crate) fn prepare(
    world: vk::Image,
    format: vk::Format,
    copy: vk::Image,
    copy_view: vk::ImageView,
    width: u32,
    height: u32,
) -> Result<(), String> {
    let guard = CONVERTER.lock().unwrap();
    let c = guard.as_ref().ok_or("no converter")?;
    let mut prepared = PREPARED.lock().unwrap();
    if prepared.contains_key(&world.as_raw()) {
        return Ok(());
    }
    let view_info = vk::ImageViewCreateInfo {
        image: world,
        view_type: vk::ImageViewType::TYPE_2D,
        format,
        subresource_range: vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::DEPTH,
            level_count: 1,
            layer_count: 1,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut depth_view = vk::ImageView::null();
    check(
        unsafe { (c.functions.create_view)(c.device, &raw const view_info, core::ptr::null(), &raw mut depth_view) },
        "depth view",
    )?;
    let allocate_info = vk::DescriptorSetAllocateInfo {
        descriptor_pool: c.pool,
        descriptor_set_count: 1,
        p_set_layouts: &raw const c.set_layout,
        ..Default::default()
    };
    let mut set = vk::DescriptorSet::null();
    check(
        unsafe { (c.functions.allocate_sets)(c.device, &raw const allocate_info, &raw mut set) },
        "descriptor set",
    )?;
    let depth_info = vk::DescriptorImageInfo {
        image_view: depth_view,
        image_layout: vk::ImageLayout::GENERAL,
        ..Default::default()
    };
    let copy_info = vk::DescriptorImageInfo {
        image_view: copy_view,
        image_layout: vk::ImageLayout::GENERAL,
        ..Default::default()
    };
    let writes = [
        vk::WriteDescriptorSet {
            dst_set: set,
            dst_binding: 0,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
            p_image_info: &raw const depth_info,
            ..Default::default()
        },
        vk::WriteDescriptorSet {
            dst_set: set,
            dst_binding: 1,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
            p_image_info: &raw const copy_info,
            ..Default::default()
        },
    ];
    unsafe { (c.functions.update_sets)(c.device, 2, writes.as_ptr(), 0, core::ptr::null()) };
    prepared.insert(
        world.as_raw(),
        Prepared {
            set,
            copy,
            width,
            height,
        },
    );
    Ok(())
}

pub(crate) fn is_prepared(world: vk::Image) -> bool {
    PREPARED.lock().unwrap().contains_key(&world.as_raw())
}

/// Records the conversion of `world` into its copy: the image taken to GENERAL from the layout
/// its rendering left it in and back after, the copy fenced against every earlier and later use.
#[allow(clippy::too_many_lines)]
pub(crate) fn record(buffer: vk::CommandBuffer, world: vk::Image, layout: vk::ImageLayout) -> bool {
    let guard = CONVERTER.lock().unwrap();
    let Some(c) = guard.as_ref() else {
        return false;
    };
    let prepared = PREPARED.lock().unwrap();
    let Some(p) = prepared.get(&world.as_raw()) else {
        return false;
    };
    let f = &c.functions;
    let depth_range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
        level_count: 1,
        layer_count: 1,
        ..Default::default()
    };
    let color_range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        level_count: 1,
        layer_count: 1,
        ..Default::default()
    };
    let before = [
        vk::ImageMemoryBarrier {
            src_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
            dst_access_mask: vk::AccessFlags::SHADER_READ,
            old_layout: layout,
            new_layout: vk::ImageLayout::GENERAL,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image: world,
            subresource_range: depth_range,
            ..Default::default()
        },
        vk::ImageMemoryBarrier {
            src_access_mask: vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
            dst_access_mask: vk::AccessFlags::SHADER_WRITE,
            old_layout: vk::ImageLayout::GENERAL,
            new_layout: vk::ImageLayout::GENERAL,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image: p.copy,
            subresource_range: color_range,
            ..Default::default()
        },
    ];
    let after = [
        vk::ImageMemoryBarrier {
            src_access_mask: vk::AccessFlags::SHADER_READ,
            dst_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                | vk::AccessFlags::MEMORY_READ,
            old_layout: vk::ImageLayout::GENERAL,
            new_layout: layout,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image: world,
            subresource_range: depth_range,
            ..Default::default()
        },
        vk::ImageMemoryBarrier {
            src_access_mask: vk::AccessFlags::SHADER_WRITE,
            dst_access_mask: vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
            old_layout: vk::ImageLayout::GENERAL,
            new_layout: vk::ImageLayout::GENERAL,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image: p.copy,
            subresource_range: color_range,
            ..Default::default()
        },
    ];
    unsafe {
        (f.barrier)(
            buffer,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            0,
            core::ptr::null(),
            0,
            core::ptr::null(),
            2,
            before.as_ptr(),
        );
        (f.bind_pipeline)(buffer, vk::PipelineBindPoint::COMPUTE, c.pipeline);
        (f.bind_sets)(
            buffer,
            vk::PipelineBindPoint::COMPUTE,
            c.layout,
            0,
            1,
            &raw const p.set,
            0,
            core::ptr::null(),
        );
        (f.dispatch)(buffer, p.width.div_ceil(GROUP), p.height.div_ceil(GROUP), 1);
        (f.barrier)(
            buffer,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            0,
            core::ptr::null(),
            0,
            core::ptr::null(),
            2,
            after.as_ptr(),
        );
    }
    RECORDED.fetch_add(1, Ordering::Relaxed);
    true
}

/// The compute shader: `copy[id] = 1 - depth[id]` for every texel of the copy.
fn shader() -> Vec<u32> {
    let op = |op: u32, len: u32| op | (len << 16);
    let mut c = vec![0x0723_0203, 0x0001_0300, 0, 60, 0];
    c.extend([op(17, 2), 1]); // Capability Shader
    c.extend([op(17, 2), 50]); // Capability ImageQuery
    c.extend([op(14, 3), 0, 1]); // MemoryModel Logical GLSL450
    c.extend([op(15, 6), 5, 10, 0x6e69_616d, 0, 11]); // EntryPoint GLCompute %10 "main" %11
    c.extend([op(16, 6), 10, 17, 8, 8, 1]); // ExecutionMode LocalSize 8 8 1
    c.extend([op(71, 4), 11, 11, 28]); // Decorate %11 BuiltIn GlobalInvocationId
    c.extend([op(71, 4), 12, 34, 0]); // Decorate %12 DescriptorSet 0
    c.extend([op(71, 4), 12, 33, 0]); // Decorate %12 Binding 0
    c.extend([op(71, 4), 13, 34, 0]); // Decorate %13 DescriptorSet 0
    c.extend([op(71, 4), 13, 33, 1]); // Decorate %13 Binding 1
    c.extend([op(19, 2), 1]); // %1 void
    c.extend([op(33, 3), 2, 1]); // %2 fn void
    c.extend([op(22, 3), 3, 32]); // %3 float
    c.extend([op(21, 4), 4, 32, 0]); // %4 uint
    c.extend([op(21, 4), 6, 32, 1]); // %6 int
    c.extend([op(23, 4), 7, 4, 3]); // %7 v3uint
    c.extend([op(23, 4), 8, 4, 2]); // %8 v2uint
    c.extend([op(23, 4), 9, 6, 2]); // %9 v2int
    c.extend([op(23, 4), 14, 3, 4]); // %14 v4float
    c.extend([op(20, 2), 15]); // %15 bool
    c.extend([op(23, 4), 16, 15, 2]); // %16 v2bool
    c.extend([op(25, 9), 18, 3, 1, 0, 0, 0, 1, 0]); // %18 image float 2D sampled=1 Unknown
    c.extend([op(25, 9), 19, 3, 1, 0, 0, 0, 2, 3]); // %19 image float 2D sampled=2 R32f
    c.extend([op(32, 4), 20, 0, 18]); // %20 ptr UniformConstant %18
    c.extend([op(32, 4), 21, 0, 19]); // %21 ptr UniformConstant %19
    c.extend([op(32, 4), 22, 1, 7]); // %22 ptr Input v3uint
    c.extend([op(43, 4), 3, 23, 1.0f32.to_bits()]); // %23 = 1.0
    c.extend([op(43, 4), 6, 24, 0]); // %24 = int 0
    c.extend([op(59, 4), 22, 11, 1]); // %11 = Variable Input
    c.extend([op(59, 4), 20, 12, 0]); // %12 = Variable UniformConstant (depth)
    c.extend([op(59, 4), 21, 13, 0]); // %13 = Variable UniformConstant (copy)
    c.extend([op(54, 5), 1, 10, 0, 2]); // %10 = Function
    c.extend([op(248, 2), 30]); // Label
    c.extend([op(61, 4), 7, 31, 11]); // %31 = Load gid
    c.extend([op(79, 7), 8, 32, 31, 31, 0, 1]); // %32 = xy
    c.extend([op(61, 4), 19, 33, 13]); // %33 = Load copy image
    c.extend([op(104, 4), 8, 34, 33]); // %34 = ImageQuerySize
    c.extend([op(176, 5), 16, 35, 32, 34]); // %35 = ULessThan
    c.extend([op(155, 4), 15, 36, 35]); // %36 = All
    c.extend([op(247, 3), 38, 0]); // SelectionMerge %38
    c.extend([op(250, 4), 36, 37, 38]); // BranchConditional
    c.extend([op(248, 2), 37]); // Label then
    c.extend([op(61, 4), 18, 39, 12]); // %39 = Load depth image
    c.extend([op(124, 4), 9, 40, 32]); // %40 = Bitcast v2int
    c.extend([op(95, 7), 14, 41, 39, 40, 2, 24]); // %41 = ImageFetch Lod %24
    c.extend([op(81, 5), 3, 42, 41, 0]); // %42 = .x
    c.extend([op(131, 5), 3, 43, 23, 42]); // %43 = 1 - x
    c.extend([op(80, 7), 14, 44, 43, 43, 43, 43]); // %44 = v4
    c.extend([op(99, 4), 33, 40, 44]); // ImageWrite
    c.extend([op(249, 2), 38]); // Branch merge
    c.extend([op(248, 2), 38]); // Label merge
    c.extend([op(253, 1)]); // Return
    c.extend([op(56, 1)]); // FunctionEnd
    c
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_shader_parses_as_whole_instructions() {
        let code = super::shader();
        let mut at = 5;
        let mut ops = Vec::new();
        while at < code.len() {
            let len = (code[at] >> 16) as usize;
            assert!(len > 0 && at + len <= code.len());
            ops.push(code[at] & 0xffff);
            at += len;
        }
        assert_eq!(ops[3], 15);
        assert_eq!(&code[5 + 2 + 2 + 3..5 + 2 + 2 + 3 + 6], &[15 | (6 << 16), 5, 10, 0x6e69_616d, 0, 11]);
        assert!(ops.ends_with(&[99, 249, 248, 253, 56]));
    }
}
