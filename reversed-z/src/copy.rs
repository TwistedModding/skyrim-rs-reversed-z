//! 1 - depth of each depth target that holds reversed depth, in a float texture the target's shader
//! view is swapped for (and that stands in wherever the engine binds the target's own view), so
//! every reader (shadow masks, SAO, water, soft effects, depth of field, SSR, TAA, the volumetric
//! apply, the image-space passes) gets the standard depth it was written for.

use core::ffi::c_void;
use std::cell::Cell;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D11::{
    D3D11_BIND_SHADER_RESOURCE, D3D11_BIND_UNORDERED_ACCESS, D3D11_SHADER_MAX_INTERFACES,
    D3D11_SHADER_RESOURCE_VIEW_DESC, D3D11_TEXTURE2D_DESC, D3D11_USAGE_DEFAULT,
    ID3D11ClassInstance, ID3D11ComputeShader, ID3D11Device, ID3D11DeviceContext,
    ID3D11ShaderResourceView, ID3D11Texture2D, ID3D11UnorderedAccessView,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT_R24_UNORM_X8_TYPELESS, DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS,
    DXGI_SAMPLE_DESC,
};
use windows::core::{Interface, s};

use crate::context::{TARGET_SHADER_VIEW, TARGET_TEXTURE, device, immediate, target_field};

/// The depth targets holding reversed depth: the scene depth and the first-person target, drawn
/// mirrored, and the second scene depth, which the engine copies the scene depth into.
pub(crate) const REVERSED_TARGETS: [usize; 3] = [0, 1, 7];

const SHADER: &str = "Texture2D<float> depth : register(t0);\nRWTexture2D<float> standard : register(u0);\n[numthreads(8, 8, 1)]\nvoid main(uint3 id : SV_DispatchThreadID) { uint width, height; standard.GetDimensions(width, height); if (id.x < width && id.y < height) standard[id.xy] = 1.0 - depth[id.xy]; }\n";
const GROUP: u32 = 8;

struct Copy {
    /// Keeps the texture the views are of alive.
    _standard: ID3D11Texture2D,
    view: ID3D11ShaderResourceView,
    target: ID3D11UnorderedAccessView,
    /// The engine's shader view of the depth target, which its slot holds when not swapped.
    engine_view: ID3D11ShaderResourceView,
    width: u32,
    height: u32,
}

struct Copies {
    shader: ID3D11ComputeShader,
    copies: Vec<Copy>,
}

static COPIES: Mutex<Option<Copies>> = Mutex::new(None);
thread_local! { static COPYING: Cell<bool> = const { Cell::new(false) }; }
/// While the copies stand in, per target of [`REVERSED_TARGETS`]: the engine's view and the copy's
/// (atomics, read inside hooks a copy itself triggers while holding `COPIES`).
static ENGINE_VIEWS: [AtomicUsize; 3] = [const { AtomicUsize::new(0) }; 3];
static COPY_VIEWS: [AtomicUsize; 3] = [const { AtomicUsize::new(0) }; 3];
static COUNT: AtomicU64 = AtomicU64::new(0);

pub(crate) fn count() -> u64 {
    COUNT.load(Ordering::Relaxed)
}

fn create(device: &ID3D11Device, target: usize) -> Result<Copy, String> {
    let texture =
        unsafe { *target_field(target, TARGET_TEXTURE).ok_or("no renderer")? } as *mut c_void;
    let texture = unsafe { ID3D11Texture2D::from_raw_borrowed(&texture) }
        .ok_or_else(|| format!("no depth texture {target}"))?;
    let engine_view =
        unsafe { *target_field(target, TARGET_SHADER_VIEW).ok_or("no renderer")? } as *mut c_void;
    create_copy(device, texture, engine_view)
}

fn create_copy(
    device: &ID3D11Device,
    texture: &ID3D11Texture2D,
    engine_view: *mut c_void,
) -> Result<Copy, String> {
    let mut depth = D3D11_TEXTURE2D_DESC::default();
    unsafe { texture.GetDesc(&raw mut depth) };
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
    Ok(Copy {
        _standard: standard,
        view: view.ok_or("copy view")?,
        target: unordered.ok_or("copy target")?,
        engine_view: unsafe { ID3D11ShaderResourceView::from_raw_borrowed(&engine_view) }
            .cloned()
            .ok_or("no source depth view")?,
        width: depth.Width,
        height: depth.Height,
    })
}

fn create_all(device: &ID3D11Device) -> Result<Copies, String> {
    let code = compile()?;
    let bytes = unsafe {
        core::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
    };
    let mut shader = None;
    unsafe { device.CreateComputeShader(bytes, None, Some(&raw mut shader)) }
        .map_err(|e| format!("copy shader: {e}"))?;
    let copies = REVERSED_TARGETS
        .iter()
        .map(|&t| create(device, t))
        .collect::<Result<_, _>>()?;
    Ok(Copies {
        shader: shader.ok_or("copy shader")?,
        copies,
    })
}

fn compile() -> Result<ID3DBlob, String> {
    let mut code = None;
    let mut errors = None;
    unsafe {
        D3DCompile(
            SHADER.as_ptr().cast(),
            SHADER.len(),
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

/// Writes 1 - depth of depth target `target` (one of [`REVERSED_TARGETS`]) into its copy.
pub(crate) fn run(target: usize) {
    let Some(k) = REVERSED_TARGETS.iter().position(|&t| t == target) else {
        return;
    };
    if !crate::active() {
        return;
    }
    let copies = COPIES.lock().unwrap();
    let (Some(all), Some(context)) = (copies.as_ref(), immediate()) else {
        return;
    };
    let c = &all.copies[k];
    crate::context::mark_copied(target);
    convert(&context, &all.shader, c);
    COUNT.fetch_add(1, Ordering::Relaxed);
}

fn preserve_resources(context: &ID3D11DeviceContext, run: impl FnOnce()) {
    macro_rules! preserve {
        ($($name:ident: $get:ident => $set:ident),*) => {
            $(
                let mut $name: [Option<ID3D11ShaderResourceView>; 128] = core::array::from_fn(|_| None);
                unsafe { context.$get(0, Some(&mut $name)) };
            )*
            run();
            $(unsafe { context.$set(0, Some(&$name)) };)*
        };
    }
    preserve!(
        vertex: VSGetShaderResources => VSSetShaderResources,
        hull: HSGetShaderResources => HSSetShaderResources,
        domain: DSGetShaderResources => DSSetShaderResources,
        geometry: GSGetShaderResources => GSSetShaderResources,
        pixel: PSGetShaderResources => PSSetShaderResources,
        compute: CSGetShaderResources => CSSetShaderResources
    );
}

fn convert(context: &ID3D11DeviceContext, shader: &ID3D11ComputeShader, c: &Copy) {
    let previous_copying = COPYING.replace(true);
    preserve_resources(context, || {
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
                Some(&instances[..instance_count as usize]),
            );
        });
    });
    COPYING.set(previous_copying);
}

/// Puts the copy's shader view (or the engine's back) in its depth target's slot, holding the
/// reference the slot owns.
fn swap(target: usize, c: &Copy, on: bool) {
    let Some(slot) = target_field(target, TARGET_SHADER_VIEW) else {
        return;
    };
    let ours = c.view.as_raw();
    let current = unsafe { *slot } as *mut c_void;
    let wanted = if on { ours } else { c.engine_view.as_raw() };
    if current == wanted {
        return;
    }
    if on {
        core::mem::forget(c.view.clone());
    } else if current == ours {
        drop(unsafe { ID3D11ShaderResourceView::from_raw(ours) });
    }
    unsafe { *slot = wanted as usize };
}

/// The copy's view standing in for `view` when `view` is a reversed depth target's own.
pub(crate) fn stand_in(view: *mut c_void) -> Option<*mut c_void> {
    if view.is_null() || COPYING.get() {
        return None;
    }
    let k = ENGINE_VIEWS
        .iter()
        .position(|e| e.load(Ordering::Relaxed) == view as usize)
        .or_else(|| {
            let textures = REVERSED_TARGETS.map(|target| {
                target_field(target, TARGET_TEXTURE).map_or(0, |slot| unsafe { *slot })
            });
            alias_index(view, &textures)
        })?;
    let copy = COPY_VIEWS[k].load(Ordering::Relaxed);
    if copy == 0 {
        return None;
    }
    crate::context::refresh_depth(REVERSED_TARGETS[k]);
    Some(copy as *mut c_void)
}

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

/// Swaps the copies in or out, making them first if needed.
pub(crate) fn set_swapped(on: bool) {
    let mut copies = COPIES.lock().unwrap();
    if on && copies.is_none() {
        match device()
            .ok_or_else(|| "no device".to_owned())
            .and_then(|d| create_all(&d))
        {
            Ok(c) => *copies = Some(c),
            Err(e) => crate::log(&format!("depth copies unavailable: {e}")),
        }
    }
    if let Some(all) = copies.as_ref() {
        for (k, (&target, c)) in REVERSED_TARGETS.iter().zip(&all.copies).enumerate() {
            swap(target, c, on);
            ENGINE_VIEWS[k].store(c.engine_view.as_raw() as usize, Ordering::Relaxed);
            COPY_VIEWS[k].store(
                if on { c.view.as_raw() as usize } else { 0 },
                Ordering::Relaxed,
            );
        }
    }
}

pub(crate) fn release() {
    set_swapped(false);
    *COPIES.lock().unwrap() = None;
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
        let code = compile().unwrap();
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
        crate::context::remove();
    }
}
