//! 1 - depth of each depth target that holds reversed depth, in a float texture the target's shader
//! view is swapped for (and that stands in wherever the engine binds the target's own view), so
//! every reader (shadow masks, SAO, water, soft effects, depth of field, SSR, TAA, the volumetric
//! apply, the image-space passes) gets the standard depth it was written for.

use core::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D11::{
    D3D11_BIND_SHADER_RESOURCE, D3D11_BIND_UNORDERED_ACCESS, D3D11_TEXTURE2D_DESC,
    D3D11_USAGE_DEFAULT, ID3D11ComputeShader, ID3D11Device, ID3D11ShaderResourceView,
    ID3D11Texture2D, ID3D11UnorderedAccessView,
};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32_FLOAT, DXGI_SAMPLE_DESC};
use windows::core::{Interface, s};

use crate::context::{TARGET_SHADER_VIEW, TARGET_TEXTURE, device, immediate, target_field};

/// The depth targets holding reversed depth: the scene depth and the first-person target, drawn
/// mirrored, and the second scene depth, which the engine copies the scene depth into.
pub(crate) const REVERSED_TARGETS: [usize; 3] = [0, 1, 7];

const SHADER: &str = "Texture2D<float> depth : register(t0);\nRWTexture2D<float> standard : register(u0);\n[numthreads(8, 8, 1)]\nvoid main(uint3 id : SV_DispatchThreadID) { standard[id.xy] = 1.0 - depth[id.xy]; }\n";
const GROUP: u32 = 8;

struct Copy {
    /// Keeps the texture the views are of alive.
    _standard: ID3D11Texture2D,
    view: ID3D11ShaderResourceView,
    target: ID3D11UnorderedAccessView,
    /// The engine's shader view of the depth target, which its slot holds when not swapped.
    engine_view: *mut c_void,
    width: u32,
    height: u32,
}

struct Copies {
    shader: ID3D11ComputeShader,
    copies: Vec<Copy>,
}

unsafe impl Send for Copies {}

static COPIES: Mutex<Option<Copies>> = Mutex::new(None);
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
    let engine_view =
        unsafe { *target_field(target, TARGET_SHADER_VIEW).ok_or("no renderer")? } as *mut c_void;
    Ok(Copy {
        _standard: standard,
        view: view.ok_or("copy view")?,
        target: unordered.ok_or("copy target")?,
        engine_view,
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
    crate::context::without_targets(&context, || unsafe {
        let mut shader = None;
        context.CSGetShader(&raw mut shader, None, None);
        let mut inputs: [Option<ID3D11ShaderResourceView>; 1] = Default::default();
        context.CSGetShaderResources(0, Some(&mut inputs));
        let mut outputs: [Option<ID3D11UnorderedAccessView>; 1] = Default::default();
        context.CSGetUnorderedAccessViews(0, Some(&mut outputs));

        context.CSSetShader(&all.shader, None);
        crate::context::set_compute_resource_as_is(&context, c.engine_view);
        let unordered = [Some(c.target.clone())];
        context.CSSetUnorderedAccessViews(0, 1, Some(unordered.as_ptr()), None);
        context.Dispatch(c.width.div_ceil(GROUP), c.height.div_ceil(GROUP), 1);

        context.CSSetUnorderedAccessViews(0, 1, Some(outputs.as_ptr()), None);
        context.CSSetShaderResources(0, Some(&inputs));
        context.CSSetShader(shader.as_ref(), None);
    });
    COUNT.fetch_add(1, Ordering::Relaxed);
}

/// Puts the copy's shader view (or the engine's back) in its depth target's slot, holding the
/// reference the slot owns.
fn swap(target: usize, c: &Copy, on: bool) {
    let Some(slot) = target_field(target, TARGET_SHADER_VIEW) else {
        return;
    };
    let ours = c.view.as_raw();
    let current = unsafe { *slot } as *mut c_void;
    let wanted = if on { ours } else { c.engine_view };
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
    let k = ENGINE_VIEWS
        .iter()
        .position(|e| e.load(Ordering::Relaxed) == view as usize && !view.is_null())?;
    let copy = COPY_VIEWS[k].load(Ordering::Relaxed);
    (copy != 0).then_some(copy as *mut c_void)
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
            ENGINE_VIEWS[k].store(c.engine_view as usize, Ordering::Relaxed);
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
