// Copyright (c) 2026 TwistedModding. SPDX-License-Identifier: MPL-2.0-no-copyleft-exception
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not
// distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.
//! Captured bytecode and mirrored shaders owned by their original D3D objects.

use core::ffi::c_void;
use std::cell::Cell;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use windows::Win32::Foundation::HMODULE;
use windows::Win32::Graphics::Direct3D::{D3D_DRIVER_TYPE, D3D_FEATURE_LEVEL};
use windows::Win32::Graphics::Direct3D11::{
    ID3D11Device, ID3D11DeviceChild, ID3D11PixelShader, ID3D11VertexShader,
};
use windows::Win32::Graphics::Dxgi::DXGI_SWAP_CHAIN_DESC;
use windows::Win32::System::LibraryLoader::{GetModuleHandleW, GetProcAddress, LoadLibraryW};
use windows::Win32::System::SystemInformation::GetSystemDirectoryW;
use windows::core::{GUID, HRESULT, Interface, PCWSTR, s, w};

use crate::context::original;

type CreateDeviceFn = unsafe extern "system" fn(
    *mut c_void,
    D3D_DRIVER_TYPE,
    HMODULE,
    u32,
    *const D3D_FEATURE_LEVEL,
    u32,
    u32,
    *mut *mut c_void,
    *mut D3D_FEATURE_LEVEL,
    *mut *mut c_void,
) -> HRESULT;
type CreateSwapChainFn = unsafe extern "system" fn(
    *mut c_void,
    D3D_DRIVER_TYPE,
    HMODULE,
    u32,
    *const D3D_FEATURE_LEVEL,
    u32,
    u32,
    *const DXGI_SWAP_CHAIN_DESC,
    *mut *mut c_void,
    *mut *mut c_void,
    *mut D3D_FEATURE_LEVEL,
    *mut *mut c_void,
) -> HRESULT;
type CreateVertexFn = unsafe extern "system" fn(
    *mut c_void,
    *const c_void,
    usize,
    *mut c_void,
    *mut *mut c_void,
) -> HRESULT;

const BYTECODE: GUID = GUID::from_u128(0xd477a0ab_d726_49e6_950b_e489c39b014a);
const TWIN: GUID = GUID::from_u128(0x68471e87_b4a1_43de_842f_6907faeae34e);
const CONSTANTS: GUID = GUID::from_u128(0x66b348c6_c52c_47dc_81c5_650970d4c881);
const NO_CONSTANTS: u32 = u32::MAX;
const FAILED: u32 = u32::MAX - 1;
const UNCHANGED: u32 = u32::MAX - 2;

static CREATE_DEVICE: Mutex<Option<hook::Hook<CreateDeviceFn>>> = Mutex::new(None);
static CREATE_SWAP_CHAIN: Mutex<Option<hook::Hook<CreateSwapChainFn>>> = Mutex::new(None);
static CREATE_VERTEX: Mutex<Option<hook::Hook<CreateVertexFn>>> = Mutex::new(None);
static CREATE_PIXEL: Mutex<Option<hook::Hook<CreateVertexFn>>> = Mutex::new(None);
static CAPTURED: AtomicU64 = AtomicU64::new(0);
static MADE: AtomicU64 = AtomicU64::new(0);
static PRECISE: AtomicU64 = AtomicU64::new(0);
static FAILURES: AtomicU64 = AtomicU64::new(0);
static MISSED: AtomicU64 = AtomicU64::new(0);
static PIXEL_MADE: AtomicU64 = AtomicU64::new(0);
static PIXEL_FAILURES: AtomicU64 = AtomicU64::new(0);
thread_local! { static BUILDING: Cell<bool> = const { Cell::new(false) }; }

fn install_hook<F: Copy>(
    slot: &Mutex<Option<hook::Hook<F>>>,
    target: usize,
    detour: F,
) -> Result<(), String> {
    let mut slot = slot.lock().unwrap();
    if let Some(hook) = slot.as_ref() {
        return if hook.target() == target {
            Ok(())
        } else {
            Err("a different D3D device implementation is already hooked".into())
        };
    }
    let hook = unsafe { hook::Hook::create(target, detour) }.map_err(|e| e.to_string())?;
    *slot = Some(hook);
    if let Err(e) = slot.as_ref().unwrap().enable() {
        slot.take();
        return Err(e.to_string());
    }
    Ok(())
}

/// DXVK's D3D11, loaded by Community Shaders.
pub(crate) fn dxvk() -> Option<HMODULE> {
    unsafe { GetModuleHandleW(w!("dxvk_d3d11.dll")) }.ok()
}

/// The system's own D3D11, by its full path: a wrapper in the game folder (ENB) takes the name.
pub(crate) fn system_d3d11() -> Option<HMODULE> {
    let mut dir = [0u16; 260];
    let len = unsafe { GetSystemDirectoryW(Some(&mut dir)) } as usize;
    let path: Vec<u16> = dir[..len]
        .iter()
        .copied()
        .chain("\\d3d11.dll".encode_utf16())
        .chain(Some(0))
        .collect();
    unsafe { LoadLibraryW(PCWSTR(path.as_ptr())) }.ok()
}

/// Hooks device creation on DXVK's D3D11 when Community Shaders loaded it, else the system's, so
/// bytecode is captured on the device every draw finally reaches.
pub(crate) fn install() -> Result<(), String> {
    let module = dxvk().or_else(system_d3d11).ok_or("no D3D11 module")?;
    let device =
        unsafe { GetProcAddress(module, s!("D3D11CreateDevice")) }.ok_or("no D3D11CreateDevice")?;
    let swap_chain = unsafe { GetProcAddress(module, s!("D3D11CreateDeviceAndSwapChain")) }
        .ok_or("no D3D11CreateDeviceAndSwapChain")?;
    // both are created before either is live, so a failure leaves nothing to remove under a caller
    let device = unsafe { hook::Hook::create(device as usize, create_device as CreateDeviceFn) }
        .map_err(|e| e.to_string())?;
    let swap_chain =
        match unsafe { hook::Hook::create(swap_chain as usize, create_swap_chain as CreateSwapChainFn) } {
            Ok(hook) => hook,
            Err(e) => {
                let _ = unsafe { device.remove() };
                return Err(e.to_string());
            }
        };
    *CREATE_DEVICE.lock().unwrap() = Some(device);
    *CREATE_SWAP_CHAIN.lock().unwrap() = Some(swap_chain);
    crate::context::enable_created(&CREATE_DEVICE)?;
    crate::context::enable_created(&CREATE_SWAP_CHAIN)
}

pub(crate) fn install_device(device: &ID3D11Device) -> Result<(), String> {
    install_hook(
        &CREATE_VERTEX,
        device.vtable().CreateVertexShader as usize,
        create_vertex as CreateVertexFn,
    )?;
    install_hook(
        &CREATE_PIXEL,
        device.vtable().CreatePixelShader as usize,
        create_pixel as CreateVertexFn,
    )
}

pub(crate) fn remove() {
    macro_rules! remove { ($($slot:ident),*) => { $(
        if let Some(hook) = $slot.lock().unwrap().take() { let _ = unsafe { hook.remove() }; }
    )* }; }
    remove!(
        CREATE_SWAP_CHAIN,
        CREATE_DEVICE,
        CREATE_VERTEX,
        CREATE_PIXEL
    );
}

fn created(device: *mut *mut c_void, result: HRESULT) {
    if result.is_ok()
        && !device.is_null()
        && let Some(device) = unsafe { ID3D11Device::from_raw_borrowed(&*device) }
        && let Err(e) = install_device(device)
    {
        crate::log(&format!("vertex shader capture hook unavailable: {e}"));
    }
}

#[allow(clippy::too_many_arguments)]
unsafe extern "system" fn create_device(
    adapter: *mut c_void,
    driver: D3D_DRIVER_TYPE,
    software: HMODULE,
    flags: u32,
    levels: *const D3D_FEATURE_LEVEL,
    count: u32,
    sdk: u32,
    device: *mut *mut c_void,
    level: *mut D3D_FEATURE_LEVEL,
    context: *mut *mut c_void,
) -> HRESULT {
    let result = unsafe {
        original(&CREATE_DEVICE)(
            adapter, driver, software, flags, levels, count, sdk, device, level, context,
        )
    };
    created(device, result);
    result
}

#[allow(clippy::too_many_arguments)]
unsafe extern "system" fn create_swap_chain(
    adapter: *mut c_void,
    driver: D3D_DRIVER_TYPE,
    software: HMODULE,
    flags: u32,
    levels: *const D3D_FEATURE_LEVEL,
    count: u32,
    sdk: u32,
    desc: *const DXGI_SWAP_CHAIN_DESC,
    swap_chain: *mut *mut c_void,
    device: *mut *mut c_void,
    level: *mut D3D_FEATURE_LEVEL,
    context: *mut *mut c_void,
) -> HRESULT {
    let result = unsafe {
        original(&CREATE_SWAP_CHAIN)(
            adapter, driver, software, flags, levels, count, sdk, desc, swap_chain, device, level,
            context,
        )
    };
    created(device, result);
    result
}

macro_rules! creation_hook {
    ($name:ident, $slot:ident) => {
        unsafe extern "system" fn $name(
            device: *mut c_void,
            code: *const c_void,
            size: usize,
            linkage: *mut c_void,
            shader: *mut *mut c_void,
        ) -> HRESULT {
            let result = unsafe { original(&$slot)(device, code, size, linkage, shader) };
            if result.is_ok()
                && !shader.is_null()
                && !code.is_null()
                && !BUILDING.get()
                && let Some(shader) = unsafe { ID3D11DeviceChild::from_raw_borrowed(&*shader) }
                && let Ok(size) = u32::try_from(size)
            {
                match unsafe { shader.SetPrivateData(&BYTECODE, size, Some(code)) } {
                    Ok(()) => {
                        CAPTURED.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(e) => crate::log(&format!("shader bytecode capture failed: {e}")),
                }
            }
            result
        }
    };
}
creation_hook!(create_vertex, CREATE_VERTEX);
creation_hook!(create_pixel, CREATE_PIXEL);

fn bytecode(shader: &ID3D11DeviceChild) -> Option<Vec<u8>> {
    let mut size = 0;
    unsafe { shader.GetPrivateData(&BYTECODE, &raw mut size, None) }.ok()?;
    let mut code = vec![0; size as usize];
    unsafe { shader.GetPrivateData(&BYTECODE, &raw mut size, Some(code.as_mut_ptr().cast())) }
        .ok()?;
    Some(code)
}

pub(crate) fn mirrored(
    shader: &ID3D11VertexShader,
    fallback: impl FnOnce() -> Option<Vec<u8>>,
) -> Option<(ID3D11VertexShader, Option<u32>)> {
    mirror(
        shader,
        fallback,
        |code| crate::twin::mirrored(code).map(crate::twin::Pixel::Mirrored),
        |device| device.vtable().CreateVertexShader,
        true,
    )
}

pub(crate) fn pixel(shader: &ID3D11PixelShader) -> Option<ID3D11PixelShader> {
    mirror(
        shader,
        || None,
        crate::twin::pixel,
        |device| device.vtable().CreatePixelShader,
        false,
    )
    .map(|(shader, _)| shader)
}

fn mirror<S: Interface + core::ops::Deref<Target = ID3D11DeviceChild>>(
    shader: &S,
    fallback: impl FnOnce() -> Option<Vec<u8>>,
    rewrite: impl FnOnce(&[u8]) -> Option<crate::twin::Pixel>,
    create: impl FnOnce(&ID3D11Device) -> CreateVertexFn,
    vertex: bool,
) -> Option<(S, Option<u32>)> {
    let mut slot = NO_CONSTANTS;
    let mut size = 4;
    if unsafe { shader.GetPrivateData(&CONSTANTS, &raw mut size, Some((&raw mut slot).cast())) }
        .is_ok()
    {
        if slot == FAILED {
            return None;
        }
        if slot == UNCHANGED {
            return Some((shader.clone(), None));
        }
        let mut twin: *mut c_void = core::ptr::null_mut();
        size = 8;
        unsafe { shader.GetPrivateData(&TWIN, &raw mut size, Some((&raw mut twin).cast())) }
            .ok()?;
        if twin.is_null() {
            return None;
        }
        return Some((
            unsafe { S::from_raw(twin) },
            (slot != NO_CONSTANTS).then_some(slot),
        ));
    }
    let Some(code) = bytecode(shader).or_else(fallback) else {
        if MISSED.fetch_add(1, Ordering::Relaxed) == 0 {
            crate::log(if vertex {
                "world vertex shader bytecode unavailable; shaders created before capture may not be reversed"
            } else {
                "world pixel shader bytecode unavailable; shaders created before capture may not be reversed"
            });
        }
        return None;
    };
    let Some(rewritten) = rewrite(&code) else {
        slot = FAILED;
        let _ = unsafe { shader.SetPrivateData(&CONSTANTS, 4, Some((&raw const slot).cast())) };
        let failures = if vertex { &FAILURES } else { &PIXEL_FAILURES };
        if failures.fetch_add(1, Ordering::Relaxed) == 0 {
            crate::log(if vertex {
                "world vertex shader rewrite refused"
            } else {
                "world pixel shader rewrite refused"
            });
        }
        return None;
    };
    let crate::twin::Pixel::Mirrored(rewritten) = rewritten else {
        slot = UNCHANGED;
        let _ = unsafe { shader.SetPrivateData(&CONSTANTS, 4, Some((&raw const slot).cast())) };
        return Some((shader.clone(), None));
    };
    let made = (|| {
        let device = unsafe { shader.GetDevice() }.ok()?;
        let mut twin = core::ptr::null_mut();
        let previous = BUILDING.replace(true);
        let result = unsafe {
            create(&device)(
                device.as_raw(),
                rewritten.code.as_ptr().cast(),
                rewritten.code.len(),
                core::ptr::null_mut(),
                &raw mut twin,
            )
        };
        BUILDING.set(previous);
        result.ok().ok()?;
        if twin.is_null() {
            return None;
        }
        let twin = unsafe { S::from_raw(twin) };
        let child: &ID3D11DeviceChild = shader;
        let key = TWIN;
        unsafe {
            (child.vtable().SetPrivateDataInterface)(child.as_raw(), &raw const key, twin.as_raw())
        }
        .ok()
        .ok()?;
        Some((twin, rewritten.constants))
    })();
    slot = made
        .as_ref()
        .map_or(FAILED, |(_, slot)| slot.unwrap_or(NO_CONSTANTS));
    let _ = unsafe { shader.SetPrivateData(&CONSTANTS, 4, Some((&raw const slot).cast())) };
    if made.is_some() {
        (if vertex { &MADE } else { &PIXEL_MADE }).fetch_add(1, Ordering::Relaxed);
        if slot != NO_CONSTANTS {
            PRECISE.fetch_add(1, Ordering::Relaxed);
        }
    } else if (if vertex { &FAILURES } else { &PIXEL_FAILURES }).fetch_add(1, Ordering::Relaxed)
        == 0
    {
        crate::log(if vertex {
            "world vertex shader creation failed after rewriting"
        } else {
            "world pixel shader creation failed after rewriting"
        });
    }
    made
}

pub(crate) fn counters() -> String {
    format!(
        "vertex shader twins {} (precise {}, not made {}, bytecode misses {}), captured shaders {}, pixel shader twins {} (not made {})",
        MADE.load(Ordering::Relaxed),
        PRECISE.load(Ordering::Relaxed),
        FAILURES.load(Ordering::Relaxed),
        MISSED.load(Ordering::Relaxed),
        CAPTURED.load(Ordering::Relaxed),
        PIXEL_MADE.load(Ordering::Relaxed),
        PIXEL_FAILURES.load(Ordering::Relaxed)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::twin::tests::{CONSTANT, ROWS, compile, compile_stage, occupied_slots};
    use windows::Win32::Graphics::Direct3D::{
        D3D_DRIVER_TYPE_WARP, D3D_PRIMITIVE_TOPOLOGY_POINTLIST,
    };
    use windows::Win32::Graphics::Direct3D11::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32G32B32A32_FLOAT;

    fn buffer(device: &ID3D11Device, data: &[f32], bind: D3D11_BIND_FLAG) -> ID3D11Buffer {
        let desc = D3D11_BUFFER_DESC {
            ByteWidth: u32::try_from(size_of_val(data)).unwrap(),
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: bind.0.cast_unsigned(),
            ..Default::default()
        };
        let initial = D3D11_SUBRESOURCE_DATA {
            pSysMem: data.as_ptr().cast(),
            ..Default::default()
        };
        let mut buffer = None;
        unsafe {
            device.CreateBuffer(
                &raw const desc,
                Some(&raw const initial),
                Some(&raw mut buffer),
            )
        }
        .unwrap();
        buffer.unwrap()
    }

    #[allow(clippy::too_many_lines)]
    fn execute(
        device: &ID3D11Device,
        context: &ID3D11DeviceContext,
        shader: &ID3D11VertexShader,
        source: &[u8],
        projection: [f32; 4],
        constants: Option<u32>,
        distance: f32,
    ) -> [f32; 5] {
        let [a, b, orthographic, scale] = projection;
        let rows = [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            a,
            b,
            0.0,
            0.0,
            1.0 - orthographic,
            orthographic,
        ];
        let matrix = buffer(device, &rows, D3D11_BIND_CONSTANT_BUFFER);
        let foreign = buffer(device, &[3.0, 4.0, 5.0, 6.0], D3D11_BIND_CONSTANT_BUFFER);
        let vertex = buffer(device, &[1.0, 0.0, distance, 1.0], D3D11_BIND_VERTEX_BUFFER);
        let output = buffer(device, &[0.0; 5], D3D11_BIND_STREAM_OUTPUT);
        let mut layout = None;
        let input = [D3D11_INPUT_ELEMENT_DESC {
            SemanticName: s!("POSITION"),
            Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
            InputSlotClass: D3D11_INPUT_PER_VERTEX_DATA,
            ..Default::default()
        }];
        unsafe { device.CreateInputLayout(&input, source, Some(&raw mut layout)) }.unwrap();
        let geometry = compile_stage(
            "struct V { float4 p : SV_Position; float fog : FOG; };\n[maxvertexcount(1)] void main(point V input[1], inout PointStream<V> output) { output.Append(input[0]); }",
            s!("gs_5_0"),
        );
        let declarations = [
            D3D11_SO_DECLARATION_ENTRY {
                SemanticName: s!("SV_Position"),
                ComponentCount: 4,
                ..Default::default()
            },
            D3D11_SO_DECLARATION_ENTRY {
                SemanticName: s!("FOG"),
                ComponentCount: 1,
                ..Default::default()
            },
        ];
        let mut gs = None;
        unsafe {
            device.CreateGeometryShaderWithStreamOutput(
                &geometry,
                Some(&declarations),
                Some(&[20]),
                D3D11_SO_NO_RASTERIZED_STREAM,
                None,
                Some(&raw mut gs),
            )
        }
        .unwrap();
        let read_desc = D3D11_BUFFER_DESC {
            ByteWidth: 20,
            Usage: D3D11_USAGE_STAGING,
            CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
            ..Default::default()
        };
        let mut read = None;
        unsafe { device.CreateBuffer(&raw const read_desc, None, Some(&raw mut read)) }.unwrap();
        let read = read.unwrap();
        unsafe {
            context.IASetInputLayout(layout.as_ref());
            context.IASetVertexBuffers(0, 1, Some([Some(vertex)].as_ptr()), Some(&16), Some(&0));
            context.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_POINTLIST);
            context.VSSetConstantBuffers(2, Some(&[Some(matrix)]));
            context.VSSetConstantBuffers(12, Some(&[Some(foreign.clone()), Some(foreign.clone())]));
            if let Some(slot) = constants {
                context.VSSetConstantBuffers(slot, Some(&[Some(foreign.clone())]));
            }
            context.VSSetShader(shader, None);
            context.GSSetShader(gs.as_ref(), None);
            context.SOSetTargets(1, Some([Some(output.clone())].as_ptr()), Some(&0));
            if let Some(slot) = constants {
                crate::context::draw_with_constants(
                    context,
                    slot,
                    [scale, a - 1.0, -b, orthographic],
                    || context.Draw(1, 0),
                );
                let mut restored = [None];
                context.VSGetConstantBuffers(slot, Some(&mut restored));
                assert_eq!(
                    restored[0].as_ref().map(Interface::as_raw),
                    Some(foreign.as_raw())
                );
            } else {
                context.Draw(1, 0);
            }
            context.SOSetTargets(0, None, None);
            context.CopyResource(&read, &output);
            let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
            context
                .Map(&read, 0, D3D11_MAP_READ, 0, Some(&raw mut mapped))
                .unwrap();
            let values = mapped.pData.cast::<[f32; 5]>().read_unaligned();
            context.Unmap(&read, 0);
            values
        }
    }

    #[test]
    fn captured_shaders_execute_with_private_constants_and_survive_replacement() {
        install().unwrap();
        let mut device = None;
        let mut context = None;
        unsafe {
            D3D11CreateDevice(
                None,
                D3D_DRIVER_TYPE_WARP,
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
        let device = device.unwrap();
        let context = context.unwrap();
        crate::context::create_vertex_constants(&device).unwrap();
        for source in [ROWS.to_owned(), occupied_slots(&[12, 13])] {
            let code = compile(&source);
            for near in [2.0f32, 15.0] {
                for distance in [51_720.0f32, 90_000.0] {
                    let a = 350_000.0 / (350_000.0 - near);
                    let b = -a * near;
                    let projection = [a, b, 0.0, near / 15.0];
                    let mut original = None;
                    unsafe { device.CreateVertexShader(&code, None, Some(&raw mut original)) }
                        .unwrap();
                    let original = original.unwrap();
                    assert_eq!(bytecode(&original).unwrap(), code);
                    let vanilla = execute(
                        &device, &context, &original, &code, projection, None, distance,
                    );
                    let (twin, slot) = mirrored(&original, || {
                        panic!("captured shaders must not use engine records")
                    })
                    .unwrap();
                    assert_eq!(slot, Some(if source == ROWS { 13 } else { 11 }));
                    let cached = mirrored(&original, || panic!("cached twin")).unwrap();
                    assert_eq!(cached.0.as_raw(), twin.as_raw());
                    drop(cached);
                    drop(original);
                    let reversed =
                        execute(&device, &context, &twin, &code, projection, slot, distance);
                    let push = 0.5 * ((distance * a + b - 70_000.0) * 0.0001).clamp(0.0, 1.0);
                    let expected = f64::from(-b)
                        - f64::from(a - 1.0) * f64::from(distance)
                        - f64::from(push * near / 15.0);
                    assert!(
                        (f64::from(reversed[2]) - expected).abs() < 0.000_01,
                        "near {near}, distance {distance}: {reversed:?}, expected {expected}"
                    );
                    for i in [0, 1, 3, 4] {
                        assert_eq!(
                            reversed[i], vanilla[i],
                            "position and fog must be preserved"
                        );
                    }
                }
            }
            let mut original = None;
            unsafe { device.CreateVertexShader(&code, None, Some(&raw mut original)) }.unwrap();
            let original = original.unwrap();
            let (twin, slot) = mirrored(&original, || None).unwrap();
            let projection = [0.001, -0.01, 1.0, 1.0];
            let vanilla = execute(&device, &context, &original, &code, projection, None, 100.0);
            let reversed = execute(&device, &context, &twin, &code, projection, slot, 100.0);
            assert!((reversed[2] - (vanilla[3] - vanilla[2])).abs() < 0.000_001);
        }
        let code = compile(CONSTANT);
        let mut original = None;
        unsafe { device.CreateVertexShader(&code, None, Some(&raw mut original)) }.unwrap();
        let original = original.unwrap();
        let (twin, slot) = mirrored(&original, || None).unwrap();
        assert_eq!(slot, None);
        // A shader predating interception must be retryable when its engine record becomes available.
        unsafe { original.SetPrivateData(&BYTECODE, 0, None) }.unwrap();
        unsafe { original.SetPrivateData(&CONSTANTS, 0, None) }.unwrap();
        assert!(mirrored(&original, || None).is_none());
        assert!(mirrored(&original, || Some(code)).is_some());
        drop(twin);
        unsafe { context.ClearState() };
        pixel_depth(&device, &context);
        unsafe { context.ClearState() };
        crate::copy::tests::exercise(&device, &context);
        unsafe { context.ClearState() };
        crate::context::release_push_scale();
        remove();
    }

    #[allow(clippy::too_many_lines)]
    fn pixel_depth(device: &ID3D11Device, context: &ID3D11DeviceContext) {
        use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_D32_FLOAT, DXGI_SAMPLE_DESC};
        let vertex_code = compile(
            "float4 main(uint i : SV_VertexID) : SV_Position { return float4(i == 2 ? 3 : -1, i == 1 ? 3 : -1, 0.75, 1); }",
        );
        let mut vertex = None;
        unsafe { device.CreateVertexShader(&vertex_code, None, Some(&raw mut vertex)) }.unwrap();
        let color_desc = D3D11_TEXTURE2D_DESC {
            Width: 2,
            Height: 2,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: D3D11_BIND_RENDER_TARGET.0.cast_unsigned(),
            ..Default::default()
        };
        let mut color = None;
        unsafe { device.CreateTexture2D(&raw const color_desc, None, Some(&raw mut color)) }
            .unwrap();
        let color = color.unwrap();
        let mut target = None;
        unsafe { device.CreateRenderTargetView(&color, None, Some(&raw mut target)) }.unwrap();
        let depth_desc = D3D11_TEXTURE2D_DESC {
            Format: DXGI_FORMAT_D32_FLOAT,
            BindFlags: D3D11_BIND_DEPTH_STENCIL.0.cast_unsigned(),
            ..color_desc
        };
        let mut depth = None;
        unsafe { device.CreateTexture2D(&raw const depth_desc, None, Some(&raw mut depth)) }
            .unwrap();
        let depth = depth.unwrap();
        let mut depth_view = None;
        unsafe { device.CreateDepthStencilView(&depth, None, Some(&raw mut depth_view)) }.unwrap();
        let depth_view = depth_view.unwrap();
        let state_desc = D3D11_DEPTH_STENCIL_DESC {
            DepthEnable: true.into(),
            DepthWriteMask: D3D11_DEPTH_WRITE_MASK_ALL,
            DepthFunc: D3D11_COMPARISON_ALWAYS,
            ..Default::default()
        };
        let mut state = None;
        unsafe { device.CreateDepthStencilState(&raw const state_desc, Some(&raw mut state)) }
            .unwrap();
        let raster_desc = D3D11_RASTERIZER_DESC {
            FillMode: D3D11_FILL_SOLID,
            CullMode: D3D11_CULL_NONE,
            DepthClipEnable: true.into(),
            ..Default::default()
        };
        let mut raster = None;
        unsafe { device.CreateRasterizerState(&raw const raster_desc, Some(&raw mut raster)) }
            .unwrap();
        for (semantic, expression, expected) in [
            ("SV_Depth", "0.1 + 0.4 * p.z", 0.8f32),
            ("SV_DepthGreaterEqual", "max(p.z, 0.45)", 0.55),
            ("SV_DepthLessEqual", "min(p.z, 0.15)", 0.85),
        ] {
            let source = format!(
                "float4 main(noperspective centroid float4 p : SV_Position, out float depth : {semantic}) : SV_Target {{ depth = {expression}; return float4(p.z, p.x, p.y, p.w); }}"
            );
            let code = compile_stage(&source, s!("ps_5_0"));
            let mut original = None;
            unsafe { device.CreatePixelShader(&code, None, Some(&raw mut original)) }.unwrap();
            let original = original.unwrap();
            let shader =
                pixel(&original).unwrap_or_else(|| panic!("pixel rewrite failed for {semantic}"));
            assert_ne!(shader.as_raw(), original.as_raw());
            assert_eq!(pixel(&original).unwrap().as_raw(), shader.as_raw());
            unsafe {
                context.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                context.VSSetShader(vertex.as_ref(), None);
                context.PSSetShader(&shader, None);
                context.OMSetRenderTargets(Some(&[target.clone()]), &depth_view);
                context.OMSetDepthStencilState(state.as_ref(), 0);
                context.RSSetState(raster.as_ref());
                context.RSSetViewports(Some(&[D3D11_VIEWPORT {
                    Width: 2.0,
                    Height: 2.0,
                    MaxDepth: 1.0,
                    ..Default::default()
                }]));
                context.ClearDepthStencilView(&depth_view, D3D11_CLEAR_DEPTH.0, 0.0, 0);
                context.Draw(3, 0);
                context.OMSetRenderTargets(None, None);
            }
            let mut values = Vec::new();
            for (texture, desc, components) in
                [(&color, color_desc, 4usize), (&depth, depth_desc, 1)]
            {
                let desc = D3D11_TEXTURE2D_DESC {
                    Usage: D3D11_USAGE_STAGING,
                    BindFlags: 0,
                    CPUAccessFlags: D3D11_CPU_ACCESS_READ.0.cast_unsigned(),
                    ..desc
                };
                let mut read = None;
                unsafe { device.CreateTexture2D(&raw const desc, None, Some(&raw mut read)) }
                    .unwrap();
                let read = read.unwrap();
                unsafe {
                    context.CopyResource(&read, texture);
                    let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
                    context
                        .Map(&read, 0, D3D11_MAP_READ, 0, Some(&raw mut mapped))
                        .unwrap();
                    for component in 0..components {
                        let data = mapped
                            .pData
                            .cast::<u8>()
                            .add(component * 4)
                            .cast::<[u8; 4]>()
                            .read_unaligned();
                        values.push(f32::from_ne_bytes(data));
                    }
                    context.Unmap(&read, 0);
                }
            }
            assert_eq!(
                &values[..4],
                &[0.25, 0.5, 0.5, 1.0],
                "{semantic}: pixel input must stay conventional"
            );
            assert!(
                (values[4] - expected).abs() < 0.000_001,
                "{semantic}: {values:?}"
            );
        }
        let code = compile_stage(
            "float4 main(float4 p : SV_Position) : SV_Target { return p.xxxx; }",
            s!("ps_5_0"),
        );
        let mut original = None;
        unsafe { device.CreatePixelShader(&code, None, Some(&raw mut original)) }.unwrap();
        let original = original.unwrap();
        assert_eq!(pixel(&original).unwrap().as_raw(), original.as_raw());
    }
}
