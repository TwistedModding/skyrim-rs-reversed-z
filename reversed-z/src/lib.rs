// Copyright (c) 2026 TwistedModding. SPDX-License-Identifier: MPL-2.0-no-copyleft-exception
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not
// distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.
//! Reversed Z for the world view: the world and first-person passes rasterise with depth 1 at the
//! near plane and 0 at the far plane into 32-bit float depth targets, so far surfaces keep their
//! precision at any near plane (no landscape LOD z-fighting from a lowered near). Shadow maps and
//! every other pass stay standard, and every shader that reads the world's depth is given
//! 1 - depth, the exact value it was written for (RE `graphics.md`, request IC12).
//!
//! The pieces: the depth targets with stencil created as D32 float + S8 (byte patches before the
//! renderer starts); while a world depth target is bound, vertex shaders swapped for twins that
//! output w - z ([`twin`]) and the context's viewport depth ranges, depth-stencil and rasterizer
//! states and depth clears mirrored ([`context`]); conventional depth views substituted at shader
//! bindings, including textures copied from world depth ([`copy`]).

use core::ffi::CStr;
use std::io::Write as _;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use framework::Host;
use framework::hooks::ops::Raw;
use framework::hooks::{Location, Options};
use skyrim::input::{Arg, Command, CommandContext, Param};
use sre::ScriptParamType;
use sre::raw::fns;

mod context;
mod copy;
mod shader;
mod spirv;
mod stream;
mod twin;
mod vulkan;

framework::plugin! {
    name: "reversed-z",
    load: load,
    unload: unload,
}

static HOST: OnceLock<&'static Host> = OnceLock::new();
/// The player's switch (setting `enabled`, console `ReversedZ on|off`).
static WANTED: AtomicBool = AtomicBool::new(true);
/// The context hooks are in and the world depth target is a float target.
static READY: AtomicBool = AtomicBool::new(false);
/// The far plane the engine gives a world with LOD.
const ENGINE_FAR: f32 = 353_840.0;
/// The far plane kept for such a world: past the farthest LOD.
const FAR: f32 = 2.0 * ENGINE_FAR;

/// `Renderer::CreateDepthStencilObjects` takes a stencil target's formats from one immediate
/// (`D24_UNORM_S8_UINT`) and register offsets of it, and the 16-bit targets' from the same register:
/// these bytes give `D32_FLOAT_S8X24_UINT` with its typeless, depth and stencil views and keep the
/// 16-bit formats. The site, original byte, new byte.
fn format_patches() -> [(Location, u8, u8); 6] {
    use fns::Renderer__CreateDepthStencilObjects as CREATE;
    [
        (framework::location!(CREATE, 0x32), 0x2d, 0x14),
        (framework::location!(CREATE, 0x42), 0x08, 0x21),
        (framework::location!(CREATE, 0x45), 0x0a, 0x23),
        (framework::location!(CREATE, 0x48), 0x0b, 0x24),
        (framework::location!(CREATE, 0x75), 0x2e, 0x15),
        (framework::location!(CREATE, 0x1fa), 0x2f, 0x16),
    ]
}

pub(crate) fn log(line: &str) {
    if let Some(host) = HOST.get()
        && let Ok(mut f) = skse::log::append(host.skse().runtime_version, "reversed-z")
    {
        let _ = writeln!(f, "{line}");
    }
}

/// Keeps the far plane of a world with LOD at `FAR`, through the scene graph's own far override,
/// while the plugin is loaded. The engine's far plane is flat and nearer than the farthest LOD, so
/// far land and water were cut along a line that swept over them as the camera turned. A world
/// whose far is not the engine's LOD distance is left alone, and so is an override someone else
/// set (the map menu's). Once a frame, on the main thread.
fn keep_far() {
    let Some(graph) = sre::g_scene_graph() else {
        return;
    };
    // +0x144 the override, +0x149 whether it is used
    let overridden = graph.field_148 >> 8 != 0;
    #[allow(clippy::cast_possible_truncation)]
    let value = (graph.field_140 >> 32) as u32;
    if overridden && value != FAR.to_bits() {
        return;
    }
    if overridden {
        graph.clear_far_distance_override();
    }
    // slot 62, the far distance, through the graph's own table
    let far: unsafe extern "C" fn(*const sre::SceneGraph) -> f32 =
        unsafe { core::mem::transmute((*graph.__vftable).slot_62) };
    if READY.load(Ordering::Acquire) && unsafe { far(graph) }.to_bits() == ENGINE_FAR.to_bits() {
        graph.set_far_distance_override(FAR);
    }
}

type CubeMapCamera = unsafe extern "C" fn(usize, usize, usize, usize) -> usize;

/// The water's cube-map camera takes the main camera's far and raises its own near to a ten
/// thousandth of it for good: it is shown the engine's far. The far plane is only moved where this
/// is in place.
fn keep_cube_map_far(host: &'static Host) -> Result<(), String> {
    // looked up without a manifest record: a runtime that cannot place it keeps its own far plane
    let at = skyrim::reloc()
        .location(fns::TESWaterReflections__UpdateCubeMap)
        .map_err(|e| e.to_string())?;
    host.hooks()
        .around(
            Raw::<CubeMapCamera>::at(at),
            Options::new(host.name()),
            |args, next| {
                let far = sre::get_main_camera()
                    .map(|camera| (&raw const camera.frustumFar).cast_mut())
                    .filter(|&far| unsafe { far.read() } > ENGINE_FAR);
                let moved = far.map(|far| unsafe { far.replace(ENGINE_FAR) });
                let result = unsafe { next.call(args) };
                if let (Some(far), Some(moved)) = (far, moved) {
                    unsafe { far.write(moved) };
                }
                result
            },
        )
        .map(|_| ())
        .map_err(|e| e.to_string())
}

/// The world view is drawn reversed.
pub(crate) fn active() -> bool {
    READY.load(Ordering::Acquire) && WANTED.load(Ordering::Relaxed)
}

fn address(id: u64) -> Option<usize> {
    HOST.get()?;
    skyrim::reloc().address(id).ok()
}

fn load(host: &'static Host) -> bool {
    if skyrim::init(host.skse()).is_err() {
        return false;
    }
    let _ = HOST.set(host);
    WANTED.store(host.settings().get_or("enabled", true), Ordering::Relaxed);
    for (at, old, new) in format_patches() {
        if let Err(e) = unsafe { host.hooks().patch(host.name(), at, &[old], &[new]) } {
            log(&format!("depth format patch refused: {e}"));
            return false;
        }
    }
    match keep_cube_map_far(host) {
        Ok(()) => host.on_frame(keep_far),
        Err(e) => log(&format!("far plane left at the engine's: {e}")),
    }
    if let Err(e) = shader::install() {
        log(&format!("replacement shader capture unavailable: {e}"));
    }
    if let Err(e) = vulkan::install() {
        log(&format!("vulkan resolution hooks unavailable: {e}"));
    }
    let _ = host.register_command::<ReversedZ>();
    if context::immediate().is_some() {
        engage();
    } else {
        host.on_message(|msg| {
            if msg.kind == skse::MessagingInterface::DATA_LOADED {
                engage();
            }
        });
    }
    true
}

/// Hooks the context once the renderer exists: at `DATA_LOADED`, or at once for a build loaded
/// into a running game.
fn engage() {
    match context::install() {
        Ok(()) => {
            READY.store(true, Ordering::Release);
            log(&format!(
                "ready: world depth targets are D32 float, {}",
                if active() { "on" } else { "off" }
            ));
            match vulkan::engage() {
                Ok(found) => log(&format!("vulkan: {found}")),
                Err(e) => log(&format!("vulkan: off, {e}")),
            }
        }
        Err(e) => log(&format!("not engaged: {e}")),
    }
}

fn unload() {
    READY.store(false, Ordering::Release);
    keep_far();
    context::remove();
    vulkan::remove();
    context::release_push_scale();
    copy::release();
    shader::remove();
}

/// Runs on the console's thread: the render thread picks the switch up at its next output-target
/// bind, so the immediate context is only ever used from the render thread.
fn set_wanted(on: bool) {
    WANTED.store(on, Ordering::Relaxed);
    if let Some(host) = HOST.get() {
        host.settings().set("enabled", on);
    }
}

static PARAMS: [Param; 1] = [Param::new(c"state", ScriptParamType::String, true)];

struct ReversedZ;

impl Command for ReversedZ {
    const NAME: &'static CStr = c"ReversedZ";
    const HELP: &'static CStr = c"Reversed Z for the world view: ReversedZ [on|off]";
    const PARAMS: &'static [Param] = &PARAMS;

    fn execute(context: &mut CommandContext<'_>) -> bool {
        match context.args().first() {
            Some(Arg::Str(s)) if s.eq_ignore_ascii_case("on") => set_wanted(true),
            Some(Arg::Str(s)) if s.eq_ignore_ascii_case("off") => set_wanted(false),
            _ => {}
        }
        skyrim::input::print(&format!(
            "reversed-z: {} (ready {}, wanted {}), {}, {}",
            if active() { "on" } else { "off" },
            READY.load(Ordering::Acquire),
            WANTED.load(Ordering::Relaxed),
            context::counters(),
            vulkan::counters(),
        ));
        true
    }
}
