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
use framework::hooks::Location;
use skyrim::input::{Arg, Command, CommandContext, Param};
use sre::ScriptParamType;
use sre::raw::fns;

mod context;
mod copy;
mod shader;
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

/// `Renderer::CreateDepthStencilObjects` takes a stencil target's formats from one immediate
/// (`D24_UNORM_S8_UINT`) and register offsets of it, and the 16-bit targets' from the same register:
/// these bytes give `D32_FLOAT_S8X24_UINT` with its typeless, depth and stencil views and keep the
/// 16-bit formats. Offsets from the function's entry, original byte, new byte.
const FORMAT_PATCHES: [(usize, u8, u8); 6] = [
    (0x32, 0x2d, 0x14),
    (0x42, 0x08, 0x21),
    (0x45, 0x0a, 0x23),
    (0x48, 0x0b, 0x24),
    (0x75, 0x2e, 0x15),
    (0x1fa, 0x2f, 0x16),
];

pub(crate) fn log(line: &str) {
    if let Some(host) = HOST.get()
        && let Ok(mut f) = skse::log::append(host.skse().runtime_version, "reversed-z")
    {
        let _ = writeln!(f, "{line}");
    }
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
    if let Err(e) = shader::install() {
        log(&format!("replacement shader capture unavailable: {e}"));
    }
    if let Err(e) = vulkan::install() {
        log(&format!("vulkan resolution hooks unavailable: {e}"));
    }
    for (offset, old, new) in FORMAT_PATCHES {
        if let Err(e) = unsafe {
            host.hooks().patch(
                host.name(),
                Location::id(fns::Renderer__CreateDepthStencilObjects, offset),
                &[old],
                &[new],
            )
        } {
            log(&format!(
                "depth format patch at +{offset:#x} refused: {e}; staying off"
            ));
            return true;
        }
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
