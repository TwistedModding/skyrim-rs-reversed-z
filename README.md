# Reversed Z for Skyrim Runtime Extender

A Skyrim Runtime Extender plugin that draws the world view in reversed Z into 32-bit float depth targets: depth 1 at the near plane and 0 at the far plane, so far surfaces keep their precision at any near plane. Made for Improved Camera's lowered near plane, where the vanilla 24-bit depth lets landscape LOD and far mountains fight. Shadow maps and every other pass stay as they are, and every shader that reads the world's depth is given the value it was written for. Nothing else in the game changes.

## Requirements

- Skyrim Special Edition 1.7.104 from Steam, SKSE 2.3.1 and Address Library for SKSE Plugins (`versionlib-1-7-104-0.bin` in `Data\SKSE\Plugins`).
- The Skyrim Runtime Extender loader (`Data\SKSE\Plugins\skyrim_runtime_extender.dll`), from its own package.

## Install

Install the package with a mod manager, or copy its `SKSE` folder into the game's `Data` folder. You end up with `Data\SKSE\Plugins\skyrim-runtime-extender\skyrim_rs_reversed_z.dll`, which `skyrim_runtime_extender.dll` loads. The depth targets are switched to float at startup, so install it with the game closed.

## Use

- On by default. `reversed-z.log` in `Documents\My Games\Skyrim Special Edition\SKSE\` says `ready: world depth targets are D32 float, on` after the game data loads; `not engaged: ...` or `depth format patch ... refused` says why it stayed off.
- The console command `ReversedZ` prints the state; `ReversedZ off` and `ReversedZ on` switch it at the next frame. The choice is kept in `Data\SKSE\Plugins\skyrim-runtime-extender\reversed-z.settings.json` (`"enabled"`).
- To check it: with Improved Camera's near plane lowered (or `fNearDistance` set low), look at far mountains from a high place; with it on, landscape LOD no longer pokes through them as the camera turns.
- Send back `reversed-z.log` and `skyrim-runtime-extender.log` with anything that looks wrong.

## Community Shaders

Experimental support for stock Community Shaders 1.9.1 requires no CS patches. D3D11 WARP tests and in-game smoke tests cover Terrain Blending 1.2.0, Upscaling 1.4.0, distant mountains, Riverwood, first-person hands and the Sleeping Giant Inn. The tested paths reported no shader rewrite failures, missing bytecode or unsupported depth copies. Other feature combinations and DLSS modes have not been validated.

With RS Improved Camera and CS, use the Skyrim Runtime Extender loader with API 5 or newer (reported in `skyrim-runtime-extender.log`). Hook chaining is the default and preserves IC body-part hiding without an INI change. An explicit `compat = refuse` setting disables chaining globally. Earlier loaders require `compat = stack` in `Data\SKSE\Plugins\skyrim-runtime-extender.ini`.

`ReversedZ off` keeps the D32 depth targets created at startup. With Community Shaders, a translucent riverbank band can appear in this mode; the original DLL also reproduces it. For a stock depth-format comparison, disable this mod and restart the game.

## Build and package

```
git clone --recurse-submodules https://github.com/TwistedModding/skyrim-rs-reversed-z.git
cargo build --release
powershell -File scripts\package.ps1
```

skyrim-rs is a submodule at `extern/skyrim-rs` (`git submodule update --init` in an existing clone). The DLL lands in `target\release\`; the package script writes `dist\skyrim-rs-reversed-z-<version>.zip` with the DLL at its install path and this README.

The package version is `<major.minor>.<commit count>`, with `-dirty` when the tree has uncommitted changes. To package every commit and merge on `master` in the background, with the output in `dist\package.log`, enable the hooks once per clone:

```
git config core.hooksPath .githooks
```
