# Reversed Z for Skyrim Runtime Extender

A Skyrim Runtime Extender plugin that draws the world view in reversed Z into 32-bit float depth targets: depth 1 at the near plane and 0 at the far plane, so far surfaces keep their precision at any near plane. Made for Improved Camera's lowered near plane, where the vanilla 24-bit depth lets landscape LOD and far mountains fight. Shadow maps and every other pass stay as they are, and every shader that reads the world's depth is given the value it was written for. Nothing else in the game changes.

## Requirements

- Skyrim Special Edition 1.7.104, 1.6.1170, 1.6.640, 1.6.353 or 1.5.97 from Steam, or 1.6.1179 from GOG, with the script extender and Address Library file the loader's README lists for that version.
- The Skyrim Runtime Extender loader (`Data\SKSE\Plugins\skyrim_runtime_extender.dll`), 0.6 or later, from its own package, and 1.0.23 or later on 1.6.640.

## Install

Install the package with a mod manager, or copy its `SKSE` folder into the game's `Data` folder. You end up with `Data\SKSE\Plugins\skyrim-runtime-extender\reversed_z.dll`, which `skyrim_runtime_extender.dll` loads. The depth targets are switched to float at startup, so install it with the game closed.

## Use

- On by default. `reversed-z.log` in `Documents\My Games\Skyrim Special Edition\SKSE\` says `ready: world depth targets are D32 float, on` after the game data loads; `not engaged: ...` or `depth format patch ... refused` says why it stayed off.
- The console command `ReversedZ` prints the state; `ReversedZ off` and `ReversedZ on` switch it at the next frame. The choice is kept in `Data\SKSE\Plugins\skyrim-runtime-extender\reversed-z.settings.json` (`"enabled"`).
- The far plane of a world with LOD is twice the game's distance while the plugin is loaded, with `ReversedZ off` as well. The game cuts everything past a flat plane 353,840 units ahead, nearer than its farthest LOD, so distant land and water at the horizon appeared and vanished as the camera turned. Interiors, worlds without LOD, the world map, fog, sun shadows and the water's reflection cube map keep the game's distances; volumetric lighting reaches as far as the new plane. On 1.5.97 the far plane stays the game's (`reversed-z.log` says `far plane left at the engine's`).
- While Reversed Z is on, the push the game gives distant land and water in depth grows evenly from 16,384 units instead of starting at 70,000: it is the game's at 80,000, less before and more beyond, and nothing for land nearer than its start. The game works its push out at each corner of a triangle, and distant water is a few very large triangles, so water and the seabed under it were pushed differently and the seabed showed through in patches that changed as the camera turned.
- To check it: with Improved Camera's near plane lowered (or `fNearDistance` set low), look at far mountains from a high place; with it on, landscape LOD no longer pokes through them as the camera turns.
- Send back `reversed-z.log` and `skyrim-runtime-extender.log` with anything that looks wrong.

## Community Shaders

Experimental support for stock Community Shaders 1.9.1 requires no CS patches. D3D11 WARP tests and in-game smoke tests cover Terrain Blending 1.2.0, Upscaling 1.4.0, distant mountains, Riverwood, first-person hands and the Sleeping Giant Inn. The tested paths reported no shader rewrite failures, missing bytecode or unsupported depth copies. Other feature combinations and DLSS modes have not been validated.

With RS Improved Camera and CS, use the Skyrim Runtime Extender loader with API 5 or newer (reported in `skyrim-runtime-extender.log`). Hook chaining is the default and preserves IC body-part hiding without an INI change. An explicit `compat = refuse` setting disables chaining globally. Earlier loaders require `compat = stack` in `Data\SKSE\Plugins\skyrim-runtime-extender.ini`.

With the Community Shaders test build that runs the game on DXVK and draws the world through Drawcall Limit Fix, the plugin also mirrors DCLF's own Vulkan renderings into the world depth targets (`reversed-z.log` says `vulkan: 3 world images on DXVK's device`). DCLF's occlusion culling and its far-depth precision are not yet covered; the console line reports the Vulkan counters.

`ReversedZ off` keeps the D32 depth targets created at startup. With Community Shaders, a translucent riverbank band can appear in this mode; the original DLL also reproduces it. For a stock depth-format comparison, disable this mod and restart the game.

## Tessellated terrain

Mods that tessellate the landscape, such as Dynamic Terrain Deformation, place the final vertex in a domain shader. Reversed Z reverses that shader's output instead of the vertex shader's, so raised snow and trails sort correctly. Checked with Dynamic Terrain Deformation 1.2 on all four game versions, alone and with ENB.

## ENB

With ENB, Reversed Z works on the Direct3D 11 device underneath ENB instead of the one ENB hands the game, so ENB's own passes are reversed too: its terrain blending, and the depth its effects read. `reversed-z.log` says `reversed depth runs on the system's below it`. If that device can't be hooked in a session, the log says so and Reversed Z falls back to the game's device; ENB's terrain blending is then wrong on 1.6.1170, 1.6.353 and 1.5.97 until the next start. Checked with ENB v0.505 and its default settings on all four game versions. ENB's shaders are rewritten at runtime with the permission of ENB's author.

## Build and package

```
git clone --recurse-submodules https://github.com/TwistedModding/skyrim-rs-reversed-z.git
cargo build --release
powershell -File scripts\package.ps1
```

skyrim-rs is a submodule at `extern/skyrim-rs` (`git submodule update --init` in an existing clone). The DLL lands in `target\release\`; the package script writes `dist\reversed-z-<version>.zip` with the DLL at its install path and this README.

The package version is `<major.minor>.<commits since major.minor was set>`, with `-dirty` when the tree has uncommitted changes. To package every commit and merge on `master` in the background, with the output in `dist\package.log`, enable the hooks once per clone:

```
git config core.hooksPath .githooks
```

## License

Copyright (c) 2026 TwistedModding, under the Mozilla Public License 2.0 (`LICENSE`), marked "Incompatible With Secondary Licenses" (Exhibit B), so no part of it may be relicensed under the GPL, LGPL or AGPL. It includes skyrim-rs by TwistedModding under the same terms. Third-party components keep their own licenses.
