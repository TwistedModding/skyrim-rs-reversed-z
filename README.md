# Reversed Z for skyrim-rs

A skyrim-rs framework plugin that draws the world view in reversed Z into 32-bit float depth targets: depth 1 at the near plane and 0 at the far plane, so far surfaces keep their precision at any near plane. Made for Improved Camera's lowered near plane, where the vanilla 24-bit depth lets landscape LOD and far mountains fight. Shadow maps and every other pass stay as they are, and every shader that reads the world's depth is given the value it was written for. Nothing else in the game changes.

## Requirements

- Skyrim Special Edition 1.7.104 from Steam, SKSE 2.3.1 and Address Library for SKSE Plugins (`versionlib-1-7-104-0.bin` in `Data\SKSE\Plugins`).
- The skyrim-rs loader (`Data\SKSE\Plugins\skyrim_rs.dll`), from its own package.

## Install

Copy the package's `Data` folder over the game's (or install the package with a mod manager). You end up with `Data\SKSE\Plugins\skyrim-rs\skyrim_rs_reversed_z.dll`, which `skyrim_rs.dll` loads. The depth targets are switched to float at startup, so install it with the game closed.

## Use

- On by default. `reversed-z.log` in `Documents\My Games\Skyrim Special Edition\SKSE\` says `ready: world depth targets are D32 float, on` after the game data loads; `not engaged: ...` or `depth format patch ... refused` says why it stayed off.
- The console command `ReversedZ` prints the state; `ReversedZ off` and `ReversedZ on` switch it at the next frame. The choice is kept in `Data\SKSE\Plugins\skyrim-rs\reversed-z.settings.json` (`"enabled"`).
- To check it: with Improved Camera's near plane lowered (or `fNearDistance` set low), look at far mountains from a high place; with it on, landscape LOD no longer pokes through them as the camera turns.
- Send back `reversed-z.log` and `skyrim-rs.log` with anything that looks wrong.

## Build and package

Community Shaders compatibility is in development. The plugin captures replacement vertex shaders through Direct3D and gives precise twins their own projection constants in an unused buffer slot. This has passed synthetic D3D11 WARP tests; it has not yet been validated in game with Community Shaders or its features.

```
git clone --recurse-submodules https://github.com/TwistedModding/skyrim-rs-reversed-z.git
cargo build --release
powershell -File scripts\package.ps1
```

skyrim-rs is a submodule at `extern/skyrim-rs` (`git submodule update --init` in an existing clone). The DLL lands in `target\release\`; the package script writes `dist\skyrim-rs-reversed-z-<version>.zip` with the DLL at its install path and this README.
