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

```
git submodule update --init
cargo build --release
powershell -File scripts\package.ps1
```

The DLL lands in `target\release\`; `scripts\package.ps1` builds it and writes `dist\skyrim-rs-reversed-z-<version>.zip` with the DLL at its install path and this README. `CLAUDE.md` explains the skyrim-rs submodule. How it works is in `reversed-z/src/lib.rs`; its state and in-game findings are in `docs/STATUS.md`.
