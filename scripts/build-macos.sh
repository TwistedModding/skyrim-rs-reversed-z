#!/usr/bin/env bash
# Cross-compile skyrim-rs for x86_64-pc-windows-msvc on macOS (Apple Silicon).
# Needs: rustup (brew), LLVM (clang-cl, llvm-lib), and an xwin-style sysroot ($WINSYSROOT) with crt/ and sdk/.
#   scripts/build-macos.sh [cargo build args...]     default: --release (whole workspace)
set -euo pipefail

BREW="$(command -v brew >/dev/null && brew --prefix || echo /opt/homebrew)"
WINSYSROOT="${WINSYSROOT:?set WINSYSROOT to an xwin-style splat (cargo install xwin; xwin --accept-license splat --output <dir>)}"
LLVM="${LLVM:-$BREW/opt/llvm/bin}"
RUSTUP_BIN="${RUSTUP_BIN:-$BREW/opt/rustup/bin}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

die() { echo "build-macos.sh: $*" >&2; exit 1; }
export PATH="$RUSTUP_BIN:$LLVM:$PATH"

command -v rustup >/dev/null || die "rustup not found (brew install rustup; Homebrew's rust has no Windows rust-std). Set RUSTUP_BIN if it is elsewhere."
[ -x "$LLVM/clang-cl" ] || die "clang-cl not found in $LLVM (brew install llvm, or set LLVM=<bin dir>)"
for f in crt/include/vcruntime.h sdk/include/um/windows.h sdk/lib/um/x86_64/kernel32.Lib crt/lib/x86_64/msvcrt.lib; do
  [ -e "$WINSYSROOT/$f" ] || [ -e "$WINSYSROOT/$(echo "$f" | sed 's/\.Lib$/.lib/')" ] \
    || die "WINSYSROOT '$WINSYSROOT' lacks $f (need an xwin-style splat: cargo install xwin; xwin --accept-license splat --output <dir>)"
done
# rust-toolchain.toml lists the target, but rustup installs it only when it first runs a toolchain command; make it explicit.
rustup target list --installed --toolchain stable 2>/dev/null | grep -qx x86_64-pc-windows-msvc \
  || rustup target add x86_64-pc-windows-msvc >&2

export CARGO_BUILD_TARGET=x86_64-pc-windows-msvc
# cc-rs (minhook builds C): clang-cl with the sysroot headers.
export CC_x86_64_pc_windows_msvc="$LLVM/clang-cl"
export CXX_x86_64_pc_windows_msvc="$LLVM/clang-cl"
export AR_x86_64_pc_windows_msvc="$LLVM/llvm-lib"
export CFLAGS_x86_64_pc_windows_msvc="--target=x86_64-pc-windows-msvc -Wno-unused-command-line-argument -imsvc $WINSYSROOT/crt/include -imsvc $WINSYSROOT/sdk/include/ucrt -imsvc $WINSYSROOT/sdk/include/shared -imsvc $WINSYSROOT/sdk/include/um"
# rust-lld (lld-link flavour) reads LIB for import libraries and the CRT.
export LIB="$WINSYSROOT/crt/lib/x86_64;$WINSYSROOT/sdk/lib/ucrt/x86_64;$WINSYSROOT/sdk/lib/um/x86_64"

cd "$ROOT"
[ $# -gt 0 ] || set -- --release
exec cargo build "$@"
