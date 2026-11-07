# Copyright (c) 2026 TwistedModding. SPDX-License-Identifier: MPL-2.0-no-copyleft-exception
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not
# distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.
# Builds release and writes dist\reversed-z-<version>.zip: the DLL at its install path and the README.
# The version is <major.minor>.<commits since the commit that set major.minor>, with -dirty for a tree with uncommitted changes.
$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
$Release = "1.0"
$since = git -C $Root log -1 --format=%H -G '^\$Release = ' -- scripts/package.ps1
$commits = git -C $Root rev-list --count "$since..HEAD"
if ($LASTEXITCODE -ne 0 -or -not $since) { throw "git log failed" }
$version = "$Release.$commits"
if (git -C $Root status --porcelain) { $version += "-dirty" }
$env:SKYRIM_RS_VERSION = $version
cargo build --release --workspace --manifest-path "$Root\Cargo.toml"
if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }
$entries = [ordered]@{
    "SKSE/Plugins/skyrim-runtime-extender/reversed_z.dll" = "$Root\target\release\reversed_z.dll"
    "README.md" = "$Root\README.md"
    "LICENSE.txt" = "$Root\LICENSE"
}
New-Item -ItemType Directory -Force "$Root\dist" | Out-Null
$out = "$Root\dist\reversed-z-$version.zip"
Remove-Item -Force $out -ErrorAction SilentlyContinue
Add-Type -AssemblyName System.IO.Compression, System.IO.Compression.FileSystem
$zip = [IO.Compression.ZipFile]::Open($out, "Create")
try {
    foreach ($e in $entries.GetEnumerator()) { [void][IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $e.Value, $e.Key, "Optimal") }
} finally { $zip.Dispose() }
$out
