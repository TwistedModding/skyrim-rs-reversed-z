# Builds release and writes dist\reversed-z-<version>.zip: the DLL at its install path and the README.
# The version is <major.minor>.<commits on HEAD>, with -dirty for a tree with uncommitted changes.
$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
$Release = "0.2"
$commits = git -C $Root rev-list --count HEAD
if ($LASTEXITCODE -ne 0) { throw "git rev-list failed" }
$version = "$Release.$commits"
if (git -C $Root status --porcelain) { $version += "-dirty" }
$env:SKYRIM_RS_VERSION = $version
cargo build --release --workspace --manifest-path "$Root\Cargo.toml"
if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }
$entries = [ordered]@{
    "SKSE/Plugins/skyrim-runtime-extender/reversed_z.dll" = "$Root\target\release\reversed_z.dll"
    "README.md" = "$Root\README.md"
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
