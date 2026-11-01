# Builds release and writes dist\skyrim-rs-reversed-z-<version>.zip: the DLL at its install path and the README.
$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
cargo build --release --workspace --manifest-path "$Root\Cargo.toml"
if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }
$version = ((cargo metadata --no-deps --format-version 1 --manifest-path "$Root\Cargo.toml" | ConvertFrom-Json).packages | Where-Object name -eq "reversed-z").version
$entries = [ordered]@{
    "Data/SKSE/Plugins/skyrim-rs/skyrim_rs_reversed_z.dll" = "$Root\target\release\skyrim_rs_reversed_z.dll"
    "README.md" = "$Root\README.md"
}
New-Item -ItemType Directory -Force "$Root\dist" | Out-Null
$out = "$Root\dist\skyrim-rs-reversed-z-$version.zip"
Remove-Item -Force $out -ErrorAction SilentlyContinue
Add-Type -AssemblyName System.IO.Compression, System.IO.Compression.FileSystem
$zip = [IO.Compression.ZipFile]::Open($out, "Create")
try {
    foreach ($e in $entries.GetEnumerator()) { [void][IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $e.Value, $e.Key, "Optimal") }
} finally { $zip.Dispose() }
$out
