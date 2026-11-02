# Runs package.ps1 for the git hooks, one run at a time and again if HEAD moved during the build.
$Root = Split-Path $PSScriptRoot -Parent
$running = New-Object Threading.Mutex($false, "Local\skyrim-rs-package")
try { if (-not $running.WaitOne(0)) { exit } } catch [Threading.AbandonedMutexException] {}
New-Item -ItemType Directory -Force "$Root\dist" | Out-Null
do {
    $head = git -C $Root rev-parse HEAD
    cmd /c "powershell -NoProfile -ExecutionPolicy Bypass -File `"$PSScriptRoot\package.ps1`" > `"$Root\dist\package.log`" 2>&1"
} while ((git -C $Root rev-parse HEAD) -ne $head)
