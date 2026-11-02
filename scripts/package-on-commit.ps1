# Runs package.ps1 for the git hooks, one run at a time and again if HEAD moved during the build, and publishes each clean package to the website.
$Root = Split-Path $PSScriptRoot -Parent
$running = New-Object Threading.Mutex($false, "Local\package-$($Root -replace '[\\:]', '_')")
try { if (-not $running.WaitOne(0)) { exit } } catch [Threading.AbandonedMutexException] {}
New-Item -ItemType Directory -Force "$Root\dist" | Out-Null
do {
    $head = git -C $Root rev-parse HEAD
    cmd /c "powershell -NoProfile -ExecutionPolicy Bypass -File `"$PSScriptRoot\package.ps1`" > `"$Root\dist\package.log`" 2>&1"
    if ((Get-Content "$Root\dist\package.log" -Tail 1) -match '^.*\\reversed-z-(\d+\.\d+\.\d+)\.zip$') {
        cmd /c "powershell -NoProfile -ExecutionPolicy Bypass -File `"$Root\..\Website\scripts\Publish-Download.ps1`" -Project reversed-z -Version $($Matches[1]) -Archive `"$($Matches[0])`" >> `"$Root\dist\package.log`" 2>&1"
    }
} while ((git -C $Root rev-parse HEAD) -ne $head)
