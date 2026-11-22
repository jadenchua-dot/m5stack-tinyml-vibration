# Downloads the newest automatic build from GitHub and flashes it to an M5Stack Fire.
# Usage:  powershell -ExecutionPolicy Bypass -File tools\flash-latest.ps1 -Port COM5
#         add -KeepModel to flash only the app and keep the learned model
param(
  [Parameter(Mandatory = $true)][string]$Port,
  [switch]$KeepModel
)
$ErrorActionPreference = "Stop"
$repo = "jadenchua-dot/m5stack-tinyml-vibration"
$file = if ($KeepModel) { "tinyml-vibration-app.bin" } else { "tinyml-vibration-factory.bin" }
$addr = if ($KeepModel) { "0x10000" } else { "0x0" }
$out = Join-Path $env:TEMP $file
Write-Host "Downloading $file from the latest release..."
Invoke-WebRequest "https://github.com/$repo/releases/download/latest/$file" -OutFile $out
python -m pip install --quiet --upgrade esptool
python -m esptool --chip esp32 --port $Port --baud 921600 write_flash $addr $out
Write-Host "Done. The Fire restarts by itself."
