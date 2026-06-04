$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$activate = Join-Path $scriptDir "..\venv\Scripts\Activate.ps1"

if (Test-Path $activate) {
    . $activate
} else {
    Write-Host "Could not find the virtual environment at $activate"
    Write-Host "Expected path from the new/ folder: ..\venv\Scripts\Activate.ps1"
}

Set-Location $scriptDir
python main.py
