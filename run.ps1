$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvActivate = Join-Path $root "venv\Scripts\Activate.ps1"
$newDir = Join-Path $root "new"

if (Test-Path $venvActivate) {
    . $venvActivate
} else {
    Write-Host "Virtual environment not found at: $venvActivate"
    exit 1
}

try {
    $ollamaListening = Get-NetTCPConnection -LocalPort 11434 -ErrorAction SilentlyContinue |
        Where-Object { $_.State -eq "Listen" }
} catch {
    $ollamaListening = $null
}

if (-not $ollamaListening) {
    Start-Process -WindowStyle Hidden -FilePath "ollama" -ArgumentList "serve"
    Start-Sleep -Seconds 3
}

Set-Location $newDir
python main.py
