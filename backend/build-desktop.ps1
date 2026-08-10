$ErrorActionPreference = "Stop"

$backendRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $backendRoot

$python = Join-Path $backendRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Backend virtual environment was not found. Create backend/.venv first."
}

& $python -m pip install "pyinstaller>=6.0,<7.0"
if ($LASTEXITCODE -ne 0) { throw "Desktop build dependencies could not be installed." }

& $python -m PyInstaller --noconfirm --clean "local-rag-api.spec"
if ($LASTEXITCODE -ne 0) { throw "The backend executable could not be built." }

Write-Host "Backend sidecar created at backend/dist/local-rag-api.exe"
