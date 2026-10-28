param([switch]$AI, [switch]$DownloadModels, [string]$Python='py')
$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
$setupWork=Join-Path $PSScriptRoot 'work'
foreach ($folder in @('work/temp','work/pip-cache','work/pycache','data','models','results')) {
    New-Item -ItemType Directory -Path (Join-Path $PSScriptRoot $folder) -Force | Out-Null
}
$env:TEMP=Join-Path $setupWork 'temp'
$env:TMP=$env:TEMP
$env:PIP_CACHE_DIR=Join-Path $setupWork 'pip-cache'
$env:PYTHONPYCACHEPREFIX=Join-Path $setupWork 'pycache'
$setupPython=Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
function Run-Checked([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed (exit $LASTEXITCODE): $Program" }
}
if (-not (Test-Path -LiteralPath $setupPython)) {
    $setupPrefix=@()
    if ($Python -eq 'py') { $setupPrefix=@('-3.12') }
    Run-Checked $Python ($setupPrefix + @('-c','import sys; assert sys.version_info[:2] == (3,12), "Use Python 3.12"'))
    Run-Checked $Python ($setupPrefix + @('-m','venv','.venv'))
}
Run-Checked $setupPython @('-m','pip','install','--upgrade','pip')
Run-Checked $setupPython @('-m','pip','install','-r','requirements.txt')
if ($AI -or $DownloadModels) {
    Run-Checked $setupPython @('-m','pip','install','torch==2.8.0','torchvision==0.23.0','--index-url','https://download.pytorch.org/whl/cu128')
    Run-Checked $setupPython @('-m','pip','install','-r','requirements-ai.txt')
    Run-Checked $setupPython @('-c','import torch; print("CUDA available:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name() if torch.cuda.is_available() else "none - AI inference needs an NVIDIA CUDA GPU")')
}
if ($DownloadModels) { Run-Checked $setupPython @('scripts/download_models.py') }
Write-Host 'Setup complete. Run Start.cmd and open http://127.0.0.1:8517/.'
