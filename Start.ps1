param([switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$dentalRoot = $PSScriptRoot
$dentalPython = Join-Path $dentalRoot '.venv\Scripts\python.exe'
$dentalScript = Join-Path $dentalRoot 'app\app.py'
$dentalWork = Join-Path $dentalRoot 'work'
if (-not (Test-Path -LiteralPath $dentalPython)) { throw 'The isolated Python environment is missing.' }
$env:TEMP = Join-Path $dentalWork 'temp'
$env:TMP = $env:TEMP
$env:PYTHONPYCACHEPREFIX = Join-Path $dentalWork 'pycache'
$env:PIP_CACHE_DIR = Join-Path $dentalWork 'pip-cache'
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = 'false'
$env:TORCH_HOME = Join-Path $dentalRoot 'models\cache'
$env:XDG_CACHE_HOME = Join-Path $dentalWork 'cache'
New-Item -ItemType Directory -Path $env:TEMP -Force | Out-Null
$dentalPidFile = Join-Path $dentalWork 'server.pid'
$dentalAlreadyRunning = $false
if (Test-Path -LiteralPath $dentalPidFile) {
    $dentalSavedPid = [int](Get-Content -LiteralPath $dentalPidFile)
    $dentalProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$dentalSavedPid" -ErrorAction SilentlyContinue
    $dentalAlreadyRunning = $dentalProcess -and $dentalProcess.ExecutablePath -eq $dentalPython -and $dentalProcess.CommandLine.Contains($dentalScript)
}
if (-not $dentalAlreadyRunning) {
    if (Get-NetTCPConnection -LocalPort 8517 -State Listen -ErrorAction SilentlyContinue) { throw 'Port 8517 is already used by another process. No process was stopped.' }
    $dentalArguments = @(('"' + $dentalScript + '"'))
    $dentalProcess = Start-Process -FilePath $dentalPython -ArgumentList $dentalArguments -WorkingDirectory $dentalRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $dentalWork 'server.log') -RedirectStandardError (Join-Path $dentalWork 'server-error.log')
    $dentalProcess.Id | Set-Content -LiteralPath $dentalPidFile
}
if (-not $NoBrowser) { Start-Process 'http://127.0.0.1:8517' }
Write-Output 'http://127.0.0.1:8517'
