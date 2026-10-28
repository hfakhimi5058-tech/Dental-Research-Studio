param([ValidateSet('all','prepare','preprocess','train','resume')][string]$Stage='all',[int]$Minutes=120)
$ErrorActionPreference='Stop'
$dentalRoot=$PSScriptRoot
$dentalPython=Join-Path $dentalRoot '.venv\Scripts\python.exe'
$dentalScript=Join-Path $dentalRoot 'app\research_run.py'
$dentalWork=Join-Path $dentalRoot 'work\training'
New-Item -ItemType Directory -Path $dentalWork -Force | Out-Null
$dentalPidFile=Join-Path $dentalWork 'research.pid'
if (Test-Path -LiteralPath $dentalPidFile) {
    $dentalSavedPid=[int](Get-Content -LiteralPath $dentalPidFile)
    $dentalProcess=Get-CimInstance Win32_Process -Filter "ProcessId=$dentalSavedPid" -ErrorAction SilentlyContinue
    if ($dentalProcess -and $dentalProcess.ExecutablePath -eq $dentalPython -and $dentalProcess.CommandLine.Contains($dentalScript)) { throw 'The research worker is already running.' }
}
if ($Minutes -lt 1 -or $Minutes -gt 720) {throw 'Choose a training budget between 1 and 720 minutes.'}
$env:TEMP=Join-Path $dentalRoot 'work\temp'
$env:TMP=$env:TEMP
$env:PYTHONPYCACHEPREFIX=Join-Path $dentalRoot 'work\pycache'
$dentalStop=Join-Path $dentalWork 'STOP'
if (Test-Path -LiteralPath $dentalStop) {Remove-Item -LiteralPath $dentalStop}
$dentalStamp=Get-Date -Format 'yyyyMMdd-HHmmss'
$dentalArguments=@('"'+$dentalScript+'"','--stage',$Stage,'--minutes',$Minutes)
$dentalProcess=Start-Process -FilePath $dentalPython -ArgumentList $dentalArguments -WorkingDirectory $dentalRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $dentalWork "$dentalStamp.log") -RedirectStandardError (Join-Path $dentalWork "$dentalStamp-errors.log")
$dentalProcess.Id | Set-Content -LiteralPath $dentalPidFile
Write-Output 'Local research worker started. Status: work\training\status.json; stop from Models & jobs.'
