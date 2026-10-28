param([switch]$ServerOnly)
$ErrorActionPreference = 'Stop'
$dentalRoot = $PSScriptRoot
$dentalExpectedPython = Join-Path $dentalRoot '.venv\Scripts\python.exe'
if (-not $ServerOnly) { foreach ($dentalWorker in @(@('work\training\research.pid','app\research_run.py'),@('work\acquisition.pid','app\acquire.py'),@('work\public-data.pid','app\public_data.py'))) {
    $dentalWorkerPidFile=Join-Path $dentalRoot $dentalWorker[0]
    if (Test-Path -LiteralPath $dentalWorkerPidFile) {
        $dentalWorkerPid=[int](Get-Content -LiteralPath $dentalWorkerPidFile)
        $dentalWorkerProcess=Get-CimInstance Win32_Process -Filter "ProcessId=$dentalWorkerPid" -ErrorAction SilentlyContinue
        $dentalWorkerScript=Join-Path $dentalRoot $dentalWorker[1]
        if ($dentalWorkerProcess -and $dentalWorkerProcess.ExecutablePath -eq $dentalExpectedPython -and $dentalWorkerProcess.CommandLine.Contains($dentalWorkerScript)) {
            # The inline code receives a validated owned PID, never an arbitrary process name.
            & $dentalExpectedPython -c 'import psutil,sys; p=psutil.Process(int(sys.argv[1])); children=p.children(recursive=True); [c.terminate() for c in children if c.is_running()]; p.terminate()' $dentalWorkerPid
            Write-Output ('Stopped owned worker: '+$dentalWorker[1])
        }
        Remove-Item -LiteralPath $dentalWorkerPidFile
    }
}
}
$dentalPidFile = Join-Path $dentalRoot 'work\server.pid'
if (Test-Path -LiteralPath $dentalPidFile) {
    $dentalSavedPid = [int](Get-Content -LiteralPath $dentalPidFile)
    $dentalProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$dentalSavedPid" -ErrorAction SilentlyContinue
    $dentalExpectedPython = Join-Path $dentalRoot '.venv\Scripts\python.exe'
    $dentalExpectedScript = Join-Path $dentalRoot 'app\app.py'
    if ($dentalProcess -and $dentalProcess.ExecutablePath -eq $dentalExpectedPython -and $dentalProcess.CommandLine.Contains($dentalExpectedScript)) {
        & $dentalExpectedPython -c 'import psutil,sys; p=psutil.Process(int(sys.argv[1])); children=p.children(recursive=True); [c.terminate() for c in children if c.is_running()]; p.terminate()' $dentalSavedPid
        Write-Output 'Dental Research app stopped.'
    } else { Write-Output 'No matching app process is running. Nothing stopped.' }
    Remove-Item -LiteralPath $dentalPidFile
}
