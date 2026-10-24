$ErrorActionPreference = 'Stop'
try {
    $taskPython = $null
    $taskCandidates = @((Join-Path $PSScriptRoot '.python\python.exe'))
    $taskCommands = @(Get-Command python.exe, py.exe -ErrorAction SilentlyContinue)
    $taskCandidates += @($taskCommands | ForEach-Object Source)
    foreach ($taskCandidate in $taskCandidates) {
        if (-not (Test-Path -LiteralPath $taskCandidate)) { continue }
        $taskArgs = @()
        if ([IO.Path]::GetFileName($taskCandidate) -eq 'py.exe') { $taskArgs += '-3' }
        try {
            $taskFound = & $taskCandidate @taskArgs -c 'import sys, tkinter, venv; assert sys.version_info >= (3,10) and sys.maxsize > 2**32; print(sys.executable)' 2>$null
        } catch { continue }
        if ($LASTEXITCODE -eq 0) { $taskPython = ($taskFound | Select-Object -Last 1); break }
    }
    if (-not $taskPython) {
        $taskInstaller = Join-Path $PSScriptRoot '.python-installer.exe'
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Write-Host 'Installing Python inside this repository ...'
        Invoke-WebRequest -UseBasicParsing 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe' -OutFile $taskInstaller
        if ((Get-AuthenticodeSignature -LiteralPath $taskInstaller).Status -ne 'Valid') {
            throw 'The Python installer signature is not valid.'
        }
        $taskTarget = Join-Path $PSScriptRoot '.python'
        $taskInstallArgs = '/quiet InstallAllUsers=0 TargetDir="' + $taskTarget + '" PrependPath=0 Include_launcher=0 Include_test=0 AssociateFiles=0 Shortcuts=0 Include_pip=1 Include_tcltk=1'
        $taskProcess = Start-Process -FilePath $taskInstaller -ArgumentList $taskInstallArgs -WindowStyle Hidden -Wait -PassThru
        if ($taskProcess.ExitCode -notin @(0, 3010)) { throw "Python installer failed: $($taskProcess.ExitCode)" }
        $taskPython = Join-Path $taskTarget 'python.exe'
    }
    & $taskPython -m venv (Join-Path $PSScriptRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Could not create .venv in the repository.' }
} catch {
    Write-Host $_.Exception.Message
    exit 1
}
