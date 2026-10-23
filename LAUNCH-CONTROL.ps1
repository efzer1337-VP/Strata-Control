$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms

function Find-ControlRoot([string]$Path) {
    $candidates = @($Path, (Join-Path $Path 'Strata'), (Join-Path $Path '.strata-full\Strata'))
    foreach ($candidate in $candidates) {
        if ((Test-Path -LiteralPath (Join-Path $candidate 'strata_control_web.py')) -and
            (Test-Path -LiteralPath (Join-Path $candidate 'serve\control\index.html')) -and
            (Test-Path -LiteralPath (Join-Path $candidate '.venv\Scripts\python.exe'))) {
            return [System.IO.Path]::GetFullPath($candidate)
        }
    }
    return $null
}

try {
    $root = Find-ControlRoot $PSScriptRoot
    # A transfer copy in Strata/dist/Strata-Control uses the installed parent.
    $ancestor = [System.IO.Directory]::GetParent($PSScriptRoot)
    for ($depth = 0; -not $root -and $ancestor -and $depth -lt 3; $depth++) {
        $root = Find-ControlRoot $ancestor.FullName
        $ancestor = $ancestor.Parent
    }
    $savedInstall = Join-Path $PSScriptRoot '.strata-control-install.txt'
    if (-not $root -and (Test-Path -LiteralPath $savedInstall)) {
        $root = Find-ControlRoot ([System.IO.File]::ReadAllText($savedInstall).Trim())
    }
    if (-not $root) {
        [System.Windows.Forms.MessageBox]::Show(
            'В этой копии не найден Python. Выберите папку установленной Strata. Править BAT-файлы не нужно.',
            'Strata Control', 'OK', 'Information') | Out-Null
        $picker = New-Object System.Windows.Forms.FolderBrowserDialog
        $picker.Description = 'Выберите папку установленной Strata (с .venv и serve)'
        $picker.SelectedPath = $PSScriptRoot
        if ($picker.ShowDialog() -ne 'OK') { exit }
        $root = Find-ControlRoot $picker.SelectedPath
        if (-not $root) { throw 'В выбранной папке нет установленной Strata Control. Для новой установки запустите INSTALL.bat из полного архива.' }
        [System.IO.File]::WriteAllText($savedInstall, $root)
    }
    $python = Join-Path $root '.venv\Scripts\python.exe'
    $script = Join-Path $root 'strata_control_web.py'
    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $python
    $startInfo.Arguments = '"' + $script + '"'
    $startInfo.WorkingDirectory = $root
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.EnvironmentVariables['PYTHONUTF8'] = '1'
    $check = New-Object System.Diagnostics.ProcessStartInfo
    $check.FileName = $python
    $check.Arguments = '-c "import tkinter; import psutil"'
    $check.UseShellExecute = $false
    $check.CreateNoWindow = $true
    $check.RedirectStandardError = $true
    $diagnostic = [System.Diagnostics.Process]::Start($check)
    if (-not $diagnostic.WaitForExit(10000)) {
        $diagnostic.Kill()
        throw 'Python не ответил. Проверьте, доступен ли диск с установленной Strata.'
    }
    if ($diagnostic.ExitCode -ne 0) {
        $details = $diagnostic.StandardError.ReadToEnd()
        $repair = [System.Windows.Forms.MessageBox]::Show(
            ('Python не может открыть Control. Открыть восстановление Strata? Оно может скачать недостающие зависимости. Файлы модели остаются на диске.' + [Environment]::NewLine + $details),
            'Восстановить Strata', 'YesNo', 'Question')
        if ($repair -eq 'Yes') {
            $repairInfo = New-Object System.Diagnostics.ProcessStartInfo
            $repairInfo.FileName = $env:ComSpec
            $repairInfo.Arguments = '/d /c ""' + (Join-Path $root 'START-HERE.bat') + '" --update"'
            $repairInfo.WorkingDirectory = $root
            $repairInfo.UseShellExecute = $true
            [System.Diagnostics.Process]::Start($repairInfo) | Out-Null
        }
        exit
    }
    $diagnostic.Dispose()
    [System.Diagnostics.Process]::Start($startInfo) | Out-Null
} catch {
    [System.Windows.Forms.MessageBox]::Show($_.Exception.Message, 'Strata Control — ошибка запуска', 'OK', 'Error') | Out-Null
}
