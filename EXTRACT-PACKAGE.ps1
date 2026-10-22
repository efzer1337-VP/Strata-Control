$ErrorActionPreference = 'Stop'
$taskArchive = Join-Path $PSScriptRoot 'downloads\Strata-Control-Full-20261022-v3.zip'
$taskDestination = Join-Path $PSScriptRoot '.strata-full'
$taskMarker = Join-Path $taskDestination '.package-sha256'
if (-not [IO.File]::Exists($taskArchive)) { throw 'Full package is missing. Download the complete repository or the full release ZIP.' }
$taskHash = (Get-FileHash -LiteralPath $taskArchive -Algorithm SHA256).Hash
if ([IO.File]::Exists($taskMarker) -and [IO.File]::ReadAllText($taskMarker).Trim() -eq $taskHash) { return }
Add-Type -AssemblyName System.IO.Compression.FileSystem
[IO.Directory]::CreateDirectory($taskDestination) | Out-Null
# PowerShell 5.1 runs on Windows by default: extract entries with literal paths
# and preserve local generated files that are not present in the source package.
$taskZip = [IO.Compression.ZipFile]::OpenRead($taskArchive)
try {
    $taskBoundary = [IO.Path]::GetFullPath($taskDestination) + [IO.Path]::DirectorySeparatorChar
    foreach ($taskEntry in $taskZip.Entries) {
        $taskOutput = [IO.Path]::GetFullPath((Join-Path $taskDestination $taskEntry.FullName))
        if (-not $taskOutput.StartsWith($taskBoundary, [StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe archive path.' }
        if ($taskEntry.FullName.EndsWith('/')) { [IO.Directory]::CreateDirectory($taskOutput) | Out-Null; continue }
        [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($taskOutput)) | Out-Null
        [IO.Compression.ZipFileExtensions]::ExtractToFile($taskEntry, $taskOutput, $true)
    }
} finally { $taskZip.Dispose() }
[IO.File]::WriteAllText($taskMarker, $taskHash, [Text.Encoding]::ASCII)
