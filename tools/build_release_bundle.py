"""Build the full source bundle, optionally including a verified NVIDIA engine."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = ROOT / 'downloads'
CONTROL_FILES = (
    'CONTROL-QUICKSTART.md', 'CONTROL-RESEARCH.md', 'LAUNCH-CONTROL.ps1',
    'ORNITH-SETUP.md', 'STRATA-CONTROL.bat', 'strata_control.py',
    'strata_control_engines.py', 'strata_control_web.py', 'strata_model_catalog.py',
    'strata_control_settings.py', 'strata_control_presets.py',
    'serve/control/index.html', 'serve/control/control.js', 'serve/control/control.css',
    'docs/CONTROL-PARAMETERS.md', 'docs/CONTROL-PARAMETERS-STRATA.md',
    'docs/CONTROL-PARAMETERS-POLYSTRATA.md', 'docs/ENGINE-UPDATE-2026-10-23.md',
    'tools/bench_ornith_local.py', 'tools/test_strata_control_memory.py',
    'tools/test_strata_control_poly.py', 'tools/test_strata_control_parameters.py',
    'tools/test_strata_control_settings.py', 'tools/test_strata_control_web.py',
    'tools/test_strata_control_ui.cjs',
    'tools/test_strata_control_release.py',
    'tools/build_release_bundle.py',
    'tools/INSTALL-LOCAL.bat', 'INSTAL.bat', 'INSTALL-PYTHON.ps1', 'control_install.py',
    'strata_memory_cleanup.py', 'tools/test_control_install.py',
    'tools/test_strata_memory_cleanup.py',
)


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def private_entry(name):
    parts = PurePosixPath(name).parts
    if any(part in {'.venv', '.python', '.strata-control', 'Strata-data', '__pycache__', '.git'} for part in parts):
        return True
    leaf = parts[-1] if parts else ''
    return (leaf in {'.python-installer.exe', '.strata-install-settings.json'} or
            leaf.endswith(('.pyc', '.log', '.bak')) or
            (leaf.startswith('strata-') and '.json' in leaf and len(parts) == 2))


def engine_files(folder):
    component = json.loads((folder / 'engine-component.json').read_text(encoding='utf-8'))
    if (component['id'], component['version'], component['platform'], component['variant']) != (
            'strata-engine', '0.1.48', 'windows-x86_64', 'cuda13'):
        raise ValueError('Expected the official Strata 0.1.48 Windows CUDA 13 component')
    files = {'Strata/engine/engine-component.json': folder / 'engine-component.json'}
    for entry in component['files']:
        source = (folder / entry['path']).resolve()
        if source.parent != folder or source.stat().st_size != entry['bytes'] or sha256(source) != entry['sha256']:
            raise ValueError('Engine file does not match its component manifest: ' + entry['path'])
        files['Strata/engine/' + entry['path']] = source
    return files


def setup_compat(source):
    old = b'    va = [int(x) for x in meta.get("vision_archs", meta.get("archs", []))]'
    new = (b'    vision_archs = meta.get("vision_archs", meta.get("archs", []))\n'
           b'    if isinstance(vision_archs, str):\n'
           b'        vision_archs = re.split(r"[;,\\s]+", vision_archs.strip())\n'
           b'    va = [int(x) for x in vision_archs if str(x).strip()]')
    if source.count(old) != 1:
        raise ValueError('The pinned upstream vision parser was not found exactly once')
    return source.replace(old, new)


def build(base, output, engine=None):
    replacements = {f'Strata/{name}': ROOT / name for name in CONTROL_FILES}
    replacements['Strata/INSTALL.bat'] = ROOT / 'tools' / 'INSTALL-LOCAL.bat'
    replacements['Strata/.gitignore'] = ROOT / 'tools' / 'strata.gitignore'
    replacements.update({name: ROOT / name for name in ('README.md', 'UPSTREAM.json')})
    if engine:
        replacements.update(engine_files(engine.resolve()))
    for name, path in replacements.items():
        if not path.is_file():
            raise FileNotFoundError(f'Missing source for {name}: {path}')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp')
    wrappers = {
        'INSTALL.bat': '@echo off\r\ncall "%~dp0Strata\\INSTALL.bat" %*\r\nexit /b %errorlevel%\r\n',
        'INSTAL.bat': '@echo off\r\ncall "%~dp0INSTALL.bat" %*\r\nexit /b %errorlevel%\r\n',
        'START-CONTROL.bat': '@echo off\r\nif not exist "%~dp0Strata\\.venv\\Scripts\\python.exe" goto install\r\ncall "%~dp0Strata\\STRATA-CONTROL.bat" %*\r\nexit /b\r\n:install\r\ncall "%~dp0INSTALL.bat" %*\r\n',
    }
    seen = set()
    with zipfile.ZipFile(base) as old, zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED,
                                                    compresslevel=6, allowZip64=True, strict_timestamps=False) as new:
        for entry in old.infolist():
            name = entry.filename
            pure = PurePosixPath(name)
            if pure.is_absolute() or '..' in pure.parts or name in seen:
                raise ValueError('Unsafe or duplicate archive entry: ' + name)
            if private_entry(name) or name.startswith('Strata/engine/'):
                continue
            seen.add(name)
            if name in wrappers:
                new.writestr(entry, wrappers[name])
            elif name == 'Strata/setup.py':
                new.writestr(entry, setup_compat(old.read(entry)))
            elif name in replacements:
                new.write(replacements[name], name)
            else:
                with old.open(entry) as source, new.open(entry, 'w') as destination:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
        for name, path in replacements.items():
            if name not in seen:
                new.write(path, name)
        for name, content in wrappers.items():
            if name not in seen:
                new.writestr(name, content)
    temporary.replace(output)
    required = set(replacements) | {'Strata/setup.py', 'Strata/serve/server.py', 'INSTALL.bat', 'START-CONTROL.bat'}
    with zipfile.ZipFile(output) as package:
        bad = package.testzip()
        if bad:
            raise ValueError('Corrupt archive entry: ' + bad)
        missing = required - set(package.namelist())
        if missing:
            raise ValueError('Missing archive entries: ' + str(sorted(missing)))
        if any(private_entry(name) for name in package.namelist()):
            raise ValueError('A local generated file was included')
        for name, path in replacements.items():
            if hashlib.sha256(package.read(name)).hexdigest() != sha256(path):
                raise ValueError('Packaged source differs: ' + name)
    digest = sha256(output)
    output.with_suffix(output.suffix + '.sha256').write_text(f'{digest}  {output.name}\n', encoding='ascii')
    print(json.dumps({'archive': str(output), 'bytes': output.stat().st_size, 'sha256': digest,
                      'bundled_engine': '0.1.48' if engine else None}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=DOWNLOADS / 'Strata-Control-Full-20261012-v2.zip')
    parser.add_argument('--output', type=Path, default=DOWNLOADS / 'Strata-Control-Full-20261025-v4.3.1-install.zip')
    parser.add_argument('--engine-dir', type=Path)
    args = parser.parse_args()
    build(args.base, args.output, args.engine_dir)


if __name__ == '__main__':
    main()
