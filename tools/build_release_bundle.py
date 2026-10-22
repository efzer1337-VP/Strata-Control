"""Refresh the full Strata archive with the Control sources in this repository.

The previous full archive supplies the unchanged upstream Strata revision. Model
weights, local profiles, engines, and virtual environments are never traversed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = ROOT / "downloads"
BASE = DOWNLOADS / "Strata-Control-Full-20261012-v2.zip"
OUTPUT = DOWNLOADS / "Strata-Control-Full-20261022-v3.zip"
CONTROL_FILES = (
    "CONTROL-QUICKSTART.md",
    "CONTROL-RESEARCH.md",
    "LAUNCH-CONTROL.ps1",
    "ORNITH-SETUP.md",
    "strata_control.py",
    "strata_control_engines.py",
    "strata_control_web.py",
    "strata_model_catalog.py",
    "serve/control/index.html",
    "tools/bench_ornith_local.py",
    "tools/test_strata_control_memory.py",
    "tools/test_strata_control_poly.py",
)


def main() -> None:
    replacements = {f"Strata/{name}": ROOT / name for name in CONTROL_FILES}
    replacements["Strata/.gitignore"] = ROOT / "tools" / "strata.gitignore"
    replacements.update({name: ROOT / name for name in ("README.md", "UPSTREAM.json")})
    for name, path in replacements.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing source for {name}: {path}")

    temp = OUTPUT.with_suffix(".tmp")
    seen: set[str] = set()
    with zipfile.ZipFile(BASE, "r") as old, zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED,
                                                              compresslevel=9, allowZip64=True) as new:
        for entry in old.infolist():
            name = entry.filename
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts or name in seen:
                raise ValueError(f"Unsafe or duplicate archive entry: {name}")
            seen.add(name)
            if name in replacements:
                new.write(replacements[name], name)
            else:
                with old.open(entry, "r") as source, new.open(entry, "w") as destination:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
        for name, path in replacements.items():
            if name not in seen:
                new.write(path, name)
                seen.add(name)
    temp.replace(OUTPUT)

    required = {"Strata/setup.py", "Strata/serve/server.py", "Strata/strata_control.py",
                "Strata/strata_control_engines.py", "Strata/serve/control/index.html", "README.md"}
    with zipfile.ZipFile(OUTPUT, "r") as package:
        bad = package.testzip()
        if bad:
            raise ValueError(f"Corrupt archive entry: {bad}")
        missing = required - set(package.namelist())
        if missing:
            raise ValueError(f"Missing archive entries: {sorted(missing)}")
        print(f"Archive: {len(package.namelist())} files, {OUTPUT.stat().st_size:,} bytes")

    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    OUTPUT.with_suffix(OUTPUT.suffix + ".sha256").write_text(f"{digest}  {OUTPUT.name}\n", encoding="ascii")
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    main()
