"""Install Control and engines locally; model weights require an explicit choice."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
POLY_REVISION = "e97c8aa47ca2c01025accdd32d07a01536d794ce"
ORNITH_REVISION = "38ab997c00045ce9fb858b16d7d914193f9e9917"
ORNITH_REPO = "dealignai/Ornith-1.5-35B-A3B-UNCENSORED-GGUF"
ORNITH_FILES = {
    "Q4_K_M": (21713462848, "f1e281f07c99aa9a731b47082c3d1bfc22e9477230330688a543e3d451319d76"),
    "Q3_K_M": (17166659136, "aa2bdd157e2642932ae4814196828db55e55f46c1934edacdc3967af852c449d"),
}


def choose(prompt, options, default):
    for key, label in options.items():
        print(f"  {key}) {label}")
    while True:
        value = input(f"{prompt} [{default}]: ").strip() or default
        if value in options:
            return value


def load_setup(root):
    spec = importlib.util.spec_from_file_location("control_engine_setup", root / "setup.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Do not adopt or move models from another Strata installation on this PC.
    module.settings_path = lambda: root / ".strata-install-settings.json"
    return module


def poly_sources(data):
    destination = data / "polystrata"
    if (destination / "setup.py").is_file():
        return destination
    archive = data / f"polystrata-{POLY_REVISION}.zip"
    data.mkdir(parents=True, exist_ok=True)
    print("Downloading PolyStrata sources from GitHub ...")
    urllib.request.urlretrieve(
        f"https://github.com/VecSzn/PolyStrata/archive/{POLY_REVISION}.zip", archive)
    prefix = f"PolyStrata-{POLY_REVISION}/"
    with tempfile.TemporaryDirectory(prefix="poly-source-", dir=data) as staging, zipfile.ZipFile(archive) as source:
        unpacked = Path(staging) / "polystrata"
        for entry in source.infolist():
            if not entry.filename.startswith(prefix):
                raise ValueError("Unexpected PolyStrata archive layout")
            relative = entry.filename[len(prefix):]
            target = (unpacked / relative).resolve()
            if not target.is_relative_to(unpacked.resolve()):
                raise ValueError("Unsafe PolyStrata archive path")
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read(entry))
        if not (unpacked / "setup.py").is_file():
            raise ValueError("PolyStrata archive has no setup.py")
        if destination.exists():
            raise RuntimeError(f"Incomplete PolyStrata folder: {destination}. Rename it and retry.")
        unpacked.replace(destination)
    return destination


def install_components(module, poly=False, yes=False):
    module.pip_install(module.requirement_lines(), "Python dependencies")
    gpu = module.gpu_info()
    hip = False
    if gpu is None:
        cards = [g for g in module.amd_gpus() if module.amd_problem(g) is None]
        if not cards:
            raise RuntimeError("No supported GPU found. Install/update the graphics driver and retry.")
        gpu = min(cards, key=module.amd_rank)
        hip = True
    elif module.gpu_problem(gpu):
        raise RuntimeError(module.gpu_problem(gpu))
    toolkit = 13
    if not hip:
        toolkit, _ = module.cuda_choice([gpu["arch"]], None)
        minimum = module.CUDA12_MIN_DRIVER if toolkit == 12 else module.MIN_DRIVER
        if module.driver_major(gpu) < minimum:
            raise RuntimeError(f"Update the NVIDIA driver (minimum {minimum}) and retry.")
    print(f"Installing {'PolyStrata' if poly else 'Strata'} for {gpu['name']} ...")
    if poly:
        llama = module.get_llama_cpp()
        if hip:
            module.build_engine_poly_hip(gpu, llama)
        else:
            if toolkit == 13:
                module.get_poly_prebuilt(gpu)
            module.build_engine_poly(gpu, yes, llama, "none", toolkit)
    else:
        engine = (module.get_prebuilt_hip(module.PREBUILT_URL, gpu) if hip else
                  module.get_prebuilt(module.PREBUILT_URL, gpu, "none", toolkit=toolkit))
        if engine is None:
            llama = module.get_llama_cpp()
            if hip:
                if module.WIN:
                    raise RuntimeError("No Windows AMD engine for this version; see docs/AMD_HIP.md.")
                module.build_engine_hip(gpu, llama, "none")
            else:
                module.build_engine(gpu, "none", yes, llama, toolkit=toolkit)
        elif not hip:
            module.pip_cuda_libs(toolkit)


def add_uncensored(module):
    # Reuse the architecture contract, never the APEX weights, tokenizer or draft.
    family = copy.deepcopy(module.FAMILIES["ornith"])
    base = f"https://huggingface.co/{ORNITH_REPO}/resolve/{ORNITH_REVISION}/"
    family.update(title="Ornith 1.5 35B-A3B Uncensored", by="dealignai",
                  about="Uncensored GGUF; memory figures below are conservative setup estimates",
                  hf=base, file="Ornith-1.5-35B-A3B-CRACK-{q}.gguf",
                  tag="ornith-uncensored-", name="ornith-1.5-uncensored", mmproj_hf=base,
                  mmproj="mmproj-Ornith-1.5-35B-A3B-f16.gguf", models={}, sha256={})
    family.pop("draft", None)
    family.pop("profile", None)
    for quant, (size, digest) in ORNITH_FILES.items():
        gb = size / 1e9
        family["models"][quant] = dict(about=f"{quant}; RAM estimate, not a benchmark",
            download_gb=gb, ram_gb=gb + 6, arena_gb=gb, vram_beside_gb=6)
        family["sha256"][family["file"].format(q=quant)] = (size, digest)
    family["sha256"][family["mmproj"]] = (
        902822240, "eef27583f74f2990b7a3d97119a0b67c05aeb48c92dde9466f050e30bf1a3595")
    module.FAMILIES["ornith"] = family


def prepare_model(module, data, extra, poly=False, uncensored=False, quant="Q4_K_M"):
    if uncensored:
        add_uncensored(module)
    args = ["--setup", "--no-start", "--data-dir", str(data), "--models-dir", str(data / "models")]
    if poly:
        args += ["--family", "ornith", "--port", "8081"]
    if uncensored:
        args += ["--model", quant]
    old = sys.argv
    sys.argv = [str(module.ROOT / "setup.py"), *extra, *args]
    try:
        result = module.main()
    finally:
        sys.argv = old
    if result:
        raise RuntimeError(f"Model setup failed ({result})")
    if poly:
        prefix = "ornith-uncensored-" if uncensored else "ornith-"
        for path in module.ROOT.glob(f"strata-{prefix}*.json"):
            base = json.loads(path.read_text(encoding="utf-8-sig"))
            model = base["args"][base["args"].index("--model") + 1]
            if "--expert-profile" in base["args"]:
                expert_profile = Path(base["args"][base["args"].index("--expert-profile") + 1])
                if expert_profile.is_relative_to(data) and not expert_profile.exists():
                    expert_profile.parent.mkdir(parents=True, exist_ok=True)
                    expert_profile.touch()
            profile = dict(launcher_engine="polystrata", launcher_profile_name=base.get("display_name", path.stem),
                port=8081, open_browser=False, launcher_memory={"ram_reserve_gib": 3},
                engine_settings=dict(model=model, poly_config=str(path), context=32768,
                                     cache_ram_mib=0, vram_reserve_gib=2))
            vision = base.get("vision", {})
            if vision:
                profile["engine_settings"].update(mmproj=vision["mmproj"],
                    vision_gpu=vision.get("gpu", False), vision_tokens=vision.get("max_tokens", 1024))
            directory = ROOT / ".strata-control" / "profiles"
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / (path.stem + ".json")
            if not target.exists():
                target.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--engine", choices=["strata", "polystrata", "both"])
    parser.add_argument("--model-choice", choices=["skip", "standard", "uncensored"])
    parser.add_argument("--no-model", dest="model_choice", action="store_const", const="skip")
    parser.add_argument("--quant", choices=list(ORNITH_FILES), default="Q4_K_M")
    parser.add_argument("--yes", action="store_true", help="Install components without questions; skip weights by default")
    parser.add_argument("--no-open", action="store_true")
    args, extra = parser.parse_known_args(argv)
    if extra and args.model_choice in (None, "skip"):
        parser.error("Model setup flags require --model-choice standard or uncensored")
    if any(x.split("=")[0] in {"--data-dir", "--models-dir", "--host", "--api-key"} for x in extra):
        parser.error("INSTALL keeps data in this repository and servers on 127.0.0.1; use START-HERE for custom paths")
    engine = args.engine or ("strata" if args.yes else
        {"1": "strata", "2": "polystrata", "3": "both"}[choose("Engine", {
            "1": "Strata", "2": "PolyStrata", "3": "Strata + PolyStrata"}, "1")])
    options = {"0": "Skip model download (install components only)", "1": "Standard model"}
    if engine != "strata":
        options["2"] = "Ornith 1.5 Uncensored (Hugging Face, Q4_K_M 21.7 GB / Q3_K_M 17.2 GB)"
    model = args.model_choice or ("skip" if args.yes else
        {"0": "skip", "1": "standard", "2": "uncensored"}[choose("Download a model now?", options, "0")])
    if model == "uncensored" and engine == "strata":
        parser.error("Ornith Uncensored requires --engine polystrata or both")
    if args.yes:
        extra.append("--yes")
    quant = args.quant
    if model == "uncensored" and not args.yes:
        print(f"Model source: https://huggingface.co/{ORNITH_REPO}")
        quant = choose("Quantization", {q: f"{q} ({size / 1e9:.1f} GB)" for q, (size, _) in ORNITH_FILES.items()}, quant)
    data = ROOT / "Strata-data"
    data.mkdir(exist_ok=True)
    print(f"Install folder: {ROOT}\nData folder: {data}")
    strata = load_setup(ROOT)
    if engine in ("strata", "both"):
        install_components(strata, yes=args.yes)
        if model == "standard":
            prepare_model(strata, data, extra)
    if engine in ("polystrata", "both"):
        poly = load_setup(poly_sources(data))
        install_components(poly, poly=True, yes=args.yes)
        if model != "skip":
            prepare_model(poly, data, extra, poly=True, uncensored=model == "uncensored", quant=quant)
    print("Installation complete. No model was started.")
    if model == "skip":
        print("Model download skipped. Run INSTALL.bat again whenever you need a model.")
    if not args.no_open:
        subprocess.Popen([sys.executable, str(ROOT / "strata_control_web.py")], cwd=ROOT,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Installation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
