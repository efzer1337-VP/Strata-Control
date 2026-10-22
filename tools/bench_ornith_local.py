"""Measure one local Ornith llama.cpp profile with optional MoE offload overrides.

No model is downloaded. The server is bound to loopback and stopped after each run.
The profile must already point at a complete local GGUF and an installed engine.
"""
from __future__ import annotations

import argparse
import copy
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from strata_control_engines import command_for

try:
    import psutil
except ImportError:
    psutil = None


def request(url: str, payload: dict | None = None, timeout: int = 10) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if data else {}
    with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers), timeout=timeout) as response:
        return json.load(response)


def gpu_used_mib() -> int | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=5, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return int(output.splitlines()[0].strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def run(profile: Path, moe_layers: int, context: int, output_dir: Path) -> dict:
    cfg = json.loads(profile.read_text(encoding="utf-8-sig"))
    cfg = copy.deepcopy(cfg)
    cfg["port"] = 18080
    cfg["engine_settings"].update(cpu_moe_layers=moe_layers, context=context)
    command = command_for(cfg)
    base = "http://127.0.0.1:18080"
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", 18080)) == 0:
            raise RuntimeError("Port 18080 is already in use")
    output_dir.mkdir(parents=True, exist_ok=True)
    engine_label = str(cfg["launcher_engine"]).replace(".", "-")
    load_label = "no-mmap" if "--load-mode" in cfg["engine_settings"].get("extra_args", []) or "--no-mmap" in cfg["engine_settings"].get("extra_args", []) else "mmap"
    log_path = output_dir / f"ornith-{engine_label}-{load_label}-moe{moe_layers}-ctx{context}.log"
    gpu_before = gpu_used_mib()
    started = time.monotonic()
    with log_path.open("wb") as log:
        process = subprocess.Popen(command, cwd=str(Path(command[0]).parent), stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            while time.monotonic() - started < 300:
                if process.poll() is not None:
                    raise RuntimeError(f"llama.cpp exited with {process.returncode}; see {log_path}")
                if psutil and psutil.virtual_memory().available < 3 * 1024 ** 3:
                    raise RuntimeError("Available RAM fell below 3 GiB while loading")
                try:
                    health = request(base + "/health", timeout=3)
                    if health.get("status") == "ok":
                        break
                except (OSError, urllib.error.URLError, ValueError):
                    pass
                time.sleep(2)
            else:
                raise RuntimeError(f"Model was not ready after 5 minutes; see {log_path}")
            loaded_s = round(time.monotonic() - started, 2)
            gpu_loaded = gpu_used_mib()
            ram_loaded = round(psutil.virtual_memory().available / 1024 ** 3, 2) if psutil else None
            if ram_loaded is not None and ram_loaded < 3:
                raise RuntimeError(f"Only {ram_loaded} GiB RAM remained after loading; see {log_path}")
            prompt = "Summarize this software requirement and list three edge cases: " + (
                "A local coding agent must edit one repository file, preserve public APIs, and return a concise test report. " * 80)
            response = request(base + "/completion", {
                "prompt": prompt, "n_predict": 256, "temperature": 0.6,
                "top_p": 0.95, "top_k": 20, "min_p": 0.0,
            }, timeout=240)
            timings = response.get("timings", {})
            return {"moe_layers": moe_layers, "context": context, "engine": command[0],
                    "model": cfg["engine_settings"]["model"], "load_seconds": loaded_s,
                    "gpu_before_mib": gpu_before, "gpu_loaded_mib": gpu_loaded,
                    "ram_loaded_gib": ram_loaded,
                    "ram_available_gib": round(psutil.virtual_memory().available / 1024 ** 3, 2) if psutil else None,
                    "tokens_predicted": response.get("tokens_predicted"),
                    "tokens_evaluated": response.get("tokens_evaluated"), "timings": timings,
                    "log": str(log_path)}
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", type=Path)
    parser.add_argument("--moe", type=int, nargs="+", default=[17, 15])
    parser.add_argument("--context", type=int, default=65536)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "Strata-data" / "benchmarks")
    args = parser.parse_args()
    results = []
    for moe in args.moe:
        try:
            results.append(run(args.profile, moe, args.context, args.output))
        except Exception as error:
            results.append({"moe_layers": moe, "context": args.context, "error": str(error)})
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "ornith-results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
