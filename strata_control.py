"""Small Windows control window for the Strata server installed in this folder."""
from __future__ import annotations

import json
import argparse
import hashlib
import math
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
import copy
import threading
import queue
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from strata_model_catalog import catalog_rows, describe_file, first_shard

try:
    import psutil
except ImportError:  # setup.py installs psutil in Strata's virtual environment
    psutil = None


ROOT = Path(__file__).resolve().parent
SERVER = ROOT / "serve" / "server.py"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
LOG_PATH = Path(os.environ.get("LOCALAPPDATA", str(ROOT))) / "Strata" / "strata-control.log"
VALUE_OPTIONS = {
    "--max-context", "--kv", "--kv-resident", "--spec", "--spec-min-p",
    "--expert-cache", "--vram-reserve-mib", "--resident-budget-gib", "--mtp",
}
FLAG_OPTIONS = {"--resident-experts", "--vision"}
MANAGED_OPTIONS = VALUE_OPTIONS | FLAG_OPTIONS
BLOCKED_EXTRA_OPTIONS = {"--host", "--api-key", "--port", "--config", "--engine", "--open"}
GIB = 1024 ** 3
THEME = {
    "background": "#eef1df", "surface": "#cbd3b5", "input": "#e5ebd4",
    "border": "#b5c09b", "text": "#283023", "muted": "#5e6951",
    "accent": "#768824", "button": "#768824", "hover": "#65771d",
    "warning": "#927224", "danger": "#9a493e",
}
CONTROL_STATE = ROOT / ".strata-control-state.json"
PROFILES_DIR = ROOT / ".strata-control" / "profiles"
GPU_TOTAL_CACHE = {}


def _external_server(config_path: Path) -> int:
    from strata_control_engines import command_for
    data = json.loads(config_path.read_text(encoding="utf-8-sig"))
    command = command_for(data, materialize=True)
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in data.get("launcher_env", {}).items()})
    # Profile commands are the complete launch contract; process ownership is
    # attached to this supervisor's exact --config path by the launcher.
    with subprocess.Popen(command, cwd=str(Path(command[0]).parent), env=env,
                          stdin=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)) as child:
        return child.wait()


def _prepare_local_model(source: Path, template: Path | None, destination: Path) -> None:
    """Prepare only local files; this route never invokes setup's download steps."""
    import setup as installer
    details = describe_file(source.name)
    if not details or not details.get("menu"):
        raise ValueError("Эта модель не поддерживается Strata. Выберите Qwen3.8, Coder, Swift 1.5 или поддерживаемый Unsloth GGUF.")
    family, quant = details["family"], details["quant"]
    fam = installer.FAMILIES[family]
    shards = installer.gguf_dir_shards(source.parent, fam, quant)
    if _normalized(shards[0]) != _normalized(source):
        raise ValueError("Выберите первую часть модели: 00001-of-….gguf.")
    problem = installer.gguf_dir_problem(source.parent, source, fam, quant)
    if problem:
        raise ValueError("\n".join(problem))
    installer.check_shards(shards)
    from gguf_reader import GGUFFile
    ple = next((shard for shard in shards if any(t.name == "per_layer_token_embd.weight" for t in GGUFFile(shard).tensors)), None)
    if ple is None:
        raise ValueError("В модели нет таблицы per_layer_token_embd: этот GGUF не совместим со Strata.")
    old = json.loads(template.read_text(encoding="utf-8-sig")) if template else {}
    if not template:
        # The web UI prepares models without a Tk selection. Reuse runtime
        # library locations from an installed Strata profile, not its tuning.
        for candidate in _configs():
            try:
                installed = json.loads(candidate.read_text(encoding="utf-8-sig"))
                if installed.get("launcher_engine", "strata") == "strata" and installed.get("lib_dirs"):
                    old = installed
                    break
            except (OSError, ValueError):
                continue
    total_ram = psutil.virtual_memory().total / GIB
    reserve = min(8, total_ram / 4)
    data = {key: old[key] for key in ("lib_dirs", "gpu", "gpus_asked", "port", "open_browser") if key in old}
    data.setdefault("port", 8080)
    data.setdefault("gpu", 0)
    # New models do not inherit another family's context, steering or expert layout.
    data["args"] = ["--max-context", "32768", "--kv", "int8", "--expert-cache", "auto", "--spec", "4", "--spec-min-p", "0.5", "--prefill", "auto", "--vram-reserve-mib", "2048", "--resident-budget-gib", f"{max(1, total_ram - reserve):.6f}"]
    data["launcher_memory"] = {"ram_reserve_gib": reserve}
    data["launcher_env"] = {"STRATA_RESIDENT_HEADROOM_GIB": f"{reserve:g}"}
    key = hashlib.sha256(_normalized(source).encode()).hexdigest()[:16]
    data_root = Path(installer.load_settings().get("data_dir") or ROOT.parent / "Strata-data")
    pack = data_root / "packs" / ("control-" + key)
    mtp = _option_value(old.get("args", []), "--mtp")
    if not mtp or not (Path(mtp) / "experts.bin").is_file():
        local_mtp = data_root / "mtp" / "rt"
        mtp = str(local_mtp) if (local_mtp / "experts.bin").is_file() else ""
    print("Подготовка локальной модели. Файлы из интернета не скачиваются.", flush=True)
    print(f"Папка подготовки: {pack}", flush=True)
    command = [str(PYTHON), str(ROOT / "tools" / "iq_pack.py"), "--gguf", str(source), "--out", str(pack), *fam.get("pack_args", [])]
    # Resident mode needs the contiguous expert file, as in official setup.
    if not installer.MODELS[quant].get("budget") and (_has_flag(data.get("args", []), "--resident-experts") or _option_value(data.get("args", []), "--resident-budget-gib")):
        if not (pack / "experts.bin").exists():
            needed = installer.MODELS[quant]["arena_gb"] * 1e9 + 3 * GIB
            if shutil.disk_usage(ROOT.parent).free < needed:
                raise ValueError(f"Недостаточно места для подготовки: нужно примерно {needed / GIB:.0f} ГБ на диске {ROOT.anchor}.")
        command.append("--experts-bin")
    subprocess.run(command, check=True, cwd=str(ROOT), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    removed = {"--pack", "--native", "--ple-gguf", "--expert-profile", "--vision"}
    args = _without_options(data.get("args", []), removed)
    args += ["--pack", str(pack), "--native", str(source), "--expert-profile", str(ROOT / "data" / fam.get("profile", "expert-profile.bin"))]
    if len(shards) <= 2:
        args += ["--ple-gguf", str(ple)]
    if installer.MODELS[quant].get("budget"):
        args = _without_options(args, {"--resident-experts", "--resident-budget-gib"})
        reserve = float(data.get("launcher_memory", {}).get("ram_reserve_gib", 0))
        total = psutil.virtual_memory().total / GIB
        budget = min(total - reserve, installer.resident_budget_gib(quant, total))
        args += ["--resident-budget-gib", str(budget)]
    args = _without_options(args, {"--mtp"}) + (["--mtp", mtp] if mtp else [])
    data.update(args=args, exe=str(ROOT / "engine" / "strata.exe"), cwd=str(ROOT),
                tokenizer=str(pack / "tokenizer"), model_name=source.stem,
                log=str(destination.with_suffix(".log")))
    data.pop("vision", None)  # The user selects a matching mmproj explicitly.
    data["launcher_model"] = {"family": family, "quant": quant}
    mmproj = next((p for p in (source.parent / fam["mmproj"], source.parent.parent / fam["mmproj"]) if p.is_file()), None)
    if mmproj:
        data["launcher_mmproj"] = str(mmproj)  # Suggest it without enabling vision.
    if not template:
        build = ROOT / "engine" / "BUILD.json"
        if build.exists():
            metadata = json.loads(build.read_text(encoding="utf-8-sig"))
            data["lib_dirs"] = metadata.get("lib_dirs") or metadata.get("cuda_dirs") or []
    if destination.exists():
        raise ValueError("Профиль уже существует. Выберите его в списке моделей.")
    temp = destination.with_suffix(".json.tmp")
    temp.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp, destination)
    print("Готово. Модель добавлена; сервер можно запустить кнопкой «Запустить».", flush=True)


def _web_url(port: int) -> str:
    # localhost is a separate browser origin from 127.0.0.1, which can still
    # have another chat app's old service worker registered on this port.
    return f"http://localhost:{port}/"


def _reserve_gib(value: str, label: str, maximum: float) -> float:
    try:
        number = float(value.strip().replace(",", "."))
    except ValueError as exc:
        raise ValueError(f"{label}: введите число в ГБ, например 2 или 2,5.") from exc
    if not math.isfinite(number) or not 0 <= number < maximum:
        raise ValueError(f"{label}: допустимо от 0 до {maximum:g} ГБ (не включая верхнюю границу).")
    return number


def _gpu_total_gib(gpu: int) -> float:
    if gpu in GPU_TOTAL_CACHE:
        return GPU_TOTAL_CACHE[gpu]
    try:
        result = subprocess.run(
            ["nvidia-smi", "--id", str(gpu), "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode == 0:
            GPU_TOTAL_CACHE[gpu] = float(result.stdout.strip()) / 1024
            return GPU_TOTAL_CACHE[gpu]
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    GPU_TOTAL_CACHE[gpu] = 256.0
    return 256.0  # A non-NVIDIA backend still validates its own allocation plan.


def _guard_path(config_path: Path) -> Path:
    key = hashlib.sha256(_normalized(config_path).encode("utf-8")).hexdigest()[:16]
    return LOG_PATH.parent / "guards" / f"{key}.json"


def _terminate_tree(proc) -> None:
    if not proc.is_running():
        return
    if os.name == "nt":
        result = subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True,
            timeout=15, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode and psutil.pid_exists(proc.pid):
            raise RuntimeError("Windows не остановил процесс Strata.")
    else:
        children = proc.children(recursive=True)
        for child in reversed(children):
            child.terminate()
        proc.terminate()
        _, alive = psutil.wait_procs([*children, proc], timeout=5)
        for child in alive:
            child.kill()


def _memory_guard(pid: int, created: float, reserve_gib: float, state_path: Path) -> None:
    """A separate process keeps watching even when the control window is closed."""
    if psutil is None:
        raise RuntimeError("RAM guard requires psutil")
    low_samples = 0
    while True:
        try:
            proc = psutil.Process(pid)
            if abs(proc.create_time() - created) > 0.01 or not proc.is_running():
                return
            available = psutil.virtual_memory().available / GIB
            low_samples = low_samples + 1 if available < reserve_gib else 0
            if low_samples >= 3:
                _terminate_tree(proc)
                state_path.write_text(json.dumps({
                    "status": "stopped_for_ram", "available_gib": available,
                    "reserve_gib": reserve_gib,
                }), encoding="utf-8")
                print(f"[RAM guard] Strata stopped: {available:.2f} GiB free, reserve {reserve_gib:g} GiB", flush=True)
                return
        except psutil.NoSuchProcess:
            return
        time.sleep(0.5)


def _normalized(value: str | Path) -> str:
    return os.path.normcase(os.path.abspath(str(value))).replace("/", "\\").casefold()


def _option_value(args: list[str], option: str, default: str = "") -> str:
    for index, token in enumerate(args):
        if token == option and index + 1 < len(args):
            return str(args[index + 1])
        if token.startswith(option + "="):
            return token.split("=", 1)[1]
    return default


def _has_flag(args: list[str], flag: str) -> bool:
    return flag in args or any(token.startswith(flag + "=") for token in args)


def _without_options(args: list[str], options: set[str]) -> list[str]:
    kept: list[str] = []
    index = 0
    while index < len(args):
        token = str(args[index])
        option = token.split("=", 1)[0]
        if option in options:
            index += 1 if "=" in token or option in FLAG_OPTIONS else 2
        else:
            kept.append(token)
            index += 1
    return kept


def _configs() -> list[Path]:
    return sorted([path for path in ROOT.glob("strata-*.json") if path.is_file()] +
                  [path for path in PROFILES_DIR.glob("*.json") if path.is_file()])


_PROCESS_CACHE = {}
_PROCESS_CACHE_LOCK = threading.Lock()
_PROCESS_CACHE_TTL = 10.0


def _server_processes(config_path: Path, force: bool = False):
    if psutil is None:
        return []
    wanted_server = _normalized(SERVER)
    wanted_wrapper = _normalized(Path(__file__).resolve())
    wanted_config = _normalized(config_path)
    now = time.monotonic()
    with _PROCESS_CACHE_LOCK:
        cached = _PROCESS_CACHE.get(wanted_config)
    if not force and cached and now - cached[0] < _PROCESS_CACHE_TTL:
        alive = []
        for proc, created in cached[1]:
            try:
                # A new Process avoids psutil's cached create_time and PID reuse.
                current = psutil.Process(proc.pid)
                if current.is_running() and abs(current.create_time() - created) < 0.01:
                    alive.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                pass
        return alive
    found = []
    identities = []
    # All managed servers use Python wrappers. Command lines of hundreds of
    # unrelated Windows processes are costly and unnecessary to inspect.
    for proc in psutil.process_iter(["name"]):
        try:
            name = (proc.info.get("name") or "").casefold()
            if "python" not in name and "pypy" not in name:
                continue
            parts = [str(item) for item in proc.cmdline()]
            scripts = {_normalized(item) for item in parts[1:] if item.endswith(".py")}
            config = _option_value(parts, "--config")
            server = wanted_server in scripts or (wanted_wrapper in scripts and "--external-server" in parts)
            if server and config and _normalized(config) == wanted_config:
                proc.info["cmdline"] = parts
                found.append(proc)
                identities.append((proc, proc.create_time()))
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue
    with _PROCESS_CACHE_LOCK:
        _PROCESS_CACHE[wanted_config] = (now, identities)
    return found


def _server_roots(config_path: Path):
    matches = _server_processes(config_path, force=True)
    match_ids = {proc.pid for proc in matches}
    roots = []
    for proc in matches:
        try:
            ancestor_is_match = any(parent.pid in match_ids for parent in proc.parents())
            if not ancestor_is_match:
                roots.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue

    # A server started from run-<model>.bat has a console parent. Stop that tree so the
    # batch file's error pause does not leave an orphaned window behind.
    run_script = f"run-{config_path.stem.removeprefix('strata-')}.bat".casefold()
    result = []
    seen = set()
    for proc in roots:
        target = proc
        try:
            for parent in [proc, *proc.parents()]:
                command = " ".join(parent.cmdline()).casefold()
                if parent.name().casefold() == "cmd.exe" and run_script in command:
                    target = parent
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            pass
        if target.pid not in seen:
            seen.add(target.pid)
            result.append(target)
    return result


def _health(port: int):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"http://127.0.0.1:{port}/health", timeout=1.5) as response:
            payload = response.read(65536).decode("utf-8", errors="replace")
            try:
                data = json.loads(payload)
            except ValueError:
                return {"http_ok": True}
            return data if isinstance(data, dict) else {"http_ok": True}
    except (OSError, ValueError, urllib.error.URLError):
        return None


def _port_in_command(proc, fallback: int) -> int:
    try:
        parts = [str(item) for item in (proc.info.get("cmdline") or [])]
        if "--port" in parts:
            index = parts.index("--port")
            return int(parts[index + 1])
    except (ValueError, IndexError, TypeError):
        pass
    return fallback


class StrataControl(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Strata Control · Desktop 4.1")
        self.geometry(f"1200x{min(900, self.winfo_screenheight() - 90)}")
        self.minsize(1040, 660)
        self.config_path: Path | None = None
        self.config_data: dict = {}
        self.display_to_path: dict[str, Path] = {}
        self._busy = False
        self._pending_browser_port: int | None = None
        self._started_process = None
        self._prepare_process = None

        self._make_variables()
        self._build_ui()
        self._load_config_list()
        threading.Thread(target=_gpu_total_gib, args=(int(self.config_data.get("gpu", 0)),), daemon=True).start()
        self.after(100, self._dark_titlebar)
        self.after(1000, self._refresh_status)

    def _dark_titlebar(self) -> None:
        if os.name == "nt":
            try:
                import ctypes
                window = ctypes.windll.user32.GetParent(self.winfo_id())
                dark = ctypes.c_int(0)
                ctypes.windll.dwmapi.DwmSetWindowAttribute(window, 20, ctypes.byref(dark), ctypes.sizeof(dark))
            except (OSError, AttributeError):
                pass

    def report_callback_exception(self, exc_type, exc_value, traceback_object) -> None:
        import traceback
        detail = "".join(traceback.format_exception(exc_type, exc_value, traceback_object))
        try:
            (ROOT / ".strata-control-startup.log").write_text(detail, encoding="utf-8")
        except OSError:
            pass
        messagebox.showerror("Strata Control", f"{exc_value}\n\nПодробности сохранены в .strata-control-startup.log", parent=self)

    def _apply_theme(self) -> None:
        c = THEME
        self.configure(background=c["background"])
        self.option_add("*Font", ("Segoe UI", 10))
        self.option_add("*TCombobox*Listbox.background", c["input"])
        self.option_add("*TCombobox*Listbox.foreground", c["text"])
        self.option_add("*TCombobox*Listbox.selectBackground", c["button"])
        self.option_add("*TCombobox*Listbox.selectForeground", c["text"])
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background=c["background"], foreground=c["text"], font=("Segoe UI", 10), bordercolor=c["border"])
        style.configure("TFrame", background=c["background"])
        style.configure("Toolbar.TFrame", background=c["background"])
        style.configure("Hint.TLabel", foreground=c["muted"], font=("Segoe UI", 9))
        style.configure("TSeparator", background=c["border"])
        style.configure("TLabel", background=c["background"], foreground=c["text"])
        style.configure("Muted.TLabel", foreground=c["muted"])
        style.configure("Title.TLabel", font=("Segoe UI", 21, "bold"))
        style.configure("Status.TLabel", font=("Segoe UI", 10, "bold"))
        style.configure("TLabelframe", background=c["background"], bordercolor=c["border"], relief="solid", borderwidth=1)
        style.configure("TLabelframe.Label", background=c["background"], foreground=c["accent"], font=("Segoe UI", 10, "bold"))
        style.configure("TEntry", fieldbackground=c["input"], foreground=c["text"], insertcolor=c["text"], bordercolor=c["border"], padding=8, lightcolor=c["border"], darkcolor=c["border"])
        style.map("TEntry", bordercolor=[("focus", c["accent"])])
        style.configure("TCombobox", fieldbackground=c["input"], foreground=c["text"], arrowcolor=c["muted"], background=c["surface"], padding=8)
        style.map("TCombobox", fieldbackground=[("readonly", c["input"])], foreground=[("readonly", c["text"])], selectbackground=[("readonly", c["input"])], selectforeground=[("readonly", c["text"])])
        style.configure("TButton", background=c["surface"], foreground=c["text"], bordercolor=c["border"], padding=(14, 10), relief="flat")
        style.map("TButton", background=[("active", c["button"])], foreground=[("disabled", "#777777")])
        style.configure("Primary.TButton", background=c["accent"], foreground="#f7f9ec", font=("Segoe UI", 10, "bold"))
        style.map("Primary.TButton", background=[("disabled", c["surface"]), ("active", c["hover"])], foreground=[("disabled", c["muted"]), ("active", "#f7f9ec")])
        style.configure("Nav.TButton", anchor="w", background=c["background"], foreground=c["muted"], padding=(16, 13), borderwidth=0)
        style.configure("Selected.Nav.TButton", anchor="w", background=c["surface"], foreground=c["text"], padding=(16, 13), borderwidth=0)
        style.layout("Page.TNotebook.Tab", [])
        style.configure("Stop.TButton", foreground=c["danger"])
        style.configure("TCheckbutton", background=c["background"], foreground=c["text"], indicatorbackground=c["input"], indicatorforeground=c["accent"], padding=4)
        style.map("TCheckbutton", background=[("active", c["background"])], indicatorbackground=[("selected", c["button"])])
        style.configure("TNotebook", background=c["background"], borderwidth=0, tabmargins=(0, 0, 0, 8))
        style.configure("TNotebook.Tab", background=c["background"], foreground=c["muted"], padding=(14, 9))
        style.map("TNotebook.Tab", background=[("selected", c["surface"]), ("active", c["surface"])], foreground=[("selected", c["accent"])])
        style.configure("Vertical.TScrollbar", background=c["surface"], troughcolor=c["input"], arrowcolor=c["muted"], bordercolor=c["border"])
        style.configure("Treeview", background=c["input"], fieldbackground=c["input"], foreground=c["text"], rowheight=32, bordercolor=c["border"])
        style.configure("Treeview.Heading", background=c["surface"], foreground=c["text"], padding=8)
        style.map("Treeview", background=[("selected", c["hover"])], foreground=[("selected", c["text"])])

    def _make_variables(self) -> None:
        self.model_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Проверка сервера…")
        self.context_var = tk.StringVar()
        self.kv_var = tk.StringVar()
        self.kv_resident_var = tk.StringVar()
        self.spec_var = tk.StringVar()
        self.spec_min_p_var = tk.StringVar()
        self.expert_cache_var = tk.StringVar()
        self.port_var = tk.StringVar()
        self.resident_var = tk.BooleanVar()
        self.vision_var = tk.BooleanVar()
        self.vision_device_var = tk.StringVar(value="GPU")
        self.mmproj_var = tk.StringVar()
        self.vision_tokens_var = tk.StringVar()
        self.vram_reserve_var = tk.StringVar()
        self.ram_reserve_var = tk.StringVar()
        self.mtp_var = tk.StringVar()
        self.mtp_enabled_var = tk.BooleanVar()
        self.model_info_var = tk.StringVar()
        self.lazy_var = tk.BooleanVar()
        self.engine_var = tk.StringVar(value="Strata")
        self.external_model_var = tk.StringVar()
        self.external_layers_var = tk.StringVar(value="auto")
        self.external_threads_var = tk.StringVar(value=str(max(1, (os.cpu_count() or 4) // 2)))
        self.external_cache_k_var = tk.StringVar(value="q8_0")
        self.external_cache_v_var = tk.StringVar(value="q8_0")
        self.external_thinking_var = tk.BooleanVar(value=False)
        self.external_cpu_moe_var = tk.StringVar(value="0")
        self.external_parallel_var = tk.StringVar(value="1")
        self.external_cache_ram_var = tk.StringVar(value="8192")
        self.external_reasoning_budget_var = tk.StringVar(value="-1")
        self.external_ubatch_var = tk.StringVar(value="512")
        self.external_temp_var = tk.StringVar(value="0.8")
        self.external_top_p_var = tk.StringVar(value="0.95")
        self.external_top_k_var = tk.StringVar(value="40")
        self.external_min_p_var = tk.StringVar(value="0.05")
        self.external_budget_message_var = tk.StringVar()
        self.external_fit_var = tk.BooleanVar(value=True)
        self.poly_window_var = tk.StringVar(value="8")
        self.poly_threads_var = tk.StringVar(value="16")
        self.poly_kv_var = tk.StringVar(value="fp16")
        self.poly_vision_gpu_var = tk.BooleanVar(value=False)

    def _build_ui(self) -> None:
        self._apply_theme()
        shell = ttk.Frame(self)
        shell.pack(fill="both", expand=True)
        shell.columnconfigure(2, weight=1)
        shell.rowconfigure(0, weight=1)
        sidebar = ttk.Frame(shell, padding=(16, 24, 16, 16), width=180)
        sidebar.grid(row=0, column=0, sticky="ns")
        sidebar.columnconfigure(0, weight=1)
        sidebar.rowconfigure(3, weight=1)
        ttk.Label(sidebar, text="STRATA", font=("Segoe UI", 21, "bold"), foreground=THEME["accent"]).grid(row=0, column=0, sticky="w")
        ttk.Label(sidebar, text="CONTROL / DESKTOP 4.1", style="Hint.TLabel").grid(row=1, column=0, sticky="w", pady=(2, 28))
        nav = ttk.Frame(sidebar)
        nav.grid(row=2, column=0, sticky="new")
        nav.columnconfigure(0, weight=1)
        ttk.Label(sidebar, text="Локальное приложение", style="Hint.TLabel").grid(row=4, column=0, sticky="w", pady=(14, 0))
        ttk.Separator(shell, orient="vertical").grid(row=0, column=1, sticky="ns")
        outer = ttk.Frame(shell, padding=(22, 18, 22, 14))
        outer.grid(row=0, column=2, sticky="nsew")
        outer.columnconfigure(0, weight=1)

        header = ttk.Frame(outer)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Управление сервером", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="Модели, движки и настройки вашего сервера", style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=(2, 4))

        model_box = ttk.LabelFrame(outer, text="Рабочая конфигурация", padding=(12, 10))
        model_box.grid(row=1, column=0, sticky="ew", pady=(16, 6))
        model_box.columnconfigure(1, weight=1)
        ttk.Label(model_box, text="Модель").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.model_combo = ttk.Combobox(model_box, textvariable=self.model_var, state="readonly")
        self.model_combo.grid(row=0, column=1, sticky="ew")
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_selected)
        self.add_model_button = ttk.Button(model_box, text="Добавить модель", command=self._add_model)
        self.add_model_button.grid(row=0, column=2, padx=(8, 0))
        self.model_path_label = ttk.Label(model_box, text="", style="Muted.TLabel", wraplength=780)
        self.model_path_label.grid(row=1, column=0, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Label(model_box, textvariable=self.model_info_var, style="Muted.TLabel", wraplength=820).grid(row=2, column=0, columnspan=3, sticky="w", pady=(6, 0))
        self._build_engine_controls(model_box)

        status_box = ttk.Frame(outer, padding=(2, 6))
        status_box.grid(row=2, column=0, sticky="ew", pady=(4, 2))
        status_box.columnconfigure(1, weight=1)
        self.status_label = ttk.Label(status_box, textvariable=self.status_var, style="Status.TLabel")
        self.status_label.grid(row=0, column=0, sticky="w")
        self.address_label = ttk.Label(status_box, text="")
        self.address_label.grid(row=0, column=1, sticky="e")

        notebook = ttk.Notebook(outer, style="Page.TNotebook")
        self.notebook = notebook
        notebook.grid(row=3, column=0, sticky="nsew", pady=8)
        outer.rowconfigure(3, weight=1)
        settings_page = ttk.Frame(notebook, padding=(4, 10))
        self.strata_settings_page = settings_page
        advanced_page = ttk.Frame(notebook, padding=(4, 10))
        memory_page = ttk.Frame(notebook, padding=(4, 14))
        notebook.add(memory_page, text="Ресурсы ПК")
        self._build_monitor_page(notebook)
        notebook.add(settings_page, text="Strata · параметры")
        self._build_external_page(notebook)
        self._build_poly_page(notebook)
        notebook.add(advanced_page, text="Дополнительно")
        catalog_page = ttk.Frame(notebook, padding=14)
        notebook.add(catalog_page, text="Каталог моделей")
        self._build_catalog(catalog_page)
        self._nav_buttons = {}
        for row, tab in enumerate(notebook.tabs()):
            label = notebook.tab(tab, "text")
            button = ttk.Button(nav, text=label, style="Nav.TButton", command=lambda tab=tab: self._select_page(tab))
            button.grid(row=row, column=0, sticky="ew", pady=3)
            self._nav_buttons[tab] = button
        notebook.bind("<<NotebookTabChanged>>", lambda _event: self._sync_navigation())
        self._sync_navigation()

        memory_page.columnconfigure(0, weight=1)
        memory_page.rowconfigure(0, weight=1)
        memory_canvas = tk.Canvas(memory_page, background=THEME["background"], highlightthickness=0)
        memory_canvas.grid(row=0, column=0, sticky="nsew")
        memory_scroll = ttk.Scrollbar(memory_page, orient="vertical", command=memory_canvas.yview)
        memory_scroll.grid(row=0, column=1, sticky="ns")
        memory_canvas.configure(yscrollcommand=memory_scroll.set)
        memory_content = ttk.Frame(memory_canvas)
        memory_window = memory_canvas.create_window((0, 0), window=memory_content, anchor="nw")
        self._wire_scroll_canvas(memory_canvas, memory_content, memory_window)
        memory_page = memory_content

        memory_page.columnconfigure(0, weight=1)
        memory_page.columnconfigure(1, weight=1)
        ttk.Label(memory_page, text="Оставить память для работы за ПК", font=("Segoe UI", 13, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(2, 14))
        for column, title, variable, detail in (
            (0, "VRAM · видеопамять", self.vram_reserve_var, "ГБ оставить свободными при выборе GPU-кэша.\nРаботает с Expert cache = auto."),
            (1, "RAM · оперативная память", self.ram_reserve_var, "ГБ оставить для остального ПК.\n0 — выключить ограничение и защиту."),
        ):
            card = ttk.LabelFrame(memory_page, text=title, padding=14)
            card.grid(row=1, column=column, sticky="nsew", padx=(0, 8) if column == 0 else (8, 0))
            card.columnconfigure(0, weight=1)
            ttk.Entry(card, textvariable=variable, font=("Segoe UI", 18), width=8).grid(row=0, column=0, sticky="ew", pady=(0, 12))
            ttk.Label(card, text=detail, style="Muted.TLabel", wraplength=340).grid(row=1, column=0, sticky="w")
        self.memory_summary = ttk.Label(memory_page, text="", foreground=THEME["accent"], font=("Segoe UI", 11, "bold"))
        self.memory_summary.grid(row=2, column=0, columnspan=2, sticky="w", pady=(20, 12))
        ttk.Label(memory_page, text="Резерв RAM уменьшает кэш экспертов; остальные эксперты читаются с диска. Если свободной RAM меньше заданного запаса три проверки подряд, защита остановит сервер. Она продолжит работать после закрытия этого окна.", style="Muted.TLabel", wraplength=780).grid(row=3, column=0, columnspan=2, sticky="w", pady=(0, 12))
        ttk.Label(memory_page, text="1 ГБ здесь = 1024 МиБ. Резерв VRAM — цель при запуске, а не запрет другим приложениям занимать память.", style="Muted.TLabel", wraplength=780).grid(row=4, column=0, columnspan=2, sticky="w")
        ttk.Button(memory_page, text="Настройки с запасом · 32K контекст", command=self._apply_memory_preset).grid(row=5, column=0, columnspan=2, sticky="w", pady=12)

        settings_page.columnconfigure(0, weight=1)
        settings_page.rowconfigure(0, weight=1)
        settings_canvas = tk.Canvas(settings_page, background=THEME["background"], highlightthickness=0)
        settings_canvas.grid(row=0, column=0, sticky="nsew")
        settings_scroll = ttk.Scrollbar(settings_page, orient="vertical", command=settings_canvas.yview)
        settings_scroll.grid(row=0, column=1, sticky="ns")
        settings_canvas.configure(yscrollcommand=settings_scroll.set)
        settings_content = ttk.Frame(settings_canvas)
        settings_content.columnconfigure(0, weight=1)
        settings_window = settings_canvas.create_window((0, 0), window=settings_content, anchor="nw")
        self._wire_scroll_canvas(settings_canvas, settings_content, settings_window)
        settings = ttk.LabelFrame(settings_content, text="Параметры запуска", padding=10)
        settings.grid(row=0, column=0, sticky="new", pady=4)
        for col in (1, 3):
            settings.columnconfigure(col, weight=1)

        ttk.Label(settings, text="Контекст (токены)").grid(row=0, column=0, sticky="w", padx=6)
        ttk.Combobox(settings, textvariable=self.context_var, values=(8192, 32768, 65536, 131072, 204800, 262144, 393216, 524288), width=14).grid(row=0, column=1, sticky="ew", padx=6, pady=4)
        self._combo(settings, 0, 2, "KV cache", self.kv_var, ("fp16", "int8", "q4_0", "k8v4"))
        self._entry(settings, 1, 0, "KV в VRAM (токены; 0 — выкл.)", self.kv_resident_var)
        self._entry(settings, 1, 2, "Порт", self.port_var)
        self._entry(settings, 2, 0, "Spec decoding (токены)", self.spec_var)
        self._entry(settings, 2, 2, "Порог spec-min-p", self.spec_min_p_var)
        self._entry(settings, 3, 0, "Expert cache (auto или число)", self.expert_cache_var)
        ttk.Checkbutton(settings, text="Resident experts (без RAM-резерва)", variable=self.resident_var).grid(
            row=3, column=2, columnspan=2, sticky="w", padx=6, pady=4
        )

        ttk.Checkbutton(settings, text="Включить vision", variable=self.vision_var).grid(
            row=4, column=0, columnspan=2, sticky="w", padx=6, pady=(8, 4)
        )
        self._combo(settings, 4, 2, "Устройство vision", self.vision_device_var, ("GPU", "CPU"))

        ttk.Label(settings, text="Файл mmproj").grid(row=5, column=0, sticky="w", padx=6, pady=4)
        self.mmproj_entry = ttk.Entry(settings, textvariable=self.mmproj_var)
        self.mmproj_entry.grid(row=5, column=1, columnspan=2, sticky="ew", padx=6, pady=4)
        ttk.Button(settings, text="Обзор…", command=self._browse_mmproj).grid(row=5, column=3, sticky="ew", padx=6, pady=4)
        self._entry(settings, 6, 0, "Лимит image tokens", self.vision_tokens_var)
        ttk.Checkbutton(settings, text="MTP · черновики модели", variable=self.mtp_enabled_var).grid(row=7, column=0, columnspan=2, sticky="w", padx=6, pady=(12, 4))
        ttk.Label(settings, text="Spec = 0 отключает спекуляцию", style="Muted.TLabel").grid(row=7, column=2, columnspan=2, sticky="w", padx=6)
        ttk.Label(settings, text="Папка MTP runtime").grid(row=8, column=0, sticky="w", padx=6)
        ttk.Entry(settings, textvariable=self.mtp_var).grid(row=8, column=1, columnspan=2, sticky="ew", padx=6, pady=4)
        ttk.Button(settings, text="Обзор…", command=self._browse_mtp).grid(row=8, column=3, sticky="ew", padx=6)
        ttk.Label(settings, text="Без MTP сервер работает; черновики по совпадениям с контекстом остаются доступны при Spec > 0. Контекст выше 262144 использует YaRN и требует дополнительной памяти; качество расширения здесь не измерялось.", style="Muted.TLabel", wraplength=760).grid(row=9, column=0, columnspan=4, sticky="w", padx=6, pady=8)
        ttk.Checkbutton(settings, text="Открывать сервер без загрузки модели (lazy)", variable=self.lazy_var).grid(row=10, column=0, columnspan=4, sticky="w", padx=6, pady=4)
        ttk.Label(settings, text="В этом режиме веб-интерфейс открывается сразу. Модель и vision загружаются при первом запросе; первый ответ будет дольше.", style="Muted.TLabel", wraplength=760).grid(row=11, column=0, columnspan=4, sticky="w", padx=6, pady=4)

        advanced = ttk.LabelFrame(advanced_page, text="Дополнительные аргументы движка (по одному на строку)", padding=8)
        advanced.grid(row=0, column=0, sticky="nsew", pady=4)
        environment = ttk.LabelFrame(advanced_page, text="Переменные окружения (ИМЯ=значение)", padding=8)
        environment.grid(row=1, column=0, sticky="nsew", pady=4)
        advanced_page.columnconfigure(0, weight=1)
        advanced_page.rowconfigure(0, weight=3)
        advanced_page.rowconfigure(1, weight=1)
        advanced.columnconfigure(0, weight=1)
        advanced.rowconfigure(0, weight=1)
        self.extra_args = tk.Text(advanced, height=10, wrap="none", undo=True)
        self._theme_editor(self.extra_args)
        self.extra_args.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(advanced, orient="vertical", command=self.extra_args.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.extra_args.configure(yscrollcommand=scroll.set)
        ttk.Label(
            advanced,
            text="Каждая строка — отдельный аргумент; значение параметра укажите следующей строкой. Пути с пробелами пишите без кавычек.",
            wraplength=800,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))

        environment.columnconfigure(0, weight=1)
        environment.rowconfigure(0, weight=1)
        self.environment_vars = tk.Text(environment, height=5, wrap="none", undo=True)
        self._theme_editor(self.environment_vars)
        self.environment_vars.grid(row=0, column=0, sticky="nsew")
        env_scroll = ttk.Scrollbar(environment, orient="vertical", command=self.environment_vars.yview)
        env_scroll.grid(row=0, column=1, sticky="ns")
        self.environment_vars.configure(yscrollcommand=env_scroll.set)
        ttk.Label(
            environment,
            text="Необязательно. По одной переменной на строку; строки с # игнорируются.",
            wraplength=800,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))

        buttons = ttk.Frame(outer, style="Toolbar.TFrame")
        buttons.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        for col in range(3):
            buttons.columnconfigure(col, weight=1)
        self.start_button = ttk.Button(buttons, text="Запустить сервер", style="Primary.TButton", command=self._start_server)
        self.stop_button = ttk.Button(buttons, text="Остановить", style="Stop.TButton", command=self._stop_server)
        self.restart_button = ttk.Button(buttons, text="Перезапустить", command=self._restart_server)
        self.save_button = ttk.Button(buttons, text="Сохранить настройки", command=self._save_settings)
        self.web_button = ttk.Button(buttons, text="Открыть веб-интерфейс", command=self._open_web)
        self.log_button = ttk.Button(buttons, text="Журнал", command=self._open_log)
        for index, button in enumerate((self.start_button, self.stop_button, self.restart_button,
                                         self.save_button, self.web_button, self.log_button)):
            button.grid(row=index // 3, column=index % 3, sticky="ew", padx=4, pady=4)

        ttk.Label(
            outer,
            text="Изменения применяются при следующем запуске. Сохраните их в профиль, чтобы использовать снова.",
            style="Hint.TLabel",
            wraplength=800,
        ).grid(row=5, column=0, sticky="w", pady=(6, 0))
        self._bind_page_scroll(memory_content, memory_canvas)
        self._bind_page_scroll(settings_content, settings_canvas)

    def _select_page(self, tab):
        if self.notebook.tab(tab, "state") != "disabled":
            self.notebook.select(tab)
            self._sync_navigation()

    def _sync_navigation(self):
        selected = self.notebook.select()
        for tab, button in getattr(self, "_nav_buttons", {}).items():
            state = "disabled" if self.notebook.tab(tab, "state") == "disabled" else "normal"
            style = "Selected.Nav.TButton" if tab == selected else "Nav.TButton"
            if str(button.cget("state")) != state:
                button.configure(state=state)
            if button.cget("style") != style:
                button.configure(style=style)

    def _wire_scroll_canvas(self, canvas, content, window_id):
        pending = [None]
        last = {"width": None, "region": None}
        def fit():
            pending[0] = None
            if not canvas.winfo_exists():
                return
            width = canvas.winfo_width()
            if width != last["width"]:
                canvas.itemconfigure(window_id, width=width)
                last["width"] = width
            region = canvas.bbox("all")
            if region != last["region"]:
                canvas.configure(scrollregion=region)
                last["region"] = region
        def request(_event):
            if pending[0] is None:
                pending[0] = canvas.after_idle(fit)
        content.bind("<Configure>", request)
        canvas.bind("<Configure>", request)

    def _bind_page_scroll(self, content, canvas) -> None:
        """Scroll settings locally without stealing wheel input from editors."""
        def wheel(event):
            if isinstance(event.widget, (ttk.Combobox, tk.Text)):
                return None
            bounds = canvas.bbox("all")
            if bounds and bounds[3] > canvas.winfo_height():
                delta = -1 if event.delta > 0 or getattr(event, "num", 0) == 4 else 1
                canvas.yview_scroll(delta * 3, "units")
                return "break"
            return None

        def attach(widget):
            widget.bind("<MouseWheel>", wheel, add="+")
            widget.bind("<Button-4>", wheel, add="+")
            widget.bind("<Button-5>", wheel, add="+")
            for child in widget.winfo_children():
                attach(child)

        attach(content)
        canvas.bind("<MouseWheel>", wheel, add="+")
        canvas.bind("<Button-4>", wheel, add="+")
        canvas.bind("<Button-5>", wheel, add="+")

    def _engine_choices(self):
        from strata_control_engines import engines
        choices = engines()
        self._engine_names = {item["name"]: key for key, item in choices.items()}
        self.engine_combo.configure(values=list(self._engine_names))
        return choices

    def _selected_engine(self):
        return self._engine_names.get(self.engine_var.get(), "strata")

    def _build_engine_controls(self, parent) -> None:
        ttk.Label(parent, text="Движок").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=(8, 0))
        self.engine_combo = ttk.Combobox(parent, textvariable=self.engine_var, state="readonly")
        self.engine_combo.grid(row=3, column=1, sticky="ew", pady=(8, 0))
        self.engine_combo.bind("<<ComboboxSelected>>", self._on_engine_selected)
        ttk.Button(parent, text="Движки / установить…", command=self._manage_engines).grid(row=3, column=2, padx=(8, 0), pady=(8, 0))
        actions = ttk.Frame(parent)
        actions.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        for index, (label, callback) in enumerate((("Сохранить профиль как…", self._save_profile_as), ("Импорт профиля…", self._import_profile), ("Экспорт профиля…", self._export_profile))):
            ttk.Button(actions, text=label, command=callback).grid(row=0, column=index, padx=(0, 8), sticky="ew")
            actions.columnconfigure(index, weight=1)
        self._engine_choices()

    def _build_monitor_page(self, notebook) -> None:
        page = ttk.Frame(notebook, padding=16)
        notebook.add(page, text="Сервер и ресурсы")
        page.columnconfigure(0, weight=1)
        self.server_resources = tk.StringVar(value="Проверяем RAM, VRAM и загрузку процессора…")
        ttk.Label(page, text="Ресурсы компьютера", style="Heading.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(page, textvariable=self.server_resources, wraplength=700).grid(row=1, column=0, sticky="w", pady=16)
        ttk.Label(page, text="Адрес API и журнал доступны в панели сервера. Для управления откройте Strata Control.",
                  style="Muted.TLabel", wraplength=700).grid(row=2, column=0, sticky="w")
        ttk.Button(page, text="Открыть панель управления", command=lambda: subprocess.Popen(
            [str(PYTHON), str(ROOT / "strata_control_web.py")], creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )).grid(row=3, column=0, sticky="w", pady=12)

    def _build_poly_page(self, notebook) -> None:
        outer = self.poly_page = ttk.Frame(notebook, padding=8)
        notebook.add(outer, text="PolyStrata · Ornith")
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)
        canvas = tk.Canvas(outer, background=THEME["background"], highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=scrollbar.set)
        page = ttk.Frame(canvas, padding=10)
        frame_id = canvas.create_window((0, 0), window=page, anchor="nw")
        self._wire_scroll_canvas(canvas, page, frame_id)
        page.columnconfigure(1, weight=1)
        ttk.Label(page, text="Ornith · отдельный движок PolyStrata", font=("Segoe UI", 14, "bold")).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 12))
        ttk.Label(page, text="GGUF модели").grid(row=1, column=0, sticky="w")
        ttk.Entry(page, textvariable=self.external_model_var).grid(row=1, column=1, sticky="ew", padx=8)
        ttk.Button(page, text="Обзор…", command=self._browse_external_model).grid(row=1, column=2)
        self._entry(page, 2, 0, "Контекст · токены", self.context_var)
        self._entry(page, 3, 0, "Порт локального API", self.port_var)
        self._entry(page, 4, 0, "Оставить VRAM · ГБ", self.vram_reserve_var)
        self._entry(page, 5, 0, "Оставить RAM · ГБ", self.ram_reserve_var)
        self._combo(page, 6, 0, "KV cache", self.poly_kv_var, ("fp16", "q8_0"))
        self._entry(page, 7, 0, "Кэш бесед в RAM · МиБ", self.external_cache_ram_var)
        self._entry(page, 8, 0, "Потоки экспертов CPU", self.poly_threads_var)
        self._entry(page, 9, 0, "Окно MTP · 1–16", self.poly_window_var)
        ttk.Label(page, text="Vision · файл mmproj").grid(row=10, column=0, sticky="w", pady=6)
        ttk.Entry(page, textvariable=self.mmproj_var).grid(row=10, column=1, sticky="ew", padx=8)
        ttk.Button(page, text="Обзор…", command=self._browse_mmproj).grid(row=10, column=2)
        ttk.Checkbutton(page, text="Vision на GPU · быстрее, но занимает VRAM", variable=self.poly_vision_gpu_var).grid(row=11, column=0, columnspan=3, sticky="w", pady=6)
        self._entry(page, 12, 0, "Image tokens · максимум", self.vision_tokens_var)
        self._entry(page, 13, 0, "Температура", self.external_temp_var)
        self._entry(page, 14, 0, "Top P", self.external_top_p_var)
        self._entry(page, 15, 0, "Top K", self.external_top_k_var)
        ttk.Label(page, text="На RTX 4080 обработка изображений на CPU оставляет больше VRAM для экспертов модели. Параметры применяются после сохранения и перезапуска.",
                  style="Muted.TLabel", wraplength=740).grid(row=16, column=0, columnspan=3, sticky="w", pady=(20, 0))

    def _build_external_page(self, notebook) -> None:
        outer = self.external_page = ttk.Frame(notebook, padding=8)
        notebook.add(outer, text="Другой движок")
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)
        canvas = tk.Canvas(outer, background=THEME["background"], highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=scrollbar.set)
        page = ttk.Frame(canvas, padding=8)
        frame_id = canvas.create_window((0, 0), window=page, anchor="nw")
        self._wire_scroll_canvas(canvas, page, frame_id)
        page.columnconfigure(1, weight=1)
        ttk.Label(page, text="Модель GGUF / файл движка").grid(row=0, column=0, sticky="w", pady=6)
        ttk.Entry(page, textvariable=self.external_model_var).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(page, text="Обзор…", command=self._browse_external_model).grid(row=0, column=2)
        self._entry(page, 1, 0, "Контекст (токены)", self.context_var)
        port_box = ttk.Frame(page)
        port_box.grid(row=1, column=2, sticky="e")
        ttk.Label(port_box, text="Порт").pack(side="left", padx=4)
        ttk.Entry(port_box, textvariable=self.port_var, width=7).pack(side="left")
        self._entry(page, 2, 0, "GPU layers · auto или число", self.external_layers_var)
        self._entry(page, 3, 0, "Потоки CPU", self.external_threads_var)
        self._combo(page, 4, 0, "KV · ключи", self.external_cache_k_var, ("f16", "q8_0", "q4_0"))
        self._combo(page, 5, 0, "KV · значения", self.external_cache_v_var, ("f16", "q8_0", "q4_0"))
        self._entry(page, 6, 0, "Слои экспертов MoE в RAM · 0–256", self.external_cpu_moe_var)
        self._entry(page, 7, 0, "Параллельные слоты", self.external_parallel_var)
        self._entry(page, 8, 0, "Кэш RAM · МиБ", self.external_cache_ram_var)
        self._entry(page, 9, 0, "Лимит размышления · -1 без лимита", self.external_reasoning_budget_var)
        self._entry(page, 10, 0, "Физический пакет · ubatch", self.external_ubatch_var)
        self._entry(page, 11, 0, "Температура", self.external_temp_var)
        self._entry(page, 12, 0, "Top P", self.external_top_p_var)
        self._entry(page, 13, 0, "Top K", self.external_top_k_var)
        self._entry(page, 14, 0, "Min P", self.external_min_p_var)
        self._entry(page, 15, 0, "Сообщение при лимите размышления", self.external_budget_message_var)
        ttk.Checkbutton(page, text="Автоподбор VRAM · fit", variable=self.external_fit_var).grid(row=16, column=0, columnspan=3, sticky="w", pady=6)
        ttk.Label(page, text="mmproj (необязательно)").grid(row=17, column=0, sticky="w", pady=6)
        ttk.Entry(page, textvariable=self.mmproj_var).grid(row=17, column=1, sticky="ew", padx=8)
        ttk.Button(page, text="Обзор…", command=self._browse_mmproj).grid(row=17, column=2)
        ttk.Checkbutton(page, text="Размышление · для моделей с enable_thinking", variable=self.external_thinking_var).grid(row=18, column=0, columnspan=3, sticky="w", pady=6)
        ttk.Label(page, text="Аргументы · по одному токену на строку. Для пользовательского EXE используйте {model}, {context}, {port}, {host}.", style="Muted.TLabel", wraplength=740).grid(row=19, column=0, columnspan=3, sticky="w", pady=6)
        self.external_args = tk.Text(page, height=5, wrap="none")
        self._theme_editor(self.external_args)
        self.external_args.grid(row=20, column=0, columnspan=3, sticky="nsew")
        page.rowconfigure(20, weight=1)
        ttk.Label(page, text="RAM-резерв включает сторож доступной памяти. VRAM-резерв задаёт цель fit, когда автоподбор включён. Пользовательский EXE должен быть локальным HTTP-сервером.", style="Muted.TLabel", wraplength=760).grid(row=21, column=0, columnspan=3, sticky="w", pady=8)
        ttk.Label(page, text="Поля GPU, MoE, KV и размышления применяются к llama.cpp. Пользовательские движки используют свой шаблон аргументов.", style="Muted.TLabel", wraplength=760).grid(row=22, column=0, columnspan=3, sticky="w", pady=4)
        self._bind_page_scroll(page, canvas)

    def _on_engine_selected(self, _event=None):
        if self._busy:
            old = self.config_data.get("launcher_engine", "strata")
            self.engine_var.set(self._engine_choices().get(old, {"name": "Strata"})["name"])
            return
        self.config_path = None
        self.config_data = {}
        self._load_config_list(restore=False)
        selected = self._selected_engine()
        external = selected != "strata"
        self.notebook.tab(self.strata_settings_page, state="disabled" if external else "normal")
        self.notebook.tab(self.external_page, state="normal" if external and selected != "polystrata" else "disabled")
        self.notebook.tab(self.poly_page, state="normal" if selected == "polystrata" else "disabled")
        if external:
            self.notebook.select(self.poly_page if selected == "polystrata" else self.external_page)
        self._sync_navigation()

    def _browse_external_model(self):
        selected = filedialog.askopenfilename(parent=self, title="Выберите локальный файл модели", filetypes=[("Модели", "*.gguf *.ninfer"), ("Все файлы", "*.*")])
        if selected:
            self.external_model_var.set(selected)

    def _new_external_profile(self):
        self._browse_external_model()
        model = self.external_model_var.get().strip()
        if not model:
            return
        PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        path = PROFILES_DIR / (uuid.uuid4().hex + ".json")
        data = {"launcher_engine": self._selected_engine(), "profile_name": Path(model).stem,
                "model_name": Path(model).stem, "port": 8080,
                "engine_settings": {"model": model, "context": 8192, "gpu_layers": "auto", "threads": max(1, (os.cpu_count() or 4) // 2), "cache_k": "q8_0", "cache_v": "q8_0", "vram_reserve_gib": 4, "extra_args": []},
                "launcher_memory": {"ram_reserve_gib": 8}, "launcher_env": {}}
        if data["launcher_engine"] == "polystrata":
            from strata_control_engines import POLY_CONFIG
            data["open_browser"] = False
            data["launcher_memory"] = {"ram_reserve_gib": 3}
            mmproj = next(iter(Path(model).parent.glob("mmproj*.gguf")), None)
            data["engine_settings"] = {"model": model, "poly_config": str(POLY_CONFIG), "context": 32768,
                                       "cache_ram_mib": 0, "poly_threads": 16, "poly_kv": "fp16",
                                       "poly_window": 8, "vram_reserve_gib": 2,
                                       "mmproj": str(mmproj) if mmproj else "", "vision_gpu": False,
                                       "vision_tokens": 1024, "temperature": 1.0, "top_p": 0.95, "top_k": 20}
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        self._load_config_list(path)

    def _load_external_config(self, path, data):
        self.config_path, self.config_data = path, data
        settings = data.get("engine_settings", {})
        self.external_model_var.set(settings.get("model", ""))
        self.context_var.set(str(settings.get("context", 8192)))
        self.port_var.set(str(data.get("port", 8080)))
        self.external_layers_var.set(str(settings.get("gpu_layers", "auto")))
        self.external_threads_var.set(str(settings.get("threads", 4)))
        self.external_cache_k_var.set(settings.get("cache_k", "q8_0"))
        self.external_cache_v_var.set(settings.get("cache_v", "q8_0"))
        self.external_thinking_var.set(bool(settings.get("thinking", False)))
        self.external_cpu_moe_var.set(str(settings.get("cpu_moe_layers", 0)))
        self.external_parallel_var.set(str(settings.get("parallel", 1)))
        self.external_cache_ram_var.set(str(settings.get("cache_ram_mib", 8192)))
        self.poly_window_var.set(str(settings.get("poly_window", 8)))
        self.poly_threads_var.set(str(settings.get("poly_threads", 16)))
        self.poly_kv_var.set(str(settings.get("poly_kv", "fp16")))
        self.poly_vision_gpu_var.set(bool(settings.get("vision_gpu", False)))
        self.vision_tokens_var.set(str(settings.get("vision_tokens", 1024)))
        self.external_reasoning_budget_var.set(str(settings.get("reasoning_budget", -1)))
        self.external_ubatch_var.set(str(settings.get("ubatch_size", 512)))
        self.external_temp_var.set(str(settings.get("temperature", 1.0 if data.get("launcher_engine") == "polystrata" else 0.8)))
        self.external_top_p_var.set(str(settings.get("top_p", 0.95)))
        self.external_top_k_var.set(str(settings.get("top_k", 20 if data.get("launcher_engine") == "polystrata" else 40)))
        self.external_min_p_var.set(str(settings.get("min_p", 0.05)))
        self.external_budget_message_var.set(settings.get("reasoning_budget_message", ""))
        self.external_fit_var.set(bool(settings.get("fit", True)))
        self.mmproj_var.set(settings.get("mmproj", ""))
        self.ram_reserve_var.set(str(data.get("launcher_memory", {}).get("ram_reserve_gib", 8)))
        self.vram_reserve_var.set(str(settings.get("vram_reserve_gib", 4)))
        self.external_args.delete("1.0", "end")
        self.external_args.insert("1.0", "\n".join(settings.get("extra_args", [])))
        self.environment_vars.delete("1.0", "end")
        self.environment_vars.insert("1.0", "\n".join(f"{key}={value}" for key, value in data.get("launcher_env", {}).items()))
        self.model_path_label.configure(text=settings.get("model", ""))
        self.model_info_var.set(f"{self.engine_var.get()} · отдельный профиль настроек")
        self.address_label.configure(text=_web_url(self._selected_port()))
        CONTROL_STATE.write_text(json.dumps({"profile": str(path.relative_to(ROOT)), "engine": data["launcher_engine"]}), encoding="utf-8")
        self.notebook.tab(self.strata_settings_page, state="disabled")
        is_poly = data.get("launcher_engine") == "polystrata"
        self.notebook.tab(self.external_page, state="disabled" if is_poly else "normal")
        self.notebook.tab(self.poly_page, state="normal" if is_poly else "disabled")
        self._sync_navigation()
        if is_poly:
            self.notebook.select(self.poly_page)

    def _updated_external_config(self):
        from strata_control_engines import command_for
        data = copy.deepcopy(self.config_data)
        settings = data.setdefault("engine_settings", {})
        if data.get("launcher_engine") == "polystrata":
            settings.update(model=self.external_model_var.get().strip(),
                            context=self._int_value(self.context_var.get(), "Контекст", 16384, 1048576),
                            cache_ram_mib=self._int_value(self.external_cache_ram_var.get(), "Кэш бесед", 0, 65536),
                            poly_threads=self._int_value(self.poly_threads_var.get(), "Потоки экспертов CPU", 1, 256),
                            poly_kv=self.poly_kv_var.get(),
                            poly_window=self._int_value(self.poly_window_var.get(), "Окно MTP", 1, 16),
                            temperature=self.external_temp_var.get().strip(),
                            top_p=self.external_top_p_var.get().strip(),
                            top_k=self._int_value(self.external_top_k_var.get(), "Top K", 0, 1000),
                            vram_reserve_gib=_reserve_gib(self.vram_reserve_var.get(), "Резерв VRAM", 64),
                            mmproj=self.mmproj_var.get().strip(), vision_gpu=self.poly_vision_gpu_var.get(),
                            vision_tokens=self._int_value(self.vision_tokens_var.get(), "Image tokens", 1, 8192))
            data["port"] = self._int_value(self.port_var.get(), "Порт", 1, 65535)
            data["launcher_memory"] = {"ram_reserve_gib": _reserve_gib(self.ram_reserve_var.get(), "RAM-резерв", psutil.virtual_memory().total / GIB)}
            command_for(data)
            return data
        settings.update(model=self.external_model_var.get().strip(), context=self._int_value(self.context_var.get(), "Контекст", 512, 2097152),
                        gpu_layers=self.external_layers_var.get().strip(), threads=self._int_value(self.external_threads_var.get(), "Потоки CPU", 1, 1024),
                        cache_k=self.external_cache_k_var.get(), cache_v=self.external_cache_v_var.get(), mmproj=self.mmproj_var.get().strip(),
                        thinking=self.external_thinking_var.get(), vram_reserve_gib=_reserve_gib(self.vram_reserve_var.get(), "VRAM-резерв", 256),
                        cpu_moe_layers=self._int_value(self.external_cpu_moe_var.get(), "Слои MoE в RAM", 0, 256),
                        parallel=self._int_value(self.external_parallel_var.get(), "Параллельные слоты", 1, 64),
                        cache_ram_mib=self._int_value(self.external_cache_ram_var.get(), "Кэш RAM", 0, 1048576),
                        reasoning_budget=self._int_value(self.external_reasoning_budget_var.get(), "Лимит размышления", -1, 1048576),
                        ubatch_size=self._int_value(self.external_ubatch_var.get(), "Физический пакет", 1, 8192),
                        temperature=self.external_temp_var.get().strip(), top_p=self.external_top_p_var.get().strip(),
                        top_k=self._int_value(self.external_top_k_var.get(), "Top K", 0, 1000),
                        min_p=self.external_min_p_var.get().strip(),
                        reasoning_budget_message=self.external_budget_message_var.get().strip(),
                        fit=self.external_fit_var.get(),
                        extra_args=[line.strip() for line in self.external_args.get("1.0", "end").splitlines() if line.strip()])
        if not Path(settings["model"]).is_file():
            raise ValueError("Выберите существующий локальный файл модели.")
        data["port"] = self._int_value(self.port_var.get(), "Порт", 1, 65535)
        data["launcher_memory"] = {"ram_reserve_gib": _reserve_gib(self.ram_reserve_var.get(), "RAM-резерв", psutil.virtual_memory().total / GIB)}
        data["launcher_env"] = self._read_environment()
        from strata_control_engines import engines
        if engines().get(data.get("launcher_engine"), {}).get("installed"):
            command_for(data)  # Uninstalled engines can still have saved profiles.
        return data

    def _save_profile_as(self):
        if self._busy:
            return
        if self.config_path is None:
            messagebox.showinfo("Профили", "Сначала добавьте модель.", parent=self)
            return
        name = simpledialog.askstring("Сохранить профиль", "Имя профиля, например: Coder · 128K · работа", parent=self)
        if not name or not name.strip():
            return
        try:
            data = self._updated_config()
            data["profile_name"] = name.strip()[:120]
            PROFILES_DIR.mkdir(parents=True, exist_ok=True)
            path = PROFILES_DIR / (uuid.uuid4().hex + ".json")
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            self._load_config_list(path)
            self.status_var.set("Профиль сохранён. Сервер продолжает работать со своими прежними настройками.")
        except (OSError, ValueError) as exc:
            messagebox.showerror("Профили", str(exc), parent=self)

    def _import_profile(self):
        if self._busy:
            return
        selected = filedialog.askopenfilename(parent=self, title="Импорт профиля", filetypes=[("Профиль JSON", "*.json")])
        if not selected:
            return
        try:
            data = json.loads(Path(selected).read_text(encoding="utf-8-sig"))
            if not isinstance(data, dict) or not (isinstance(data.get("args"), list) or isinstance(data.get("engine_settings"), dict)):
                raise ValueError("Это не профиль Strata Control.")
            engine = data.get("launcher_engine", "strata")
            choices = self._engine_choices()
            if engine not in choices:
                raise ValueError("Сначала подключите движок этого профиля в «Движки / установить».")
            PROFILES_DIR.mkdir(parents=True, exist_ok=True)
            destination = PROFILES_DIR / (uuid.uuid4().hex + ".json")
            data.setdefault("profile_name", Path(selected).stem)
            destination.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            self.engine_var.set(choices[engine]["name"])
            self._load_config_list(destination)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Импорт профиля", str(exc), parent=self)

    def _export_profile(self):
        if self.config_path is None:
            return
        selected = filedialog.asksaveasfilename(parent=self, title="Экспорт профиля", defaultextension=".json", filetypes=[("Профиль JSON", "*.json")])
        if selected:
            try:
                data = self._updated_config()
                Path(selected).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            except (OSError, ValueError) as exc:
                messagebox.showerror("Экспорт профиля", str(exc), parent=self)

    def _manage_engines(self):
        if self._busy:
            return
        from strata_control_engines import install_llama, install_zip, register_engine
        import threading
        import queue
        window = tk.Toplevel(self)
        window.title("Движки · установка и подключение")
        window.geometry("720x450")
        window.configure(background=THEME["background"])
        body = ttk.Frame(window, padding=20)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        ttk.Label(body, text="Добавить движок", font=("Segoe UI", 17, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 12))
        info = tk.StringVar(value="PolyStrata обнаруживается автоматически в Strata-data/polystrata и запускает Ornith. llama.cpp скачивается с официального GitHub. Для других движков выберите локальный ZIP или серверный EXE.")
        ttk.Label(body, textvariable=info, wraplength=660, style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=8)
        progress = ttk.Progressbar(body, mode="indeterminate")
        progress.grid(row=7, column=0, sticky="ew", pady=14)
        controls = []
        def job(work):
            q = queue.Queue()
            for button in controls:
                button.configure(state="disabled")
            progress.start()
            def execute():
                try:
                    result = work(lambda message: q.put(("progress", message)))
                    q.put(("done", result))
                except Exception as exc:
                    q.put(("error", str(exc)))
            threading.Thread(target=execute, daemon=True).start()
            def poll():
                if not window.winfo_exists():
                    return
                finished = False
                while not q.empty():
                    kind, value = q.get_nowait()
                    if kind == "progress":
                        info.set(value)
                    else:
                        finished = True
                        progress.stop()
                        for button in controls:
                            button.configure(state="normal")
                        if kind == "error":
                            info.set("Ошибка: " + value)
                        else:
                            choices = self._engine_choices()
                            self.engine_var.set(choices[value]["name"])
                            self._on_engine_selected()
                            info.set("Движок подключён. Теперь добавьте модель и сохраните профиль.")
                if not finished:
                    window.after(150, poll)
            window.after(150, poll)
        def local_zip():
            filename = filedialog.askopenfilename(parent=window, title="ZIP движка", filetypes=[("ZIP", "*.zip")])
            if filename:
                name = simpledialog.askstring("Движок", "Имя движка", parent=window)
                if name:
                    job(lambda update: install_zip(Path(filename), name, "custom", update))
        def executable():
            filename = filedialog.askopenfilename(parent=window, title="Серверный EXE движка", filetypes=[("EXE", "*.exe")])
            if filename:
                name = simpledialog.askstring("Движок", "Имя движка", parent=window)
                if name:
                    kind = "llama.cpp" if Path(filename).name.lower() == "llama-server.exe" else "custom"
                    job(lambda update: register_engine(name, Path(filename), kind))
        for row, label, callback in ((2, "Установить llama.cpp · NVIDIA CUDA", lambda: job(lambda update: install_llama("cuda", update))),
                                     (3, "Установить llama.cpp · CPU", lambda: job(lambda update: install_llama("cpu", update))),
                                     (4, "Установить другой движок из ZIP…", local_zip),
                                     (5, "Подключить установленный серверный EXE…", executable)):
            button = ttk.Button(body, text=label, command=callback)
            button.grid(row=row, column=0, sticky="ew", pady=4)
            controls.append(button)

    def _build_catalog(self, page) -> None:
        page.columnconfigure(0, weight=1)
        page.rowconfigure(1, weight=1)
        ttk.Label(page, text="Семейство и квантование — два отдельных выбора", font=("Segoe UI", 14, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))
        columns = ("family", "quant", "ram", "disk", "status")
        tree = ttk.Treeview(page, columns=columns, show="headings", height=7)
        tree.grid(row=1, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(page, orient="vertical", command=tree.yview)
        scroll.grid(row=1, column=1, sticky="ns")
        tree.configure(yscrollcommand=scroll.set)
        for name, label, width in zip(columns, ("Семейство", "Квантовка", "RAM, ГБ ≈", "GGUF, ГБ ≈", "Поддержка"), (260, 115, 90, 100, 120)):
            tree.heading(name, text=label)
            tree.column(name, width=width, minwidth=60)
        for row in catalog_rows():
            tree.insert("", "end", values=row)
        detail = ttk.Label(page, text="Ориентиры штатного setup, не гарантия запуска. RAM и VRAM не складываются в один общий лимит; большой GPU помогает в режиме low-RAM.", style="Muted.TLabel", wraplength=790)
        detail.grid(row=2, column=0, columnspan=2, sticky="w", pady=12)
        def selected(_event):
            rows = tree.selection()
            if rows:
                title, quant, *_ = tree.item(rows[0], "values")
                from strata_model_catalog import catalog, DESCRIPTIONS
                _, families = catalog()
                family = next((key for key, item in families.items() if item["title"] == title), "qwen")
                detail.configure(text=DESCRIPTIONS[family])
        tree.bind("<<TreeviewSelect>>", selected)
        ttk.Label(page, text="UD-Q6_K_XL и OrcaRouter IQ3_XXS описаны в документации как ручные workflows; их автоматическая подготовка здесь не включена.", style="Muted.TLabel", wraplength=790).grid(row=3, column=0, columnspan=2, sticky="w", pady=(0, 8))
        ttk.Button(page, text="Открыть документацию моделей", command=lambda: os.startfile(str(ROOT / "docs" / "MODELS.md"))).grid(row=4, column=0, sticky="w")

    def _browse_mtp(self) -> None:
        selected = filedialog.askdirectory(parent=self, title="Папка MTP runtime (с experts.bin)")
        if selected:
            self.mtp_var.set(selected)
            self.mtp_enabled_var.set(True)

    def _apply_memory_preset(self) -> None:
        # An explicit user action; existing profiles are never changed on load.
        total = psutil.virtual_memory().total / GIB if psutil else 32
        self.ram_reserve_var.set(f"{min(8, total / 4):g}")
        self.vram_reserve_var.set("2")
        self.expert_cache_var.set("auto")
        self.context_var.set("32768")
        self.kv_var.set("int8")
        self.kv_resident_var.set("0")
        self.status_var.set("Выбран профиль с запасом памяти. Сохраните настройки перед запуском.")

    @staticmethod
    def _theme_editor(editor) -> None:
        editor.configure(background=THEME["input"], foreground=THEME["text"], insertbackground=THEME["accent"], selectbackground=THEME["button"], selectforeground=THEME["text"], font=("Cascadia Mono", 10), relief="flat", highlightthickness=1, highlightbackground=THEME["border"], highlightcolor=THEME["accent"], padx=12, pady=10)

    @staticmethod
    def _entry(parent, row: int, col: int, label: str, variable: tk.StringVar) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w", padx=6, pady=4)
        ttk.Entry(parent, textvariable=variable, width=14).grid(row=row, column=col + 1, sticky="ew", padx=6, pady=4)

    @staticmethod
    def _combo(parent, row: int, col: int, label: str, variable: tk.StringVar, values) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w", padx=6, pady=4)
        ttk.Combobox(parent, textvariable=variable, values=values, state="readonly", width=12).grid(
            row=row, column=col + 1, sticky="ew", padx=6, pady=4
        )

    def _load_config_list(self, selected: Path | None = None, restore=True) -> None:
        paths = _configs()
        choices = self._engine_choices()
        if restore and selected is None and self.config_path is None:
            try:
                state = json.loads(CONTROL_STATE.read_text(encoding="utf-8"))
                remembered = ROOT / state["profile"]
                if remembered.is_file():
                    selected = remembered
            except (OSError, ValueError, KeyError):
                pass
        if selected is not None:
            try:
                key = json.loads(selected.read_text(encoding="utf-8-sig")).get("launcher_engine", "strata")
                if key in choices:
                    self.engine_var.set(choices[key]["name"])
            except (OSError, ValueError):
                pass
        engine = self._selected_engine()
        self.display_to_path.clear()
        for path in paths:
            try:
                data = json.loads(path.read_text(encoding="utf-8-sig"))
                if data.get("launcher_engine", "strata") != engine:
                    continue
                label = str(data.get("profile_name") or data.get("model_name", path.stem))
                base, number = label, 2
                while label in self.display_to_path:
                    label = f"{base} ({number})"
                    number += 1
            except (OSError, ValueError):
                label = path.name
            self.display_to_path[label] = path
        labels = list(self.display_to_path)
        self.model_combo.configure(values=labels)
        if labels:
            if selected is None:
                try:
                    selected = ROOT / json.loads(CONTROL_STATE.read_text(encoding="utf-8"))["profile"]
                except (OSError, ValueError, KeyError):
                    pass
            label = next((label for label, path in self.display_to_path.items() if path == selected), labels[0])
            self.model_var.set(label)
            self._load_config(self.display_to_path[label])
            self.save_button.configure(state="normal")
        else:
            self.config_path = None
            self.config_data = {}
            self.model_var.set("")
            self.model_info_var.set("")
            self.model_path_label.configure(text="")
            self.status_var.set("Добавьте скачанную модель кнопкой «+ Добавить модель»")
            for widget in (self.start_button, self.stop_button, self.restart_button, self.save_button):
                widget.configure(state="disabled")

    def _on_model_selected(self, _event=None) -> None:
        path = self.display_to_path.get(self.model_var.get())
        if path:
            self._load_config(path)

    def _load_config(self, path: Path) -> None:
        try:
            if self.config_path != path:
                self._pending_browser_port = None
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            choices = self._engine_choices()
            key = data.get("launcher_engine", "strata")
            if key in choices:
                self.engine_var.set(choices[key]["name"])
            if key != "strata":
                self._load_external_config(path, data)
                return
            self.notebook.tab(self.strata_settings_page, state="normal")
            self.notebook.tab(self.external_page, state="disabled")
            self.notebook.tab(self.poly_page, state="disabled")
            self._sync_navigation()
            args = [str(arg) for arg in data.get("args", [])]
            self.config_path = path
            self.config_data = data
            self.model_path_label.configure(text=_option_value(args, "--native", "Путь к модели не указан"))
            details = describe_file(_option_value(args, "--native"))
            if details:
                mode = "RAM-кэш с бюджетом" if _option_value(args, "--resident-budget-gib") else "Resident / low-RAM" if _has_flag(args, "--resident-experts") else "Стандартный режим"
                info = f"{details['title']} · {details['quant']} · {mode}"
                if "arena_gb" in details:
                    info += f" · эксперты ≈ {details['arena_gb'] / 1.073741824:.1f} ГиБ"
                if details.get("experimental") or not details.get("menu"):
                    info += " · экспериментальная / ручная настройка"
                self.model_info_var.set(info)
            else:
                self.model_info_var.set("Пользовательский профиль — совместимость определяет движок Strata")
            try:
                CONTROL_STATE.write_text(json.dumps({"profile": str(path.relative_to(ROOT)), "engine": "strata"}), encoding="utf-8")
            except OSError:
                pass
            self.context_var.set(_option_value(args, "--max-context", "32768"))
            self.kv_var.set(_option_value(args, "--kv", "int8"))
            self.kv_resident_var.set(_option_value(args, "--kv-resident", "0"))
            self.spec_var.set(_option_value(args, "--spec", "4"))
            self.mtp_var.set(_option_value(args, "--mtp") or data.get("launcher_mtp", ""))
            self.mtp_enabled_var.set(bool(_option_value(args, "--mtp")))
            self.lazy_var.set(bool(data.get("lazy_load", False)))
            self.spec_min_p_var.set(_option_value(args, "--spec-min-p", "0.5"))
            self.expert_cache_var.set(_option_value(args, "--expert-cache", "auto"))
            self.resident_var.set(_has_flag(args, "--resident-experts") or bool(_option_value(args, "--resident-budget-gib")))
            self.port_var.set(str(data.get("port", 8080)))
            self.vram_reserve_var.set(f"{float(_option_value(args, '--vram-reserve-mib', '700')) / 1024:.6g}")
            memory = data.get("launcher_memory", {})
            self.ram_reserve_var.set(str(memory.get("ram_reserve_gib", 0)))
            vision = data.get("vision") if isinstance(data.get("vision"), dict) else {}
            self.vision_var.set(bool(vision))
            self.vision_device_var.set("GPU" if vision.get("gpu", True) else "CPU")
            self.mmproj_var.set(str(vision.get("mmproj") or data.get("launcher_mmproj", "")))
            self.vision_tokens_var.set(str(vision.get("max_tokens", 1024)))
            self.extra_args.delete("1.0", "end")
            self.extra_args.insert("1.0", "\n".join(_without_options(args, MANAGED_OPTIONS)))
            env_vars = data.get("launcher_env", {})
            if not isinstance(env_vars, dict):
                env_vars = {}
            if memory.get("ram_reserve_gib", 0):
                env_vars = {name: value for name, value in env_vars.items()
                            if name.upper() != "STRATA_RESIDENT_HEADROOM_GIB"}
            self.environment_vars.delete("1.0", "end")
            self.environment_vars.insert(
                "1.0", "\n".join(f"{name}={value}" for name, value in env_vars.items())
            )
            self.address_label.configure(text=_web_url(int(data.get('port', 8080))))
        except (OSError, ValueError, TypeError) as exc:
            messagebox.showerror("Не удалось прочитать конфигурацию", str(exc), parent=self)

    def _add_model(self) -> None:
        if self._busy and self._prepare_process is None:
            self.status_var.set("Дождитесь завершения текущей операции.")
            return
        if self._selected_engine() != "strata":
            self._new_external_profile()
            return
        if self._prepare_process is not None:
            self._prepare_window.lift()
            return
        if any(_server_processes(path, force=True) for path in _configs()):
            messagebox.showinfo("Добавить модель", "Остановите текущий сервер перед подготовкой другой модели, чтобы освободить память.", parent=self)
            return
        selected = filedialog.askopenfilename(parent=self, title="Выберите модель GGUF (первую часть 00001)",
                                              filetypes=[("Модель GGUF", "*.gguf")])
        if not selected:
            return
        source = first_shard(selected)
        details = describe_file(source.name)
        if not details:
            messagebox.showerror("Модель не распознана", "Strata специализируется на Qwen3.8 Flash Next. Произвольные GGUF не поддерживаются. Список доступен на вкладке «Какие модели выбрать».", parent=self)
            return
        if not details.get("menu"):
            manual_doc = "UNSLOTH_Q6.md" if details["quant"] == "UD-Q6_K_XL" else "ORCA.md"
            messagebox.showinfo("Нужна ручная подготовка", details["description"] + f"\n\nИнструкция: docs/{manual_doc}\nПодготовленный strata-*.json можно разместить в папке Strata; он появится в списке после открытия приложения.", parent=self)
            return
        if not source.is_file():
            messagebox.showerror("Не найдена первая часть", f"Для подготовки нужен файл:\n{source}", parent=self)
            return
        key = hashlib.sha256(_normalized(source).encode()).hexdigest()[:16]
        destination = ROOT / f"strata-local-{key}.json"
        if destination.exists():
            self._load_config_list(destination)
            return
        note = f"{details['title']} · {details['quant']}\n{details['description']}\n\nОриентир setup: {details['ram_gb']} ГБ RAM. GGUF ≈ {details['download_gb']:g} ГБ; файлы уже должны быть скачаны."
        if not messagebox.askokcancel("Подготовить модель", f"{note}\n\nВсе части GGUF должны лежать рядом. Подготовка локальная; для обычного Resident-режима может потребоваться 20–80 ГБ дополнительного места. Unsloth читает экспертов из GGUF без такой копии.\n\nНачать подготовку?", parent=self):
            return
        log_path = ROOT / ".strata-control-prepare.log"
        command = [str(PYTHON), str(Path(__file__).resolve()), "--prepare-model", str(source), "--destination", str(destination)]
        if self.config_path:
            command += ["--template", str(self.config_path)]
        try:
            with log_path.open("wb") as log:
                self._prepare_process = subprocess.Popen(command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    env={**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"})
        except OSError as exc:
            messagebox.showerror("Подготовка модели", str(exc), parent=self)
            return
        window = self._prepare_window = tk.Toplevel(self)
        window.title("Добавление модели")
        window.geometry("800x450")
        window.configure(background=THEME["background"])
        ttk.Label(window, text="Подготавливаем локальную модель", font=("Segoe UI", 16, "bold")).pack(anchor="w", padx=20, pady=16)
        output = tk.Text(window, wrap="word")
        self._theme_editor(output)
        output.pack(fill="both", expand=True, padx=20)
        progress = ttk.Progressbar(window, mode="indeterminate")
        progress.pack(fill="x", padx=20, pady=12)
        progress.start()
        window.protocol("WM_DELETE_WINDOW", lambda: messagebox.showinfo("Подготовка", "Дождитесь окончания подготовки. Она не запускает сервер.", parent=window))
        self._busy = True
        self.add_model_button.configure(state="disabled")
        def poll():
            try:
                content = log_path.read_text(encoding="utf-8", errors="replace")[-20000:]
            except OSError:
                content = "Ожидание журнала…"
            output.configure(state="normal")
            output.delete("1.0", "end")
            output.insert("1.0", content)
            output.see("end")
            output.configure(state="disabled")
            code = self._prepare_process.poll()
            if code is None:
                self.after(1000, poll)
                return
            self._prepare_process = None
            self._busy = False
            self.add_model_button.configure(state="normal")
            progress.stop()
            progress.destroy()
            window.protocol("WM_DELETE_WINDOW", window.destroy)
            ttk.Button(window, text="Готово" if code == 0 else "Закрыть", command=window.destroy).pack(pady=12)
            if code == 0 and destination.exists():
                self._load_config_list(destination)
                self.status_var.set("Модель добавлена — нажмите «Запустить»")
            else:
                self.status_var.set("Не удалось добавить модель — причина в окне подготовки")
        self.after(500, poll)

    def _browse_mmproj(self) -> None:
        selected = filedialog.askopenfilename(
            parent=self,
            title="Выберите mmproj GGUF",
            filetypes=(("GGUF", "*.gguf"), ("Все файлы", "*.*")),
            initialdir=str(Path(self.mmproj_var.get()).parent) if self.mmproj_var.get() else str(ROOT),
        )
        if selected:
            self.mmproj_var.set(selected)

    def _read_extra_args(self) -> list[str]:
        lines = [line.strip() for line in self.extra_args.get("1.0", "end").splitlines()]
        extras = [line for line in lines if line and not line.startswith("#")]
        for token in extras:
            key = token.split("=", 1)[0]
            if key in MANAGED_OPTIONS:
                raise ValueError(f"Параметр {key} задаётся отдельным полем сверху.")
            if key in BLOCKED_EXTRA_OPTIONS:
                raise ValueError(f"{key} управляется приложением и не принимается в дополнительных аргументах.")
        return extras

    def _read_environment(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for line_number, raw in enumerate(self.environment_vars.get("1.0", "end").splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            name, separator, value = line.partition("=")
            name = name.strip()
            if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise ValueError(f"Переменная окружения, строка {line_number}: формат ИМЯ=значение.")
            key = name.upper()
            if key in {existing.upper() for existing in result}:
                raise ValueError(f"Переменная окружения {name} указана больше одного раза.")
            if "\0" in value:
                raise ValueError(f"Переменная окружения {name}: значение содержит недопустимый символ.")
            result[name] = value.strip()
        return result

    @staticmethod
    def _int_value(value: str, label: str, minimum: int, maximum: int) -> int:
        try:
            number = int(value.strip())
        except ValueError as exc:
            raise ValueError(f"{label}: введите целое число.") from exc
        if not minimum <= number <= maximum:
            raise ValueError(f"{label}: допустимо от {minimum} до {maximum}.")
        return number

    def _updated_config(self) -> dict:
        if self.config_path is None:
            raise ValueError("Сначала выберите конфигурацию модели.")
        if self.config_data.get("launcher_engine", "strata") != "strata":
            return self._updated_external_config()
        context = self._int_value(self.context_var.get(), "Контекст", 1024, 524288)
        if context % 1024:
            raise ValueError("Контекст должен делиться на 1024.")
        kv_resident = self._int_value(self.kv_resident_var.get(), "KV в VRAM", 0, context)
        if kv_resident % 1024:
            raise ValueError("KV в VRAM должен делиться на 1024.")
        spec = self._int_value(self.spec_var.get(), "Spec decoding", 0, 16)
        port = self._int_value(self.port_var.get(), "Порт", 1, 65535)
        try:
            spec_min_p = float(self.spec_min_p_var.get().strip())
        except ValueError as exc:
            raise ValueError("Порог spec-min-p должен быть числом от 0 до 1.") from exc
        if not 0 <= spec_min_p <= 1:
            raise ValueError("Порог spec-min-p должен быть от 0 до 1.")
        expert_cache = self.expert_cache_var.get().strip().lower()
        if expert_cache != "auto":
            self._int_value(expert_cache, "Expert cache", 1, 4096)
        kv = self.kv_var.get().strip()
        if kv not in ("fp16", "int8", "q4_0", "k8v4"):
            raise ValueError("KV cache: выберите fp16, int8, q4_0 или k8v4.")
        extras = self._read_extra_args()
        launcher_env = self._read_environment()
        vram_reserve_gib = _reserve_gib(self.vram_reserve_var.get(), "Резерв VRAM", _gpu_total_gib(int(self.config_data.get("gpu", 0))))
        if vram_reserve_gib and expert_cache != "auto":
            raise ValueError("Для резерва VRAM выберите Expert cache = auto: фиксированный кэш может занять этот запас.")
        if psutil is None:
            raise ValueError("Для расчёта резерва RAM нужен psutil из окружения Strata.")
        total_ram_gib = psutil.virtual_memory().total / GIB
        ram_reserve_gib = _reserve_gib(self.ram_reserve_var.get(), "Резерв RAM", total_ram_gib)
        if ram_reserve_gib and total_ram_gib - ram_reserve_gib < 1:
            raise ValueError("После резерва RAM для Strata должно оставаться хотя бы 1 ГБ.")

        args = list(extras)
        mtp_enabled = self.mtp_enabled_var.get() if hasattr(self, "mtp_enabled_var") else bool(_option_value(self.config_data.get("args", []), "--mtp"))
        mtp = self.mtp_var.get().strip() if hasattr(self, "mtp_var") else _option_value(self.config_data.get("args", []), "--mtp")
        if mtp_enabled:
            if not (Path(mtp) / "experts.bin").is_file():
                raise ValueError("MTP: выберите подготовленную папку runtime с experts.bin или выключите MTP. Сервер умеет работать без него.")
            args += ["--mtp", mtp]
        if context > 262144 or self.config_data.get("launcher_context_auto_rope"):
            args = _without_options(args, {"--rope-scaling", "--rope-scale"})
            if context > 262144:
                args += ["--rope-scaling", "yarn", "--rope-scale", f"{context / 262144:g}"]
        for option, value in (
            ("--max-context", str(context)), ("--kv", kv), ("--spec", str(spec)),
            ("--spec-min-p", str(spec_min_p)), ("--expert-cache", expert_cache),
        ):
            args.extend((option, value))
        if kv_resident:
            args.extend(("--kv-resident", str(kv_resident)))
        args.extend(("--vram-reserve-mib", str(math.ceil(vram_reserve_gib * 1024))))
        old_reserve = self.config_data.get("launcher_memory", {}).get("ram_reserve_gib", 0)
        old_budget = _option_value(self.config_data.get("args", []), "--resident-budget-gib")
        if ram_reserve_gib:
            budget = total_ram_gib - ram_reserve_gib
            details = describe_file(_option_value(extras, "--native"))
            if details and details.get("budget"):
                # setup reserves 24 GiB beside Unsloth's resident experts.
                cell_bytes = {"fp16": 2048, "int8": 1056, "k8v4": 816, "q4_0": 576}[kv]
                kv_gb = context * 13 * cell_bytes / 1e9 if kv_resident else 0
                budget = min(budget, max(8, round(total_ram_gib) - 24 - math.ceil(kv_gb)), int(details["arena_gb"] / 1.073741824))
            args.extend(("--resident-budget-gib", f"{budget:.6f}"))
            launcher_env["STRATA_RESIDENT_HEADROOM_GIB"] = f"{ram_reserve_gib:g}"
        elif old_budget and not old_reserve:
            args.extend(("--resident-budget-gib", old_budget))
        elif self.resident_var.get():
            args.append("--resident-experts")
        if old_reserve and not ram_reserve_gib:
            launcher_env = {name: value for name, value in launcher_env.items()
                            if name.upper() != "STRATA_RESIDENT_HEADROOM_GIB"}

        data = dict(self.config_data)
        if mtp:
            data["launcher_mtp"] = mtp
        if hasattr(self, "mmproj_var") and self.mmproj_var.get().strip():
            data["launcher_mmproj"] = self.mmproj_var.get().strip()
        if hasattr(self, "lazy_var"):
            if self.lazy_var.get():
                data["lazy_load"] = True
            else:
                data.pop("lazy_load", None)
        if context > 262144:
            data["launcher_context_auto_rope"] = True
        elif data.get("launcher_context_auto_rope"):
            data.pop("launcher_context_auto_rope", None)
        if self.vision_var.get():
            mmproj = Path(self.mmproj_var.get().strip())
            if not mmproj.is_file():
                raise ValueError("Укажите существующий файл mmproj GGUF.")
            vision_tokens = self._int_value(self.vision_tokens_var.get(), "Лимит image tokens", 1, 8192)
            native_model = _option_value(extras, "--native", "")
            old_vision = data.get("vision") if isinstance(data.get("vision"), dict) else {}
            if not native_model:
                native_model = str(old_vision.get("model", ""))
            if not native_model:
                raise ValueError("В дополнительных аргументах не найден --native с путём к основной модели.")
            details = describe_file(native_model)
            if details and details.get("mmproj"):
                canonical = {"mmproj-Qwen3.8-Flash-Next-BF16.gguf", "mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf"}
                if mmproj.name in canonical and mmproj.name != details.get("mmproj"):
                    raise ValueError(f"Этот mmproj относится к другому семейству. Для {details['title']} нужен {details.get('mmproj')}.")
            vision_gpu = self.vision_device_var.get() == "GPU"
            vision_config = {
                "exe": str(old_vision.get("exe") or ROOT / "engine" / "strata-vision.exe"),
                "mmproj": str(mmproj),
                "model": native_model,
                "gpu": vision_gpu,
                "max_tokens": vision_tokens,
            }
            if not vision_gpu:
                vision_config["threads"] = max(1, (os.cpu_count() or 8) // 2)
            data["vision"] = vision_config
            args.append("--vision")
        else:
            data.pop("vision", None)

        data["args"] = args
        data["port"] = port
        data["launcher_env"] = launcher_env
        data["launcher_memory"] = {"ram_reserve_gib": ram_reserve_gib}
        return data

    def _save_settings(self, silent: bool = False):
        try:
            data = self._updated_config()
            path = self.config_path
            assert path is not None
            backup = path.with_suffix(path.suffix + ".launcher.bak")
            if not backup.exists():
                backup.write_bytes(path.read_bytes())
            temp = path.with_suffix(path.suffix + ".tmp")
            temp.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            os.replace(temp, path)
            self.config_data = data
            self.address_label.configure(text=_web_url(data['port']))
            self.status_var.set("Настройки сохранены. Для применения перезапустите сервер.")
            if not silent:
                messagebox.showinfo("Strata", "Настройки сохранены.", parent=self)
            return data
        except (OSError, ValueError, AssertionError) as exc:
            messagebox.showerror("Не удалось сохранить настройки", str(exc), parent=self)
            return None

    def _selected_port(self) -> int:
        try:
            return int(self.config_data.get("port", 8080))
        except (TypeError, ValueError):
            return 8080

    def _dispose_status_poll(self, event) -> None:
        if event.widget is self:
            self._status_closed = True
            if getattr(self, "_status_after_id", None):
                try:
                    self.after_cancel(self._status_after_id)
                except tk.TclError:
                    pass
                self._status_after_id = None

    @staticmethod
    def _status_snapshot(config_path, port, finished_code=None, force=False):
        """Read OS/network state in a worker, without accessing Tk objects."""
        result = {"path": str(config_path) if config_path else None, "port": port,
                  "matches": [], "health": None, "memory": None, "guard": None,
                  "finished_code": finished_code, "log": "", "gpu_memory": None, "cpu_percent": None}
        if psutil is not None:
            memory = psutil.virtual_memory()
            result["memory"] = (memory.available / GIB, memory.total / GIB)
            result["cpu_percent"] = psutil.cpu_percent(interval=None)
        try:
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                 timeout=3, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if gpu.returncode == 0:
                result["gpu_memory"] = [float(piece.strip()) for piece in gpu.stdout.splitlines()[0].split(",")]
        except (OSError, ValueError, subprocess.TimeoutExpired, IndexError):
            pass
        if config_path:
            matches = _server_processes(config_path, force=force)
            result["matches"] = matches
            if matches:
                port = _port_in_command(matches[0], port)
            result["port"] = port
            result["health"] = _health(port)
            if not matches:
                try:
                    result["guard"] = json.loads(_guard_path(config_path).read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    pass
        if finished_code is not None:
            try:
                with LOG_PATH.open("rb") as log:
                    log.seek(0, os.SEEK_END)
                    log.seek(max(0, log.tell() - 12000))
                    result["log"] = log.read().decode("utf-8", errors="replace")[-5000:]
            except OSError:
                result["log"] = "Журнал недоступен."
        return result

    def _schedule_status_poll(self, delay: int) -> None:
        if not self._status_closed:
            self._status_after_id = self.after(delay, self._refresh_status)

    def _refresh_status(self) -> None:
        # Explicit refreshes share one timer with normal polling.
        if not hasattr(self, "_status_queue"):
            self._status_queue = queue.Queue(maxsize=1)
            self._status_thread = None
            self._status_after_id = None
            self._status_closed = False
            self._status_cached = None
            self.bind("<Destroy>", self._dispose_status_poll, add="+")
        if self._status_closed:
            return
        if self._status_after_id:
            self.after_cancel(self._status_after_id)
            self._status_after_id = None
        try:
            snapshot = self._status_queue.get_nowait()
        except queue.Empty:
            snapshot = None
        if snapshot is not None:
            self._status_thread = None
            selected = str(self.config_path) if self.config_path else None
            if snapshot["path"] == selected:
                self._status_cached = snapshot
                self._apply_status_snapshot(snapshot)
                self._schedule_status_poll(3000)
                return
        if self._status_thread is not None:
            self._schedule_status_poll(100)
            return
        finished_code = None
        if self._started_process is not None and self._started_process.poll() is not None:
            if self._pending_browser_port is not None:
                finished_code = self._started_process.returncode
            self._started_process = None
        config_path, port = self.config_path, self._selected_port()
        force = self._pending_browser_port is not None
        output_queue = self._status_queue
        def collect():
            try:
                result = self._status_snapshot(config_path, port, finished_code, force)
            except Exception as exc:
                result = {"path": str(config_path) if config_path else None, "port": port,
                          "matches": [], "health": None, "memory": None, "guard": None,
                          "finished_code": None, "log": "", "error": str(exc)}
            output_queue.put(result)
        self._status_thread = threading.Thread(target=collect, name="control-status", daemon=True)
        self._status_thread.start()
        self._schedule_status_poll(100)

    @staticmethod
    def _status_widget(widget, **options) -> None:
        changed = {key: value for key, value in options.items() if str(widget.cget(key)) != str(value)}
        if changed:
            widget.configure(**changed)

    def _apply_status_snapshot(self, snapshot) -> None:
        if snapshot.get("finished_code") is not None:
            self._pending_browser_port = None
            messagebox.showerror("Сервер не запустился", f"Сервер завершился с кодом {snapshot['finished_code']}.\n\n{snapshot['log']}", parent=self)
        memory = snapshot.get("memory")
        if memory:
            protection = "защита RAM выключена" if self.ram_reserve_var.get() in ("", "0", "0.0") else f"запас RAM: {self.ram_reserve_var.get()} ГБ"
            self._status_widget(self.memory_summary, text=f"RAM: свободно {memory[0]:.1f} из {memory[1]:.1f} ГБ · {protection}")
        gpu_memory = snapshot.get("gpu_memory")
        ram_text = f"RAM {memory[1] - memory[0]:.1f}/{memory[1]:.1f} ГБ" if memory else "RAM —"
        gpu_text = (f"VRAM {gpu_memory[0] / 1024:.1f}/{gpu_memory[1] / 1024:.1f} ГБ · GPU {gpu_memory[2]:.0f}% · "
                    f"{gpu_memory[3]:.0f} °C · {gpu_memory[4]:.0f} Вт") if gpu_memory else "VRAM / GPU —"
        cpu = snapshot.get("cpu_percent")
        self.server_resources.set(ram_text + " · " + gpu_text + (f" · CPU {cpu:.0f}%" if cpu is not None else ""))
        if self.config_path is None:
            return
        if psutil is None:
            if self.status_var.get() != "Не найден psutil — управление запущенным процессом недоступно":
                self.status_var.set("Не найден psutil — управление запущенным процессом недоступно")
            self._status_widget(self.status_label, foreground=THEME["danger"])
            self._status_widget(self.start_button, state="disabled")
            self._status_widget(self.stop_button, state="disabled")
            self._status_widget(self.restart_button, state="disabled")
            self._status_widget(self.web_button, state="disabled")
            return
        port, matches, health = snapshot["port"], snapshot["matches"], snapshot["health"]
        engine = self.config_data.get("launcher_engine", "strata")
        if engine == "strata":
            ready = bool(health and health.get("service") == "strata" and (health.get("loaded") or self.config_data.get("lazy_load")))
        else:
            ready = bool(matches and health and health.get("status", "ok") not in ("loading", "error"))
        if ready:
            if matches:
                text, color = ("●  API готов · модель загрузится при запросе" if engine == "strata" and not health.get("loaded") else "●  Работает"), THEME["accent"]
            else:
                text, color = "●  Strata работает на этом порту", THEME["accent"]
            self._status_widget(self.web_button, state="normal")
            if matches and self._pending_browser_port == port:
                self._pending_browser_port = None
                if self.config_data.get("open_browser") is not False:
                    threading.Thread(target=webbrowser.open, args=(_web_url(port),), daemon=True).start()
        elif matches:
            text, color = "●  Запускается — ожидаем готовность API", THEME["warning"]
            if engine not in ("strata", "llama.cpp", "llama_cpp", "llama"):
                text, color = "●  Процесс работает · API не подтверждён", THEME["accent"]
            self._status_widget(self.web_button, state="disabled")
        elif health:
            text, color = "Порт занят другим сервисом", THEME["danger"]
            self._status_widget(self.web_button, state="disabled")
        else:
            text, color = "○  Остановлен", THEME["muted"]
            guard = snapshot.get("guard") or {}
            if guard.get("status") == "stopped_for_ram":
                text, color = f"Остановлен защитой RAM (резерв {guard.get('reserve_gib', 0):g} ГБ)", THEME["danger"]
            self._status_widget(self.web_button, state="disabled")
        if snapshot.get("error"):
            text, color = "Не удалось проверить состояние сервера", THEME["warning"]
        if self.status_var.get() != text:
            self.status_var.set(text)
        self._status_widget(self.status_label, foreground=color)
        busy = self._busy
        self._status_widget(self.start_button, state="disabled" if busy or matches or health else "normal")
        self._status_widget(self.stop_button, state="normal" if not busy and matches else "disabled")
        self._status_widget(self.restart_button, state="normal" if not busy and matches else "disabled")

    def _start_server(self, save: bool = True) -> None:
        if self.config_data.get("launcher_engine", "strata") == "strata":
            self._start_strata_server(save)
            return
        if self.config_path is None or self._busy:
            return
        data = self._save_settings(silent=True) if save else self.config_data
        if data is None:
            return
        try:
            from strata_control_engines import command_for
            command_for(data)
            if _server_processes(self.config_path, force=True):
                self.status_var.set("Этот профиль уже запущен.")
                return
            if _health(int(data["port"])):
                raise ValueError("Порт занят. Остановите предыдущий сервер или выберите другой порт.")
            reserve = float(data.get("launcher_memory", {}).get("ram_reserve_gib", 0))
            if reserve and psutil.virtual_memory().available / GIB <= reserve + 0.25:
                raise ValueError("Недостаточно свободной RAM рядом с заданным резервом.")
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            guard_path = _guard_path(self.config_path)
            guard_path.parent.mkdir(parents=True, exist_ok=True)
            guard_path.write_text(json.dumps({"status": "monitoring" if reserve else "disabled"}), encoding="utf-8")
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            with LOG_PATH.open("ab") as log:
                log.write(f"\n--- {self.engine_var.get()} · {self.config_path.name} ---\n".encode("utf-8"))
                proc = subprocess.Popen([str(PYTHON), str(Path(__file__).resolve()), "--external-server", "--config", str(self.config_path), "--port", str(data["port"])],
                    cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=flags, env={**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"})
                self._started_process = proc
                if reserve:
                    try:
                        created = psutil.Process(proc.pid).create_time()
                        subprocess.Popen([str(PYTHON), str(Path(__file__).resolve()), "--memory-guard", "--pid", str(proc.pid), "--created", str(created), "--ram-reserve-gib", str(reserve), "--state", str(guard_path)],
                            cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
                    except (OSError, psutil.Error):
                        try:
                            _terminate_tree(psutil.Process(proc.pid))
                        except psutil.NoSuchProcess:
                            pass
                        raise OSError("Не удалось включить защиту RAM. Сервер остановлен.")
            self._pending_browser_port = int(data["port"])
            self.status_var.set("Запускаю выбранный движок…")
            self._refresh_status()
        except (OSError, ValueError) as exc:
            messagebox.showerror("Запуск движка", str(exc), parent=self)

    def _start_strata_server(self, save: bool = True) -> None:
        if self.config_path is None or self._busy:
            return
        data = self._save_settings(silent=True) if save else self.config_data
        if data is None:
            return
        if _server_processes(self.config_path, force=True):
            self.status_var.set("Сервер уже запускается или работает — используйте «Перезапустить».")
            return
        health = _health(int(data.get("port", 8080)))
        if health:
            messagebox.showerror("Порт занят", "На выбранном порту уже отвечает другой сервер.", parent=self)
            return
        if not PYTHON.exists() or not SERVER.exists():
            messagebox.showerror("Не найден Strata", "Не найдены .venv или serve/server.py в папке Strata.", parent=self)
            return
        details = describe_file(_option_value(data.get("args", []), "--native"))
        build_path = ROOT / "engine" / "BUILD.json"
        if details and build_path.is_file():
            try:
                build = json.loads(build_path.read_text(encoding="utf-8-sig"))
                backend = build.get("backend", "cuda" if build.get("cuda") else "")
                if details.get("nvidia_only") and backend not in ("cuda", ""):
                    messagebox.showerror("Модель и движок не совместимы", f"{details['quant']} в штатном setup поддерживается только на NVIDIA. Текущий backend: {backend}.", parent=self)
                    return
                version = tuple(int(part) for part in build.get("version", "0").split(".") if part.isdigit())
                minimum = details.get("engine", (0, 1, 32) if details.get("budget") else (0,))
                if version < minimum:
                    messagebox.showerror("Нужен более новый движок", f"Для {details['quant']} нужен Strata {'.'.join(map(str, minimum))} или новее. Установлен {build.get('version')}. Обновите установку Strata.", parent=self)
                    return
            except (OSError, ValueError, TypeError):
                pass
        for label, value in (("Движок Strata", data.get("exe", "")),
                             ("Модель GGUF", _option_value(data.get("args", []), "--native")),
                             ("Папка подготовленной модели", _option_value(data.get("args", []), "--pack")),
                             ("Токенизатор", data.get("tokenizer", ""))):
            if not value or not Path(value).exists():
                messagebox.showerror("Не найден файл модели", f"{label}:\n{value or 'Путь не указан'}\n\nПроверьте доступность диска. Если модель перенесена, добавьте её кнопкой «+ Добавить модель».", parent=self)
                return
        ram_reserve = float(data.get("launcher_memory", {}).get("ram_reserve_gib", 0))
        if ram_reserve and psutil.virtual_memory().available / GIB <= ram_reserve + 0.25:
            messagebox.showerror("Недостаточно свободной RAM", f"Для запуска нужна свободная RAM больше резерва {ram_reserve:g} ГБ. Закройте другие программы или уменьшите резерв.", parent=self)
            return
        try:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            guard_path = _guard_path(self.config_path)
            guard_path.parent.mkdir(parents=True, exist_ok=True)
            guard_path.write_text(json.dumps({"status": "monitoring" if ram_reserve else "disabled"}), encoding="utf-8")
            with LOG_PATH.open("ab") as log:
                log.write(f"\n\n--- запуск {self.config_path.name} ---\n".encode("utf-8"))
                flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
                process_env = os.environ.copy()
                process_env.update({str(name): str(value) for name, value in data.get("launcher_env", {}).items()})
                server_proc = subprocess.Popen(
                    [str(PYTHON), str(SERVER), "--engine", "strata", "--config", str(self.config_path),
                     "--port", str(data.get("port", 8080)), "--host", "127.0.0.1"],
                    cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=flags, close_fds=True, env=process_env,
                )
                self._started_process = server_proc
                if ram_reserve:
                    try:
                        created = psutil.Process(server_proc.pid).create_time()
                        subprocess.Popen(
                            [str(PYTHON), str(Path(__file__).resolve()), "--memory-guard",
                             "--pid", str(server_proc.pid), "--created", str(created),
                             "--ram-reserve-gib", str(ram_reserve), "--state", str(guard_path)],
                            cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                            creationflags=flags, close_fds=True,
                        )
                    except (OSError, psutil.Error):
                        try:
                            _terminate_tree(psutil.Process(server_proc.pid))
                        except psutil.NoSuchProcess:
                            pass
                        raise OSError("Не удалось включить защиту RAM. Сервер остановлен.")
            self._pending_browser_port = int(data.get("port", 8080))
            self.status_var.set("Запускаю Strata…")
        except OSError as exc:
            messagebox.showerror("Не удалось запустить Strata", str(exc), parent=self)

    def _stop_server(self, quiet: bool = False, on_stopped=None) -> bool:
        self._pending_browser_port = None
        if self.config_path is None or self._busy:
            return False
        if psutil is None:
            if not quiet:
                messagebox.showerror("Не удалось остановить Strata", "В окружении Strata не найден psutil.", parent=self)
            return False
        self._busy = True
        path = self.config_path
        self.status_var.set("Останавливаю сервер…")
        self.engine_combo.configure(state="disabled")
        self.model_combo.configure(state="disabled")
        result = queue.Queue()
        def stop():
            try:
                for proc in _server_roots(path):
                    try:
                        _terminate_tree(proc)
                    except psutil.NoSuchProcess:
                        pass
                result.put(None)
            except Exception as exc:
                result.put(str(exc))
        threading.Thread(target=stop, name="control-stop", daemon=True).start()
        def finish():
            try:
                error = result.get_nowait()
            except queue.Empty:
                self.after(100, finish)
                return
            self._busy = False
            self.engine_combo.configure(state="readonly")
            self.model_combo.configure(state="readonly")
            if error:
                if not quiet:
                    messagebox.showerror("Не удалось остановить сервер", error, parent=self)
            elif on_stopped:
                self.after(500, on_stopped)
            self._refresh_status()
        self.after(100, finish)
        return True

    def _restart_server(self) -> None:
        data = self._save_settings(silent=True)
        if data is None:
            return
        self._stop_server(on_stopped=lambda: self._start_server(save=False))

    def _open_web(self) -> None:
        snapshot = getattr(self, "_status_cached", None)
        if snapshot and snapshot["path"] == str(self.config_path) and snapshot.get("health"):
            threading.Thread(target=webbrowser.open, args=(_web_url(snapshot["port"]),), daemon=True).start()
        else:
            self.status_var.set("Проверяем готовность веб-интерфейса…")
            self._refresh_status()

    def _open_log(self) -> None:
        if not LOG_PATH.exists():
            messagebox.showinfo("Журнал Strata", "Журнал появится после первого запуска через приложение.", parent=self)
            return
        try:
            os.startfile(str(LOG_PATH))
        except (AttributeError, OSError) as exc:
            messagebox.showerror("Журнал Strata", str(exc), parent=self)


if __name__ == "__main__":
    if "--external-server" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--external-server", action="store_true")
        parser.add_argument("--config", type=Path, required=True)
        parser.add_argument("--port", type=int)
        options = parser.parse_args()
        try:
            raise SystemExit(_external_server(options.config))
        except (OSError, ValueError) as exc:
            print(f"Не удалось запустить движок: {exc}", flush=True)
            raise SystemExit(1)
    elif "--prepare-model" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--prepare-model", type=Path, required=True)
        parser.add_argument("--destination", type=Path, required=True)
        parser.add_argument("--template", type=Path)
        options = parser.parse_args()
        try:
            _prepare_local_model(options.prepare_model, options.template, options.destination)
        except Exception as exc:
            print(f"Не удалось подготовить модель: {exc}", flush=True)
            raise SystemExit(1)
    elif "--memory-guard" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--memory-guard", action="store_true")
        parser.add_argument("--pid", type=int, required=True)
        parser.add_argument("--created", type=float, required=True)
        parser.add_argument("--ram-reserve-gib", type=float, required=True)
        parser.add_argument("--state", type=Path, required=True)
        options = parser.parse_args()
        _memory_guard(options.pid, options.created, options.ram_reserve_gib, options.state)
    else:
        try:
            from strata_control_web import main
            main()
        except Exception:
            import traceback
            detail = traceback.format_exc()
            try:
                (ROOT / ".strata-control-startup.log").write_text(detail, encoding="utf-8")
            except OSError:
                pass
            error_window = tk.Tk()
            error_window.withdraw()
            messagebox.showerror("Не удалось открыть Strata Control", "Причина ошибки сохранена в .strata-control-startup.log\n\n" + detail[-2500:], parent=error_window)
            error_window.destroy()
