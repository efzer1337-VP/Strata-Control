"""Small Windows control window for the Strata server installed in this folder."""
from __future__ import annotations

import json
import argparse
import hashlib
import math
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

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
    "--expert-cache", "--vram-reserve-mib", "--resident-budget-gib",
}
FLAG_OPTIONS = {"--resident-experts", "--vision"}
MANAGED_OPTIONS = VALUE_OPTIONS | FLAG_OPTIONS
BLOCKED_EXTRA_OPTIONS = {"--host", "--api-key", "--port", "--config", "--engine", "--open"}
GIB = 1024 ** 3
THEME = {
    "background": "#0d1815", "surface": "#15251f", "input": "#101d17",
    "border": "#30443a", "text": "#e2eee4", "muted": "#a5bbad",
    "accent": "#a5dfb5", "button": "#244f36", "hover": "#326e49",
    "warning": "#e8c382", "danger": "#efaaa1",
}


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
    try:
        result = subprocess.run(
            ["nvidia-smi", "--id", str(gpu), "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode == 0:
            return float(result.stdout.strip()) / 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
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
    return sorted(path for path in ROOT.glob("strata-*.json") if path.is_file())


def _server_processes(config_path: Path):
    if psutil is None:
        return []
    wanted_server = _normalized(SERVER)
    wanted_config = _normalized(config_path)
    found = []
    for proc in psutil.process_iter(["cmdline", "name"]):
        try:
            parts = [str(item) for item in (proc.info.get("cmdline") or [])]
            joined = " ".join(parts).replace("/", "\\").casefold()
            if wanted_server in joined and wanted_config in joined:
                found.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue
    return found


def _server_roots(config_path: Path):
    matches = _server_processes(config_path)
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
            data = json.loads(response.read().decode("utf-8"))
            return data if isinstance(data, dict) else None
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
        self.title("Strata — управление сервером")
        self.geometry(f"960x{min(840, self.winfo_screenheight() - 100)}")
        self.minsize(860, 650)
        self.config_path: Path | None = None
        self.config_data: dict = {}
        self.display_to_path: dict[str, Path] = {}
        self._busy = False
        self._pending_browser_port: int | None = None

        self._make_variables()
        self._build_ui()
        self._load_config_list()
        self.after(100, self._dark_titlebar)
        self.after(1000, self._refresh_status)

    def _dark_titlebar(self) -> None:
        if os.name == "nt":
            try:
                import ctypes
                window = ctypes.windll.user32.GetParent(self.winfo_id())
                dark = ctypes.c_int(1)
                ctypes.windll.dwmapi.DwmSetWindowAttribute(window, 20, ctypes.byref(dark), ctypes.sizeof(dark))
            except (OSError, AttributeError):
                pass

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
        style.configure("TLabel", background=c["background"], foreground=c["text"])
        style.configure("Muted.TLabel", foreground=c["muted"])
        style.configure("Title.TLabel", font=("Segoe UI", 23, "bold"))
        style.configure("Status.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("TLabelframe", background=c["background"], bordercolor=c["border"], relief="solid", borderwidth=1)
        style.configure("TLabelframe.Label", background=c["background"], foreground=c["accent"], font=("Segoe UI", 10, "bold"))
        style.configure("TEntry", fieldbackground=c["input"], foreground=c["text"], insertcolor=c["text"], bordercolor=c["border"], padding=8, lightcolor=c["border"], darkcolor=c["border"])
        style.map("TEntry", bordercolor=[("focus", c["accent"])])
        style.configure("TCombobox", fieldbackground=c["input"], foreground=c["text"], arrowcolor=c["muted"], background=c["surface"], padding=8)
        style.map("TCombobox", fieldbackground=[("readonly", c["input"])], foreground=[("readonly", c["text"])], selectbackground=[("readonly", c["input"])], selectforeground=[("readonly", c["text"])])
        style.configure("TButton", background=c["surface"], foreground=c["text"], bordercolor=c["border"], padding=(14, 10), relief="flat")
        style.map("TButton", background=[("active", c["button"])], foreground=[("disabled", "#6a8574")])
        style.configure("Primary.TButton", background=c["button"], foreground=c["accent"], font=("Segoe UI", 10, "bold"))
        style.map("Primary.TButton", background=[("active", c["hover"])])
        style.configure("Stop.TButton", foreground=c["danger"])
        style.configure("TCheckbutton", background=c["background"], foreground=c["text"], indicatorbackground=c["input"], indicatorforeground=c["accent"], padding=4)
        style.map("TCheckbutton", background=[("active", c["background"])], indicatorbackground=[("selected", c["button"])])
        style.configure("TNotebook", background=c["background"], borderwidth=0, tabmargins=(0, 0, 0, 8))
        style.configure("TNotebook.Tab", background=c["background"], foreground=c["muted"], padding=(18, 10))
        style.map("TNotebook.Tab", background=[("selected", c["surface"]), ("active", c["surface"])], foreground=[("selected", c["accent"])])
        style.configure("Vertical.TScrollbar", background=c["surface"], troughcolor=c["input"], arrowcolor=c["muted"], bordercolor=c["border"])

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

    def _build_ui(self) -> None:
        self._apply_theme()
        outer = ttk.Frame(self, padding=24)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)

        header = ttk.Frame(outer)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Strata", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="Управление локальной моделью", style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=(2, 4))
        ttk.Label(header, text="EVERGREEN", foreground=THEME["accent"], font=("Segoe UI", 9, "bold")).grid(row=0, column=1, sticky="e")

        model_box = ttk.LabelFrame(outer, text="Модель", padding=10)
        model_box.grid(row=1, column=0, sticky="ew", pady=(12, 8))
        model_box.columnconfigure(1, weight=1)
        ttk.Label(model_box, text="Конфигурация:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.model_combo = ttk.Combobox(model_box, textvariable=self.model_var, state="readonly")
        self.model_combo.grid(row=0, column=1, sticky="ew")
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_selected)

        status_box = ttk.LabelFrame(outer, text="Сервер", padding=10)
        status_box.grid(row=2, column=0, sticky="ew", pady=8)
        status_box.columnconfigure(1, weight=1)
        self.status_label = ttk.Label(status_box, textvariable=self.status_var, style="Status.TLabel")
        self.status_label.grid(row=0, column=0, sticky="w")
        self.address_label = ttk.Label(status_box, text="")
        self.address_label.grid(row=0, column=1, sticky="e")

        notebook = ttk.Notebook(outer)
        notebook.grid(row=3, column=0, sticky="nsew", pady=8)
        outer.rowconfigure(3, weight=1)
        settings_page = ttk.Frame(notebook, padding=6)
        advanced_page = ttk.Frame(notebook, padding=6)
        memory_page = ttk.Frame(notebook, padding=14)
        notebook.add(memory_page, text="Память ПК")
        notebook.add(settings_page, text="Модель и vision")
        notebook.add(advanced_page, text="Дополнительно")

        memory_page.columnconfigure(0, weight=1)
        memory_page.columnconfigure(1, weight=1)
        ttk.Label(memory_page, text="Запас для Windows и других программ", font=("Segoe UI", 15, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(4, 20))
        for column, title, variable, detail in (
            (0, "VRAM · видеопамять", self.vram_reserve_var, "ГБ оставить свободными при выборе GPU-кэша.\nРаботает с Expert cache = auto."),
            (1, "RAM · оперативная память", self.ram_reserve_var, "ГБ оставить для остального ПК.\n0 — выключить ограничение и защиту."),
        ):
            card = ttk.LabelFrame(memory_page, text=title, padding=20)
            card.grid(row=1, column=column, sticky="nsew", padx=(0, 8) if column == 0 else (8, 0))
            card.columnconfigure(0, weight=1)
            ttk.Entry(card, textvariable=variable, font=("Segoe UI", 20), width=8).grid(row=0, column=0, sticky="ew", pady=(0, 12))
            ttk.Label(card, text=detail, style="Muted.TLabel", wraplength=340).grid(row=1, column=0, sticky="w")
        self.memory_summary = ttk.Label(memory_page, text="", foreground=THEME["accent"], font=("Segoe UI", 11, "bold"))
        self.memory_summary.grid(row=2, column=0, columnspan=2, sticky="w", pady=(20, 12))
        ttk.Label(memory_page, text="Резерв RAM уменьшает кэш экспертов; остальные эксперты читаются с диска. Если свободной RAM меньше заданного запаса три проверки подряд, защита остановит сервер. Она продолжит работать после закрытия этого окна.", style="Muted.TLabel", wraplength=780).grid(row=3, column=0, columnspan=2, sticky="w", pady=(0, 12))
        ttk.Label(memory_page, text="1 ГБ здесь = 1024 МиБ. Резерв VRAM — цель при запуске, а не запрет другим приложениям занимать память.", style="Muted.TLabel", wraplength=780).grid(row=4, column=0, columnspan=2, sticky="w")

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
        settings_content.bind("<Configure>", lambda _event: settings_canvas.configure(scrollregion=settings_canvas.bbox("all")))
        settings_canvas.bind("<Configure>", lambda event: settings_canvas.itemconfigure(settings_window, width=event.width))
        settings = ttk.LabelFrame(settings_content, text="Параметры запуска", padding=10)
        settings.grid(row=0, column=0, sticky="new", pady=4)
        for col in (1, 3):
            settings.columnconfigure(col, weight=1)

        self._entry(settings, 0, 0, "Контекст (токены)", self.context_var)
        self._combo(settings, 0, 2, "KV cache", self.kv_var, ("int8", "q4_0", "k8v4"))
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

        buttons = ttk.Frame(outer)
        buttons.grid(row=4, column=0, sticky="ew", pady=(4, 0))
        for col in range(3):
            buttons.columnconfigure(col, weight=1)
        self.start_button = ttk.Button(buttons, text="Запустить", style="Primary.TButton", command=self._start_server)
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
            text="Настройки сохраняются в профиль модели. После изменений нажмите «Перезапустить».",
            wraplength=800,
        ).grid(row=5, column=0, sticky="w", pady=(6, 0))

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

    def _load_config_list(self) -> None:
        paths = _configs()
        self.display_to_path.clear()
        for path in paths:
            try:
                data = json.loads(path.read_text(encoding="utf-8-sig"))
                label = f"{data.get('model_name', path.stem)} — {path.name}"
            except (OSError, ValueError):
                label = path.name
            self.display_to_path[label] = path
        labels = list(self.display_to_path)
        self.model_combo.configure(values=labels)
        if labels:
            self.model_var.set(labels[0])
            self._load_config(self.display_to_path[labels[0]])
        else:
            self.status_var.set("Конфигурация не найдена — сначала установите модель через START-HERE.bat")
            for widget in (self.start_button, self.stop_button, self.restart_button, self.save_button):
                widget.configure(state="disabled")

    def _on_model_selected(self, _event=None) -> None:
        path = self.display_to_path.get(self.model_var.get())
        if path:
            self._load_config(path)

    def _load_config(self, path: Path) -> None:
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            args = [str(arg) for arg in data.get("args", [])]
            self.config_path = path
            self.config_data = data
            self.context_var.set(_option_value(args, "--max-context", "32768"))
            self.kv_var.set(_option_value(args, "--kv", "int8"))
            self.kv_resident_var.set(_option_value(args, "--kv-resident", "0"))
            self.spec_var.set(_option_value(args, "--spec", "4"))
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
            self.mmproj_var.set(str(vision.get("mmproj", "")))
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
        context = self._int_value(self.context_var.get(), "Контекст", 1024, 262144)
        if context % 1024:
            raise ValueError("Контекст должен делиться на 1024.")
        kv_resident = self._int_value(self.kv_resident_var.get(), "KV в VRAM", 0, context)
        if kv_resident % 1024:
            raise ValueError("KV в VRAM должен делиться на 1024.")
        spec = self._int_value(self.spec_var.get(), "Spec decoding", 1, 16)
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
        if kv not in ("int8", "q4_0", "k8v4"):
            raise ValueError("KV cache: выберите int8, q4_0 или k8v4.")
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
            args.extend(("--resident-budget-gib", f"{total_ram_gib - ram_reserve_gib:.6f}"))
            launcher_env["STRATA_RESIDENT_HEADROOM_GIB"] = f"{ram_reserve_gib:g}"
        elif old_budget and not old_reserve:
            args.extend(("--resident-budget-gib", old_budget))
        elif self.resident_var.get():
            args.append("--resident-experts")
        if old_reserve and not ram_reserve_gib:
            launcher_env = {name: value for name, value in launcher_env.items()
                            if name.upper() != "STRATA_RESIDENT_HEADROOM_GIB"}

        data = dict(self.config_data)
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

    def _refresh_status(self) -> None:
        if psutil is not None:
            memory = psutil.virtual_memory()
            self.memory_summary.configure(text=f"RAM сейчас: свободно {memory.available / GIB:.1f} из {memory.total / GIB:.1f} ГБ")
        if self.config_path is None:
            self.after(2000, self._refresh_status)
            return
        if psutil is None:
            self.status_var.set("Не найден psutil — управление запущенным процессом недоступно")
            self.status_label.configure(foreground=THEME["danger"])
            self.start_button.configure(state="disabled")
            self.stop_button.configure(state="disabled")
            self.restart_button.configure(state="disabled")
            self.web_button.configure(state="normal" if _health(self._selected_port()) else "disabled")
            self.after(3000, self._refresh_status)
            return
        port = self._selected_port()
        matches = _server_processes(self.config_path)
        for proc in matches:
            port = _port_in_command(proc, port)
            break
        health = _health(port)
        if health and health.get("service") == "strata" and health.get("loaded"):
            if matches:
                text, color = "●  Работает", THEME["accent"]
            else:
                text, color = "●  Strata работает на этом порту", THEME["accent"]
            self.web_button.configure(state="normal")
            if matches and self._pending_browser_port == port:
                self._pending_browser_port = None
                if self.config_data.get("open_browser") is not False:
                    webbrowser.open(_web_url(port))
        elif matches:
            text, color = "●  Запускается — модель загружается", THEME["warning"]
            self.web_button.configure(state="disabled")
        elif health:
            text, color = "Порт занят другим сервисом", THEME["danger"]
            self.web_button.configure(state="disabled")
        else:
            text, color = "○  Остановлен", THEME["muted"]
            try:
                guard = json.loads(_guard_path(self.config_path).read_text(encoding="utf-8"))
                if guard.get("status") == "stopped_for_ram":
                    text, color = f"Остановлен защитой RAM (резерв {guard['reserve_gib']:g} ГБ)", THEME["danger"]
            except (OSError, ValueError, KeyError):
                pass
            self.web_button.configure(state="disabled")
        self.status_var.set(text)
        self.status_label.configure(foreground=color)
        busy = self._busy
        self.start_button.configure(state="disabled" if busy or matches or health else "normal")
        self.stop_button.configure(state="normal" if not busy and matches else "disabled")
        self.restart_button.configure(state="normal" if not busy and matches else "disabled")
        self.after(2000, self._refresh_status)

    def _start_server(self, save: bool = True) -> None:
        if self.config_path is None or self._busy:
            return
        data = self._save_settings(silent=True) if save else self.config_data
        if data is None:
            return
        if _server_processes(self.config_path):
            self.status_var.set("Сервер уже запускается или работает — используйте «Перезапустить».")
            return
        health = _health(int(data.get("port", 8080)))
        if health:
            messagebox.showerror("Порт занят", "На выбранном порту уже отвечает другой сервер.", parent=self)
            return
        if not PYTHON.exists() or not SERVER.exists():
            messagebox.showerror("Не найден Strata", "Не найдены .venv или serve/server.py в папке Strata.", parent=self)
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
                     "--port", str(data.get("port", 8080))],
                    cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=flags, close_fds=True, env=process_env,
                )
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

    def _stop_server(self, quiet: bool = False) -> bool:
        if self.config_path is None:
            return False
        if psutil is None:
            if not quiet:
                messagebox.showerror("Не удалось остановить Strata", "В окружении Strata не найден psutil.", parent=self)
            return False
        roots = _server_roots(self.config_path)
        if not roots:
            return True
        self._busy = True
        self.status_var.set("Останавливаю Strata…")
        self.update_idletasks()
        try:
            for proc in roots:
                if os.name == "nt":
                    result = subprocess.run(
                        ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                        capture_output=True, text=True, timeout=15,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                    if result.returncode and psutil.pid_exists(proc.pid):
                        raise RuntimeError(result.stderr.strip() or "Windows не остановил процесс Strata.")
                else:
                    descendants = proc.children(recursive=True)
                    for child in reversed(descendants):
                        child.terminate()
                    proc.terminate()
                    _, alive = psutil.wait_procs([*descendants, proc], timeout=5)
                    for child in alive:
                        child.kill()
            return True
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            if not quiet:
                messagebox.showerror("Не удалось остановить Strata", str(exc), parent=self)
            return False
        finally:
            self._busy = False
            self.after(300, self._refresh_status)

    def _restart_server(self) -> None:
        data = self._save_settings(silent=True)
        if data is None:
            return
        if not _server_processes(self.config_path):
            self._start_server(save=False)
            return
        if not self._stop_server():
            return
        self.status_var.set("Перезапускаю Strata…")
        self.after(1200, lambda: self._start_server(save=False))

    def _open_web(self) -> None:
        port = self._selected_port()
        if self.config_path is not None:
            matches = _server_processes(self.config_path)
            if matches:
                port = _port_in_command(matches[0], port)
        health = _health(port)
        if not health or health.get("service") != "strata":
            messagebox.showinfo("Strata", "Веб-интерфейс станет доступен после загрузки модели.", parent=self)
            return
        webbrowser.open(_web_url(port))

    def _open_log(self) -> None:
        if not LOG_PATH.exists():
            messagebox.showinfo("Журнал Strata", "Журнал появится после первого запуска через приложение.", parent=self)
            return
        try:
            os.startfile(str(LOG_PATH))
        except (AttributeError, OSError) as exc:
            messagebox.showerror("Журнал Strata", str(exc), parent=self)


if __name__ == "__main__":
    if "--memory-guard" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--memory-guard", action="store_true")
        parser.add_argument("--pid", type=int, required=True)
        parser.add_argument("--created", type=float, required=True)
        parser.add_argument("--ram-reserve-gib", type=float, required=True)
        parser.add_argument("--state", type=Path, required=True)
        options = parser.parse_args()
        _memory_guard(options.pid, options.created, options.ram_reserve_gib, options.state)
    else:
        StrataControl().mainloop()
