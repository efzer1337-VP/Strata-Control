"""Native server engine adapters. Installation never downloads model weights.

llama CLI contract: https://github.com/ggml-org/llama.cpp/tree/master/tools/server
Official assets: https://github.com/ggml-org/llama.cpp/releases
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import string
import subprocess
import tempfile
import threading
import urllib.request
import uuid
import zipfile

import strata_control_settings as control_settings

ROOT = Path(__file__).resolve().parent
STATE = ROOT / ".strata-control"
REGISTRY = STATE / "engines.json"
POLY_ROOT = ROOT / "Strata-data" / "polystrata"
POLY_EXE = POLY_ROOT / "engine-poly" / "strata-poly.exe"
POLY_SERVER = POLY_ROOT / "serve" / "server.py"
POLY_CONFIG = POLY_ROOT / "strata-ornith-uncensored-q3.json"
MAX_ARCHIVE = 3 * 1024 ** 3
MAX_EXPANDED = 8 * 1024 ** 3
_LOCK = threading.RLock()


def _read() -> dict:
    if not REGISTRY.exists():
        return {}
    value = json.loads(REGISTRY.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("Некорректный реестр движков: engines.json")
    return value


def _write(value: dict) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    temporary = STATE / ("engines-" + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, REGISTRY)
    finally:
        temporary.unlink(missing_ok=True)


def engines() -> dict:
    """Return display records including built-in Strata and installable llama.cpp."""
    with _LOCK:
        value = _read()
    value.pop("strata", None)
    result = {"strata": {"name": "Strata", "kind": "strata", "exe": str(ROOT / "engine" / "strata.exe"),
                         "installed": (ROOT / "serve" / "server.py").is_file() and (ROOT / "engine" / "strata.exe").is_file()},
              "llama.cpp": {"name": "llama.cpp", "kind": "llama.cpp", "exe": "", "installed": False},
              "polystrata": {"name": "PolyStrata", "kind": "polystrata", "exe": str(POLY_EXE),
                              "installed": POLY_EXE.is_file() and POLY_SERVER.is_file()}}
    for key, record in value.items():
        if not isinstance(record, dict):
            continue
        result[key] = dict(record, installed=Path(record.get("exe", "")).is_file())
    return result


def _validate_exe(exe) -> Path:
    path = Path(exe).expanduser().resolve()
    if not path.is_file() or path.suffix.lower() != ".exe":
        raise ValueError("Выберите существующий EXE сервера.")
    return path


def register_engine(name: str, exe, kind: str = "custom") -> str:
    """Register an already installed native Windows server without running it."""
    name = str(name).strip()
    if not name:
        raise ValueError("Укажите название движка.")
    if kind not in {"llama.cpp", "custom"}:
        raise ValueError("Тип движка должен быть llama.cpp или custom.")
    path = _validate_exe(exe)
    with _LOCK:
        records = _read()
        if kind == "llama.cpp":
            first = records.get("llama.cpp", {})
            key = "llama.cpp" if not first or str(first.get("exe", "")).casefold() == str(path).casefold() else \
                "llama.cpp-" + hashlib.sha256(str(path).lower().encode()).hexdigest()[:12]
        else:
            key = "custom-" + hashlib.sha256(str(path).lower().encode()).hexdigest()[:12]
        if kind == "custom" and name.casefold() in {"strata", "llama.cpp", "polystrata"}:
            raise ValueError("Это название зарезервировано для встроенного движка.")
        if any(record_key != key and isinstance(record, dict) and str(record.get("name", "")).casefold() == name.casefold()
               for record_key, record in records.items()):
            raise ValueError("Движок с таким названием уже существует. Выберите другое название.")
        records[key] = {"name": name, "kind": kind, "exe": str(path)}
        _write(records)
    return key


def _notify(progress, message):
    if progress:
        progress(message)


def _remove_staging(path: Path) -> None:
    resolved = path.resolve()
    boundary = STATE.resolve()
    if (resolved == boundary or not resolved.is_relative_to(boundary)
            or not boundary.is_relative_to(ROOT.resolve())
            or not path.name.startswith(("extract-", "install-"))):
        raise ValueError("Папка временной установки выходит за допустимые границы.")
    shutil.rmtree(path)


def _extract(archive: Path, destination: Path) -> None:
    """Reject traversal, symlinks, NTFS streams and decompression bombs."""
    if archive.stat().st_size > MAX_ARCHIVE:
        raise ValueError("ZIP больше допустимых 3 ГБ.")
    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.infolist()
        if len(entries) > 30000 or sum(item.file_size for item in entries) > MAX_EXPANDED:
            raise ValueError("Распакованный движок превышает допустимый размер.")
        seen = set()
        for item in entries:
            relative = PurePosixPath(item.filename.replace("\\", "/"))
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise ValueError("Небезопасный путь внутри ZIP.")
            if any(":" in part or part.endswith((".", " ")) or
                   re.fullmatch(r"(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", part)
                   for part in relative.parts):
                raise ValueError("Недопустимое имя файла внутри ZIP.")
            if stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError("Символические ссылки в ZIP не поддерживаются.")
            target = destination.joinpath(*relative.parts)
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError("Путь выходит за пределы папки движка.")
            normalized = str(target).casefold()
            if normalized in seen:
                raise ValueError("ZIP содержит повторяющиеся имена файлов.")
            seen.add(normalized)
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(item) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)


def _complete(staging: Path, name: str, kind: str, progress) -> str:
    candidates = sorted(staging.rglob("*.exe"))
    llama_servers = [path for path in candidates if path.name.lower() == "llama-server.exe"]
    if kind == "custom" and len(llama_servers) == 1:
        kind = "llama.cpp"
    if kind == "llama.cpp":
        candidates = [path for path in candidates if path.name.lower() == "llama-server.exe"]
    elif len(candidates) != 1:
        named = [path for path in candidates if "server" in path.stem.lower()]
        if len(named) == 1:
            candidates = named
    if len(candidates) != 1:
        raise ValueError("Не удалось однозначно выбрать EXE сервера. Распакуйте ZIP и подключите нужный EXE вручную.")
    relative = candidates[0].relative_to(staging)
    destination = STATE / "engines" / ("engine-" + uuid.uuid4().hex)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(destination)
    key = register_engine(name, destination / relative, kind)
    _notify(progress, "Движок установлен. Можно выбрать модель.")
    return key


def install_zip(path, name: str, kind: str = "custom", progress=None) -> str:
    """Install a user-chosen local ZIP, retaining adjacent DLLs and data files."""
    if kind not in {"custom", "llama.cpp"}:
        raise ValueError("Неподдерживаемый тип движка.")
    STATE.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="extract-", dir=STATE))
    try:
        _notify(progress, "Распаковка движка…")
        _extract(Path(path), staging)
        return _complete(staging, name, kind, progress)
    finally:
        if staging.exists():
            _remove_staging(staging)


def _request(url):
    if not url.startswith("https://"):
        raise ValueError("Загрузка движка требует HTTPS.")
    return urllib.request.Request(url, headers={"User-Agent": "Strata-Control", "Accept": "application/vnd.github+json"})


def _download(asset: dict, path: Path, progress) -> None:
    url = asset["browser_download_url"]
    if not url.startswith("https://github.com/ggml-org/llama.cpp/releases/download/"):
        raise ValueError("Релиз содержит неожиданный адрес загрузки.")
    expected_size = int(asset.get("size", 0))
    if expected_size > MAX_ARCHIVE:
        raise ValueError("Архив движка превышает 3 ГБ.")
    digest = hashlib.sha256()
    count = 0
    next_update = 0
    with urllib.request.urlopen(_request(url), timeout=60) as response, path.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            count += len(chunk)
            if count > MAX_ARCHIVE:
                raise ValueError("Архив движка превышает 3 ГБ.")
            output.write(chunk)
            digest.update(chunk)
            if count >= next_update:
                _notify(progress, f"Скачивание движка: {count // (1024 ** 2)} МБ" +
                        (f" / {expected_size // (1024 ** 2)} МБ" if expected_size else ""))
                next_update = count + 4 * 1024 ** 2
    if expected_size and count != expected_size:
        raise ValueError("Архив скачан не полностью. Повторите установку.")
    expected = asset.get("digest") or ""
    if expected:
        if not expected.startswith("sha256:") or digest.hexdigest() != expected.split(":", 1)[1].lower():
            raise ValueError("Контрольная сумма архива не совпала.")


def install_llama(variant: str = "cuda", progress=None) -> str:
    """Download current official Windows x64 CPU/CUDA server and runtime only."""
    if variant not in {"cuda", "cpu"}:
        raise ValueError("Выберите cuda или cpu.")
    _notify(progress, "Поиск официального релиза llama.cpp…")
    with urllib.request.urlopen(_request("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"), timeout=30) as response:
        release = json.loads(response.read(8 * 1024 * 1024))
    # GitHub's /latest can point at a versioned source-only release while the
    # current numbered build has the Windows binaries. Select its actual assets.
    if not any(a.get("name", "").lower().endswith(".zip") and "win" in a.get("name", "").lower()
               for a in release.get("assets", [])):
        with urllib.request.urlopen(_request("https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10"), timeout=30) as response:
            recent = json.loads(response.read(16 * 1024 * 1024))
        release = next((item for item in recent if not item.get("draft") and
                        any(a.get("name", "").lower().endswith(".zip") and "win" in a.get("name", "").lower()
                            for a in item.get("assets", []))), release)
    assets = [asset for asset in release.get("assets", []) if asset.get("name", "").lower().endswith(".zip")]
    def matching(asset):
        name = asset["name"].lower()
        return ("win" in name and "x64" in name and "arm" not in name and "cudart" not in name
                and (("cuda" in name) if variant == "cuda" else not any(label in name for label in ("cuda", "vulkan", "sycl", "hip", "opencl"))))
    choices = [asset for asset in assets if matching(asset)]
    if not choices:
        raise ValueError("В официальном релизе нет подходящего Windows x64 ZIP. Подключите локальный EXE или ZIP.")
    # Prefer the newest CUDA generation or AVX2 CPU variant when multiple builds exist.
    def score(asset):
        name = asset["name"].lower()
        cuda = re.search(r"cuda[-_]?([0-9]+)(?:[._]([0-9]+))?", name)
        return (int(cuda[1]) if cuda else 0, int(cuda[2] or 0) if cuda else 0, "avx2" in name, name)
    if variant == "cuda":
        supported_cuda = None
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=3,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=False)
            match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", result.stdout)
            if result.returncode == 0 and match:
                supported_cuda = (int(match[1]), int(match[2]))
        except (OSError, subprocess.TimeoutExpired):
            pass
        if supported_cuda:
            compatible = [asset for asset in choices if score(asset)[:2] != (0, 0) and score(asset)[:2] <= supported_cuda]
            if not compatible:
                raise ValueError(f"Релиз требует более новый NVIDIA драйвер (доступно CUDA {supported_cuda[0]}.{supported_cuda[1]}). Обновите драйвер или выберите CPU / локальный ZIP.")
            choices = compatible
            _notify(progress, f"Драйвер поддерживает CUDA {supported_cuda[0]}.{supported_cuda[1]}. Выбираем совместимый архив…")
        else:
            conservative = [asset for asset in choices if score(asset)[0] == 12]
            if conservative:
                choices = conservative
                _notify(progress, "Версия NVIDIA драйвера не определена. Выбран CUDA 12; совместимость зависит от установленного драйвера.")
            else:
                _notify(progress, "Версия NVIDIA драйвера не определена и CUDA 12 нет в релизе. Выбран текущий CUDA архив; может потребоваться обновление драйвера.")
    selected = max(choices, key=score)
    downloads = [selected]
    cuda_label = re.search(r"cuda[-_]?[0-9]+(?:[._][0-9]+)?", selected["name"].lower())
    if variant == "cuda" and cuda_label:
        companions = [asset for asset in assets if "cudart" in asset["name"].lower() and "x64" in asset["name"].lower()
                      and cuda_label[0] in asset["name"].lower()]
        if companions:
            downloads.append(companions[0])
    STATE.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="install-", dir=STATE))
    unpacked = staging / "files"
    unpacked.mkdir()
    try:
        for index, asset in enumerate(downloads):
            archive = staging / f"asset-{index}.zip"
            _download(asset, archive, progress)
            _notify(progress, "Проверка и распаковка…")
            _extract(archive, unpacked)
        return _complete(unpacked, f"llama.cpp {release.get('tag_name', '')} ({variant})", "llama.cpp", progress)
    finally:
        if staging.exists():
            _remove_staging(staging)


def _integer(value, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not re.fullmatch(r"[+-]?\d+", str(value).strip()):
        raise ValueError(f"{label}: требуется целое число.")
    number = int(value)
    if not minimum <= number <= maximum:
        raise ValueError(f"{label}: допустимо от {minimum} до {maximum}.")
    return number


def _poly_arg(args: list[str], flag: str, value: str) -> list[str]:
    result = []
    index = 0
    while index < len(args):
        if args[index] == flag:
            index += 2
        else:
            result.append(args[index])
            index += 1
    return result + [flag, value]


def poly_runtime_config(config: dict) -> dict:
    settings = config.get("engine_settings", {})
    base_path = Path(str(settings.get("poly_config", POLY_CONFIG))).expanduser().resolve()
    if not base_path.is_file() or not base_path.is_relative_to(POLY_ROOT.resolve()):
        raise ValueError("Укажите конфигурацию установленной PolyStrata.")
    base = json.loads(base_path.read_text(encoding="utf-8-sig"))
    if not isinstance(base, dict) or not isinstance(base.get("args"), list):
        raise ValueError("Некорректная конфигурация PolyStrata.")
    model = Path(str(settings.get("model", ""))).expanduser().resolve()
    if not model.is_file() or model.suffix.lower() != ".gguf":
        raise ValueError("Выберите существующий GGUF для PolyStrata.")
    args = list(base["args"])
    args = _poly_arg(args, "--model", str(model))
    args = _poly_arg(args, "--max-context", str(_integer(settings.get("context", 32768), "Контекст", 16, 1048576)))
    args = _poly_arg(args, "--conversation-cache-mib", str(_integer(settings.get("cache_ram_mib", 0), "Кэш бесед", 0, 65536)))
    reserve = float(settings.get("vram_reserve_gib", 2))
    if not math.isfinite(reserve) or not 0 <= reserve <= 64:
        raise ValueError("Запас VRAM должен быть от 0 до 64 ГБ.")
    args = _poly_arg(args, "--vram-reserve-mib", str(round(reserve * 1024)))
    if "poly_threads" in settings:
        args = _poly_arg(args, "--threads", str(_integer(settings["poly_threads"], "Потоки CPU", 0, 256)))
    if "poly_kv" in settings:
        kv = str(settings["poly_kv"])
        if kv not in {"fp16", "q8_0"}:
            raise ValueError("PolyStrata KV: выберите fp16 или q8_0.")
        args = _poly_arg(args, "--kv", kv)
    if "poly_window" in settings:
        args = _poly_arg(args, "--window", str(_integer(settings["poly_window"], "Окно MTP", 0, 8)))
    vision_path = str(settings.get("mmproj", "")).strip()
    if vision_path:
        mmproj = Path(vision_path).expanduser().resolve()
        if not mmproj.is_file() or mmproj.suffix.lower() != ".gguf":
            raise ValueError("Файл vision mmproj не найден.")
        encoder = POLY_ROOT / "engine-poly" / "strata-vision.exe"
        if not encoder.is_file():
            raise ValueError("Vision encoder PolyStrata ещё не установлен.")
        vision = base.get("vision", {})
        if not isinstance(vision, dict):
            vision = {}
        vision.update(exe=str(encoder), mmproj=str(mmproj), model=str(model),
                      gpu=bool(settings.get("vision_gpu", False)),
                      max_tokens=_integer(settings.get("vision_tokens", 1024), "Image tokens", 1, 8192))
        if not vision["gpu"]:
            vision["threads"] = max(1, (os.cpu_count() or 8) // 2)
        base["vision"] = vision
    else:
        base.pop("vision", None)
    args = [token for token in args if token != "--vision"]
    base["args"] = args
    base["exe"] = str(POLY_EXE.resolve())
    base["cwd"] = str(POLY_ROOT.resolve())
    base["model_name"] = str(config.get("model_name") or model.stem)
    base["display_name"] = str(config.get("launcher_profile_name") or model.stem)
    base["log"] = str(STATE / "polystrata-engine.log")
    base["port"] = _integer(config.get("port", 8081), "Порт", 1, 65535)
    base["host"] = "127.0.0.1"
    return control_settings.apply_poly_runtime(config, base)


def command_for(config: dict, materialize: bool = False) -> list[str]:
    """Build an external server argv. Strata continues using its existing launcher."""
    key = config.get("launcher_engine", "strata")
    if key == "strata":
        raise ValueError("Для Strata используйте встроенный запуск сервера.")
    engine = engines().get(key)
    if not engine or not engine["installed"]:
        raise ValueError("Сначала установите или подключите выбранный движок.")
    if engine["kind"] == "polystrata":
        runtime = poly_runtime_config(config)
        digest = hashlib.sha256(json.dumps(runtime, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
        path = STATE / "poly-runtime" / (digest + ".json")
        if materialize:
            path.parent.mkdir(parents=True, exist_ok=True)
            serialized = json.dumps(runtime, ensure_ascii=False, indent=2)
            if not path.is_file() or path.read_text(encoding="utf-8") != serialized:
                temporary = path.with_suffix(".tmp")
                temporary.write_text(serialized, encoding="utf-8")
                os.replace(temporary, path)
        python = ROOT / ".venv" / "Scripts" / "python.exe"
        if not python.is_file():
            raise ValueError("Не найдено Python окружение Strata.")
        return [str(python), str(POLY_SERVER), "--engine", "strata", "--config", str(path),
                "--host", "127.0.0.1", "--port", str(runtime["port"])]
    exe = str(_validate_exe(engine["exe"]))
    settings = config.get("engine_settings", {})
    if not isinstance(settings, dict):
        raise ValueError("Некорректные настройки движка.")
    model = Path(str(settings.get("model", config.get("model", "")))).expanduser().resolve()
    if not model.is_file():
        raise ValueError("Выберите существующий файл модели.")
    context = _integer(settings.get("context", 131072), "Контекст", 512, 2097152)
    port = _integer(config.get("port", 8080), "Порт", 1, 65535)
    if str(settings.get("host", config.get("host", "127.0.0.1"))) not in {"127.0.0.1", "localhost"}:
        raise ValueError("Движок должен слушать только 127.0.0.1.")
    extra = settings.get("extra_args", [])
    if not isinstance(extra, list) or any(not isinstance(token, str) or "\x00" in token for token in extra):
        raise ValueError("Дополнительные параметры должны быть списком аргументов.")
    blocked = {"--host", "--port", "--listen", "--bind", "-H", "--address", "--rpc", "--model-url", "-mu", "--hf-repo", "-hf", "-hfr", "--docker-repo", "-dr"}
    custom_bindings = {"--host", "--port", "--listen", "--bind", "-H", "--address"} if engine["kind"] == "custom" else set()
    if any(token.split("=", 1)[0] in blocked - custom_bindings for token in extra):
        raise ValueError("Адрес, порт и скачивание моделей нельзя задавать дополнительными параметрами.")
    if engine["kind"] == "custom":
        if not extra:
            raise ValueError("Укажите аргументы custom сервера с {model}, {context}, {host} и {port}.")
        values = {"model": str(model), "context": str(context), "host": "127.0.0.1", "port": str(port)}
        for index, token in enumerate(extra):
            flag = token.split("=", 1)[0]
            if flag in custom_bindings:
                value = token.split("=", 1)[1] if "=" in token else (extra[index + 1] if index + 1 < len(extra) else "")
                if value != ("{port}" if flag == "--port" else "{host}"):
                    raise ValueError("Адрес custom сервера задавайте как {host}, порт — как {port}.")
        result = []
        for token in extra:
            for _, field, spec, conversion in string.Formatter().parse(token):
                if field is not None and (field not in values or spec or conversion):
                    raise ValueError("Поддерживаются только {model}, {context}, {host}, {port}.")
            result.append(token.format_map(values))
        # A custom adapter must explicitly specify its documented loopback binding.
        if not any("{host}" in token for token in extra):
            raise ValueError("Аргументы custom сервера должны содержать {host} для локального запуска.")
        if any(value in {"0.0.0.0", "::", "[::]"} or value.endswith("=0.0.0.0") for value in result):
            raise ValueError("Удалённый доступ отключён.")
        return [exe, *result]
    gpu = str(settings.get("gpu_layers", "auto"))
    if gpu != "auto":
        gpu = str(_integer(gpu, "GPU layers", 0, 9999))
    threads = _integer(settings.get("threads", max(1, (os.cpu_count() or 4) // 2)), "Потоки", 1, 1024)
    cache_types = {"f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"}
    cache_k, cache_v = settings.get("cache_k", "q8_0"), settings.get("cache_v", "q8_0")
    if cache_k not in cache_types or cache_v not in cache_types:
        raise ValueError("Неподдерживаемый формат KV-кэша llama.cpp.")
    reserve = float(settings.get("vram_reserve_gib", 4))
    if not math.isfinite(reserve) or not 0 <= reserve <= 1024:
        raise ValueError("Резерв VRAM должен быть от 0 до 1024 ГБ.")
    fit = settings.get("fit", True)
    if not isinstance(fit, bool):
        raise ValueError("Автоподбор VRAM должен быть true или false.")
    command = [exe, "--model", str(model), "--ctx-size", str(context), "--host", "127.0.0.1", "--port", str(port),
               "--alias", str(config.get("profile_name", config.get("launcher_profile_name", config.get("name", model.stem)))),
               "--gpu-layers", gpu, "--threads", str(threads), "--cache-type-k", cache_k, "--cache-type-v", cache_v,
               "--flash-attn", "on", "--fit", "on" if fit else "off", "--fit-target", str(math.ceil(reserve * 1024))]
    optional = (("cpu_moe_layers", "--n-cpu-moe", "Слои MoE в RAM", 0, 256),
                ("parallel", "--parallel", "Параллельные слоты", 1, 64),
                ("cache_ram_mib", "--cache-ram", "Кэш RAM", 0, 1048576),
                ("reasoning_budget", "--reasoning-budget", "Лимит размышления", -1, 1048576),
                ("ubatch_size", "--ubatch-size", "Физический пакет", 1, 8192))
    for key, flag, label, minimum, maximum in optional:
        if key in settings:
            value = _integer(settings[key], label, minimum, maximum)
            if key != "cpu_moe_layers" or value:
                command.extend([flag, str(value)])
    if "top_k" in settings:
        command.extend(["--top-k", str(_integer(settings["top_k"], "Top K", 0, 1000))])
    for key, flag, label, maximum in (("temperature", "--temp", "Температура", 2),
                                       ("top_p", "--top-p", "Top P", 1),
                                       ("min_p", "--min-p", "Min P", 1)):
        if key in settings:
            try:
                value = float(str(settings[key]).replace(",", "."))
            except (TypeError, ValueError):
                raise ValueError(f"{label}: требуется число.") from None
            if not math.isfinite(value) or not 0 <= value <= maximum:
                raise ValueError(f"{label}: допустимо от 0 до {maximum}.")
            command.extend([flag, str(value)])
    if "reasoning_budget_message" in settings:
        message = settings["reasoning_budget_message"]
        if not isinstance(message, str) or len(message) > 500 or "\n" in message or "\r" in message:
            raise ValueError("Сообщение лимита размышления должно быть одной строкой до 500 символов.")
        if message:
            command.extend(["--reasoning-budget-message", message])
    mmproj = settings.get("mmproj", "")
    if mmproj:
        projector = Path(mmproj).expanduser().resolve()
        if not projector.is_file():
            raise ValueError("Файл mmproj не найден.")
        command.extend(["--mmproj", str(projector)])
    if "thinking" in settings:
        if not isinstance(settings["thinking"], bool):
            raise ValueError("Режим размышления должен быть true или false.")
        command.extend(["--chat-template-kwargs", json.dumps({"enable_thinking": settings["thinking"]})])
    managed = {command[index] for index in range(1, len(command), 2)}
    if any(token.split("=", 1)[0] in managed or token.split("=", 1)[0] in {"-m", "-c", "-ngl", "-t", "-ctk", "-ctv", "-a", "-fit", "-fitt"} for token in extra):
        raise ValueError("Меняйте основные параметры через поля профиля.")
    return command + extra
