"""Loopback-only responsive web control for Strata and local model engines."""
from __future__ import annotations
import argparse
import copy
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import socket
import struct
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
import urllib.request
import uuid
import webbrowser
import traceback
import strata_control as control
import strata_control_engines as engines
import strata_control_settings as control_settings
import strata_memory_cleanup as memory_cleanup
from strata_control_presets import presets

ROOT = Path(__file__).resolve().parent
VERSION = '4.3'
INSTANCE = ROOT / '.strata-control' / 'web-instance.json'
FLAGS = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
TOKEN = secrets.token_urlsafe(30)

class Value:
    def __init__(self, value): self.value = value
    def get(self, *args): return self.value

def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, path)

def log_tail():
    try:
        with control.LOG_PATH.open('rb') as stream:
            stream.seek(max(0, stream.seek(0, 2) - 65536))
            return stream.read().decode('utf-8', 'replace')
    except OSError: return ''

def validate_contract(data):
    if not isinstance(data, dict): raise ValueError('Профиль должен быть JSON объектом.')
    for key in ('engine_settings','launcher_env','launcher_memory'):
        if not isinstance(data.get(key, {}), dict): raise ValueError(key + ': требуется JSON объект.')
    if data.get('sampling') is not None and not isinstance(data['sampling'], dict):
        raise ValueError('sampling: требуется JSON объект.')
    if not isinstance(data.get('args', []), list) or any(not isinstance(arg, str) for arg in data.get('args', [])):
        raise ValueError('args: требуется список строк.')
    for key in ('launcher_engine','launcher_profile_name','profile_description'):
        if key in data and not isinstance(data[key], str): raise ValueError(key + ': требуется строка.')
    names = set()
    for name, value in data.get('launcher_env', {}).items():
        if not control.re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name) or '\0' in str(value) or name.upper() in names:
            raise ValueError('Переменная окружения: некорректное имя, значение или повтор.')
        names.add(name.upper())

@lru_cache(maxsize=32)
def model_contract(path, size, modified):
    """Compare prepared assets across quantizations without reading tensor data."""
    from tools.gguf_reader import GGUFFile
    header = GGUFFile(Path(path))
    architecture = header.metadata.get('general.architecture')
    if not architecture: raise ValueError('В GGUF отсутствует general.architecture.')
    metadata = {key:value for key,value in header.metadata.items()
                if key == 'general.architecture' or key.startswith(('tokenizer.', architecture + '.'))}
    shapes = sorted((tensor.name,tensor.shape) for tensor in header.tensors)
    return hashlib.sha256(json.dumps([metadata,shapes],sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def compatible_poly_models(model, prepared):
    signatures = []
    for value in (model, prepared):
        path = Path(value).expanduser().resolve()
        try:
            stat = path.stat()
            signatures.append(model_contract(str(path),stat.st_size,stat.st_mtime_ns))
        except (OSError,ValueError,KeyError,TypeError,OverflowError,struct.error) as exc:
            raise ValueError('Не удалось сверить GGUF с подготовленной моделью PolyStrata. Укажите JSON этой модели: ' + str(exc)) from exc
    return signatures[0] == signatures[1]

@lru_cache(maxsize=32)
def _poly_model_capabilities(path, size, modified):
    from tools.gguf_reader import GGUFFile
    metadata = GGUFFile(Path(path)).metadata
    architecture = metadata.get('general.architecture')
    supported = None
    if architecture:
        supported = architecture == 'qwen35moe'
        if supported:
            prefix = architecture + '.attention.'
            heads = metadata.get(prefix + 'head_count_kv')
            dimension = metadata.get(prefix + 'key_length')
            if dimension is None:
                count = metadata.get(prefix + 'head_count')
                embedding = metadata.get(architecture + '.embedding_length')
                if isinstance(count,int) and count > 0 and isinstance(embedding,int): dimension = embedding // count
            supported = (heads * dimension) % 32 == 0 if isinstance(heads,int) and isinstance(dimension,int) else None
    return {'architecture':architecture, 'q8_supported':supported}

def poly_model_capabilities(data):
    if data.get('launcher_engine') != 'polystrata': return {}
    try:
        path = Path(data.get('engine_settings', {}).get('model','')).expanduser().resolve()
        stat = path.stat()
        return _poly_model_capabilities(str(path),stat.st_size,stat.st_mtime_ns)
    except (OSError,ValueError,KeyError,TypeError,OverflowError,struct.error): return {}

def _legacy_field_values(data):
    args = data.get('args', [])
    option = lambda name, default='': control._option_value(args, name, str(default))
    ext = data.get('engine_settings', {})
    vision = data.get('vision', {}) if isinstance(data.get('vision'), dict) else {}
    external = data.get('launcher_engine', 'strata') != 'strata'
    extras = control._without_options(args, control.MANAGED_OPTIONS)
    result = {
        'model': ext.get('model', '') if external else option('--native'),
        'context': ext.get('context', 131072) if external else option('--max-context', 4096),
        'port': data.get('port', 8080), 'kv': option('--kv', 'fp16'),
        'kv_resident': option('--kv-resident', 0), 'spec': option('--spec', 0),
        'spec_min_p': option('--spec-min-p', 0), 'expert_cache': option('--expert-cache', 'auto'),
        'vram_reserve': ext.get('vram_reserve_gib', 4) if external else float(option('--vram-reserve-mib', 0)) / 1024,
        'ram_reserve': data.get('launcher_memory', {}).get('ram_reserve_gib', 0),
        'resident': control._has_flag(args, '--resident-experts'),
        'mtp_enabled': bool(option('--mtp')), 'mtp': option('--mtp', data.get('launcher_mtp', '')),
        'vision': bool(data.get('vision')), 'mmproj': ext.get('mmproj', '') if external else vision.get('mmproj', data.get('launcher_mmproj', '')),
        'vision_tokens': ext.get('vision_tokens', vision.get('max_tokens', 1024)) if external else vision.get('max_tokens', 1024), 'vision_device': 'GPU' if vision.get('gpu', True) else 'CPU',
        'lazy': bool(data.get('lazy_load')), 'gpu_layers': ext.get('gpu_layers', 'auto'),
        'threads': ext.get('threads', max(1, (os.cpu_count() or 4)//2)),
        'cache_k': ext.get('cache_k', 'q8_0'), 'cache_v': ext.get('cache_v', 'q8_0'),
        'thinking': ext.get('thinking', True),
        'cpu_moe_layers': ext.get('cpu_moe_layers', 0), 'parallel': ext.get('parallel', 1),
        'cache_ram_mib': ext.get('cache_ram_mib', 8192), 'reasoning_budget': ext.get('reasoning_budget', -1),
        'fit': ext.get('fit', True),
        'poly_window': ext.get('poly_window', 8),
        'poly_config': ext.get('poly_config', str(engines.POLY_CONFIG)),
        'poly_threads': ext.get('poly_threads', max(1,(os.cpu_count() or 4)//2)),
        'poly_kv': ext.get('poly_kv', 'fp16'),
        'vision_gpu': ext.get('vision_gpu', False),
        'ubatch_size': ext.get('ubatch_size', 512),
        'temperature': ext.get('temperature', 0.8), 'top_p': ext.get('top_p', 0.95),
        'top_k': ext.get('top_k', 40), 'min_p': ext.get('min_p', 0.05),
        'reasoning_budget_message': ext.get('reasoning_budget_message', ''),
        'extra_args': '\n'.join(ext.get('extra_args', [])) if external else '\n'.join(extras),
        'environment': '\n'.join(f'{k}={v}' for k,v in data.get('launcher_env', {}).items())}
    if data.get('launcher_engine') == 'polystrata':
        for key, default in {'context':32768,'vram_reserve_gib':2,'cache_ram_mib':0,'temperature':1.0,'top_k':20}.items():
            result['vram_reserve' if key == 'vram_reserve_gib' else key] = ext.get(key, default)
        base_path = Path(ext.get('poly_config', engines.POLY_CONFIG)).resolve()
        if base_path.is_relative_to(engines.POLY_ROOT.resolve()):
            try:
                base_args = json.loads(base_path.read_text(encoding='utf-8-sig')).get('args', [])
                for key, flag in [('poly_threads','--threads'),('poly_window','--window'),('poly_kv','--kv')]:
                    if key not in ext: result[key] = control._option_value(base_args, flag, str(result[key]))
            except (OSError,ValueError,AttributeError): pass
    if result['kv'] == 'q4': result['kv'] = 'q4_0'
    return result

def field_values(data):
    result = _legacy_field_values(data)
    if data.get('launcher_engine', 'strata') == 'strata':
        extras, _ = control_settings.split_args(result['extra_args'].splitlines())
        result['extra_args'] = '\n'.join(extras)
    result.update(control_settings.field_values(data))
    return result

DESCRIPTIONS = {
 'model': ('Модель', 'Локальный файл модели', 'path', 'Модель'),
 'context': ('Контекст', 'Максимальное число токенов', 'number', 'Память'),
 'port': ('Порт', 'Порт локального API', 'number', 'Сервер'),
 'kv': ('KV cache', 'Формат хранения кэша контекста', 'select', 'Память'),
 'kv_resident': ('KV в VRAM, токенов', '0 оставляет весь KV в VRAM. Streaming хранит полную копию в RAM; движок поднимает ненулевой запрос минимум до 20480 токенов.', 'number', 'Память'),
 'spec': ('Spec decoding', 'Число черновых токенов', 'number', 'Скорость'),
 'spec_min_p': ('Spec min-p', 'Порог вероятности черновых токенов', 'float', 'Скорость'),
 'expert_cache': ('Expert cache', 'auto сохраняет заданный запас VRAM', 'text', 'Память'),
 'vram_reserve': ('Запас VRAM', 'ГБ для других программ; не жёсткий лимит', 'float', 'Память'),
 'ram_reserve': ('Запас RAM', 'ГБ для ПК; защита остановит сервер при нехватке', 'float', 'Память'),
 'resident': ('Резидентные эксперты', 'Хранить эксперты в оперативной памяти', 'bool', 'Память'),
 'mtp_enabled': ('MTP', 'Использовать подготовленный черновой runtime', 'bool', 'Скорость'),
 'mtp': ('MTP runtime', 'Папка с experts.bin', 'path', 'Скорость'),
 'vision': ('Vision', 'Обработка изображений', 'bool', 'Изображения'),
 'mmproj': ('mmproj', 'Проектор изображений для выбранной модели', 'path', 'Изображения'),
 'vision_tokens': ('Image tokens', 'Максимум токенов изображения', 'number', 'Изображения'),
 'vision_device': ('Vision device', 'Устройство обработки изображений', 'select', 'Изображения'),
 'lazy': ('Lazy загрузка', 'Загрузить модель при первом запросе', 'bool', 'Сервер'),
 'gpu_layers': ('GPU layers', 'auto — автоматическое размещение', 'text', 'Память'),
 'threads': ('Потоки CPU', 'Число потоков обработки', 'number', 'Скорость'),
 'cache_k': ('KV keys', 'Квантование ключей KV cache', 'select', 'Память'),
 'cache_v': ('KV values', 'Квантование значений KV cache', 'select', 'Память'),
 'thinking': ('Размышление', 'Использовать режим thinking модели', 'bool', 'Модель'),
 'cpu_moe_layers': ('MoE в RAM', 'Число первых слоёв экспертов в RAM; 0 — все на GPU', 'number', 'Память'),
 'parallel': ('Параллельные слоты', 'Для одного пользователя обычно 1', 'number', 'Сервер'),
 'cache_ram_mib': ('Кэш RAM, МиБ', 'Предел кэша промптов llama.cpp', 'number', 'Память'),
 'reasoning_budget': ('Лимит размышления', '-1 — без ограничения, 0 — сразу ответ', 'number', 'Модель'),
 'fit': ('Автоподбор VRAM', 'Включить автоматическую подгонку llama.cpp', 'bool', 'Память'),
 'ubatch_size': ('Физический пакет', 'Размер пакета обработки промпта', 'number', 'Скорость'),
 'temperature': ('Температура', 'Для нового Ornith авторы рекомендуют 1,0', 'float', 'Модель'),
 'top_p': ('Top P', 'Порог вероятности для выборки токенов', 'float', 'Модель'),
 'top_k': ('Top K', 'Число кандидатов для выборки токенов', 'number', 'Модель'),
 'min_p': ('Min P', '0 отключает фильтр', 'float', 'Модель'),
 'reasoning_budget_message': ('Фраза после размышления', 'Одна строка при достижении лимита', 'text', 'Модель'),
 'poly_window': ('Окно MTP', '0 выбирает автоматически; 1 отключает угадывание; максимум 8', 'number', 'Скорость'),
 'poly_config': ('Конфигурация PolyStrata', 'Подготовленный JSON с токенизатором и профилем экспертов именно этой модели', 'path', 'Модель'),
 'poly_threads': ('Потоки экспертов CPU', '0 выбирает автоматически; больше 0 задаёт число потоков экспертов вне VRAM', 'number', 'Скорость'),
 'poly_kv': ('KV cache', 'fp16; q8_0 доступен только qwen35moe с поддерживаемой размерностью', 'select', 'Память'),
 'vision_gpu': ('Vision на GPU', 'Быстрее обработка изображений, но меньше VRAM для экспертов', 'bool', 'Изображения'),
 'extra_args': ('Дополнительные параметры', 'Один аргумент на строку; пути без кавычек', 'textarea', 'Дополнительно'),
 'environment': ('Переменные окружения', 'ИМЯ=значение, одна переменная на строку', 'textarea', 'Дополнительно')}

def fields(data):
    values = field_values(data)
    engine = data.get('launcher_engine', 'strata')
    external = engine != 'strata'
    keys = ['model','context','port','vram_reserve','ram_reserve']
    kind = engines.engines().get(engine, {}).get('kind', 'custom' if external else 'strata')
    if engine == 'polystrata':
        keys += ['poly_config','poly_kv','cache_ram_mib','poly_threads','poly_window','mmproj','vision_gpu','vision_tokens','temperature','top_p','top_k','environment']
    elif kind == 'custom':
        keys = ['model','context','port','ram_reserve','extra_args','environment']
    else:
        keys += ['gpu_layers','cpu_moe_layers','threads','parallel','cache_k','cache_v','cache_ram_mib','ubatch_size','thinking','reasoning_budget','reasoning_budget_message','temperature','top_p','top_k','min_p','fit','mmproj'] if external else ['kv','kv_resident','expert_cache','resident','mtp_enabled','mtp','spec','spec_min_p','vision','mmproj','vision_tokens','vision_device','lazy']
        keys += ['extra_args','environment']
    choices = {'kv':['fp16','int8','q4_0','k8v4'], 'poly_kv':['fp16','q8_0'], 'vision_device':['GPU','CPU'], 'cache_k':['f32','f16','bf16','q8_0','q4_0','q4_1','iq4_nl','q5_0','q5_1'], 'cache_v':['f32','f16','bf16','q8_0','q4_0','q4_1','iq4_nl','q5_0','q5_1']}
    if poly_model_capabilities(data).get('q8_supported') is False: choices['poly_kv'] = ['fp16']
    result = []
    optional = control_settings.keys(data)
    for key in keys:
        if key in optional: continue
        label, desc, kind, section = DESCRIPTIONS[key]
        if engine == 'polystrata' and key == 'cache_ram_mib':
            label, desc = 'Кэш бесед, МиБ', 'Память для нескольких разговоров; 0 отключает'
        if key in {'vram_reserve', 'ram_reserve'}:
            label = label + ', ГиБ'
            desc = desc.replace('ГБ', 'ГиБ')
        row = dict(key=key,label=label,description=desc,type=kind,section=section,value=values[key])
        if key in choices: row['choices'] = choices[key]
        if engine == 'polystrata':
            limits = {'context':(16,1048576),'poly_threads':(0,256),'poly_window':(0,8)}
            if key in limits: row['min'],row['max'] = limits[key]
        result.append(row)
    return result + control_settings.fields(data)

def _updated_legacy(path, data, changes, validate=False):
    validate_contract(data)
    if not isinstance(changes, dict): raise ValueError('Некорректные параметры профиля.')
    changes = dict(changes)
    description = changes.pop('profile_description', None)
    values = _legacy_field_values(data)
    if validate:
        control.StrataControl._read_environment(SimpleNamespace(environment_vars=Value(str(values['environment']))))
    unknown = set(changes) - set(values)
    if unknown: raise ValueError('Неизвестные параметры: ' + ', '.join(unknown))
    for key, value in changes.items():
        if isinstance(values[key], bool) and not isinstance(value, bool):
            raise ValueError(f'{key}: требуется true или false.')
        if not isinstance(value, (str,int,float,bool)): raise ValueError('Некорректное значение параметра.')
    # A form submits displayed fallback values too. Do not turn those into new
    # runtime flags when an existing profile never specified them.
    effective = {key: value for key, value in changes.items() if str(value) != str(values[key])}
    incompatible = set(effective) - {field['key'] for field in fields(data)}
    if incompatible: raise ValueError('Движок не поддерживает параметры: ' + ', '.join(sorted(incompatible)))
    if not effective and not validate:
        result = copy.deepcopy(data)
        if description is not None: result['profile_description'] = str(description)[:2000]
        return result
    values.update(changes)
    if control.psutil is None:
        raise ValueError('Для управления памятью требуется psutil из окружения Strata.')
    total = control.psutil.virtual_memory().total / control.GIB
    reserve = control._reserve_gib(str(values['ram_reserve']), 'Резерв RAM', total)
    if reserve and total - reserve < 1:
        raise ValueError('После резерва RAM должен оставаться хотя бы 1 ГиБ.')
    if data.get('launcher_engine') == 'polystrata':
        result = copy.deepcopy(data)
        settings = result.setdefault('engine_settings', {})
        limits = {'context':(16,1048576), 'cache_ram_mib':(0,65536), 'poly_window':(0,8),
                  'poly_threads':(0,256), 'vision_tokens':(1,8192), 'top_k':(1,64)}
        for key in (values if validate else effective):
            if key in limits:
                settings[key] = engines._integer(values[key], DESCRIPTIONS[key][0], *limits[key])
            elif key in {'model','mmproj','poly_config','poly_kv','vision_gpu','temperature','top_p'}:
                settings[key] = values[key]
            elif key == 'vram_reserve':
                settings['vram_reserve_gib'] = control._reserve_gib(str(values[key]), 'Резерв VRAM', 64)
        if settings.get('poly_kv', 'fp16') not in {'fp16', 'q8_0'}: raise ValueError('PolyStrata KV: выберите fp16 или q8_0.')
        for key, fallback, low, high in [('temperature',1.0,0,None),('top_p',.95,0,1)]:
            value = float(settings.get(key, fallback))
            if not math.isfinite(value) or value < low or (key == 'top_p' and value == 0) or (high is not None and value > high):
                raise ValueError('Проверьте температуру и Top P.')
        result['port'] = engines._integer(values['port'], 'Порт', 1, 65535)
        if 'ram_reserve' in effective: result.setdefault('launcher_memory', {})['ram_reserve_gib'] = reserve
        if 'environment' in effective:
            result['launcher_env'] = control.StrataControl._read_environment(SimpleNamespace(environment_vars=Value(str(values['environment']))))
        if description is not None: result['profile_description'] = str(description)[:2000]
        return result
    # Validation must never invoke nvidia-smi on an HTTP request thread. The
    # monitor replaces this conservative fallback with measured GPU capacity.
    control.GPU_TOTAL_CACHE.setdefault(int(data.get('gpu', 0)), 256.0)
    adapter = SimpleNamespace(config_path=path, config_data=copy.deepcopy(data))
    mapping = {'context':'context_var','port':'port_var','kv':'kv_var','kv_resident':'kv_resident_var','spec':'spec_var','spec_min_p':'spec_min_p_var','expert_cache':'expert_cache_var','vram_reserve':'vram_reserve_var','ram_reserve':'ram_reserve_var','resident':'resident_var','mtp_enabled':'mtp_enabled_var','mtp':'mtp_var','vision':'vision_var','mmproj':'mmproj_var','vision_tokens':'vision_tokens_var','vision_device':'vision_device_var','lazy':'lazy_var','model':'external_model_var','gpu_layers':'external_layers_var','threads':'external_threads_var','cache_k':'external_cache_k_var','cache_v':'external_cache_v_var','thinking':'external_thinking_var','cpu_moe_layers':'external_cpu_moe_var','parallel':'external_parallel_var','cache_ram_mib':'external_cache_ram_var','reasoning_budget':'external_reasoning_budget_var','reasoning_budget_message':'external_budget_message_var','ubatch_size':'external_ubatch_var','temperature':'external_temp_var','top_p':'external_top_p_var','top_k':'external_top_k_var','min_p':'external_min_p_var','fit':'external_fit_var'}
    for key, attr in mapping.items(): setattr(adapter, attr, Value(values[key] if isinstance(values[key],bool) else str(values[key])))
    adapter.extra_args = Value(str(values['extra_args']))
    adapter.external_args = Value(str(values['extra_args']))
    adapter.environment_vars = Value(str(values['environment']))
    adapter._int_value = control.StrataControl._int_value
    adapter._read_extra_args = lambda: control.StrataControl._read_extra_args(adapter)
    adapter._read_environment = lambda: control.StrataControl._read_environment(adapter)
    adapter._updated_external_config = lambda: control.StrataControl._updated_external_config(adapter)
    if data.get('launcher_engine', 'strata') == 'strata' and str(values['model']) != field_values(data)['model']:
        raise ValueError('Для другой Strata модели используйте «Добавить модель»: необходима подготовка экспертов.')
    result = control.StrataControl._updated_config(adapter)
    if data.get('launcher_engine', 'strata') != 'strata':
        mapping = {'vram_reserve':'vram_reserve_gib'}
        settings = result['engine_settings']
        original = data.get('engine_settings', {})
        for key in set(settings) - set(original):
            if key not in {mapping.get(k,k) for k in effective}: settings.pop(key)
    result['launcher_memory'] = dict(data.get('launcher_memory', {}), **result.get('launcher_memory', {}))
    if description is not None: result['profile_description'] = str(description)[:2000]
    return result

def updated(path, data, changes, validate=False):
    validate_contract(data)
    if not isinstance(changes, dict): raise ValueError('Некорректные параметры профиля.')
    values = field_values(data)
    unknown = set(changes) - set(values) - {'profile_description'}
    if unknown: raise ValueError('Неизвестные параметры: ' + ', '.join(sorted(unknown)))
    effective = {key:value for key,value in changes.items()
                 if key == 'profile_description' or str(value) != str(values[key])}
    supported = {field['key'] for field in fields(data)} | {'profile_description'}
    incompatible = set(effective) - supported
    if incompatible: raise ValueError('Движок не поддерживает параметры: ' + ', '.join(sorted(incompatible)))
    optional = control_settings.keys(data)
    advanced = {key:value for key,value in effective.items() if key in optional}
    legacy = {key:value for key,value in effective.items() if key not in optional}
    if 'extra_args' in legacy and data.get('launcher_engine','strata') == 'strata':
        control_settings.assert_no_conflicts(legacy['extra_args'],data)
        _, managed = control_settings.split_args(data.get('args', []))
        legacy['extra_args'] = str(legacy['extra_args']) + ('\n' + '\n'.join(managed) if managed else '')
    if validate:
        # Validate legacy values without adopting its reconstructed defaults.
        _updated_legacy(path,data,{},validate=True)
    result = _updated_legacy(path,data,legacy) if legacy else copy.deepcopy(data)
    return control_settings.apply(result,advanced,validate=validate)

class Controller:
    def __init__(self):
        self.lock = threading.RLock()
        self.operation_lock = threading.Lock()
        self.memory_cleanup_lock = threading.Lock()
        self.tasks = []
        self.profiles = {}
        self.profile_errors = []
        self.selected = ''
        self.selected_engine = 'strata'
        self.engine_profiles = {}
        self.servers = []
        self.processes = {}
        self.status = {'running':False,'ready':False,'port':8080,'log':''}
        self.last_seen = time.monotonic()
        self.reload()
        try:
            stored = json.loads(control.CONTROL_STATE.read_text(encoding='utf-8-sig'))
            self.engine_profiles = stored.get('web_engine_profiles', {})
            if not isinstance(self.engine_profiles, dict): self.engine_profiles = {}
            selected = stored.get('web_profile', '')
            if selected in self.profiles: self.selected = selected
            self.selected_engine = stored.get('web_engine', self.profiles.get(self.selected, (None, {}))[1].get('launcher_engine', 'strata'))
            if self.selected_engine not in engines.engines(): self.selected_engine = 'strata'
            if self.selected and self.profiles[self.selected][1].get('launcher_engine','strata') != self.selected_engine:
                self.selected = ''
        except (OSError, ValueError): pass
        if self.selected: self.selected_engine = self.profiles[self.selected][1].get('launcher_engine','strata')
        threading.Thread(target=self.monitor, daemon=True).start()
    def reload(self):
        profiles = {}
        errors = []
        for path in control._configs():
            try:
                data = json.loads(path.read_text(encoding='utf-8-sig'))
                validate_contract(data)
                profiles[str(path.resolve())] = (path, data)
            except (OSError, ValueError, TypeError) as exc: errors.append(path.name + ': ' + str(exc))
        with self.lock:
            self.profiles = profiles
            self.profile_errors = errors
            if self.selected not in profiles:
                self.selected = next((ident for ident, (_, data) in profiles.items() if data.get('launcher_engine','strata') == self.selected_engine), '')
    def profile(self, ident=None):
        with self.lock:
            ident = ident or self.selected
            if ident not in self.profiles: raise ValueError('Выберите профиль модели.')
            path, data = self.profiles[ident]
            return path, copy.deepcopy(data)
    def select(self, ident):
        _, data = self.profile(ident)
        with self.lock:
            self.selected = ident
            self.selected_engine = data.get('launcher_engine', 'strata')
            self.engine_profiles[self.selected_engine] = ident
            self.persist_selection()
    def persist_selection(self):
        try: stored = json.loads(control.CONTROL_STATE.read_text(encoding='utf-8-sig'))
        except (OSError,ValueError): stored = {}
        stored.update(web_profile=self.selected, web_engine=self.selected_engine, web_engine_profiles=self.engine_profiles)
        write_json(control.CONTROL_STATE, stored)
    def select_engine(self, engine):
        if engine not in engines.engines(): raise ValueError('Неизвестный движок.')
        with self.lock:
            self.selected_engine = engine
            candidates = [ident for ident, (_, data) in self.profiles.items() if data.get('launcher_engine','strata') == engine]
            remembered = self.engine_profiles.get(engine)
            self.selected = remembered if remembered in candidates else next(iter(candidates), '')
            if self.selected: self.engine_profiles[engine] = self.selected
            self.persist_selection()
    def command(self, path, data, materialize=False):
        if data.get('launcher_engine', 'strata') != 'strata': return engines.command_for(data, materialize=materialize)
        return [str(control.PYTHON), str(control.SERVER), '--engine','strata','--config',str(path),'--host','127.0.0.1','--port',str(data.get('port',8080))]
    def preflight(self, path, data):
        errors = []
        try:
            updated(path, data, {}, validate=True)
            if not control.PYTHON.is_file(): raise ValueError('Не найдено Python окружение Strata.')
            if control.psutil is None: raise ValueError('Не найден psutil: управление процессами недоступно.')
            if data.get('launcher_engine','strata') == 'strata':
                details = control.describe_file(control._option_value(data.get('args',[]),'--native'))
                build_path = ROOT / 'engine' / 'BUILD.json'
                if details and build_path.is_file():
                    build = json.loads(build_path.read_text(encoding='utf-8-sig'))
                    backend = build.get('backend', 'cuda' if build.get('cuda') else '')
                    if details.get('nvidia_only') and backend not in {'cuda',''}:
                        raise ValueError(details['quant'] + ' в setup поддерживается только на NVIDIA. Backend: ' + backend)
                    version = tuple(int(part) for part in str(build.get('version','0')).split('.') if part.isdigit())
                    minimum = details.get('engine', (0,1,32) if details.get('budget') else (0,))
                    if version < minimum: raise ValueError('Для этой модели требуется более новый Strata: ' + '.'.join(map(str,minimum)))
                for label, value in [('Движок',data.get('exe','')),('GGUF',control._option_value(data.get('args',[]),'--native')),('Pack',control._option_value(data.get('args',[]),'--pack')),('Токенизатор',data.get('tokenizer',''))]:
                    if not value or not Path(value).exists(): raise ValueError(label + ': путь не найден: ' + str(value))
            else:
                if data.get('launcher_engine') == 'polystrata':
                    settings = data.get('engine_settings', {})
                    if settings.get('poly_kv') == 'q8_0' and poly_model_capabilities(data).get('q8_supported') is False:
                        raise ValueError('PolyStrata q8_0 KV поддерживается только qwen35moe с кратной 32 размерностью KV. Для этой модели выберите fp16.')
                    base = json.loads(Path(settings.get('poly_config', engines.POLY_CONFIG)).read_text(encoding='utf-8-sig'))
                    prepared_model = control._option_value(base.get('args', []), '--model')
                    if (prepared_model and control._normalized(prepared_model) != control._normalized(settings.get('model',''))
                            and not compatible_poly_models(settings.get('model',''), prepared_model)):
                        raise ValueError('GGUF не соответствует подготовленной конфигурации PolyStrata. Выберите JSON этой модели: токенизатор и профиль экспертов нельзя переносить между моделями.')
                    for label, value in [('Токенизатор PolyStrata',base.get('tokenizer')),('Профиль экспертов PolyStrata',control._option_value(base.get('args',[]),'--expert-profile'))]:
                        if value and not Path(value).exists(): raise ValueError(label + ': файл или папка не найдены: ' + value)
                self.command(path, data)
        except (ValueError, TypeError, OSError, KeyError) as exc: errors.append(str(exc))
        return errors
    def preview(self, path, data):
        try:
            tokens = self.command(path,data)
            quote = lambda value: "'" + str(value).replace("'", "''") + "'"
            if any(not control.re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', str(name)) for name in data.get('launcher_env',{})):
                raise ValueError('Некорректное имя переменной окружения.')
            env = '\n'.join(f"$env:{name} = {quote(value)}" for name,value in data.get('launcher_env',{}).items())
            if data.get('launcher_engine','strata') not in {'strata','polystrata'}:
                return (env + '\n' if env else '') + '& ' + ' `\n  '.join(quote(v) for v in tokens)
            # Preview contains the edited profile, not a saved file whose args
            # may differ. The text is only executed when the user copies it.
            runtime = engines.poly_runtime_config(data) if data.get('launcher_engine') == 'polystrata' else data
            profile_json = json.dumps(runtime,ensure_ascii=False,indent=2)
            prefix = "$configPath = Join-Path $env:TEMP ('strata-control-' + [guid]::NewGuid().ToString('N') + '.json')\n$profile = @'\n" + profile_json + "\n'@\n[IO.File]::WriteAllText($configPath, $profile, [Text.UTF8Encoding]::new($false))\n"
            if env: prefix += env + '\n'
            rendered = []
            for index,value in enumerate(tokens):
                rendered.append('$configPath' if index and tokens[index-1] == '--config' else quote(value))
            return prefix + '& ' + ' `\n  '.join(rendered) + '\n\n# RAM-сторож включается кнопкой запуска в Control; ручная команда его не запускает.'
        except ValueError as exc: return '# ' + str(exc)
    def state(self):
        self.last_seen = time.monotonic()
        with self.lock:
            rows = [{'id':ident,'name':data.get('launcher_profile_name',data.get('profile_name',path.stem)),'engine':data.get('launcher_engine','strata'),'model':field_values(data)['model']} for ident,(path,data) in self.profiles.items()]
            selected = self.profiles.get(self.selected)
            status = dict(self.status)
            active = copy.deepcopy(self.servers)
            tasks = copy.deepcopy(self.tasks[-20:])
            ident = self.selected
            engine = self.selected_engine
        if selected:
            path,data = selected
            config = copy.deepcopy(data)
            fs = fields(data)
            command = self.preview(path,data)
            errors = self.preflight(path,data)
            if status.get('profile_id') != ident:
                status = {k:v for k,v in status.items() if k.startswith(('gpu_', 'vram_', 'memory_', 'cpu_'))}
                status.update(profile_id=ident, running=False, ready=False, checking=True, port=data.get('port',8080))
        else:
            config,fs,command,errors = {},[],'',[]
            status = {k:v for k,v in status.items() if k.startswith(('gpu_', 'vram_', 'memory_', 'cpu_'))}
            status.update(profile_id='', running=False, ready=False)
        hardware = dict(status)
        if 'memory_total_gib' not in hardware and control.psutil:
            hardware['memory_total_gib'] = control.psutil.virtual_memory().total / control.GIB
        hardware['wsl'] = bool(os.environ.get('WSL_DISTRO_NAME') or os.environ.get('WSL_INTEROP'))
        return dict(app_version=VERSION,root=str(ROOT),profiles=rows,profile_errors=self.profile_errors,engines=engines.engines(),engine_profiles=dict(self.engine_profiles),selected_engine=engine,selected_profile={'id':ident,'config':config},status=status,active_servers=active,validation_errors=errors,fields=fs,presets=presets(config,hardware) if selected else [],command=command,tasks=tasks)
    def task(self, message, function, serialize=False):
        item = {'id':uuid.uuid4().hex,'status':'queued' if serialize else 'running','message':message, 'action':message,'serialized':serialize}
        with self.lock: self.tasks.append(item)
        def progress(message):
            with self.lock: item['message'] = message
        def run():
            try:
                if serialize:
                    with self.operation_lock:
                        with self.lock: item['status'] = 'running'
                        result = function(progress)
                else:
                    result = function(progress)
                with self.lock: item.update(status='done',message='Готово',result=result)
            except Exception as exc:
                with self.lock: item.update(status='error',message=str(exc),error=str(exc))
        threading.Thread(target=run,daemon=True).start()
        return {'task_id':item['id']}
    def active_servers(self):
        with self.lock: profiles = list(self.profiles.items())
        active = []
        for ident, (path, data) in profiles:
            matches = control._server_processes(path, force=True)
            if matches:
                active.append(dict(id=ident, engine=data.get('launcher_engine','strata'), name=data.get('launcher_profile_name', data.get('profile_name',path.stem)), port=control._port_in_command(matches[0], int(data.get('port',8080)))))
        return active
    def collect_status(self):
        with self.lock: ident = self.selected
        active = self.active_servers()
        with self.lock: self.servers = active
        state = {'profile_id':ident, 'running':False, 'ready':False, 'log':log_tail()}
        if not ident: return state
        path, data = self.profile(ident)
        server = next((row for row in active if row['id'] == ident), None)
        port = server['port'] if server else int(data.get('port',8080))
        health = control._health(port) if server else None
        ready = bool(health) and health.get('status','ok') not in {'loading','error'}
        if health and data.get('launcher_engine','strata') in {'strata','polystrata'}:
            ready = ready and health.get('service') == 'strata'
            state['model_loaded'] = health.get('loaded')
        state.update(running=bool(server), ready=ready, port=port, phase='ready' if ready else 'starting' if server else 'stopped')
        try: state['guard'] = json.loads(control._guard_path(path).read_text(encoding='utf-8'))
        except (OSError,ValueError): pass
        with self.lock: launched = self.processes.get(ident)
        if launched and launched.poll() is not None and not server:
            state.update(exit_code=launched.returncode, phase='error' if launched.returncode else 'stopped')
        if ready and data.get('launcher_engine','strata') in {'strata','polystrata'}:
            try:
                headers = {'Authorization':'Bearer ' + str(data['api_key'])} if data.get('api_key') else {}
                request = urllib.request.Request(f'http://127.0.0.1:{port}/metrics', headers=headers)
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(request, timeout=1) as response: metrics = json.loads(response.read(2*1024*1024))
                # Only management metrics, never request bodies or conversation content.
                state['metrics'] = {key:metrics.get(key,{}) for key in ('engine','live','totals','conversation_cache')}
            except (OSError,ValueError): pass
        return state
    def monitor(self):
        while True:
            if time.monotonic() - self.last_seen > 30:
                time.sleep(3)
                continue
            try:
                state = self.collect_status()
            except Exception as exc: state = {'running':False,'ready':False,'error':str(exc),'log':log_tail()}
            if control.psutil:
                memory = control.psutil.virtual_memory()
                state.update(memory_available_gib=memory.available/control.GIB,memory_total_gib=memory.total/control.GIB,cpu_util=control.psutil.cpu_percent())
            try:
                result = subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=2,creationflags=FLAGS)
                row = result.stdout.splitlines()[0].split(',')
                state.update(gpu_name=row[0].strip(),vram_used_gib=float(row[1])/1024,vram_total_gib=float(row[2])/1024,gpu_util=float(row[3]))
                control.GPU_TOTAL_CACHE[0] = float(row[2])/1024
            except (OSError,ValueError,IndexError,subprocess.SubprocessError): pass
            with self.lock: self.status = state
            time.sleep(3)
    def save(self, ident, values, name='', new=False, description=None):
        path,data = self.profile(ident)
        data = updated(path,data,values)
        if name.strip(): data['launcher_profile_name'] = name.strip()[:200]
        if description is not None: data['profile_description'] = str(description)[:2000]
        if new: path = control.PROFILES_DIR / (uuid.uuid4().hex+'.json')
        write_json(path,data)
        self.reload()
        self.select(str(path.resolve()))
        return {'profile_id':str(path.resolve())}
    def start(self, ident, values):
        active = self.active_servers()
        if active: raise ValueError('Уже работает профиль ' + active[0]['name'] + '. Остановите его перед запуском другого.')
        with self.lock: original_path,_ = self.profile(ident)
        if control._server_processes(original_path,force=True): raise ValueError('Этот профиль уже работает.')
        path, data = self.profile(ident)
        candidate = updated(path, data, values)
        errors = self.preflight(path, candidate)
        if errors: raise ValueError('\n'.join(errors))
        result = self.save(ident,values)
        path,data = self.profile(result['profile_id'])
        command = self.command(path,data,materialize=True)
        if not control.PYTHON.is_file(): raise ValueError('Не найдено Python окружение Strata.')
        if data.get('launcher_engine','strata') == 'strata':
            for label,value in [('Движок',data.get('exe','')),('GGUF',control._option_value(data.get('args',[]),'--native')),('Pack',control._option_value(data.get('args',[]),'--pack')),('Токенизатор',data.get('tokenizer',''))]:
                if not value or not Path(value).exists(): raise ValueError(label+': путь не найден — '+value)
        port = int(data.get('port',8080))
        with socket.socket() as sock:
            try: sock.bind(('127.0.0.1',port))
            except OSError: raise ValueError('Порт занят. Выберите другой порт или остановите предыдущий сервер.')
        reserve = float(data.get('launcher_memory',{}).get('ram_reserve_gib',0))
        if reserve and control.psutil.virtual_memory().available/control.GIB <= reserve+.25: raise ValueError('Свободной RAM недостаточно для заданного резерва.')
        if data.get('launcher_engine','strata') != 'strata': command = [str(control.PYTHON),str(Path(control.__file__).resolve()),'--external-server','--config',str(path),'--port',str(port)]
        env = dict(os.environ,PYTHONUTF8='1',PYTHONUNBUFFERED='1')
        env.update({str(name):str(value) for name,value in data.get('launcher_env',{}).items()})
        control.LOG_PATH.parent.mkdir(parents=True,exist_ok=True)
        guard = control._guard_path(path)
        write_json(guard,{'status':'monitoring' if reserve else 'disabled'})
        with control.LOG_PATH.open('ab') as log:
            log.write(('\n--- '+path.name+' ---\n').encode())
            proc = subprocess.Popen(command,cwd=str(ROOT),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,creationflags=FLAGS,close_fds=True)
            with self.lock: self.processes[str(path.resolve())] = proc
            if reserve:
                try:
                    created = control.psutil.Process(proc.pid).create_time()
                    subprocess.Popen([str(control.PYTHON),str(Path(control.__file__).resolve()),'--memory-guard','--pid',str(proc.pid),'--created',str(created),'--ram-reserve-gib',str(reserve),'--state',str(guard)],cwd=str(ROOT),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,creationflags=FLAGS,close_fds=True)
                except Exception:
                    control._terminate_tree(control.psutil.Process(proc.pid))
                    raise
        control._server_processes(path,force=True)
        return {'pid':proc.pid,'port':port}
    def stop(self, ident):
        path,_ = self.profile(ident)
        for process in control._server_processes(path,force=True):
            try: control._terminate_tree(process)
            except control.psutil.NoSuchProcess: pass
        control._server_processes(path,force=True)
        with self.lock: launched = self.processes.get(str(path.resolve()))
        if launched: launched.wait(timeout=5)
        return {'stopped':True}
    def restart(self, ident, values):
        path, data = self.profile(ident)
        data = updated(path, data, values)
        errors = self.preflight(path, data)
        if errors: raise ValueError('\n'.join(errors))
        self.stop(ident)
        return self.start(ident, values)
    def clear_memory(self, progress):
        if not self.memory_cleanup_lock.acquire(blocking=False):
            raise ValueError('Очистка памяти уже выполняется.')
        try:
            before = control.psutil.virtual_memory().available if control.psutil else None
            progress('Очищаем системный файловый кэш RAM. Подтвердите права администратора в окне Windows.')
            memory_cleanup.clear_system_cache()
            after = control.psutil.virtual_memory().available if control.psutil else None
            message = ('Файловый кэш RAM очищен. Программы продолжают работать. '
                       'VRAM: общесистемная очистка без участия приложений недоступна.')
            if before is not None and after is not None:
                message += f' Доступно RAM: {before / control.GIB:.2f} → {after / control.GIB:.2f} ГиБ (весь ПК).'
            return {'ram_cache_cleared': True, 'vram_cleared': False,
                    'ram_available_before': before, 'ram_available_after': after, 'message': message}
        finally:
            self.memory_cleanup_lock.release()
    def prepare(self, source, progress, name=''):
        if self.active_servers(): raise ValueError('Остановите работающий сервер перед подготовкой модели.')
        source = Path(source).resolve()
        if not source.is_file(): raise ValueError('Файл модели не найден.')
        source = control.first_shard(source)
        destination = control.PROFILES_DIR/(uuid.uuid4().hex+'.json')
        destination.parent.mkdir(parents=True,exist_ok=True)
        command = [str(control.PYTHON),str(Path(control.__file__).resolve()),'--prepare-model',str(source),'--destination',str(destination)]
        progress('Подготовка локальных экспертов. Веса не скачиваются.')
        control.LOG_PATH.parent.mkdir(parents=True,exist_ok=True)
        with control.LOG_PATH.open('ab') as log: subprocess.run(command,cwd=str(ROOT),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,creationflags=FLAGS,check=True)
        if name:
            data = json.loads(destination.read_text(encoding='utf-8-sig'))
            data['launcher_profile_name'] = str(name)[:200]
            write_json(destination, data)
        self.reload(); self.select(str(destination.resolve()))
        return {'profile_id':str(destination.resolve())}
    def new(self, engine,model,name,poly_config=''):
        if engine == 'strata': return self.task('Подготовка модели',lambda p:self.prepare(model,p,name),serialize=True)
        if engine not in engines.engines(): raise ValueError('Неизвестный движок.')
        if not Path(model).is_file(): raise ValueError('Выберите существующую локальную модель.')
        if engine == 'polystrata':
            mmproj = next(iter(Path(model).parent.glob('mmproj*.gguf')), None)
            data = dict(launcher_engine=engine,launcher_profile_name=name or Path(model).stem,port=8081,open_browser=False,
                        launcher_env={},launcher_memory={'ram_reserve_gib':3},
                        engine_settings=dict(model=model,poly_config=poly_config or str(engines.POLY_CONFIG),context=32768,
                                             cache_ram_mib=0,poly_threads=16,poly_kv='fp16',poly_window=8,vram_reserve_gib=2,
                                             mmproj=str(mmproj) if mmproj else '',vision_gpu=False,vision_tokens=1024,
                                             temperature=1.0,top_p=0.95,top_k=20))
        else:
            data = dict(launcher_engine=engine,launcher_profile_name=name or Path(model).stem,port=8081,launcher_env={},launcher_memory={'ram_reserve_gib':4},engine_settings=dict(model=model,context=32768,gpu_layers='auto',threads=max(1,(os.cpu_count() or 4)//2),cache_k='q8_0',cache_v='q8_0',thinking=True,vram_reserve_gib=4,extra_args=[]))
        path = control.PROFILES_DIR/(uuid.uuid4().hex+'.json')
        write_json(path,data); self.reload(); self.select(str(path.resolve()))
        return {'profile_id':str(path.resolve())}

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def allowed(self):
        host = self.headers.get('Host','')
        if host not in {f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}: return False
        origin = self.headers.get('Origin')
        if origin and origin not in {f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}'}: return False
        return self.client_address[0] == '127.0.0.1'
    def send(self,payload,status=200,mime='application/json; charset=utf-8'):
        body = json.dumps(payload,ensure_ascii=False).encode() if isinstance(payload,(dict,list)) else payload
        self.send_response(status); self.send_header('Content-Type',mime); self.send_header('Content-Length',str(len(body))); self.send_header('Cache-Control','no-store'); self.send_header('X-Content-Type-Options','nosniff'); self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        if not self.allowed(): return self.send({'error':'Forbidden'},403)
        route = urlsplit(self.path).path
        if route == '/identity':
            return self.send({'app_version':VERSION, 'root':str(ROOT), 'pid':os.getpid()})
        if route.startswith('/api/') and not secrets.compare_digest(self.headers.get('X-Control-Token',''),TOKEN): return self.send({'error':'Forbidden'},403)
        if route == '/':
            try:
                html = (ROOT/'serve'/'control'/'index.html').read_text(encoding='utf-8').replace('__CONTROL_TOKEN__',json.dumps(TOKEN))
                return self.send(html.encode(),mime='text/html; charset=utf-8')
            except OSError: return self.send({'error':'Файл интерфейса не найден.'},500)
        if route in {'/control.js','/control.css'}:
            mime = 'text/javascript; charset=utf-8' if route.endswith('.js') else 'text/css; charset=utf-8'
            try: return self.send((ROOT/'serve'/'control'/route[1:]).read_bytes(),mime=mime)
            except OSError: return self.send({'error':'Файл интерфейса не найден.'},404)
        if route == '/api/state': return self.send(self.server.controller.state())
        if route == '/api/catalog': return self.send({'models':list(control.catalog_rows())})
        if route == '/api/log': return self.send({'log':log_tail()})
        return self.send({'error':'Not found'},404)
    def do_POST(self):
        if not self.allowed() or not secrets.compare_digest(self.headers.get('X-Control-Token',''),TOKEN): return self.send({'error':'Forbidden'},403)
        if self.headers.get('Content-Type','').split(';')[0] != 'application/json': return self.send({'error':'JSON required'},415)
        try:
            size = int(self.headers.get('Content-Length','0'))
            if not 0 <= size <= 1024*1024: return self.send({'error':'Request too large'},413)
            body = json.loads(self.rfile.read(size))
            if not isinstance(body,dict): raise ValueError('Требуется JSON объект.')
            c = self.server.controller; route = urlsplit(self.path).path
            ident = body.get('profile_id'); values = body.get('values',{})
            if route in {'/api/start','/api/stop','/api/restart'}:
                ident = str(c.profile(ident)[0].resolve())
            if route == '/api/select': c.select(body['id']); result = {'selected':True}
            elif route == '/api/select-engine': c.select_engine(body['engine']); result = {'selected':True}
            elif route == '/api/preview':
                path,data = c.profile(ident); candidate = updated(path,data,values)
                result = {'command':c.preview(path,candidate), 'validation_errors':c.preflight(path,candidate)}
            elif route in {'/api/save','/api/save-as'}: result = c.save(ident,values,str(body.get('name','')),route.endswith('save-as'),body.get('description'))
            elif route == '/api/export':
                path,data = c.profile(ident); result = {'config':updated(path,data,values)}
            elif route == '/api/start': result = c.task('Запуск сервера',lambda p:c.start(ident,values),serialize=True)
            elif route == '/api/stop': result = c.task('Остановка сервера',lambda p:c.stop(ident),serialize=True)
            elif route == '/api/clear-memory': result = c.task('Очистка RAM + VRAM',c.clear_memory)
            elif route == '/api/restart': result = c.task('Перезапуск сервера',lambda p:c.restart(ident,values),serialize=True)
            elif route == '/api/install': result = c.task('Установка движка',lambda p:{'engine':engines.install_llama(body.get('variant','cuda'),p)})
            elif route == '/api/register':
                kind = 'llama.cpp' if Path(body['exe']).name.casefold() == 'llama-server.exe' else 'custom'
                result = {'engine':engines.register_engine(body['name'],body['exe'],kind)}
            elif route == '/api/install-zip': result = c.task('Установка ZIP движка',lambda p:{'engine':engines.install_zip(body['path'],body.get('name','Локальный движок'),body.get('kind','custom'),progress=p)})
            elif route == '/api/prepare': result = c.task('Подготовка модели',lambda p:c.prepare(body['model'],p),serialize=True)
            elif route == '/api/new': result = c.new(body.get('engine','strata'),body['model'],body.get('name',''),body.get('poly_config',''))
            elif route == '/api/import':
                data = body['config']
                if not isinstance(data,dict) or not isinstance(data.get('launcher_env',{}),dict): raise ValueError('Некорректный профиль.')
                path = control.PROFILES_DIR/(uuid.uuid4().hex+'.json')
                # Reuse every editable-field validation before saving an imported contract.
                data = updated(path,data,{},validate=True)
                write_json(path,data); c.reload(); c.select(str(path.resolve())); result = {'profile_id':str(path.resolve())}
            elif route == '/api/browse':
                kind = body.get('kind','model')
                if kind not in {'model','mmproj','exe','profile','zip','folder'}: raise ValueError('Неизвестный тип файла.')
                result = {'path':subprocess.check_output([str(control.PYTHON),str(Path(__file__).resolve()),'--browse-dialog',kind],creationflags=FLAGS,text=True,encoding='utf-8',env=dict(os.environ,PYTHONUTF8='1')).strip()}
            else: return self.send({'error':'Not found'},404)
            return self.send(result)
        except (ValueError,KeyError,TypeError,OSError,subprocess.SubprocessError) as exc: return self.send({'error':str(exc)},400)
        except Exception as exc: return self.send({'error':str(exc)},500)

class Server(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, address):
        super().__init__(address,Handler)
        self.controller = Controller()

def browse(kind):
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk(); root.withdraw(); root.attributes('-topmost',True)
    types = {'model':[('Модель','*.gguf *.ninfer'),('Все файлы','*.*')],'mmproj':[('Vision GGUF','*.gguf')],'exe':[('Сервер','*.exe')],'profile':[('Профиль','*.json')],'zip':[('Архив','*.zip')]}
    path = filedialog.askdirectory(parent=root) if kind == 'folder' else filedialog.askopenfilename(parent=root,filetypes=types[kind])
    root.destroy(); print(path,flush=True)

def existing_url():
    try:
        info = json.loads(INSTANCE.read_text(encoding='utf-8'))
        if info.get('version') != VERSION or info.get('root') != str(ROOT): return None
        proc = control.psutil.Process(info['pid'])
        if abs(proc.create_time()-info['created']) > .01: return None
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"http://127.0.0.1:{int(info['port'])}/identity", timeout=.5) as response:
            live = json.loads(response.read(4096))
        if live.get('root') != str(ROOT) or live.get('app_version') != VERSION or live.get('pid') != info['pid']: return None
        return info['url']
    except (OSError,ValueError,KeyError,TypeError,AttributeError,control.psutil.Error if control.psutil else OSError): return None

def open_window(url):
    # Use a regular installed browser app window; no altered security flags.
    if os.name == 'nt':
        for base in [os.environ.get('ProgramFiles(x86)', ''), os.environ.get('ProgramFiles', ''), os.environ.get('LOCALAPPDATA', '')]:
            edge = Path(base) / 'Microsoft' / 'Edge' / 'Application' / 'msedge.exe'
            if base and edge.is_file():
                try:
                    subprocess.Popen([str(edge), '--app=' + url], creationflags=FLAGS)
                    return
                except OSError: pass
    webbrowser.open(url)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-browser',action='store_true'); parser.add_argument('--browse-dialog')
    args = parser.parse_args()
    if args.browse_dialog: browse(args.browse_dialog); return
    url = existing_url()
    if url:
        if not args.no_browser: open_window(url)
        return
    server = None
    for port in range(8787,8798):
        try: server = Server(('127.0.0.1',port)); break
        except OSError: continue
    if server is None: raise RuntimeError('Не удалось выбрать свободный локальный порт запуска.')
    url = f'http://127.0.0.1:{server.server_port}/'
    created = control.psutil.Process().create_time() if control.psutil else time.time()
    write_json(INSTANCE,dict(version=VERSION,root=str(ROOT),pid=os.getpid(),created=created,port=server.server_port,url=url))
    if not args.no_browser: open_window(url)
    try: server.serve_forever(poll_interval=.5)
    finally: server.server_close()

if __name__ == '__main__':
    try: main()
    except Exception:
        path = ROOT / '.strata-control' / 'web-control.log'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as output:
            output.write(time.strftime('%Y-%m-%d %H:%M:%S') + '\n' + traceback.format_exc() + '\n')
        raise
