"""Loopback-only responsive web control for Strata and local model engines."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import secrets
import socket
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

ROOT = Path(__file__).resolve().parent
VERSION = '4.0'
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

def field_values(data):
    args = data.get('args', [])
    option = lambda name, default='': control._option_value(args, name, str(default))
    ext = data.get('engine_settings', {})
    vision = data.get('vision', {}) if isinstance(data.get('vision'), dict) else {}
    external = data.get('launcher_engine', 'strata') != 'strata'
    extras = control._without_options(args, control.MANAGED_OPTIONS)
    return {
        'model': ext.get('model', '') if external else option('--native'),
        'context': ext.get('context', 131072) if external else option('--max-context', 32768),
        'port': data.get('port', 8080), 'kv': option('--kv', 'int8'),
        'kv_resident': option('--kv-resident', 0), 'spec': option('--spec', 4),
        'spec_min_p': option('--spec-min-p', .5), 'expert_cache': option('--expert-cache', 'auto'),
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
        'poly_threads': ext.get('poly_threads', max(1,(os.cpu_count() or 4)//2)),
        'poly_kv': ext.get('poly_kv', 'fp16'),
        'vision_gpu': ext.get('vision_gpu', False),
        'ubatch_size': ext.get('ubatch_size', 512),
        'temperature': ext.get('temperature', 0.8), 'top_p': ext.get('top_p', 0.95),
        'top_k': ext.get('top_k', 40), 'min_p': ext.get('min_p', 0.05),
        'reasoning_budget_message': ext.get('reasoning_budget_message', ''),
        'extra_args': '\n'.join(ext.get('extra_args', [])) if external else '\n'.join(extras),
        'environment': '\n'.join(f'{k}={v}' for k,v in data.get('launcher_env', {}).items())}

DESCRIPTIONS = {
 'model': ('Модель', 'Локальный файл модели', 'path', 'Модель'),
 'context': ('Контекст', 'Максимальное число токенов', 'number', 'Память'),
 'port': ('Порт', 'Порт локального API', 'number', 'Сервер'),
 'kv': ('KV cache', 'Формат хранения кэша контекста', 'select', 'Память'),
 'kv_resident': ('KV в VRAM', 'Резидентная часть контекста; 0 — без streaming', 'number', 'Память'),
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
 'poly_window': ('Окно MTP', 'Максимум проверяемых токенов; 1 отключает угадывание', 'number', 'Скорость'),
 'poly_threads': ('Потоки экспертов CPU', 'Сколько потоков используют эксперты вне VRAM', 'number', 'Скорость'),
 'poly_kv': ('KV cache', 'fp16 — штатное качество; q8_0 экономит VRAM', 'select', 'Память'),
 'vision_gpu': ('Vision на GPU', 'Быстрее обработка изображений, но меньше VRAM для экспертов', 'bool', 'Изображения'),
 'extra_args': ('Дополнительные параметры', 'Один аргумент на строку; пути без кавычек', 'textarea', 'Дополнительно'),
 'environment': ('Переменные окружения', 'ИМЯ=значение, одна переменная на строку', 'textarea', 'Дополнительно')}

def fields(data):
    values = field_values(data)
    engine = data.get('launcher_engine', 'strata')
    external = engine != 'strata'
    keys = ['model','context','port','vram_reserve','ram_reserve']
    if engine == 'polystrata':
        keys += ['poly_kv','cache_ram_mib','poly_threads','poly_window','mmproj','vision_gpu','vision_tokens','temperature','top_p','top_k']
    else:
        keys += ['gpu_layers','cpu_moe_layers','threads','parallel','cache_k','cache_v','cache_ram_mib','ubatch_size','thinking','reasoning_budget','reasoning_budget_message','temperature','top_p','top_k','min_p','fit','mmproj'] if external else ['kv','kv_resident','expert_cache','resident','mtp_enabled','mtp','spec','spec_min_p','vision','mmproj','vision_tokens','vision_device','lazy']
        keys += ['extra_args','environment']
    choices = {'kv':['int8','q4_0','k8v4'], 'poly_kv':['fp16','q8_0'], 'vision_device':['GPU','CPU'], 'cache_k':['f16','bf16','q8_0','q4_0','q4_1','iq4_nl','q5_0','q5_1'], 'cache_v':['f16','bf16','q8_0','q4_0','q4_1','iq4_nl','q5_0','q5_1']}
    result = []
    for key in keys:
        label, desc, kind, section = DESCRIPTIONS[key]
        if engine == 'polystrata' and key == 'cache_ram_mib':
            label, desc = 'Кэш бесед, МиБ', 'Память для нескольких разговоров; 0 отключает'
        row = dict(key=key,label=label,description=desc,type=kind,section=section,value=values[key])
        if key in choices: row['choices'] = choices[key]
        result.append(row)
    return result

def updated(path, data, changes):
    if not isinstance(changes, dict): raise ValueError('Некорректные параметры профиля.')
    changes = dict(changes)
    description = changes.pop('profile_description', None)
    values = field_values(data)
    unknown = set(changes) - set(values)
    if unknown: raise ValueError('Неизвестные параметры: ' + ', '.join(unknown))
    for key, value in changes.items():
        if isinstance(values[key], bool) and not isinstance(value, bool):
            raise ValueError(f'{key}: требуется true или false.')
        if not isinstance(value, (str,int,float,bool)): raise ValueError('Некорректное значение параметра.')
    values.update(changes)
    if data.get('launcher_engine') == 'polystrata':
        result = copy.deepcopy(data)
        settings = result.setdefault('engine_settings', {})
        settings.update(model=str(values['model']), context=int(values['context']),
                        vram_reserve_gib=float(values['vram_reserve']),
                        cache_ram_mib=int(values['cache_ram_mib']),
                        poly_window=int(values['poly_window']), poly_threads=int(values['poly_threads']),
                        poly_kv=str(values['poly_kv']), mmproj=str(values['mmproj']),
                        vision_gpu=bool(values['vision_gpu']), vision_tokens=int(values['vision_tokens']),
                        temperature=float(values['temperature']), top_p=float(values['top_p']),
                        top_k=int(values['top_k']))
        result['port'] = int(values['port'])
        result['launcher_memory'] = {'ram_reserve_gib':float(values['ram_reserve'])}
        if description is not None: result['profile_description'] = str(description)[:2000]
        engines.command_for(result)
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
    if description is not None: result['profile_description'] = str(description)[:2000]
    return result

class Controller:
    def __init__(self):
        self.lock = threading.RLock()
        self.operation_lock = threading.Lock()
        self.tasks = []
        self.profiles = {}
        self.selected = ''
        self.status = {'running':False,'ready':False,'port':8080,'cpu_util':0,'gpu_util':0,'log':''}
        self.last_seen = time.monotonic()
        self.reload()
        try:
            stored = json.loads(control.CONTROL_STATE.read_text(encoding='utf-8-sig'))
            selected = stored.get('web_profile', '')
            if selected in self.profiles: self.selected = selected
        except (OSError, ValueError): pass
        threading.Thread(target=self.monitor, daemon=True).start()
    def reload(self):
        profiles = {}
        for path in control._configs():
            try:
                data = json.loads(path.read_text(encoding='utf-8-sig'))
                if isinstance(data, dict): profiles[str(path.resolve())] = (path, data)
            except (OSError, ValueError): pass
        with self.lock:
            self.profiles = profiles
            if self.selected not in profiles: self.selected = next(iter(profiles), '')
    def profile(self, ident=None):
        with self.lock:
            ident = ident or self.selected
            if ident not in self.profiles: raise ValueError('Выберите профиль модели.')
            path, data = self.profiles[ident]
            return path, copy.deepcopy(data)
    def select(self, ident):
        self.profile(ident)
        with self.lock: self.selected = ident
        try: stored = json.loads(control.CONTROL_STATE.read_text(encoding='utf-8-sig'))
        except (OSError,ValueError): stored = {}
        stored['web_profile'] = ident
        write_json(control.CONTROL_STATE, stored)
    def command(self, path, data):
        if data.get('launcher_engine', 'strata') != 'strata': return engines.command_for(data, materialize=data.get('launcher_engine') == 'polystrata')
        return [str(control.PYTHON), str(control.SERVER), '--engine','strata','--config',str(path),'--host','127.0.0.1','--port',str(data.get('port',8080))]
    def preview(self, path, data):
        try:
            tokens = self.command(path,data)
            quote = lambda value: "'" + str(value).replace("'", "''") + "'"
            if any(not control.re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', str(name)) for name in data.get('launcher_env',{})):
                raise ValueError('Некорректное имя переменной окружения.')
            env = '\n'.join(f"$env:{name} = {quote(value)}" for name,value in data.get('launcher_env',{}).items())
            if data.get('launcher_engine','strata') != 'strata':
                return (env + '\n' if env else '') + '& ' + ' `\n  '.join(quote(v) for v in tokens)
            # Preview contains the edited profile, not a saved file whose args
            # may differ. The text is only executed when the user copies it.
            profile_json = json.dumps(data,ensure_ascii=False,indent=2)
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
            tasks = copy.deepcopy(self.tasks[-20:])
            ident = self.selected
        if selected:
            path,data = selected
            config = copy.deepcopy(data)
            fs = fields(data)
            command = self.preview(path,data)
        else: config,fs,command = {},[],''
        return dict(app_version=VERSION,root=str(ROOT),profiles=rows,engines=engines.engines(),selected_profile={'id':ident,'config':config},status=status,fields=fs,command=command,tasks=tasks)
    def task(self, message, function, serialize=False):
        item = {'id':uuid.uuid4().hex,'status':'running','message':message}
        with self.lock: self.tasks.append(item)
        def progress(message):
            with self.lock: item['message'] = message
        def run():
            try:
                if serialize:
                    with self.operation_lock: result = function(progress)
                else:
                    result = function(progress)
                with self.lock: item.update(status='done',message='Готово',result=result)
            except Exception as exc:
                with self.lock: item.update(status='error',message=str(exc),error=str(exc))
        threading.Thread(target=run,daemon=True).start()
        return {'task_id':item['id']}
    def monitor(self):
        while True:
            if time.monotonic() - self.last_seen > 30:
                time.sleep(3)
                continue
            try:
                with self.lock: ident = self.selected
                path,data = self.profile(ident)
                processes = control._server_processes(path)
                port = int(data.get('port',8080))
                if processes:
                    parts = processes[0].info.get('cmdline', [])
                    port = int(control._option_value(parts, '--port', str(port)))
                health = control._health(port) if processes else None
                state = {'running':bool(processes),'ready':bool(health),'port':port,'log':log_tail(),'profile_id':ident}
                if health and data.get('launcher_engine','strata') == 'strata': state['ready'] = health.get('service') == 'strata' and bool(health.get('loaded') or data.get('lazy_load'))
            except (ValueError, OSError): state = {'running':False,'ready':False,'port':8080,'log':log_tail()}
            if control.psutil:
                memory = control.psutil.virtual_memory()
                state.update(memory_available_gib=memory.available/control.GIB,memory_total_gib=memory.total/control.GIB,cpu_util=control.psutil.cpu_percent())
            try:
                result = subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=2,creationflags=FLAGS)
                row = result.stdout.splitlines()[0].split(',')
                state.update(gpu_name=row[0].strip(),vram_used_gib=float(row[1])/1024,vram_total_gib=float(row[2])/1024,gpu_util=float(row[3]))
                control.GPU_TOTAL_CACHE[0] = float(row[2])/1024
            except (OSError,ValueError,IndexError,subprocess.SubprocessError): pass
            with self.lock: self.status.update(state)
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
        with self.lock: original_path,_ = self.profile(ident)
        if control._server_processes(original_path,force=True): raise ValueError('Этот профиль уже работает.')
        result = self.save(ident,values)
        path,data = self.profile(result['profile_id'])
        command = self.command(path,data)
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
        env.update(data.get('launcher_env',{}))
        control.LOG_PATH.parent.mkdir(parents=True,exist_ok=True)
        guard = control._guard_path(path)
        write_json(guard,{'status':'monitoring' if reserve else 'disabled'})
        with control.LOG_PATH.open('ab') as log:
            log.write(('\n--- '+path.name+' ---\n').encode())
            proc = subprocess.Popen(command,cwd=str(ROOT),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,creationflags=FLAGS,close_fds=True)
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
        return {'stopped':True}
    def restart(self, ident, values):
        path, data = self.profile(ident)
        data = updated(path, data, values)
        self.command(path, data)
        self.stop(ident)
        return self.start(ident, values)
    def prepare(self, source, progress, name=''):
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
    def new(self, engine,model,name):
        if engine == 'strata': return self.task('Подготовка модели',lambda p:self.prepare(model,p,name))
        if engine not in engines.engines(): raise ValueError('Неизвестный движок.')
        if not Path(model).is_file(): raise ValueError('Выберите существующую локальную модель.')
        if engine == 'polystrata':
            mmproj = next(iter(Path(model).parent.glob('mmproj*.gguf')), None)
            data = dict(launcher_engine=engine,launcher_profile_name=name or Path(model).stem,port=8081,open_browser=False,
                        launcher_env={},launcher_memory={'ram_reserve_gib':3},
                        engine_settings=dict(model=model,poly_config=str(engines.POLY_CONFIG),context=32768,
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
        if route == '/api/state': return self.send(self.server.controller.state())
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
            if route == '/api/select': c.select(body['id']); result = {'selected':True}
            elif route == '/api/preview':
                path,data = c.profile(ident); result = {'command':c.preview(path,updated(path,data,values))}
            elif route in {'/api/save','/api/save-as'}: result = c.save(ident,values,str(body.get('name','')),route.endswith('save-as'),body.get('description'))
            elif route == '/api/export':
                path,data = c.profile(ident); result = {'config':updated(path,data,values)}
            elif route == '/api/start': result = c.task('Запуск сервера',lambda p:c.start(ident,values),serialize=True)
            elif route == '/api/stop': result = c.task('Остановка сервера',lambda p:c.stop(ident),serialize=True)
            elif route == '/api/restart': result = c.task('Перезапуск сервера',lambda p:c.restart(ident,values),serialize=True)
            elif route == '/api/install': result = c.task('Установка движка',lambda p:{'engine':engines.install_llama(body.get('variant','cuda'),p)})
            elif route == '/api/register': result = {'engine':engines.register_engine(body['name'],body['exe'])}
            elif route == '/api/install-zip': result = c.task('Установка ZIP движка',lambda p:{'engine':engines.install_zip(body['path'],body.get('name','Локальный движок'),body.get('kind','custom'),progress=p)})
            elif route == '/api/prepare': result = c.task('Подготовка модели',lambda p:c.prepare(body['model'],p))
            elif route == '/api/new': result = c.new(body.get('engine','strata'),body['model'],body.get('name',''))
            elif route == '/api/import':
                data = body['config']
                if not isinstance(data,dict) or not isinstance(data.get('launcher_env',{}),dict): raise ValueError('Некорректный профиль.')
                path = control.PROFILES_DIR/(uuid.uuid4().hex+'.json')
                # Reuse every editable-field validation before saving an imported contract.
                data = updated(path,data,{})
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
