"""Optional Control settings backed by engine arguments or server JSON keys.

An empty form value removes an override. Reading a profile never inserts engine
defaults, and applying one setting preserves unrelated arguments and metadata.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import math
import re


@dataclass(frozen=True)
class Setting:
    key: str
    label: str
    description: str
    kind: str = 'number'
    minimum: float | None = 0
    maximum: float | None = None
    flag: str = ''
    target: str = 'config'
    choices: tuple[str, ...] = ()
    default_hint: str = 'По умолчанию движка'
    exclusive_minimum: bool = False


STRATA = (
    Setting('prefill', 'Пакет чтения промпта', 'auto, auto:16384, auto:32768 или число токенов; 0 отключает пакетное чтение.', 'prefill', flag='--prefill', target='args', default_hint='Не задано: сохраняется выбор движка'),
    Setting('pool_workers', 'Рабочие потоки CPU', 'Число работников пула экспертов; 0 выбирает автоматически. Проверяйте на своём CPU.', maximum=2147483647, flag='--pool-workers', target='args'),
    Setting('prompt_cache', 'Точки сохранения промпта', 'Число точек восстановления KV; 0 отключает. Требуют RAM.', maximum=2147483647, flag='--prompt-cache', target='args', default_hint='Движок: 6 точек'),
    Setting('prompt_cache_every', 'Шаг сохранения промпта', 'Число токенов между точками восстановления; 0 отключает промежуточные точки, сохраняя конец хода.', flag='--prompt-cache-every', target='args', default_hint='Движок: 16384 токена'),
    Setting('conversation_cache_mib', 'Кэш бесед, МиБ', 'Бюджет RAM для припаркованных бесед; 0 отключает. Нужен включённый кэш промпта.', maximum=8796093022207, flag='--conversation-cache-mib', target='args', default_hint='Не задано: выбор установленной версии движка'),
    Setting('conversation_cache_slots', 'Слоты кэша бесед', 'Максимум припаркованных бесед; ограничивается бюджетом RAM.', maximum=2147483647, flag='--conversation-cache-slots', target='args', default_hint='Движок: 4 слота'),
    Setting('conversation_cache_min_free_mib', 'Свободная RAM для кэша, МиБ', 'Минимум свободной RAM для сохранения бесед. Не заменяет RAM-сторож Control.', maximum=8796093022207, flag='--conversation-cache-min-free-mib', target='args', default_hint='Движок: 2560 МиБ'),
    Setting('parallel', 'Параллельные запросы', '1 выполняет запросы по очереди; до 8 запросов вместе требуют больше памяти. Изменение заменяет старые --batch/--slots; пустое поле возвращает штатное поведение.', minimum=1, maximum=8, default_hint='Сервер: запросы по очереди'),
)

POLY = (
    Setting('poly_prompt_batch', 'Пакет чтения промпта', 'Максимум токенов в пакете промпта. Больший пакет требует больше VRAM.', minimum=1, maximum=4096, flag='--prompt-batch', target='engine_settings'),
    Setting('poly_stage_mib', 'Буфер экспертов, МиБ', 'Размер GPU-буфера для переноса экспертов. Требует свободной VRAM.', minimum=1, flag='--stage-mib', target='engine_settings'),
    Setting('poly_stage_ahead', 'Порог предзагрузки, токенов', 'Размер пакета, начиная с которого эксперты копируются заранее; 0 отключает.', flag='--stage-ahead', target='engine_settings'),
    Setting('poly_prompt_cache', 'Точки сохранения промпта', 'Число точек восстановления KV; 0 отключает кэш промпта.', flag='--prompt-cache', target='engine_settings'),
    Setting('poly_prompt_cache_every', 'Шаг сохранения промпта', 'Число токенов между точками восстановления промпта.', flag='--prompt-cache-every', target='engine_settings'),
    Setting('poly_lookup', 'Поиск черновых токенов', 'Глубина поиска повторяющихся последовательностей; 0 отключает lookup.', flag='--lookup', target='engine_settings'),
    Setting('poly_no_draft', 'Отключить черновую модель', 'Включено отключает draft. Lookup управляется отдельным параметром.', 'tristate', flag='--no-draft', target='engine_settings', choices=('', 'true', 'false')),
    Setting('poly_kv_whole', 'Весь KV в VRAM', 'Выделить память сразу для всего контекста. Уменьшает доступную VRAM для экспертов.', 'tristate', flag='--kv-whole', target='engine_settings', choices=('', 'true', 'false')),
)

SERVER = (
    Setting('idle_unload_s', 'Выгрузка при простое, с', 'Выгрузить модель после простоя; 0 оставляет её загруженной. Следующий запрос загрузит модель снова.', 'float'),
    Setting('engine_silence_s', 'Ожидание движка, с', 'Сколько ждать без сообщений от движка во время запроса; 0 отключает лимит.', 'float', default_hint='Сервер Strata: 300 с; PolyStrata: из конфигурации'),
    Setting('min_free_vram_mib', 'Свободная VRAM перед загрузкой, МиБ', 'Минимум свободной VRAM перед загрузкой модели. Это проверка перед загрузкой, а не лимит памяти.'),
    Setting('reasoning_effort', 'Уровень размышления', 'none отключает размышление; low, medium и high задают его уровень. Запрос API может переопределить значение.', 'select', choices=('', 'none', 'low', 'medium', 'high')),
    Setting('reasoning_budget_tokens', 'Бюджет размышления, токенов', '0 означает без ограничения. Значение запроса API имеет приоритет.'),
)

SAMPLING = (
    Setting('temperature', 'Температура', '0 выбирает наиболее вероятный токен. Действует, если клиент не передал своё значение.', 'float'),
    Setting('top_p', 'Top P', 'Порог суммарной вероятности; больше 0 и до 1 включительно.', 'float', minimum=0, maximum=1, exclusive_minimum=True),
    Setting('top_k', 'Top K', 'Число кандидатов для выборки; от 1 до 64.', minimum=1, maximum=64),
    Setting('min_p', 'Min P', 'Относительный порог вероятности; 0 отключает фильтр.', 'float', maximum=1),
    Setting('repetition_penalty', 'Штраф повторения', 'Множитель штрафа; 1 отключает. Значение должно быть больше 0.', 'float', exclusive_minimum=True),
    Setting('presence_penalty', 'Штраф присутствия', 'Неотрицательный штраф за уже встречавшийся токен; 0 отключает.', 'float'),
    Setting('frequency_penalty', 'Штраф частоты', 'Неотрицательный штраф с учётом числа повторов токена; 0 отключает.', 'float'),
    Setting('penalty_last_n', 'Окно штрафов, токенов', 'Число последних токенов для штрафов повторения. При включённых штрафах 0 использует штатное окно сервера.'),
    Setting('seed', 'Seed', 'Положительное целое для генератора выборки. Сам по себе не гарантирует одинаковый текст.', minimum=1),
)

PARALLEL_FLAGS = frozenset({'--batch', '--slots'})
STRATA_FLAGS = frozenset(spec.flag for spec in STRATA if spec.flag) | PARALLEL_FLAGS
POLY_FLAGS = frozenset(spec.flag for spec in POLY if spec.flag)
SAMPLING_KEYS = frozenset(spec.key for spec in SAMPLING)
SERVER_KEYS = frozenset(spec.key for spec in SERVER)
SAMPLING_ALIASES = {'repetition_penalty': 'repeat_penalty', 'penalty_last_n': 'repeat_last_n'}


def _engine(data):
    return data.get('launcher_engine', 'strata')


def _specs(data):
    engine = _engine(data)
    return (*STRATA, *SERVER, *SAMPLING) if engine == 'strata' else (*POLY, *SERVER, *SAMPLING) if engine == 'polystrata' else ()


def keys(data):
    return {spec.key for spec in _specs(data)}


def split_args(args):
    """Separate web-managed Strata options without changing either list's order."""
    unmanaged, managed = [], []
    index = 0
    while index < len(args):
        token = args[index]
        destination = managed if str(token).split('=', 1)[0] in STRATA_FLAGS else unmanaged
        destination.append(token)
        if destination is managed and '=' not in str(token) and index + 1 < len(args):
            index += 1
            destination.append(args[index])
        index += 1
    return unmanaged, managed


def _argument(args, flag):
    flags = {flag} if isinstance(flag, str) else flag
    value = ''
    for index, token in enumerate(args):
        if token in flags:
            value = args[index + 1] if index + 1 < len(args) else ''
        elif isinstance(token, str) and token.split('=', 1)[0] in flags and '=' in token:
            value = token.split('=', 1)[1]
    return value


def _source(data, spec):
    if spec.target == 'args':
        return _argument(data.get('args', []), spec.flag)
    if spec.key == 'parallel':
        batch = _argument(data.get('args', []), PARALLEL_FLAGS)
        if batch != '':
            return batch
    if spec in SAMPLING:
        target = data.get('sampling', {}) if _engine(data) == 'strata' else data.get('engine_settings', {})
        if target is None:
            target = {}
    elif spec.target == 'engine_settings':
        target = data.get('engine_settings', {})
    else:
        target = data
    if not isinstance(target, dict):
        raise ValueError('Некорректный блок настроек: ' + ('sampling' if spec in SAMPLING else spec.target))
    if spec in SAMPLING and _engine(data) == 'strata' and target.get(spec.key) is None:
        return target.get(SAMPLING_ALIASES.get(spec.key), '')
    return target.get(spec.key, '')


def _display(value, spec):
    if value is None:
        return ''
    if spec.kind == 'tristate' and isinstance(value, bool):
        return 'true' if value else 'false'
    if value != '' and not isinstance(value, bool) and spec.kind in {'number', 'float'}:
        try:
            normalized = _normalize(value, spec)
            # JSON numbers outside this range lose integer precision in browsers.
            return str(normalized) if isinstance(normalized, int) and abs(normalized) > 9007199254740991 else normalized
        except ValueError:
            pass
    return value


def field_values(data):
    return {spec.key: _display(_source(data, spec), _existing_spec(data, spec)) for spec in _specs(data)}


def _existing_spec(data, spec):
    if spec.key == 'parallel' and _argument(data.get('args', []), PARALLEL_FLAGS) != '':
        # Native batch-MTP profiles may use more slots than the JSON form offers;
        # zero also has a meaning in existing arguments (batching disabled).
        return replace(spec, minimum=0, maximum=None)
    return spec


def fields(data):
    values = field_values(data)
    output = []
    for spec in _specs(data):
        section = 'Генерация' if spec in SAMPLING else 'Сервер' if spec in SERVER or spec.key == 'parallel' else 'Кэш и чтение промпта'
        hint = 'Наследуется из конфигурации PolyStrata' if _engine(data) == 'polystrata' else spec.default_hint
        row = dict(key=spec.key, label=spec.label, description=spec.description,
                   type='text' if spec.kind == 'prefill' else 'select' if spec.kind == 'tristate' else spec.kind,
                   section=section, value=values[spec.key], optional=True, default_hint=hint)
        if spec.choices:
            names = {'': 'По умолчанию', 'true': 'Включено', 'false': 'Выключено',
                     'none': 'Без размышления', 'low': 'Низкий', 'medium': 'Средний', 'high': 'Высокий'}
            row['choices'] = [dict(value=value, label=names.get(value, value)) for value in spec.choices]
        if spec.minimum is not None and spec.kind in {'number', 'float'}:
            row['min'] = spec.minimum
        if spec.maximum is not None and spec.kind in {'number', 'float'}:
            row['max'] = spec.maximum
        output.append(row)
    return output


def _normalize(value, spec):
    if value is None or value == '':
        return ''
    if spec.kind == 'tristate':
        if isinstance(value, bool):
            return value
        if value in ('true', 'false'):
            return value == 'true'
        raise ValueError(spec.label + ': выберите значение из списка.')
    if spec.kind == 'select':
        if not isinstance(value, str) or value not in spec.choices:
            raise ValueError(spec.label + ': выберите значение из списка.')
        return value
    if spec.kind == 'prefill':
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError(spec.label + ': auto, auto:N или целое число токенов от 0.')
        text = str(value).strip()
        if text == 'auto' or re.fullmatch(r'auto:[1-9][0-9]*', text) or re.fullmatch(r'[0-9]+', text):
            return text
        raise ValueError(spec.label + ': auto, auto:N или целое число токенов от 0.')
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(spec.label + ': требуется число.')
    if isinstance(value, str) and not value.strip():
        return ''
    if spec.kind == 'number':
        if isinstance(value, str):
            if not re.fullmatch(r'[+-]?[0-9]+', value.strip()):
                raise ValueError(spec.label + ': требуется целое число.')
            try:
                number = int(value.strip())
            except ValueError:
                raise ValueError(spec.label + ': требуется целое число.') from None
        elif isinstance(value, int):
            number = value
        elif not math.isfinite(value) or not value.is_integer():
            raise ValueError(spec.label + ': требуется конечное целое число.')
        else:
            number = int(value)
    else:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(spec.label + ': требуется число.') from None
        if not math.isfinite(number):
            raise ValueError(spec.label + ': требуется конечное число.')
    if spec.minimum is not None and (number <= spec.minimum if spec.exclusive_minimum else number < spec.minimum):
        raise ValueError(spec.label + ': значение должно быть ' + ('больше ' if spec.exclusive_minimum else 'не меньше ') + str(spec.minimum) + '.')
    if spec.maximum is not None and number > spec.maximum:
        raise ValueError(spec.label + ': значение должно быть не больше ' + str(spec.maximum) + '.')
    return int(number) if spec.kind == 'number' else number


def _replace_arg(args, flag, value, boolean=False):
    output = []
    index = 0
    while index < len(args):
        token = args[index]
        if str(token).split('=', 1)[0] == flag:
            index += 1 if boolean or '=' in str(token) else 2
        else:
            output.append(token)
            index += 1
    if value != '':
        if boolean:
            if value:
                output.append(flag)
        else:
            output.extend((flag, str(value)))
    return output


def apply(data, changes, validate=False):
    if not isinstance(changes, dict):
        raise ValueError('Некорректные параметры профиля.')
    unknown = set(changes) - keys(data)
    if unknown:
        raise ValueError('Движок не поддерживает параметры: ' + ', '.join(sorted(unknown)))
    if validate and _engine(data) == 'strata':
        sampling = data.get('sampling', {})
        if isinstance(sampling, dict):
            for key, alias in SAMPLING_ALIASES.items():
                if sampling.get(key) is not None and sampling.get(alias) is not None:
                    raise ValueError('Параметры ' + key + ' и ' + alias + ' задают одну настройку; оставьте один.')
    result = copy.deepcopy(data)
    for spec in _specs(data):
        existing = _source(data, spec)
        existing_spec = _existing_spec(data, spec)
        if validate and existing not in ('', None):
            _normalize(existing, existing_spec)
        if spec.key not in changes:
            continue
        if existing_spec != spec:
            try:
                if _normalize(changes[spec.key], existing_spec) == _normalize(existing, existing_spec):
                    continue
            except ValueError:
                pass
        value = _normalize(changes[spec.key], spec)
        try:
            unchanged = value == _normalize(existing, existing_spec)
        except ValueError:
            unchanged = False
        if unchanged:
            continue
        if spec.key == 'parallel':
            for flag in PARALLEL_FLAGS:
                result['args'] = _replace_arg(result.get('args', []), flag, '')
        if spec.target == 'args':
            result['args'] = _replace_arg(result.get('args', []), spec.flag, value)
            continue
        if spec in SAMPLING:
            target_key = 'sampling' if _engine(data) == 'strata' else 'engine_settings'
        else:
            target_key = spec.target
        target = result if target_key == 'config' else result.setdefault(target_key, {})
        if target is None and target_key == 'sampling':
            target = result[target_key] = {}
        if not isinstance(target, dict):
            raise ValueError('Некорректный блок настроек: ' + target_key)
        if target_key == 'sampling' and spec.key in SAMPLING_ALIASES:
            target.pop(SAMPLING_ALIASES[spec.key], None)
        if value == '':
            target.pop(spec.key, None)
            if not target and target_key != 'config' and target_key not in data:
                result.pop(target_key, None)
        else:
            target[spec.key] = value
    return result


def assert_no_conflicts(text, data):
    managed = STRATA_FLAGS if _engine(data) == 'strata' else POLY_FLAGS if _engine(data) == 'polystrata' else ()
    tokens = text.splitlines() if isinstance(text, str) else text
    for raw in tokens:
        token = str(raw).strip()
        if token.split('=', 1)[0] in managed:
            raise ValueError('Параметр ' + token.split('=', 1)[0] + ' задаётся отдельным полем профиля.')


def apply_poly_runtime(config, runtime):
    """Overlay only explicit optional values onto the prepared Poly runtime."""
    if _engine(config) != 'polystrata':
        return copy.deepcopy(runtime)
    apply(config, {}, validate=True)
    result = copy.deepcopy(runtime)
    args = list(result.get('args', []))
    for spec in POLY:
        raw = _source(config, spec)
        if raw not in ('', None):
            args = _replace_arg(args, spec.flag, _normalize(raw, spec), boolean=spec.kind == 'tristate')
    if 'args' in runtime or args:
        result['args'] = args
    for spec in SERVER:
        raw = _source(config, spec)
        if raw not in ('', None):
            result[spec.key] = _normalize(raw, spec)
    sampling = None
    for spec in SAMPLING:
        raw = _source(config, spec)
        if raw not in ('', None):
            if sampling is None:
                sampling = copy.deepcopy(result.get('sampling', {}))
                if not isinstance(sampling, dict):
                    raise ValueError('Некорректный блок настроек: sampling')
            sampling[spec.key] = _normalize(raw, spec)
            if spec.key in SAMPLING_ALIASES:
                sampling.pop(SAMPLING_ALIASES[spec.key], None)
    if sampling is not None:
        result['sampling'] = sampling
    return result
