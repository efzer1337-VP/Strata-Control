"""Opt-in starting points; never applied while opening or saving a profile."""
from __future__ import annotations

import math


def presets(data, hardware=None):
    engine = data.get('launcher_engine', 'strata')
    if engine not in {'strata', 'polystrata'}:
        return []
    hardware = hardware or {}
    total = hardware.get('memory_total_gib')
    if isinstance(total, bool) or not isinstance(total, (int, float)) or not math.isfinite(total) or total <= 0:
        total = None
    reserve = min(8, total / 4) if total else 4
    conversation_cache = min(2048, int(total * 64)) if total else 1024
    common = {'context':32768, 'ram_reserve':reserve, 'vram_reserve':2}
    if engine == 'strata':
        common.update(kv='int8', kv_resident=0, expert_cache='auto',
                      conversation_cache_mib=0, parallel=1)
        conversations = dict(common, conversation_cache_mib=conversation_cache,
                             conversation_cache_slots=4, conversation_cache_min_free_mib=2560,
                             prompt_cache=6, prompt_cache_every=16384)
        long_context = dict(common, context=131072, kv_resident=0 if hardware.get('wsl') else 32768)
    else:
        common.update(poly_kv='fp16', cache_ram_mib=0)
        conversations = dict(common, cache_ram_mib=conversation_cache, poly_prompt_cache=6,
                             poly_prompt_cache_every=2048)
        long_context = dict(common, context=131072)
    return [
        {'id':'memory', 'name':'С запасом памяти',
         'description':'Контекст 32K, кэш бесед выключен, запас VRAM 2 ГиБ. '
                       'Запас RAM: четверть установленной памяти, максимум 8 ГиБ. '
                       'Это начальные значения, вместимость модели проверяется отдельно.',
         'values':dict(common)},
        {'id':'conversations', 'name':'Несколько клиентов',
         'description':f'Контекст 32K, RAM-кэш бесед {conversation_cache} МиБ для чередующихся запросов. '
                       'Кэш помогает повторно использовать контекст; он не включает параллельное выполнение '
                       'и не гарантирует, что все длинные беседы поместятся.',
         'values':conversations},
        {'id':'long-context', 'name':'Длинный контекст',
         'description':'Контекст 128K. Проверьте предел выбранной модели и свободную память. '
                       + ('KV хранит полную копию контекста в RAM, 32K остаётся в VRAM. '
                          if engine == 'strata' and not hardware.get('wsl') else
                          'В WSL KV streaming не включается. ' if engine == 'strata' else
                          'PolyStrata сохраняет fp16 KV; параметры streaming Strata к ней не применяются. ')
                       + 'Качество и скорость на этом размере контекста не измерялись.',
         'values':long_context},
    ]
