"""Settings contracts against temporary Control profiles; no engine or GPU needed."""
from __future__ import annotations

import copy
import importlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import strata_control_web as web


class ParameterIntegrationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.path = self.root / 'profile.json'
        self.model = self.root / 'fixture.gguf'
        self.model.write_bytes(b'configuration fixture only')
        records = {
            'strata': {'name': 'Strata', 'kind': 'strata', 'installed': False},
            'polystrata': {'name': 'PolyStrata', 'kind': 'polystrata', 'installed': False},
            'llama.cpp': {'name': 'llama.cpp', 'kind': 'llama.cpp', 'installed': False},
            'custom': {'name': 'Custom', 'kind': 'custom', 'installed': False},
        }
        for item in [
            patch.object(web.engines, 'engines', return_value=records),
            patch.object(web.control.psutil, 'virtual_memory', return_value=SimpleNamespace(
                total=64 * web.control.GIB, available=48 * web.control.GIB)),
            patch.object(web.control, '_gpu_total_gib', return_value=12),
        ]:
            item.start()
            self.addCleanup(item.stop)

    def native(self, *extra):
        return {
            'launcher_engine': 'strata', 'port': 8081,
            'args': ['--native', str(self.model), '--pack', 'prepared-pack',
                     '--max-context', '32768', '--kv', 'int8', *extra],
            'launcher_env': {}, 'launcher_memory': {'ram_reserve_gib': 0},
            'user_metadata': {'retain': ['exact', 'value']},
        }

    def poly(self):
        return {
            'launcher_engine': 'polystrata', 'port': 8081,
            'engine_settings': {'model': str(self.model), 'context': 32768},
            'launcher_env': {}, 'launcher_memory': {'ram_reserve_gib': 0},
            'user_metadata': {'retain': True},
        }

    def test_noop_preserves_arguments_and_unknown_fields(self):
        data = self.native('--prefill=auto:32768', '--pool-workers', '5', '--custom-switch', 'keep')
        data['sampling'] = {'temperature': 0.7, 'repetition_penalty': 1.13}
        before = copy.deepcopy(data)
        self.assertEqual(web.updated(self.path, data, web.field_values(data)), before)
        self.assertEqual(data, before)
        self.assertFalse(self.path.exists())

    def test_sampling_updates_json_and_leaves_native_arguments_exact(self):
        data = self.native('--prefill', 'auto', '--custom-switch', 'keep')
        data['sampling'] = {'top_k': 17, 'repetition_penalty': 1.1}
        before = copy.deepcopy(data)
        result = web.updated(self.path, data, {'temperature': '0.65'})
        self.assertEqual(result['args'], before['args'])
        self.assertEqual(result['sampling'], {'temperature': 0.65, 'top_k': 17, 'repetition_penalty': 1.1})
        self.assertNotIn('--temperature', result['args'])
        self.assertEqual(data, before)

    def test_clear_sampling_and_reasoning_override_inherits(self):
        data = self.native()
        data.update(reasoning_effort='low', reasoning_budget_tokens=256)
        data['sampling'] = {'temperature': 0.4, 'top_k': 20}
        result = web.updated(self.path, data, {'temperature': '', 'reasoning_effort': '', 'reasoning_budget_tokens': ''})
        self.assertEqual(result['sampling'], {'top_k': 20})
        self.assertNotIn('reasoning_effort', result)
        self.assertNotIn('reasoning_budget_tokens', result)
        self.assertEqual(result['args'], data['args'])

    def test_zero_budget_is_preserved_as_unlimited(self):
        result = web.updated(self.path, self.native(), {'reasoning_budget_tokens': 0})
        self.assertEqual(result['reasoning_budget_tokens'], 0)
        self.assertNotIn('--reasoning-budget', result['args'])

    def test_optional_absence_and_metadata_are_visible(self):
        data = self.native()
        values = web.field_values(data)
        rows = {row['key']: row for row in web.fields(data)}
        for key in ['prefill', 'pool_workers', 'prompt_cache', 'prompt_cache_every',
                    'conversation_cache_mib', 'conversation_cache_slots',
                    'conversation_cache_min_free_mib', 'reasoning_effort',
                    'reasoning_budget_tokens', 'temperature', 'top_p', 'top_k', 'min_p']:
            with self.subTest(key=key):
                self.assertEqual(values[key], '')
                self.assertTrue(rows[key]['optional'])
                self.assertTrue(rows[key]['default_hint'])

    def test_legacy_context_edit_keeps_advanced_and_unknown_arguments(self):
        data = self.native('--prefill=auto:32768', '--pool-workers', '5', '--custom-switch', 'keep')
        result = web.updated(self.path, data, {'context': 65536})
        self.assertEqual(web.control._option_value(result['args'], '--max-context'), '65536')
        self.assertEqual(web.control._option_value(result['args'], '--prefill'), 'auto:32768')
        self.assertEqual(web.control._option_value(result['args'], '--pool-workers'), '5')
        self.assertIn('--custom-switch', result['args'])
        self.assertIn('keep', result['args'])
        self.assertEqual(result['user_metadata'], data['user_metadata'])

    def test_advanced_arguments_are_hidden_from_extra_and_not_duplicated(self):
        data = self.native('--prefill=auto:32768', '--pool-workers', '5', '--custom-switch', 'keep')
        values = web.field_values(data)
        self.assertEqual(values['prefill'], 'auto:32768')
        self.assertEqual(str(values['pool_workers']), '5')
        self.assertNotIn('--prefill', values['extra_args'])
        self.assertNotIn('--pool-workers', values['extra_args'])
        self.assertIn('--custom-switch', values['extra_args'])
        result = web.updated(self.path, data, {'pool_workers': 3})
        self.assertEqual(sum(token.split('=', 1)[0] == '--pool-workers' for token in result['args']), 1)
        self.assertEqual(web.control._option_value(result['args'], '--pool-workers'), '3')
        self.assertEqual(web.control._option_value(result['args'], '--prefill'), 'auto:32768')

    def test_extra_argument_cannot_override_a_dedicated_setting(self):
        for extra in ['--prefill\nauto', '--pool-workers=7', '--conversation-cache-mib\n2048']:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                web.updated(self.path, self.native(), {'extra_args': extra})

    def test_clearing_advanced_flag_removes_only_its_override(self):
        data = self.native('--pool-workers', '5', '--prefill=auto:32768', '--custom-switch', 'keep')
        result = web.updated(self.path, data, {'pool_workers': ''})
        self.assertNotIn('--pool-workers', result['args'])
        self.assertIn('--prefill=auto:32768', result['args'])
        self.assertEqual(web.field_values(result)['pool_workers'], '')

    def test_combined_legacy_and_advanced_edit_keeps_unrelated_values(self):
        data = self.native('--prefill=auto:32768', '--pool-workers', '5', '--custom-switch', 'keep')
        data['sampling'] = {'top_k': 23, 'frequency_penalty': 0.2}
        result = web.updated(self.path, data, {
            'context': 65536, 'pool_workers': 3, 'temperature': 0.6, 'reasoning_effort': 'medium',
        })
        self.assertEqual(web.control._option_value(result['args'], '--max-context'), '65536')
        self.assertEqual(web.control._option_value(result['args'], '--pool-workers'), '3')
        self.assertEqual(web.control._option_value(result['args'], '--prefill'), 'auto:32768')
        self.assertIn('--custom-switch', result['args'])
        self.assertEqual(result['sampling'], {'temperature': 0.6, 'top_k': 23, 'frequency_penalty': 0.2})
        self.assertEqual(result['reasoning_effort'], 'medium')
        self.assertEqual(result['user_metadata'], data['user_metadata'])

    def test_advanced_flag_edit_removes_all_previous_spellings(self):
        data = self.native('--pool-workers=4', '--pool-workers', '6', '--prefill=auto')
        result = web.updated(self.path, data, {'pool_workers': 2})
        self.assertEqual(sum(token.split('=', 1)[0] == '--pool-workers' for token in result['args']), 1)
        self.assertEqual(web.control._option_value(result['args'], '--pool-workers'), '2')
        self.assertIn('--prefill=auto', result['args'])

    def test_poly_sampling_and_reasoning_use_poly_contract(self):
        data = self.poly()
        result = web.updated(self.path, data, {
            'temperature': '0.7', 'top_p': '0.9', 'top_k': 32,
            'reasoning_effort': 'low', 'reasoning_budget_tokens': 512,
        })
        self.assertEqual(result['reasoning_effort'], 'low')
        self.assertEqual(result['reasoning_budget_tokens'], 512)
        self.assertEqual(result['engine_settings']['temperature'], 0.7)
        self.assertNotIn('reasoning_budget', result['engine_settings'])
        self.assertNotIn('--reasoning-effort', result.get('args', []))
        settings = importlib.import_module('strata_control_settings')
        runtime = settings.apply_poly_runtime(result, {'args': ['--model', str(self.model)]})
        self.assertEqual(runtime['sampling']['temperature'], 0.7)
        self.assertEqual(runtime['reasoning_effort'], 'low')
        self.assertEqual(runtime['reasoning_budget_tokens'], 512)
        self.assertEqual(data, self.poly())

    def test_absent_native_kv_reports_fp16_and_noop_preserves_absence(self):
        data = self.native()
        data['args'] = web.control._without_options(data['args'], {'--kv'})
        self.assertEqual(web.field_values(data)['kv'], 'fp16')
        self.assertEqual(web.updated(self.path, data, web.field_values(data)), data)
        self.assertIn('fp16', next(row['choices'] for row in web.fields(data) if row['key'] == 'kv'))

    def test_fp16_streaming_budget_uses_2048_bytes_for_full_context(self):
        data = self.native('--kv-resident', '32768')
        with patch.object(web.control, 'describe_file', return_value={'budget': True, 'arena_gb': 100}):
            result = web.updated(self.path, data, {'context': 262144, 'kv': 'fp16', 'ram_reserve': 4})
        self.assertEqual(web.control._option_value(result['args'], '--resident-budget-gib'), '33.000000')
        self.assertEqual(web.control._option_value(result['args'], '--kv'), 'fp16')
        self.assertEqual(web.control._option_value(result['args'], '--kv-resident'), '32768')

    def test_validate_rejects_existing_malformed_advanced_values(self):
        cases = [
            ('args', ['--pool-workers', '-1']),
            ('args', ['--prompt-cache', '1.5']),
            ('args', ['--prefill', 'auto:bogus']),
            ('engine_silence_s', float('nan')),
            ('idle_unload_s', -1),
            ('min_free_vram_mib', 1.5),
            ('reasoning_effort', 'maximum'),
            ('reasoning_budget_tokens', -1),
            ('sampling', {'top_k': 65}),
            ('sampling', {'top_p': 0}),
            ('sampling', {'temperature': float('inf')}),
        ]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                data = self.native()
                if key == 'args':
                    data['args'].extend(value)
                else:
                    data[key] = value
                with self.assertRaises(ValueError):
                    web.updated(self.path, data, {}, validate=True)


class PresetIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.presets = importlib.import_module('strata_control_presets').presets

    def test_three_opt_in_presets_for_supported_engines_only(self):
        for engine in ['strata', 'polystrata']:
            with self.subTest(engine=engine):
                data = {'launcher_engine': engine, 'args': ['--keep-this', 'value'],
                        'launcher_env': {'KEEP': 'value'}, 'engine_settings': {'model': 'keep.gguf'}}
                before = copy.deepcopy(data)
                rows = self.presets(data, {'memory_total_gib': 32})
                self.assertEqual(len(rows), 3)
                self.assertEqual({row['id'] for row in rows}, {'memory', 'conversations', 'long-context'})
                self.assertTrue(all(row['name'] and row['description'] and row['values'] for row in rows))
                self.assertEqual(data, before)
                forbidden = {'model', 'port', 'mtp', 'mtp_enabled', 'vision', 'mmproj', 'environment',
                             'poly_config', 'temperature', 'top_p', 'top_k', 'min_p',
                             'reasoning_effort', 'reasoning_budget_tokens'}
                for row in rows:
                    self.assertTrue(forbidden.isdisjoint(row['values']))
        for engine in ['llama.cpp', 'custom']:
            self.assertEqual(self.presets({'launcher_engine': engine}), [])

    def test_reserves_are_capped_for_16_32_and_64_gib_systems(self):
        for engine in ['strata', 'polystrata']:
            for total in [16, 32, 64]:
                hardware = {'memory_total_gib': total, 'memory_available_gib': total / 2}
                before = copy.deepcopy(hardware)
                with self.subTest(engine=engine, total=total):
                    rows = self.presets({'launcher_engine': engine}, hardware)
                    for row in rows:
                        self.assertEqual(row['values']['ram_reserve'], min(8, total / 4))
                        self.assertLess(row['values']['ram_reserve'], total)
                        self.assertEqual(row['values']['vram_reserve'], 2)
                    self.assertEqual(hardware, before)

    def test_cache_and_streaming_presets_are_engine_specific(self):
        for engine in ['strata', 'polystrata']:
            rows = {row['id']: row['values'] for row in self.presets(
                {'launcher_engine': engine}, {'memory_total_gib': 64})}
            self.assertEqual(rows['memory']['context'], 32768)
            self.assertEqual(rows['long-context']['context'], 131072)
            if engine == 'strata':
                self.assertEqual(rows['memory']['conversation_cache_mib'], 0)
                self.assertEqual(rows['conversations']['conversation_cache_mib'], 2048)
                self.assertEqual(rows['conversations']['conversation_cache_slots'], 4)
                self.assertEqual(rows['conversations']['parallel'], 1)
                self.assertEqual(rows['long-context']['kv_resident'], 32768)
                self.assertTrue(all('poly_kv' not in row for row in rows.values()))
            else:
                self.assertEqual(rows['memory']['cache_ram_mib'], 0)
                self.assertEqual(rows['conversations']['cache_ram_mib'], 2048)
                self.assertTrue(all(row['poly_kv'] == 'fp16' for row in rows.values()))
                self.assertTrue(all('kv_resident' not in row and 'parallel' not in row for row in rows.values()))

    def test_wsl_long_context_does_not_enable_streaming(self):
        rows = {row['id']: row['values'] for row in self.presets(
            {'launcher_engine': 'strata'}, {'memory_total_gib': 64, 'wsl': True})}
        self.assertEqual(rows['long-context']['kv_resident'], 0)

    def test_absent_or_invalid_memory_uses_finite_safe_fallback(self):
        for total in [None, 0, -1, True, '64', float('nan'), float('inf')]:
            with self.subTest(total=total):
                rows = self.presets({'launcher_engine': 'strata'}, {'memory_total_gib': total})
                self.assertEqual(rows[0]['values']['ram_reserve'], 4)


if __name__ == '__main__':
    unittest.main()
