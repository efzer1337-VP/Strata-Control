"""Optional settings contracts; no model, process or GPU is started."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import strata_control_settings as settings


class OptionalSettingsTests(unittest.TestCase):
    def strata(self):
        return {'args': ['--native', 'model.gguf', '--pack', 'prepared', '--unknown=keep'],
                'tokenizer': 'tokenizer', 'user_metadata': {'keep': True}}

    def poly(self):
        return {'launcher_engine': 'polystrata', 'engine_settings': {'model': 'model.gguf'},
                'user_metadata': {'keep': True}}

    def test_old_profiles_have_no_new_overrides(self):
        for profile in (self.strata(), self.poly()):
            with self.subTest(engine=profile.get('launcher_engine', 'strata')):
                before = copy.deepcopy(profile)
                values = settings.field_values(profile)
                self.assertTrue(values)
                self.assertTrue(all(value == '' for value in values.values()))
                self.assertEqual(settings.apply(profile, values), before)
                self.assertEqual(settings.apply(profile, {}, validate=True), before)
                self.assertEqual(profile, before)

    def test_optional_field_metadata_has_inherit_option(self):
        fields = {field['key']: field for field in settings.fields(self.poly())}
        for field in fields.values():
            self.assertTrue(field['optional'])
            self.assertTrue(field['default_hint'])
        for key in ('poly_kv_whole', 'poly_no_draft', 'reasoning_effort'):
            self.assertEqual(fields[key]['choices'][0], {'value': '', 'label': 'По умолчанию'})
        self.assertEqual(fields['top_k']['default_hint'], 'Наследуется из конфигурации PolyStrata')

    def test_settings_are_engine_specific(self):
        self.assertIn('prefill', settings.keys(self.strata()))
        self.assertNotIn('poly_prompt_batch', settings.keys(self.strata()))
        self.assertIn('poly_prompt_batch', settings.keys(self.poly()))
        self.assertNotIn('prefill', settings.keys(self.poly()))
        for engine in ('llama.cpp', 'custom'):
            profile = {'launcher_engine': engine}
            self.assertEqual(settings.fields(profile), [])
            self.assertEqual(settings.field_values(profile), {})
            with self.assertRaisesRegex(ValueError, 'не поддерживает'):
                settings.apply(profile, {'temperature': 1})

    def test_split_preserves_order_and_equals_syntax(self):
        args = ['--native', 'model', '--prefill=auto', '--unknown', 'x',
                '--pool-workers', '0', '--prompt-cache', '3', '--other=keep']
        unmanaged, managed = settings.split_args(args)
        self.assertEqual(unmanaged, ['--native', 'model', '--unknown', 'x', '--other=keep'])
        self.assertEqual(managed, ['--prefill=auto', '--pool-workers', '0', '--prompt-cache', '3'])
        self.assertEqual(args[2], '--prefill=auto')

    def test_parallel_displays_raw_batch_priority_and_preserves_noop(self):
        profile = dict(self.strata(), parallel=8)
        profile['args'] += ['--batch', '2', '--unknown-later', 'keep', '--slots=3']
        self.assertEqual(settings.field_values(profile)['parallel'], 3)
        self.assertEqual(settings.apply(profile, settings.field_values(profile)), profile)
        self.assertEqual(settings.apply(profile, {}, validate=True), profile)
        unmanaged, managed = settings.split_args(profile['args'])
        self.assertEqual(managed, ['--batch', '2', '--slots=3'])
        self.assertEqual(unmanaged, self.strata()['args'] + ['--unknown-later', 'keep'])

    def test_parallel_edit_replaces_all_legacy_flags_and_clear_inherits(self):
        profile = dict(self.strata(), parallel=8)
        profile['args'] += ['--batch=2', '--slots', '3', '--batch', '4', '--unknown-later', 'keep']
        expected_args = self.strata()['args'] + ['--unknown-later', 'keep']
        result = settings.apply(profile, {'parallel': 2})
        self.assertEqual(result['parallel'], 2)
        self.assertEqual(result['args'], expected_args)
        self.assertEqual(settings.field_values(result)['parallel'], 2)
        cleared = settings.apply(profile, {'parallel': ''})
        self.assertNotIn('parallel', cleared)
        self.assertEqual(cleared['args'], expected_args)
        self.assertEqual(settings.field_values(cleared)['parallel'], '')
        self.assertEqual(settings.apply(cleared, settings.field_values(cleared)), cleared)
        self.assertEqual(profile['parallel'], 8)

    def test_existing_native_batch_zero_and_large_slot_count_survive_validation(self):
        for batch in (0, 16, 64):
            with self.subTest(batch=batch):
                profile = dict(self.strata(), parallel=2)
                profile['args'] += ['--batch-mtp', '--batch', str(batch)]
                self.assertEqual(settings.field_values(profile)['parallel'], batch)
                self.assertEqual(settings.apply(profile, {}, validate=True), profile)
                self.assertEqual(settings.apply(profile, settings.field_values(profile), validate=True), profile)
                edited = settings.apply(profile, {'parallel': 3})
                self.assertEqual(edited['parallel'], 3)
                self.assertNotIn('--batch', edited['args'])
                self.assertIn('--batch-mtp', edited['args'])
                cleared = settings.apply(profile, {'parallel': ''})
                self.assertNotIn('parallel', cleared)
                self.assertNotIn('--batch', cleared['args'])
        with self.assertRaises(ValueError):
            settings.apply(self.strata(), {'parallel': 16})
        with self.assertRaises(ValueError):
            settings.apply(dict(self.strata(), args=['--batch', '-1']), {}, validate=True)

    def test_structured_raw_argument_display_uses_last_occurrence(self):
        profile = self.strata()
        profile['args'] += ['--pool-workers', '2', '--pool-workers=7',
                            '--prefill=auto', '--prefill', '16384']
        values = settings.field_values(profile)
        self.assertEqual(values['pool_workers'], 7)
        self.assertEqual(values['prefill'], '16384')
        self.assertEqual(settings.apply(profile, values), profile)

    def test_single_argument_change_preserves_unknown_settings(self):
        profile = self.strata()
        profile['args'] += ['--prefill=auto', '--pool-workers', '7', '--prompt-cache=6']
        before = copy.deepcopy(profile)
        result = settings.apply(profile, {'pool_workers': 3})
        self.assertEqual(result['args'], before['args'][:6] + ['--prompt-cache=6', '--pool-workers', '3'])
        self.assertEqual(result['user_metadata'], before['user_metadata'])
        self.assertEqual(profile, before)

    def test_unchanged_values_preserve_original_argument_spelling(self):
        profile = self.strata()
        profile['args'] += ['--prefill=auto:16384', '--pool-workers=0', '--prompt-cache', '6']
        self.assertEqual(settings.apply(profile, settings.field_values(profile)), profile)

    def test_removing_override_removes_all_duplicate_arguments(self):
        profile = self.strata()
        profile['args'] += ['--prompt-cache=3', '--other', 'x', '--prompt-cache', '4']
        result = settings.apply(profile, {'prompt_cache': ''})
        self.assertEqual(result['args'], self.strata()['args'] + ['--other', 'x'])
        self.assertEqual(settings.field_values(result)['prompt_cache'], '')

    def test_sampling_and_server_settings_use_json(self):
        profile = self.strata()
        profile['sampling'] = {'vendor_setting': 99}
        result = settings.apply(profile, {'temperature': .8, 'top_k': 20, 'seed': 7,
                                          'reasoning_effort': 'none', 'reasoning_budget_tokens': 0,
                                          'idle_unload_s': 30.5, 'parallel': 2})
        self.assertEqual(result['args'], profile['args'])
        self.assertEqual(result['sampling'], {'vendor_setting': 99, 'temperature': .8, 'top_k': 20, 'seed': 7})
        self.assertEqual(result['reasoning_effort'], 'none')
        self.assertEqual(result['reasoning_budget_tokens'], 0)
        self.assertEqual(result['parallel'], 2)
        self.assertEqual(result['idle_unload_s'], 30.5)
        cleared = settings.apply(result, {'temperature': '', 'reasoning_effort': ''})
        self.assertNotIn('temperature', cleared['sampling'])
        self.assertEqual(cleared['sampling']['vendor_setting'], 99)
        self.assertNotIn('reasoning_effort', cleared)

    def test_sampling_aliases_survive_noop_and_are_replaced_when_edited(self):
        profile = dict(self.strata(), sampling={'repeat_penalty': 1.1, 'repeat_last_n': 64, 'vendor': 2})
        values = settings.field_values(profile)
        self.assertEqual(values['repetition_penalty'], 1.1)
        self.assertEqual(values['penalty_last_n'], 64)
        self.assertEqual(settings.apply(profile, values), profile)
        result = settings.apply(profile, {'repetition_penalty': 1.2, 'penalty_last_n': ''})
        self.assertEqual(result['sampling'], {'repetition_penalty': 1.2, 'vendor': 2})
        invalid = dict(profile, sampling={'repeat_penalty': 1.1, 'repetition_penalty': 1.2})
        with self.assertRaisesRegex(ValueError, 'одну настройку'):
            settings.apply(invalid, {}, validate=True)

    def test_null_sampling_block_can_be_read_without_materializing(self):
        profile = dict(self.strata(), sampling=None)
        self.assertEqual(settings.apply(profile, settings.field_values(profile)), profile)
        result = settings.apply(profile, {'temperature': 1})
        self.assertEqual(result['sampling'], {'temperature': 1.0})

    def test_known_valid_prefill_forms(self):
        for value in ('auto', 'auto:8192', 'auto:16384', 'auto:32768', '0', 512):
            with self.subTest(value=value):
                result = settings.apply(self.strata(), {'prefill': value})
                self.assertEqual(settings.field_values(result)['prefill'], str(value))

    def test_ranges_types_and_nonfinite_values_are_rejected(self):
        invalid = {'parallel': [0, 9, 2.5, True], 'top_k': [0, 65, 1.5, False],
                   'pool_workers': [-1, 1.1, '1.0', True], 'top_p': [0, 1.01, True],
                   'temperature': [-.1, 'nan', float('inf'), True], 'seed': [0, -1, .5, True],
                   'repetition_penalty': [0, -.1], 'min_p': [-.1, 1.01],
                   'reasoning_effort': ['auto', True], 'prefill': ['automatic', 'auto:0', '-1', 1.5],
                   'idle_unload_s': [-1, 'nan', False], 'min_free_vram_mib': [1.5, -1]}
        for key, values in invalid.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    settings.apply(self.strata(), {key: value})

    def test_validate_checks_saved_settings_without_changes(self):
        profiles = [dict(self.strata(), sampling={'top_k': 0}),
                    dict(self.strata(), idle_unload_s=float('inf')),
                    dict(self.strata(), args=['--pool-workers', '1.5']),
                    dict(self.poly(), engine_settings={'poly_prompt_batch': 4097})]
        for profile in profiles:
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                settings.apply(profile, {}, validate=True)

    def test_integer_json_values_do_not_round_through_float(self):
        exact = 9007199254740993
        for value in (exact, str(exact)):
            with self.subTest(value=value):
                result = settings.apply(self.strata(), {'reasoning_budget_tokens': value})
                self.assertEqual(result['reasoning_budget_tokens'], exact)
                self.assertIsInstance(result['reasoning_budget_tokens'], int)
                self.assertEqual(settings.field_values(result)['reasoning_budget_tokens'], str(exact))
                self.assertEqual(settings.apply(result, settings.field_values(result)), result)
        for value in (1.5, '1.5', True, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                settings.apply(self.strata(), {'reasoning_budget_tokens': value})

    def test_confirmed_native_integer_bounds(self):
        limits = {'pool_workers': 2147483647, 'prompt_cache': 2147483647,
                  'conversation_cache_slots': 2147483647, 'conversation_cache_mib': 8796093022207,
                  'conversation_cache_min_free_mib': 8796093022207}
        for key, maximum in limits.items():
            with self.subTest(key=key):
                result = settings.apply(self.strata(), {key: str(maximum)})
                self.assertEqual(settings.field_values(result)[key], maximum)
                for value in (maximum + 1, str(maximum + 1)):
                    with self.subTest(value=value), self.assertRaises(ValueError):
                        settings.apply(self.strata(), {key: value})
                invalid = dict(self.strata(), args=[next(spec.flag for spec in settings.STRATA if spec.key == key), str(maximum + 1)])
                with self.assertRaises(ValueError):
                    settings.apply(invalid, {}, validate=True)

    def test_extra_args_conflicts_cover_both_spellings(self):
        for token in ('--prefill', '--prefill=auto', '--prompt-cache=0', ' --pool-workers ',
                      '--batch', '--slots', '--batch=3', '--slots=4'):
            with self.subTest(token=token), self.assertRaisesRegex(ValueError, 'отдельным полем'):
                settings.assert_no_conflicts(token, self.strata())
        settings.assert_no_conflicts('--unknown\nvalue', self.strata())
        settings.assert_no_conflicts('--prefill\nauto', {'launcher_engine': 'custom'})

    def test_poly_overrides_do_not_materialize_inherited_defaults(self):
        runtime = {'args': ['--prompt-batch', '128'], 'sampling': {'temperature': 1.2, 'vendor': 7},
                   'engine_specific': {'keep': 'yes'}, 'idle_unload_s': 19}
        before = copy.deepcopy(runtime)
        result = settings.apply_poly_runtime(self.poly(), runtime)
        self.assertEqual(result, runtime)
        result['engine_specific']['keep'] = 'changed'
        self.assertEqual(runtime, before)
        self.assertEqual(settings.apply_poly_runtime(self.poly(), {'args': []}), {'args': []})

    def test_poly_overlay_preserves_base_sampling_and_runtime_metadata(self):
        profile = settings.apply(self.poly(), {'poly_prompt_batch': 256, 'poly_stage_mib': 64,
                                              'poly_no_draft': 'true', 'top_k': 20,
                                              'min_p': 0, 'temperature': .8,
                                              'reasoning_effort': 'low', 'idle_unload_s': 5})
        self.assertNotIn('sampling', profile)
        self.assertEqual(profile['engine_settings']['temperature'], .8)
        self.assertEqual(profile['reasoning_effort'], 'low')
        runtime = {'args': ['--model', 'x', '--prompt-batch=128', '--native-unknown', 'keep'],
                   'sampling': {'top_p': .9, 'vendor_setting': 2}, 'metadata': 4}
        before = copy.deepcopy(runtime)
        result = settings.apply_poly_runtime(profile, runtime)
        self.assertEqual(result['sampling'], {'top_p': .9, 'vendor_setting': 2, 'temperature': .8, 'top_k': 20, 'min_p': 0})
        self.assertIn('--no-draft', result['args'])
        self.assertIn('256', result['args'])
        self.assertNotIn('--prompt-batch=128', result['args'])
        self.assertEqual(result['metadata'], 4)
        self.assertEqual(result['idle_unload_s'], 5)
        self.assertEqual(runtime, before)

    def test_poly_tristate_false_removes_base_flag_and_empty_inherits(self):
        profile = settings.apply(self.poly(), {'poly_no_draft': 'false', 'poly_kv_whole': 'true'})
        self.assertIs(profile['engine_settings']['poly_no_draft'], False)
        self.assertEqual(settings.field_values(profile)['poly_no_draft'], 'false')
        runtime = {'args': ['--model', 'x', '--no-draft', '--no-draft=true']}
        result = settings.apply_poly_runtime(profile, runtime)
        self.assertNotIn('--no-draft', result['args'])
        self.assertNotIn('--no-draft=true', result['args'])
        self.assertIn('--kv-whole', result['args'])
        cleared = settings.apply(profile, {'poly_no_draft': '', 'poly_kv_whole': ''})
        self.assertNotIn('poly_no_draft', cleared['engine_settings'])
        self.assertEqual(settings.apply_poly_runtime(cleared, runtime), runtime)

    def test_poly_sampling_overlay_replaces_base_aliases_only_when_explicit(self):
        runtime = {'args': [], 'sampling': {'repeat_penalty': 1.1, 'repeat_last_n': 64, 'vendor': 7}}
        self.assertEqual(settings.apply_poly_runtime(self.poly(), runtime), runtime)
        profile = settings.apply(self.poly(), {'repetition_penalty': 1.2, 'penalty_last_n': 128})
        result = settings.apply_poly_runtime(profile, runtime)
        self.assertEqual(result['sampling'], {'repetition_penalty': 1.2, 'penalty_last_n': 128, 'vendor': 7})
        self.assertEqual(runtime['sampling']['repeat_penalty'], 1.1)

    def test_poly_advanced_ranges(self):
        for key, value in [('poly_prompt_batch', 0), ('poly_prompt_batch', 4097),
                           ('poly_stage_mib', 0), ('poly_stage_ahead', -1),
                           ('poly_lookup', 1.5), ('poly_no_draft', 'yes'), ('poly_kv_whole', 1)]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                settings.apply(self.poly(), {key: value})
        result = settings.apply(self.poly(), {'poly_prompt_cache': 0, 'poly_prompt_cache_every': 0})
        self.assertEqual(result['engine_settings']['poly_prompt_cache'], 0)


if __name__ == '__main__':
    unittest.main()
