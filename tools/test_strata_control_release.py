"""Release boundaries and the legacy installer's engine metadata compatibility."""
import ast
import importlib.util
from pathlib import Path
import re
import textwrap
import unittest

spec = importlib.util.spec_from_file_location('bundle', Path(__file__).with_name('build_release_bundle.py'))
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)


class ReleaseTests(unittest.TestCase):
    def test_local_runtime_files_are_excluded(self):
        for name in ('Strata/.venv/python.exe', 'Strata/Strata-data/models/model.gguf',
                     'Strata/.strata-control/profiles/example.json', 'Strata/.git/config',
                     'Strata/strata-local.json', 'Strata/strata-local.json.launcher.bak',
                     'Strata/server.log', 'Strata/__pycache__/script.pyc'):
            with self.subTest(name=name):
                self.assertTrue(bundle.private_entry(name))

    def test_source_and_official_engine_files_are_allowed(self):
        for name in ('Strata/strata_control.py', 'Strata/serve/control/control.js',
                     'Strata/docs/DETAILS.md', 'Strata/engine/strata.exe',
                     'Strata/engine/engine-component.json'):
            with self.subTest(name=name):
                self.assertFalse(bundle.private_entry(name))

    def test_vision_metadata_accepts_published_string_and_legacy_list(self):
        source = textwrap.dedent('''\
        def read_archs(meta):
            va = [int(x) for x in meta.get("vision_archs", meta.get("archs", []))]
            return va
        ''').encode()
        namespace = {'re': re}
        exec(bundle.setup_compat(source), namespace)
        for value in ([75, 86, 89, 120], '75;86;89;120', '75, 86 89;120'):
            with self.subTest(value=value):
                self.assertEqual(namespace['read_archs']({'vision_archs': value}), [75, 86, 89, 120])
        self.assertEqual(namespace['read_archs']({'archs': [89]}), [89])

    def test_a_changed_upstream_parser_requires_review(self):
        with self.assertRaises(ValueError):
            bundle.setup_compat(b'def different_parser(): pass')

    def test_new_modules_and_assets_are_packaged(self):
        required = {'strata_control_settings.py', 'strata_control_presets.py',
                    'serve/control/control.js', 'serve/control/control.css',
                    'tools/test_strata_control_parameters.py'}
        self.assertTrue(required.issubset(bundle.CONTROL_FILES))


if __name__ == '__main__':
    unittest.main()
