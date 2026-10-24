"""Installer routing and relocation checks; no GPU, network or model weights."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import control_install as installer


class InstallTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="Strata [test] ")
        self.root = Path(self.temp.name)
        self.root_patch = patch.object(installer, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.addCleanup(self.temp.cleanup)

    def test_skip_installs_components_without_preparing_weights_or_opening(self):
        with patch.object(installer, "load_setup") as load, \
             patch.object(installer, "install_components") as components, \
             patch.object(installer, "prepare_model") as model, \
             patch.object(installer.subprocess, "Popen") as launch:
            self.assertEqual(installer.main(["--yes", "--no-open"]), 0)
        components.assert_called_once_with(load.return_value, yes=True)
        model.assert_not_called()
        launch.assert_not_called()
        self.assertTrue((self.root / "Strata-data").is_dir())

    def test_both_engines_skip_weights(self):
        with patch.object(installer, "load_setup") as load, \
             patch.object(installer, "poly_sources", return_value=self.root / "Strata-data" / "polystrata"), \
             patch.object(installer, "install_components") as components, \
             patch.object(installer, "prepare_model") as model:
            installer.main(["--engine", "both", "--no-model", "--yes", "--no-open"])
        self.assertEqual(components.call_count, 2)
        model.assert_not_called()

    def test_interactive_defaults_skip_model(self):
        with patch("builtins.input", side_effect=["", ""]), \
             patch.object(installer, "load_setup"), \
             patch.object(installer, "install_components"), \
             patch.object(installer, "prepare_model") as model:
            installer.main(["--no-open"])
        model.assert_not_called()

    def test_uncensored_routes_only_to_poly(self):
        with patch.object(installer, "load_setup") as load, \
             patch.object(installer, "poly_sources", return_value=self.root / "polystrata"), \
             patch.object(installer, "install_components"), \
             patch.object(installer, "prepare_model") as model:
            installer.main(["--engine", "both", "--model-choice", "uncensored", "--quant", "Q3_K_M", "--yes", "--no-open"])
        model.assert_called_once_with(load.return_value, self.root / "Strata-data", ["--yes"],
                                      poly=True, uncensored=True, quant="Q3_K_M")

    def test_failure_does_not_open_control(self):
        with patch.object(installer, "load_setup"), \
             patch.object(installer, "install_components", side_effect=RuntimeError("failed")), \
             patch.object(installer.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(RuntimeError, "failed"):
                installer.main(["--yes"])
        launch.assert_not_called()

    def test_model_flags_cannot_silently_trigger_download_or_override_local_paths(self):
        for args in (["--yes", "--family", "coder"],
                     ["--model-choice", "standard", "--data-dir=C:\\elsewhere"],
                     ["--model-choice", "standard", "--host", "0.0.0.0"],
                     ["--yes", "--model-choice", "uncensored", "--engine", "strata"]):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), \
                 self.assertRaises(SystemExit) as result:
                installer.main(args)
            self.assertEqual(result.exception.code, 2)

    def test_uncensored_uses_own_weights_hashes_and_no_apex_draft(self):
        module = SimpleNamespace(FAMILIES={"ornith": {
            "file": "APEX.gguf", "models": {"APEX": {}}, "sha256": {}, "draft": "apex-draft", "profile": "apex-profile"}})
        installer.add_uncensored(module)
        family = module.FAMILIES["ornith"]
        self.assertNotIn("draft", family)
        self.assertNotIn("profile", family)
        self.assertEqual(set(family["models"]), {"Q4_K_M", "Q3_K_M"})
        self.assertIn(installer.ORNITH_REVISION, family["hf"])
        for quant, expected in installer.ORNITH_FILES.items():
            name = family["file"].format(q=quant)
            self.assertEqual(family["sha256"][name], expected)

    def test_poly_preparation_creates_control_profile_and_keeps_it_on_rerun(self):
        poly = self.root / "Strata-data" / "polystrata"
        poly.mkdir(parents=True)
        base = poly / "strata-ornith-uncensored-q4_k_m.json"
        model = self.root / "Strata-data" / "models" / "ornith.gguf"
        def prepare():
            self.assertIn("--no-start", sys.argv)
            self.assertEqual(sys.argv[sys.argv.index("--data-dir") + 1], str(self.root / "Strata-data"))
            base.write_text(json.dumps({"args": ["--model", str(model)]}), encoding="utf-8")
            return 0
        module = SimpleNamespace(ROOT=poly, main=prepare)
        with patch.object(installer, "add_uncensored"):
            installer.prepare_model(module, self.root / "Strata-data", [], poly=True, uncensored=True)
            target = self.root / ".strata-control" / "profiles" / (base.stem + ".json")
            data = json.loads(target.read_text())
            self.assertEqual(data["engine_settings"]["poly_config"], str(base))
            target.write_text('{"custom": true}', encoding="utf-8")
            installer.prepare_model(module, self.root / "Strata-data", [], poly=True, uncensored=True)
            self.assertEqual(json.loads(target.read_text()), {"custom": True})

    @unittest.skipUnless(os.name == "nt", "Windows launcher")
    def test_bat_bootstrap_in_relocated_unicode_folder_without_weights(self):
        root = self.root / "папка [Strata] & test !"
        root.mkdir()
        source = Path(installer.__file__).parent
        for name in ("INSTALL.bat", "INSTAL.bat", "INSTALL-PYTHON.ps1", "control_install.py"):
            path = source / "tools" / "INSTALL-LOCAL.bat" if name == "INSTALL.bat" and (source / "tools" / "INSTALL-LOCAL.bat").is_file() else source / name
            shutil.copyfile(path, root / name)
        (root / "setup.py").write_text('''from pathlib import Path
ROOT = Path(__file__).parent
PREBUILT_URL = "fixture"
CUDA12_MIN_DRIVER = 528
MIN_DRIVER = 580
def requirement_lines(): return []
def pip_install(*a): (ROOT / ".dependencies-ready").touch()
def gpu_info(): return dict(name="Fixture GPU", arch=89)
def gpu_problem(gpu): return None
def cuda_choice(*a): return 13, ""
def driver_major(gpu): return 600
def get_prebuilt(*a, **kw):
    (ROOT / "engine").mkdir(exist_ok=True)
    return ROOT / "engine"
def pip_cuda_libs(*a): pass
def main(): raise AssertionError("Model setup must not run")
''', encoding="utf-8")
        # Exercise bootstrap with a real compatible interpreter, without installing Python.
        env = dict(os.environ, PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"])
        command = f'"{os.environ.get("COMSPEC", "cmd.exe")}" /d /s /c ""{root / "INSTAL.bat"}" --yes --no-model --no-open"'
        result = subprocess.run(command,
            cwd=self.root, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((root / ".venv" / "Scripts" / "python.exe").is_file())
        self.assertTrue((root / ".dependencies-ready").is_file())
        self.assertTrue((root / "Strata-data").is_dir())
        self.assertFalse(list(root.rglob("*.gguf")))


if __name__ == "__main__":
    unittest.main()
