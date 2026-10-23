"""PolyStrata profile translation, without a GPU or model download."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import strata_control_engines as engines


class PolyConfigTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        root = Path(self.folder.name)
        (root / "engine-poly").mkdir()
        (root / "engine-poly" / "strata-vision.exe").write_bytes(b"test")
        self.model = root / "ornith.gguf"
        self.model.write_bytes(b"test")
        self.mmproj = root / "mmproj.gguf"
        self.mmproj.write_bytes(b"test")
        self.base = root / "strata-ornith.json"
        self.base.write_text(json.dumps({"exe": "old.exe", "cwd": "old", "port": 8081,
                                         "args": ["--model", "old.gguf", "--max-context", "32768",
                                                  "--expert-profile", "profile.txt"],
                                         "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20}}), encoding="utf-8")
        self.root_patch = patch.object(engines, "POLY_ROOT", root)
        self.exe_patch = patch.object(engines, "POLY_EXE", root / "engine-poly" / "strata-poly.exe")
        self.root_patch.start()
        self.exe_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.addCleanup(self.exe_patch.stop)

    def config(self, **settings):
        return {"launcher_engine": "polystrata", "port": 8094, "engine_settings": {"poly_config": str(self.base), "model": str(self.model),
                "context": 16384, "cache_ram_mib": 2048, "vram_reserve_gib": 2, "poly_window": 8,
                "poly_threads": 16, "poly_kv": "q8_0",
                **settings}}

    def test_only_poly_flags_and_loopback(self):
        result = engines.poly_runtime_config(self.config())
        args = result["args"]
        self.assertEqual(result["host"], "127.0.0.1")
        self.assertEqual(result["port"], 8094)
        self.assertEqual(args[args.index("--model") + 1], str(self.model.resolve()))
        self.assertEqual(args[args.index("--window") + 1], "8")
        self.assertEqual(args[args.index("--threads") + 1], "16")
        self.assertEqual(args[args.index("--kv") + 1], "q8_0")
        self.assertEqual(result["sampling"], {"temperature": 1.0, "top_p": 0.95, "top_k": 20})
        self.assertNotIn("--vision", args)
        self.assertNotIn("vision", result)

    def test_vision_uses_selected_projector_and_model(self):
        result = engines.poly_runtime_config(self.config(mmproj=str(self.mmproj), vision_gpu=False))
        self.assertEqual(result["vision"]["mmproj"], str(self.mmproj.resolve()))
        self.assertEqual(result["vision"]["model"], str(self.model.resolve()))
        self.assertFalse(result["vision"]["gpu"])
        self.assertNotIn("--vision", result["args"])

    def test_rejects_invalid_model_or_outside_config(self):
        with self.assertRaisesRegex(ValueError, "GGUF"):
            engines.poly_runtime_config(self.config(model=str(self.model.with_name("missing.gguf"))))
        outside = Path(self.folder.name).parent / "outside.json"
        with self.assertRaisesRegex(ValueError, "конфигурацию"):
            engines.poly_runtime_config(self.config(poly_config=str(outside)))

    def test_native_auto_and_small_context_are_preserved(self):
        result = engines.poly_runtime_config(self.config(context=8192, poly_threads=0, poly_window=0))
        for flag, expected in (("--max-context", "8192"), ("--threads", "0"), ("--window", "0")):
            with self.subTest(flag=flag):
                self.assertEqual(result["args"][result["args"].index(flag) + 1], expected)

    def test_invalid_native_ranges_are_rejected_before_start(self):
        for key, value in (("context", 15), ("poly_threads", -1), ("poly_window", 9),
                           ("poly_window", 16), ("top_k", 0), ("top_k", 65)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                engines.poly_runtime_config(self.config(**{key: value}))

    def test_sampling_overrides_keep_inherited_values(self):
        base = json.loads(self.base.read_text(encoding="utf-8"))
        base["sampling"] = {"temperature": 0.7, "top_p": 0.8, "top_k": 40,
                            "min_p": 0.03, "presence_penalty": 1.2, "seed": 42}
        self.base.write_text(json.dumps(base), encoding="utf-8")
        result = engines.poly_runtime_config(self.config(temperature=0.6))
        self.assertEqual(result["sampling"], {**base["sampling"], "temperature": 0.6})
        self.assertEqual(json.loads(self.base.read_text(encoding="utf-8")), base)

    def test_no_sampling_override_keeps_model_defaults(self):
        base = json.loads(self.base.read_text(encoding="utf-8"))
        base.pop("sampling")
        self.base.write_text(json.dumps(base), encoding="utf-8")
        result = engines.poly_runtime_config(self.config())
        self.assertNotIn("sampling", result)

    def test_optional_settings_reach_runtime_without_mutating_prepared_json(self):
        base = json.loads(self.base.read_text(encoding="utf-8"))
        base["args"].extend(["--prompt-batch", "1024", "--prompt-cache", "2", "--no-draft"])
        self.base.write_text(json.dumps(base), encoding="utf-8")
        config = self.config(
            poly_prompt_batch=256, poly_prompt_cache=1, poly_lookup=0, poly_no_draft=False,
            poly_kv_whole=True, min_p=0.05, seed=42)
        config.update(reasoning_effort="low", reasoning_budget_tokens=4096)
        result = engines.poly_runtime_config(config)
        for flag, value in (("--prompt-batch", "256"), ("--prompt-cache", "1"), ("--lookup", "0")):
            self.assertEqual(result["args"].count(flag), 1)
            self.assertEqual(result["args"][result["args"].index(flag) + 1], value)
        self.assertNotIn("--no-draft", result["args"])
        self.assertIn("--kv-whole", result["args"])
        self.assertEqual(result["reasoning_effort"], "low")
        self.assertEqual(result["reasoning_budget_tokens"], 4096)
        self.assertEqual(result["sampling"], {**base["sampling"], "min_p": 0.05, "seed": 42})
        self.assertNotIn("--temperature", result["args"])
        self.assertEqual(json.loads(self.base.read_text(encoding="utf-8")), base)


if __name__ == "__main__":
    unittest.main()
