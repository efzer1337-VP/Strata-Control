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
                                                  "--expert-profile", "profile.txt"]}), encoding="utf-8")
        self.root_patch = patch.object(engines, "POLY_ROOT", root)
        self.exe_patch = patch.object(engines, "POLY_EXE", root / "engine-poly" / "strata-poly.exe")
        self.root_patch.start()
        self.exe_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.addCleanup(self.exe_patch.stop)

    def config(self, **settings):
        return {"port": 8094, "engine_settings": {"poly_config": str(self.base), "model": str(self.model),
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


if __name__ == "__main__":
    unittest.main()
