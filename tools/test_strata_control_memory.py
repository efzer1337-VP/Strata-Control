"""Memory safety tests; never start Strata or allocate model/GPU memory."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import strata_control as control


def form(ram="8", vram="3", vision=False):
    data = SimpleNamespace(
        config_path=Path("profile.json"),
        config_data={"args": ["--native", "model.gguf", "--resident-experts"]},
        _int_value=control.StrataControl._int_value,
        _read_extra_args=lambda: ["--native", "model.gguf"],
        _read_environment=lambda: {},
    )
    for name, value in {
        "context": "262144", "kv_resident": "32768", "spec": "4", "port": "8080",
        "spec_min_p": "0.5", "expert_cache": "auto", "kv": "k8v4", "vram_reserve": vram,
        "ram_reserve": ram, "resident": True, "vision": vision,
    }.items():
        setattr(data, name + "_var", SimpleNamespace(get=lambda value=value: value))
    return data


class MemoryConfigTests(unittest.TestCase):
    def setUp(self):
        self.memory = patch.object(control.psutil, "virtual_memory", return_value=SimpleNamespace(total=32 * control.GIB, available=24 * control.GIB))
        self.gpu = patch.object(control, "_gpu_total_gib", return_value=16)
        self.memory.start()
        self.gpu.start()
        self.addCleanup(self.memory.stop)
        self.addCleanup(self.gpu.stop)

    def test_reserves_apply_without_vision_and_replace_unbounded_residency(self):
        cfg = control.StrataControl._updated_config(form())
        self.assertEqual(control._option_value(cfg["args"], "--vram-reserve-mib"), "3072")
        self.assertEqual(float(control._option_value(cfg["args"], "--resident-budget-gib")), 24)
        self.assertNotIn("--resident-experts", cfg["args"])
        self.assertEqual(cfg["launcher_env"]["STRATA_RESIDENT_HEADROOM_GIB"], "8")
        self.assertEqual(cfg["launcher_memory"]["ram_reserve_gib"], 8)
        self.assertEqual(control._option_value(cfg["args"], "--max-context"), "262144")

    def test_fractional_and_comma_values(self):
        cfg = control.StrataControl._updated_config(form(ram="8,5", vram="2.5"))
        self.assertEqual(control._option_value(cfg["args"], "--vram-reserve-mib"), "2560")
        self.assertEqual(cfg["launcher_env"]["STRATA_RESIDENT_HEADROOM_GIB"], "8.5")

    def test_disable_ram_reserve_removes_managed_budget_and_headroom(self):
        f = form(ram="0")
        f.config_data = {"args": ["--resident-budget-gib", "24"], "launcher_memory": {"ram_reserve_gib": 8}}
        f._read_environment = lambda: {"STRATA_RESIDENT_HEADROOM_GIB": "8", "EXAMPLE": "value"}
        cfg = control.StrataControl._updated_config(f)
        self.assertNotIn("--resident-budget-gib", cfg["args"])
        self.assertIn("--resident-experts", cfg["args"])
        self.assertEqual(cfg["launcher_env"], {"EXAMPLE": "value"})

    def test_legacy_explicit_budget_is_preserved_without_new_reserve(self):
        f = form(ram="0")
        f.config_data = {"args": ["--resident-budget-gib", "12"]}
        cfg = control.StrataControl._updated_config(f)
        self.assertEqual(control._option_value(cfg["args"], "--resident-budget-gib"), "12")

    def test_invalid_reserves_and_fixed_gpu_cache_are_rejected(self):
        for ram, vram in [("nan", "3"), ("8", "inf"), ("-1", "3"), ("32", "3"), ("8", "16")]:
            with self.subTest(ram=ram, vram=vram), self.assertRaises(ValueError):
                control.StrataControl._updated_config(form(ram=ram, vram=vram))
        f = form()
        f.expert_cache_var = SimpleNamespace(get=lambda: "1024")
        with self.assertRaises(ValueError):
            control.StrataControl._updated_config(f)


class MemoryGuardTests(unittest.TestCase):
    def test_web_url_uses_separate_browser_origin(self):
        self.assertEqual(control._web_url(8080), "http://localhost:8080/")

    def test_open_web_uses_port_of_running_profile(self):
        app = SimpleNamespace(config_path=Path("profile.json"),
                              _status_cached={"path": "profile.json", "port": 8080,
                                              "health": {"service": "strata"}})
        with patch.object(control.threading, "Thread") as worker:
            control.StrataControl._open_web(app)
            self.assertEqual(worker.call_args.kwargs["args"], ("http://localhost:8080/",))
            worker.return_value.start.assert_called_once()

    def test_sustained_pressure_stops_only_target_and_records_reason(self):
        proc = Mock()
        proc.create_time.return_value = 123
        proc.is_running.return_value = True
        samples = [7, 9, 7, 7, 7]
        with tempfile.TemporaryDirectory() as folder:
            state = Path(folder) / "guard.json"
            with patch.object(control.psutil, "Process", return_value=proc), patch.object(control.psutil, "virtual_memory", side_effect=[SimpleNamespace(available=n * control.GIB) for n in samples]), patch.object(control, "_terminate_tree") as stop, patch.object(control.time, "sleep"):
                control._memory_guard(555, 123, 8, state)
                stop.assert_called_once_with(proc)
            record = json.loads(state.read_text())
            self.assertEqual(record["status"], "stopped_for_ram")
            self.assertEqual(record["available_gib"], 7)

    def test_reused_pid_is_never_stopped(self):
        proc = Mock()
        proc.create_time.return_value = 456
        with patch.object(control.psutil, "Process", return_value=proc), patch.object(control, "_terminate_tree") as stop:
            control._memory_guard(555, 123, 8, Path("unused.json"))
            stop.assert_not_called()

    @unittest.skipUnless(os.name == "nt", "Windows process-tree integration")
    def test_guard_cli_stops_only_its_owned_sleep_process(self):
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], creationflags=flags)
        try:
            created = control.psutil.Process(sleeper.pid).create_time()
            with tempfile.TemporaryDirectory() as folder:
                state = Path(folder) / "guard.json"
                guard = subprocess.run(
                    [sys.executable, str(Path(control.__file__).resolve()), "--memory-guard",
                     "--pid", str(sleeper.pid), "--created", str(created),
                     "--ram-reserve-gib", str(control.psutil.virtual_memory().total / control.GIB + 1),
                     "--state", str(state)],
                    capture_output=True, text=True, timeout=10, creationflags=flags,
                )
                self.assertEqual(guard.returncode, 0, guard.stderr)
                sleeper.wait(timeout=3)
                self.assertEqual(json.loads(state.read_text())["status"], "stopped_for_ram")
        finally:
            if sleeper.poll() is None:
                control._terminate_tree(control.psutil.Process(sleeper.pid))
                sleeper.wait(timeout=3)

    def test_theme_and_memory_widgets_construct_without_starting_server(self):
        with patch.object(control.StrataControl, "_start_server") as start:
            app = control.StrataControl()
            try:
                app.withdraw()
                app.update_idletasks()
                self.assertTrue(app.memory_summary.winfo_exists())
                self.assertEqual(app.start_button.cget("style"), "Primary.TButton")
                start.assert_not_called()
            finally:
                app.destroy()


if __name__ == "__main__":
    unittest.main()
