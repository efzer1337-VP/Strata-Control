"""Control contracts with temporary profiles; no model or GPU is started."""
from __future__ import annotations

import copy
import json
import os
import shutil
import socket
import struct
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
import urllib.error
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import strata_control_web as web


def write_gguf(path, architecture='qwen36moe', tokenizer='gpt2', dimensions=(2,2), quant=0):
    def string(value):
        value = value.encode()
        return struct.pack('<Q',len(value)) + value
    header = b'GGUF' + struct.pack('<IQQ',3,1,2)
    for key,value in [('general.architecture',architecture),('tokenizer.ggml.model',tokenizer)]:
        header += string(key) + struct.pack('<I',8) + string(value)
    header += string('blk.0.weight') + struct.pack('<I',len(dimensions))
    header += struct.pack('<' + 'Q' * len(dimensions),*dimensions) + struct.pack('<IQ',quant,0)
    path.write_bytes(header)


class ControlTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.model = self.root / 'ornith.gguf'
        self.model.write_bytes(b'fixture')
        self.records = {
            'strata': {'name': 'Strata', 'kind': 'strata', 'installed': False},
            'polystrata': {'name': 'PolyStrata', 'kind': 'polystrata', 'installed': False},
            'custom-test': {'name': 'Custom', 'kind': 'custom', 'installed': False},
            'llama.cpp': {'name': 'llama.cpp', 'kind': 'llama.cpp', 'installed': False},
        }
        for target, name, value in [
            (web.control, 'PROFILES_DIR', self.root / 'profiles'),
            (web.control, 'CONTROL_STATE', self.root / 'state.json'),
            (web.control, 'LOG_PATH', self.root / 'log.txt'),
        ]:
            p = patch.object(target, name, value)
            p.start()
            self.addCleanup(p.stop)
        for p in [patch.object(web.engines, 'engines', return_value=self.records),
                  patch.object(web.control, '_configs', side_effect=lambda: sorted(self.root.glob('*.profile.json')) + sorted(web.control.PROFILES_DIR.glob('*.json'))),
                  patch.object(web.Controller, 'monitor'),
                  patch.object(web.control, '_server_processes', return_value=[])]:
            p.start()
            self.addCleanup(p.stop)

    def profile(self, engine, name, **settings):
        data = dict(launcher_engine=engine, launcher_profile_name=name, port=8081,
                    launcher_env={}, launcher_memory={'ram_reserve_gib': 0},
                    engine_settings={'model':str(self.model), 'context':32768, **settings})
        path = self.root / (name + '.profile.json')
        web.write_json(path, data)
        return str(path.resolve()), data

    def test_custom_does_not_advertise_llama_capabilities(self):
        _, data = self.profile('custom-test', 'custom')
        keys = {f['key'] for f in web.fields(data)}
        self.assertIn('extra_args', keys)
        self.assertNotIn('thinking', keys)
        self.assertNotIn('gpu_layers', keys)
        self.assertNotIn('mmproj', keys)

    def test_cross_engine_settings_are_rejected(self):
        path, data = self.profile('polystrata','poly')
        with self.assertRaisesRegex(ValueError,'не поддерживает'):
            web.updated(Path(path),data,{'kv':'q4_0'})

    def test_poly_sampling_inherits_the_prepared_models_defaults(self):
        _, data = self.profile('polystrata','poly')
        values = web.field_values(data)
        self.assertEqual(values['vram_reserve'],2)
        self.assertEqual(values['cache_ram_mib'],0)
        self.assertEqual(values['temperature'],'')
        self.assertEqual(values['top_k'],'')

    def test_invalid_import_contract_does_not_get_written(self):
        with self.assertRaisesRegex(ValueError,'engine_settings'):
            web.updated(self.root/'bad.json',{'engine_settings':[]},{},validate=True)
        self.assertFalse((self.root/'bad.json').exists())

    def test_import_rejects_invalid_poly_environment(self):
        path, data = self.profile('polystrata','poly')
        data['launcher_env'] = {'BAD=NAME':'value'}
        with self.assertRaisesRegex(ValueError,'Переменная окружения'):
            web.updated(Path(path),data,{},validate=True)

    def test_import_rejects_malformed_sampling_block(self):
        path, data = self.profile('strata','native')
        data['sampling'] = []
        with self.assertRaisesRegex(ValueError,'sampling'):
            web.updated(Path(path),data,{},validate=True)

    def test_poly_preflight_rejects_another_models_prepared_assets(self):
        write_gguf(self.model)
        other = self.root/'other.gguf'
        write_gguf(other,architecture='other')
        base = self.root / 'poly-base.json'
        base.write_text(json.dumps({'args':['--model',str(other)]}))
        path, data = self.profile('polystrata','poly',poly_config=str(base))
        errors = web.Controller().preflight(Path(path),data)
        self.assertTrue(any('не соответствует' in error for error in errors),errors)

    def test_poly_allows_quantization_variants_with_the_same_prepared_contract(self):
        write_gguf(self.model,quant=0)
        other = self.root/'q4.gguf'
        write_gguf(other,quant=1)
        self.assertTrue(web.compatible_poly_models(self.model,other))
        for changes in [{'architecture':'other'}, {'tokenizer':'llama'}, {'dimensions':(3,2)}]:
            with self.subTest(changes=changes):
                write_gguf(other,**changes)
                self.assertFalse(web.compatible_poly_models(self.model,other))

    def test_poly_corrupt_header_has_a_readable_error(self):
        self.model.write_bytes(b'GGUF')
        with self.assertRaisesRegex(ValueError,'Не удалось сверить'):
            web.compatible_poly_models(self.model,self.model)

    def test_poly_hides_q8_for_an_unsupported_model_architecture(self):
        write_gguf(self.model,architecture='other')
        _, data = self.profile('polystrata','poly',poly_kv='fp16')
        field = next(row for row in web.fields(data) if row['key'] == 'poly_kv')
        self.assertEqual(field['choices'],['fp16'])

    def test_unchanged_external_defaults_are_not_materialized(self):
        path, data = self.profile('llama.cpp', 'llama')
        result = web.updated(Path(path), data, web.field_values(data))
        self.assertEqual(result, data)

    def test_unchanged_strata_profile_keeps_the_same_engine_arguments(self):
        data = {'exe':'existing-strata.exe','tokenizer':'existing-tokenizer',
                'args':['--native','existing.gguf','--pack','existing-pack','--kv','k8v4',
                        '--max-context','163840','--spec','0','--vram-reserve-mib','1024'],
                'launcher_memory':{'ram_reserve_gib':0},'user_metadata':{'keep':True}}
        original = copy.deepcopy(data)
        self.assertEqual(web.updated(self.root/'strata.json',data,web.field_values(data)),original)
        self.assertEqual(data,original)

    def test_editing_one_model_does_not_change_another_model_on_the_same_engine(self):
        first, _ = self.profile('llama.cpp','first')
        second, _ = self.profile('llama.cpp','second',model=str(self.root/'second.gguf'))
        before = Path(second).read_bytes()
        c = web.Controller()
        c.save(first,{'context':65536})
        self.assertEqual(Path(second).read_bytes(),before)

    def test_poly_rejects_fractional_integer_instead_of_truncating(self):
        path, data = self.profile('polystrata', 'poly')
        with patch.object(web.engines, 'command_for', return_value=['python']):
            with self.assertRaises(ValueError):
                web.updated(Path(path), data, {'context': 32768.5})

    def test_poly_validates_ram_reserve(self):
        path, data = self.profile('polystrata', 'poly')
        with patch.object(web.engines, 'command_for', return_value=['python']):
            for value in [-1, 'nan', 'inf']:
                with self.subTest(value=value), self.assertRaises(ValueError):
                    web.updated(Path(path), data, {'ram_reserve': value})

    def test_preview_never_materializes_poly_runtime(self):
        path, data = self.profile('polystrata', 'poly')
        c = web.Controller()
        with patch.object(web.engines, 'command_for', return_value=['python', '--config', 'runtime.json']) as command:
            c.preview(Path(path), data)
            self.assertFalse(command.call_args.kwargs.get('materialize', False))

    def test_poly_preview_includes_edited_runtime_without_writing_files(self):
        path, data = self.profile('polystrata', 'poly')
        c = web.Controller()
        before = set(self.root.rglob('*'))
        with patch.object(web.engines,'command_for',return_value=['python','--config','missing-runtime.json']), \
             patch.object(web.engines,'poly_runtime_config',return_value={'port':9001,'args':['--max-context','65536']}):
            preview = c.preview(Path(path),data)
        self.assertIn('65536', preview)
        self.assertIn('$configPath', preview)
        self.assertNotIn("'missing-runtime.json'",preview)
        self.assertEqual(set(self.root.rglob('*')),before)

    def test_engine_selection_remembers_profiles_and_has_empty_state(self):
        first, _ = self.profile('strata', 'strata')
        second, _ = self.profile('polystrata', 'poly')
        c = web.Controller()
        c.select(first)
        c.select_engine('polystrata')
        self.assertEqual(c.selected, second)
        c.select_engine('llama.cpp')
        self.assertEqual(c.selected, '')
        self.assertEqual(c.state()['fields'], [])
        c.select_engine('strata')
        self.assertEqual(c.selected, first)
        self.assertEqual(web.Controller().selected, first)

    def test_save_as_keeps_source_and_other_engine_unchanged(self):
        first, original = self.profile('llama.cpp', 'llama')
        other, _ = self.profile('polystrata', 'poly')
        before = Path(other).read_bytes()
        c = web.Controller()
        result = c.save(first, {'context': 65536}, 'copy', new=True)
        self.assertEqual(json.loads(Path(first).read_text()), original)
        self.assertEqual(Path(other).read_bytes(), before)
        _, saved = c.profile(result['profile_id'])
        self.assertEqual(saved['engine_settings']['context'], 65536)
        self.assertEqual(saved['launcher_engine'], 'llama.cpp')

    def test_start_does_not_start_a_second_managed_profile(self):
        first, _ = self.profile('llama.cpp', 'llama')
        self.profile('polystrata', 'poly')
        c = web.Controller()
        with patch.object(c, 'active_servers', return_value=[{'id': 'other', 'name': 'Other'}]), \
             patch.object(c, 'save') as save, patch.object(web.subprocess, 'Popen') as start:
            with self.assertRaisesRegex(ValueError, 'Other'):
                c.start(first, {})
            save.assert_not_called()
            start.assert_not_called()

    def test_state_does_not_reuse_another_profiles_status(self):
        first, _ = self.profile('llama.cpp', 'llama')
        second, _ = self.profile('polystrata', 'poly')
        c = web.Controller()
        c.status = {'profile_id': first, 'running': True, 'ready': True, 'port': 9999}
        c.select(second)
        state = c.state()
        self.assertFalse(state['status']['ready'])
        self.assertEqual(state['status']['profile_id'], second)
        self.assertEqual(state['status']['port'], 8081)

    def test_stopped_server_has_no_previous_gpu_or_model_metrics(self):
        first, _ = self.profile('llama.cpp', 'llama')
        c = web.Controller()
        c.select(first)
        c.status = {'gpu_util': 99, 'metrics': {'live': {'tok_s': 90}}}
        fresh = c.collect_status()
        self.assertNotIn('gpu_util', fresh)
        self.assertNotIn('metrics', fresh)

    def test_restart_validates_before_stopping(self):
        first, _ = self.profile('llama.cpp', 'llama')
        c = web.Controller()
        with patch.object(c, 'stop') as stop, patch.object(c, 'start') as start:
            with self.assertRaises(ValueError): c.restart(first, {'context': 1})
            stop.assert_not_called()
            start.assert_not_called()

    def test_clear_memory_never_stops_servers(self):
        self.profile('strata', 'first')
        c = web.Controller()
        with patch.object(web.memory_cleanup, 'clear_system_cache') as clear, \
             patch.object(c, 'stop') as stop, patch.object(web.control, '_terminate_tree') as terminate:
            result = c.clear_memory(lambda message: None)
        clear.assert_called_once_with()
        stop.assert_not_called()
        terminate.assert_not_called()
        self.assertTrue(result['ram_cache_cleared'])
        self.assertFalse(result['vram_cleared'])

    def test_clear_memory_reports_error_and_releases_lock(self):
        c = web.Controller()
        with patch.object(web.memory_cleanup, 'clear_system_cache', side_effect=RuntimeError('cancelled')):
            with self.assertRaisesRegex(RuntimeError, 'cancelled'):
                c.clear_memory(lambda message: None)
        self.assertFalse(c.memory_cleanup_lock.locked())

    def test_clear_memory_rejects_duplicate_but_ignores_model_operation_lock(self):
        c = web.Controller()
        with c.operation_lock, patch.object(web.memory_cleanup, 'clear_system_cache'):
            c.clear_memory(lambda message: None)
        with c.memory_cleanup_lock:
            with self.assertRaisesRegex(ValueError, 'уже выполняется'):
                c.clear_memory(lambda message: None)

    def test_unknown_profile_and_engine_do_not_change_selection(self):
        first, _ = self.profile('llama.cpp', 'llama')
        c = web.Controller()
        c.select(first)
        for select, value in [(c.select, 'missing'), (c.select_engine, 'missing')]:
            with self.assertRaises(ValueError): select(value)
            self.assertEqual(c.selected, first)


class HTTPTests(unittest.TestCase):
    profile = ControlTests.profile

    def setUp(self):
        ControlTests.setUp(self)
        self.ident, _ = self.profile('llama.cpp', 'llama')
        self.server = web.Server(('127.0.0.1', 0))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.url = 'http://127.0.0.1:' + str(self.server.server_port)

    def request(self, path, body=None, token=web.TOKEN, origin=None):
        headers = {'Content-Type':'application/json', 'X-Control-Token':token}
        if origin: headers['Origin'] = origin
        request = urllib.request.Request(self.url + path, headers=headers,
                    data=json.dumps(body).encode() if body is not None else None)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request) as response:
            return json.loads(response.read())

    def test_http_security_and_assets(self):
        for token, origin in [('', None), (web.TOKEN, 'https://example.com')]:
            with self.subTest(token=bool(token)), self.assertRaises(urllib.error.HTTPError) as error:
                self.request('/api/state', token=token, origin=origin)
            self.assertEqual(error.exception.code, 403)
            error.exception.close()
        for asset in ['/', '/control.js', '/control.css']:
            with urllib.request.urlopen(self.url + asset) as response:
                self.assertEqual(response.status, 200)
                self.assertIn("frame-ancestors 'none'", response.headers['Content-Security-Policy'])

    def test_clear_memory_endpoint_without_selected_profile(self):
        c = self.server.controller
        c.selected = ''
        with patch.object(web.memory_cleanup, 'clear_system_cache'):
            response = self.request('/api/clear-memory', {})
            for _ in range(100):
                task = next(task for task in c.tasks if task['id'] == response['task_id'])
                if task['status'] not in {'queued', 'running'}: break
                time.sleep(.01)
        self.assertEqual(task['status'], 'done')
        self.assertFalse(task['serialized'])
        self.assertTrue(task['result']['ram_cache_cleared'])

    def test_http_selection_copy_export_import(self):
        self.request('/api/select', {'id': self.ident})
        source = Path(self.ident).read_bytes()
        saved = self.request('/api/save-as', {'profile_id': self.ident, 'name': 'copy', 'values': {'context': 65536}})['profile_id']
        exported = self.request('/api/export', {'profile_id': saved})['config']
        imported = self.request('/api/import', {'config': exported})['profile_id']
        self.assertNotEqual(saved, imported)
        self.assertEqual(json.loads(Path(saved).read_text()), json.loads(Path(imported).read_text()))
        self.assertEqual(Path(self.ident).read_bytes(), source)
        self.request('/api/select-engine', {'engine': 'polystrata'})
        state = self.request('/api/state')
        self.assertEqual(state['selected_engine'], 'polystrata')
        self.assertEqual(state['fields'], [])

    def test_queued_operation_keeps_its_original_profile(self):
        other, _ = self.profile('polystrata','poly')
        c = self.server.controller
        c.reload()
        c.select(self.ident)
        queued = []
        def task(message, function, **kwargs):
            queued.append(function)
            return {'task_id':'fixture'}
        with patch.object(c,'task',side_effect=task), patch.object(c,'start',return_value={}) as start:
            self.request('/api/start',{'values':{}})
            c.select(other)
            queued[0](lambda message: None)
            start.assert_called_once_with(self.ident,{})


@unittest.skipUnless(os.name == 'nt', 'The installed EXE adapter is Windows-specific.')
class LifecycleTests(unittest.TestCase):
    def test_real_cpu_process_start_restart_stop_and_log(self):
        """Run a tiny HTTP process through the real custom wrapper, never inference."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ['strata_control.py','strata_control_engines.py','strata_model_catalog.py','strata_control_settings.py']:
                shutil.copyfile(web.ROOT / name, root / name)
            dummy = root / 'dummy.py'
            dummy.write_text('''import argparse, json
from http.server import BaseHTTPRequestHandler, HTTPServer
p=argparse.ArgumentParser()
p.add_argument('--host'); p.add_argument('--port',type=int); p.add_argument('--model')
a=p.parse_args()
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{"status":"ok"}')
print('CPU lifecycle fixture ready',flush=True)
HTTPServer((a.host,a.port),H).serve_forever()
''', encoding='utf-8')
            model = root / 'model.gguf'
            model.write_bytes(b'not a real model')
            profiles = root / '.strata-control' / 'profiles'
            registry = {'custom-test': {'name':'CPU fixture','kind':'custom','exe':sys.executable}}
            web.write_json(root / '.strata-control' / 'engines.json', registry)
            with socket.socket() as listener:
                listener.bind(('127.0.0.1',0))
                port = listener.getsockname()[1]
            path = profiles / 'dummy.json'
            config = dict(launcher_engine='custom-test', launcher_profile_name='CPU fixture', port=port,
                          launcher_memory={'ram_reserve_gib':0}, launcher_env={},
                          engine_settings=dict(model=str(model),context=32768,extra_args=[
                              '-u',str(dummy),'--host','{host}','--port','{port}','--model','{model}']))
            web.write_json(path, config)
            from contextlib import ExitStack
            with ExitStack() as stack:
                for target, name, value in [(web,'ROOT',root),(web.control,'ROOT',root),
                    (web.control,'__file__',str(root/'strata_control.py')),(web.control,'SERVER',root/'serve'/'server.py'),
                    (web.control,'PYTHON',Path(sys.executable)),(web.control,'PROFILES_DIR',profiles),
                    (web.control,'CONTROL_STATE',root/'state.json'),(web.control,'LOG_PATH',root/'log.txt')]:
                    stack.enter_context(patch.object(target,name,value))
                stack.enter_context(patch.object(web.engines,'engines',return_value={
                    'strata':{'name':'Strata','kind':'strata','installed':False},
                    'custom-test':dict(registry['custom-test'],installed=True)}))
                stack.enter_context(patch.object(web.Controller,'monitor'))
                c = web.Controller()
                ident = str(path.resolve())
                def ready():
                    deadline = time.monotonic() + 8
                    while time.monotonic() < deadline:
                        if web.control._health(port): return
                        time.sleep(.05)
                    self.fail('The CPU fixture did not become ready: ' + web.log_tail())
                try:
                    first = c.start(ident,{})
                    ready()
                    c.select(ident)
                    self.assertTrue(c.collect_status()['ready'])
                    restarted = c.restart(ident,{'context':65536})
                    self.assertNotEqual(first['pid'],restarted['pid'])
                    ready()
                    self.assertEqual(json.loads(path.read_text())['engine_settings']['context'],65536)
                    self.assertIn('CPU lifecycle fixture ready', web.log_tail())
                    with patch.object(web.memory_cleanup, 'clear_system_cache'):
                        cleared = c.clear_memory(lambda message: None)
                    self.assertTrue(cleared['ram_cache_cleared'])
                    self.assertTrue(c.collect_status()['ready'])
                    self.assertTrue(web.control.psutil.pid_exists(restarted['pid']))
                    c.stop(ident)
                    self.assertFalse(c.collect_status()['running'])
                    self.assertIsNone(web.control._health(port))
                finally:
                    c.stop(ident)
                    for process in c.processes.values(): process.wait(timeout=5)


def serve_fixture():
    """Serve temporary GPU-free profiles for browser regression checks."""
    fixture = ControlTests()
    fixture.setUp()
    cleanup = patch.object(web.memory_cleanup, 'clear_system_cache')
    cleanup.start()
    fixture.addCleanup(cleanup.stop)
    fixture.profile('strata', 'Strata fixture')
    fixture.profile('polystrata', 'Poly fixture', poly_window=8, poly_kv='fp16')
    fixture.profile('llama.cpp', 'llama fixture', extra_args=['--load-mode', 'none'])
    fixture.profile('custom-test', 'custom fixture', extra_args=['--host','{host}','--model','{model}'])
    fixture.records['custom-empty'] = {'name':'Empty fixture','kind':'custom','installed':False}
    server = web.Server(('127.0.0.1', 18987))
    finished = threading.Event()
    def monitor():
        while not finished.is_set():
            with server.controller.lock: server.controller.status = server.controller.collect_status()
            finished.wait(.2)
    threading.Thread(target=monitor, daemon=True).start()
    try:
        print('http://127.0.0.1:18987', flush=True)
        server.serve_forever()
    finally:
        finished.set()
        server.server_close()
        fixture.doCleanups()


if __name__ == '__main__':
    if '--serve-fixture' in sys.argv: serve_fixture()
    else: unittest.main()
