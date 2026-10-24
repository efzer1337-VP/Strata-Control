"""No system cache is changed and no administrator prompt is opened by these tests."""
import ctypes
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import strata_memory_cleanup as cleanup


@unittest.skipUnless(os.name == 'nt', 'Windows API bindings')
class CleanupTests(unittest.TestCase):
    def test_flush_uses_pointer_sized_sentinel_and_restores_privilege(self):
        kernel, security = Mock(), Mock()
        kernel.GetCurrentProcess.return_value = 42
        with patch.object(ctypes, 'WinDLL', side_effect=[kernel, security]):
            cleanup.trim_file_cache()
        sentinel = ctypes.c_size_t(-1).value
        kernel.SetSystemFileCacheSize.assert_called_once_with(sentinel, sentinel, 0)
        self.assertEqual(security.AdjustTokenPrivileges.call_count, 2)
        kernel.CloseHandle.assert_called_once()

    def test_flush_error_restores_privilege_and_closes_handle(self):
        kernel, security = Mock(), Mock()
        kernel.SetSystemFileCacheSize.return_value = 0
        with patch.object(ctypes, 'WinDLL', side_effect=[kernel, security]), \
             patch.object(ctypes, 'get_last_error', return_value=5):
            with self.assertRaises(OSError): cleanup.trim_file_cache()
        self.assertEqual(security.AdjustTokenPrivileges.call_count, 2)
        kernel.CloseHandle.assert_called_once()

    def test_access_denied_uses_one_time_elevation(self):
        with patch.object(cleanup, 'trim_file_cache', side_effect=ctypes.WinError(1300)), \
             patch.object(cleanup, '_elevated_trim') as elevate:
            cleanup.clear_system_cache()
        elevate.assert_called_once_with()

    def test_unexpected_error_does_not_elevate(self):
        with patch.object(cleanup, 'trim_file_cache', side_effect=ctypes.WinError(87)), \
             patch.object(cleanup, '_elevated_trim') as elevate:
            with self.assertRaises(OSError): cleanup.clear_system_cache()
        elevate.assert_not_called()

    def test_cancelled_uac_is_reported(self):
        with patch.object(cleanup.subprocess, 'run', return_value=Mock(returncode=1223)):
            with self.assertRaisesRegex(RuntimeError, 'отменена'):
                cleanup._elevated_trim()


if __name__ == '__main__':
    unittest.main()
