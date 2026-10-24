"""Manual Windows system file-cache trim; never stop processes or reset the GPU."""
from __future__ import annotations

import base64
import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import subprocess
import sys


def trim_file_cache():
    # https://learn.microsoft.com/windows/win32/api/memoryapi/nf-memoryapi-setsystemfilecachesize
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    security = ctypes.WinDLL('advapi32', use_last_error=True)

    class LUID(ctypes.Structure):
        _fields_ = [('low', wintypes.DWORD), ('high', wintypes.LONG)]

    class Privileges(ctypes.Structure):
        _fields_ = [('count', wintypes.DWORD), ('luid', LUID), ('attributes', wintypes.DWORD)]

    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    security.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    security.LookupPrivilegeValueW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, ctypes.POINTER(LUID)]
    security.AdjustTokenPrivileges.argtypes = [wintypes.HANDLE, wintypes.BOOL, ctypes.POINTER(Privileges),
                                              wintypes.DWORD, ctypes.POINTER(Privileges), ctypes.POINTER(wintypes.DWORD)]
    kernel.SetSystemFileCacheSize.argtypes = [ctypes.c_size_t, ctypes.c_size_t, wintypes.DWORD]
    token = wintypes.HANDLE()
    if not security.OpenProcessToken(kernel.GetCurrentProcess(), 0x20 | 0x8, ctypes.byref(token)):
        raise ctypes.WinError(ctypes.get_last_error())
    previous = Privileges()
    changed = False
    try:
        privilege = Privileges(count=1, attributes=2)
        if not security.LookupPrivilegeValueW(None, 'SeIncreaseQuotaPrivilege', ctypes.byref(privilege.luid)):
            raise ctypes.WinError(ctypes.get_last_error())
        size = wintypes.DWORD()
        ctypes.set_last_error(0)
        if not security.AdjustTokenPrivileges(token, False, ctypes.byref(privilege), ctypes.sizeof(previous),
                                               ctypes.byref(previous), ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        if ctypes.get_last_error() == 1300:
            raise ctypes.WinError(1300)
        changed = True
        if not kernel.SetSystemFileCacheSize(ctypes.c_size_t(-1).value, ctypes.c_size_t(-1).value, 0):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        if changed:
            security.AdjustTokenPrivileges(token, False, ctypes.byref(previous), 0, None, None)
        kernel.CloseHandle(token)


def _elevated_trim():
    # Only the fixed helper is elevated; profiles and user-supplied commands are never passed to it.
    quote = lambda value: "'" + str(value).replace("'", "''") + "'"
    arguments = subprocess.list2cmdline([str(Path(__file__).resolve()), '--system-file-cache'])
    script = ("$ErrorActionPreference='Stop'; try { $p = Start-Process -FilePath " + quote(sys.executable)
              + ' -ArgumentList ' + quote(arguments)
              + " -Verb RunAs -WindowStyle Hidden -Wait -PassThru; exit $p.ExitCode } catch { exit 1223 }")
    encoded = base64.b64encode(script.encode('utf-16-le')).decode('ascii')
    powershell = Path(os.environ['SystemRoot']) / 'System32/WindowsPowerShell/v1.0/powershell.exe'
    result = subprocess.run([str(powershell), '-NoProfile', '-NonInteractive', '-EncodedCommand', encoded],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), timeout=180)
    if result.returncode == 1223:
        raise RuntimeError('Очистка RAM отменена или не удалось получить права администратора.')
    if result.returncode:
        raise RuntimeError('Windows не выполнила очистку файлового кэша RAM (код ' + str(result.returncode) + ').')


def clear_system_cache():
    if os.name != 'nt':
        raise ValueError('Очистка системного файлового кэша этой кнопкой поддерживается только в Windows.')
    try:
        trim_file_cache()
    except OSError as exc:
        if exc.winerror not in {5, 1300, 1314}:
            raise
        _elevated_trim()


if __name__ == '__main__':
    if sys.argv[1:] != ['--system-file-cache'] or os.name != 'nt':
        sys.exit(2)
    try:
        trim_file_cache()
    except OSError as exc:
        sys.exit(exc.winerror or 1)
