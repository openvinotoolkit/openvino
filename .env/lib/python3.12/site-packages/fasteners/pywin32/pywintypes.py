from ctypes import c_void_p
from ctypes import Structure
from ctypes import Union
from ctypes.wintypes import DWORD
from ctypes.wintypes import HANDLE

# Definitions for OVERLAPPED.
# Refer: https://docs.microsoft.com/en-us/windows/win32/api/minwinbase/ns-minwinbase-overlapped


class _DummyStruct(Structure):
    _fields_ = [
        ('Offset', DWORD),
        ('OffsetHigh', DWORD),
    ]


class _DummyUnion(Union):
    _fields_ = [
        ('_offsets', _DummyStruct),
        ('Pointer', c_void_p),
    ]


class OVERLAPPED(Structure):
    _fields_ = [
        ('Internal', c_void_p),
        ('InternalHigh', c_void_p),
        ('_offset_or_ptr', _DummyUnion),
        ('hEvent', HANDLE),
    ]
