#   Intel® Single Event API
#
#   This file is provided under the BSD 3-Clause license.
#   Copyright (c) 2021, Intel Corporation
#   All rights reserved.
#
#   Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#       Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#       Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#       Neither the name of the Intel Corporation nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
#   IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
#   HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

from __future__ import print_function
import os
import sys
import json
import time
import platform
import threading
from ctypes import cdll, c_char_p, c_void_p, c_ulonglong, c_int, c_double, c_long, c_bool, c_short, c_wchar_p, c_uint32, POINTER, CFUNCTYPE
from sea_runtool import reset_global, global_storage


class Dummy:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class Task:
    def __init__(self, itt, name, id, parent):
        self.itt = itt
        self.name = name
        self.id = id
        self.parent = parent

    def __enter__(self):
        self.itt.lib.itt_task_begin(self.itt.domain, self.id, self.parent, self.itt.get_string_id(self.name), 0)
        return self

    def arg(self, name, value):
        try:
            value = float(value)
            self.itt.lib.itt_metadata_add(self.itt.domain, self.id, self.itt.get_string_id(name), value)
        except ValueError:
            self.itt.lib.itt_metadata_add_str(self.itt.domain, self.id, self.itt.get_string_id(name), str(value))
        return self

    def blob(self, name, pointer, size):
        self.itt.lib.itt_metadata_add_blob(self.itt.domain, self.id, self.itt.get_string_id(name), pointer, size)
        return self

    def __exit__(self, type, value, traceback):
        self.itt.lib.itt_task_end(self.itt.domain, 0)
        return False


class Track:
    def __init__(self, itt, track):
        self.itt = itt
        self.track = track

    def __enter__(self):
        self.itt.lib.itt_set_track(self.track)
        return self

    def __exit__(self, type, value, traceback):
        self.itt.lib.itt_set_track(None)
        return False


def prepare_environ(args):
    if 'sea_env' in global_storage(None):
        return global_storage('sea_env')
    env = os.environ.copy()
    if args.verbose == 'info':
        env['INTEL_SEA_VERBOSE'] = '1'
    bitness = '32' if '32' in platform.architecture()[0] else '64'
    env_name = 'INTEL_LIBITTNOTIFY' + bitness
    if env_name not in env or 'SEAPI' not in env[env_name]:
        if sys.platform == 'win32':
            dl_name = 'IntelSEAPI.dll'
        elif sys.platform == 'darwin':
            dl_name = 'libIntelSEAPI.dylib'
        else:
            dl_name = 'libIntelSEAPI.so'

        env[env_name] = os.path.join(args.bindir, dl_name)
    if args.bindir not in env['PATH']:
        env['PATH'] += os.pathsep + args.bindir

    reset_global('sea_env', env)
    return global_storage('sea_env')


class ITT(object):
    scope_global = 1
    scope_process = 2
    scope_thread = 3
    scope_task = 4

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ITT, cls).__new__(cls)
        return cls._instance

    def __init__(self, domain):
        if hasattr(self, 'lib'):
            return
        bitness = 32 if '32' in platform.architecture()[0] else 64
        env_name = 'INTEL_LIBITTNOTIFY' + str(bitness)
        self.lib = None
        self.strings = {}
        self.tracks = {}
        self.counters = {}
        env = global_storage('sea_env')
        if env_name not in env:
            print("Warning:", env_name, "is not set...")
            return
        if os.path.exists(env[env_name]):
            self.lib = cdll.LoadLibrary(env[env_name])
        if not self.lib:
            print("Warning: Failed to load", env[env_name], "...")
            return

        # void* itt_create_domain(const char* str)
        self.lib.itt_create_domain.argtypes = [c_char_p]
        self.lib.itt_create_domain.restype = c_void_p

        # void* itt_create_string(const char* str)
        self.lib.itt_create_string.argtypes = [c_char_p]
        self.lib.itt_create_string.restype = c_void_p

        # void itt_marker(void* domain, uint64_t id, void* name, int scope)
        self.lib.itt_marker.argtypes = [c_void_p, c_ulonglong, c_void_p, c_int, c_ulonglong]

        # void itt_task_begin(void* domain, uint64_t id, uint64_t parent, void* name, uint64_t timestamp)
        self.lib.itt_task_begin.argtypes = [c_void_p, c_ulonglong, c_ulonglong, c_void_p, c_ulonglong]

        # void itt_task_begin_overlapped(void* domain, uint64_t id, uint64_t parent, void* name, uint64_t timestamp)
        self.lib.itt_task_begin_overlapped.argtypes = [c_void_p, c_ulonglong, c_ulonglong, c_void_p, c_ulonglong]

        # void itt_metadata_add(void* domain, uint64_t id, void* name, double value)
        self.lib.itt_metadata_add.argtypes = [c_void_p, c_ulonglong, c_void_p, c_double]

        # void itt_metadata_add_str(void* domain, uint64_t id, void* name, const char* value)
        self.lib.itt_metadata_add_str.argtypes = [c_void_p, c_ulonglong, c_void_p, c_char_p]

        # void itt_metadata_add_blob(void* domain, uint64_t id, void* name, const void* value, uint32_t size)
        self.lib.itt_metadata_add_blob.argtypes = [c_void_p, c_ulonglong, c_void_p, c_void_p, c_uint32]

        # void itt_task_end(void* domain, uint64_t timestamp)
        self.lib.itt_task_end.argtypes = [c_void_p, c_ulonglong]

        # void itt_task_end_overlapped(void* domain, uint64_t timestamp, uint64_t taskid)
        self.lib.itt_task_end_overlapped.argtypes = [c_void_p, c_ulonglong, c_ulonglong]

        # void* itt_counter_create(void* domain, void* name)
        self.lib.itt_counter_create.argtypes = [c_void_p, c_void_p]
        self.lib.itt_counter_create.restype = c_void_p

        # void itt_set_counter(void* id, double value, uint64_t timestamp)
        self.lib.itt_set_counter.argtypes = [c_void_p, c_double, c_ulonglong]

        # void* itt_create_track(const char* group, const char* track)
        self.lib.itt_create_track.argtypes = [c_char_p, c_char_p]
        self.lib.itt_create_track.restype = c_void_p

        # void itt_set_track(void* track)
        self.lib.itt_set_track.argtypes = [c_void_p]

        # uint64_t itt_get_timestamp()
        self.lib.itt_get_timestamp.restype = c_ulonglong

        if hasattr(self.lib, 'get_gpa_version'):
            # const char* get_gpa_version()
            self.lib.get_gpa_version.restype = c_char_p

        if sys.platform == 'win32':
            # const char* resolve_pointer(const char* szModulePath, uint64_t addr)
            self.lib.resolve_pointer.argtypes = [c_char_p, c_ulonglong]
            self.lib.resolve_pointer.restype = c_char_p

            # bool ExportExeIconAsGif(LPCWSTR szExePath, LPCWSTR szGifPath)
            if hasattr(self.lib, 'ExportExeIconAsGif'):
                self.lib.ExportExeIconAsGif.argtypes = [c_wchar_p, c_wchar_p]
                self.lib.ExportExeIconAsGif.restype = c_bool

                # bool ConvertToGif(LPCWSTR szImagePath, LPCWSTR szGifPath, long width, long height)
                self.lib.ConvertToGif.argtypes = [c_wchar_p, c_wchar_p, c_long, c_long]
                self.lib.ConvertToGif.restype = c_bool

        elif 'linux' in sys.platform:
            # void itt_write_time_sync_markers()
            self.lib.itt_write_time_sync_markers.argtypes = []

        # typedef bool (*receive_t)(void* pReceiver, uint64_t time, uint16_t count, const wchar_t** names, const wchar_t** values, double progress);
        self.receive_t = CFUNCTYPE(c_bool, c_ulonglong, c_ulonglong, c_short, POINTER(c_wchar_p), POINTER(c_wchar_p), c_double)
        # typedef void* (*get_receiver_t)(const wchar_t* provider, const wchar_t* opcode, const wchar_t* taskName);
        self.get_receiver_t = CFUNCTYPE(c_ulonglong, c_wchar_p, c_wchar_p, c_wchar_p)
        if hasattr(self.lib, 'parse_standard_source'):
            # bool parse_standard_source(const char* file, get_receiver_t get_receiver, receive_t receive)
            self.lib.parse_standard_source.argtypes = [c_char_p, self.get_receiver_t, self.receive_t]
            self.lib.parse_standard_source.restype = c_bool

        self.domain = self.lib.itt_create_domain(domain.encode())

    def get_string_id(self, text):
        try:
            return self.strings[text]
        except:
            id = self.strings[text] = self.lib.itt_create_string(bytes(text, encoding='utf-8'))
            return id

    def marker(self, text, scope=scope_process, timestamp=0, id=0):
        if not self.lib:
            return
        self.lib.itt_marker(self.domain, id, self.get_string_id(text), scope, timestamp)

    def task(self, name, id=0, parent=0):
        if not self.lib:
            return Dummy()
        return Task(self, name, id, parent)

    def task_submit(self, name, timestamp, dur, id=0, parent=0):
        self.lib.itt_task_begin(self.domain, id, parent, self.get_string_id(name), timestamp)
        self.lib.itt_task_end(self.domain, timestamp + dur)

    def counter(self, name, value, timestamp=0):
        if not self.lib:
            return
        try:
            counter = self.counters[name]
        except:
            counter = self.counters[name] = self.lib.itt_counter_create(self.domain, self.get_string_id(name))
        self.lib.itt_set_counter(counter, value, timestamp)

    def track(self, group, name):
        if not self.lib:
            return Dummy()
        key = group + "/" + name
        try:
            track = self.tracks[key]
        except:
            track = self.tracks[key] = self.lib.itt_create_track(group, name)
        return Track(self, track)

    def get_timestamp(self):
        if not self.lib:
            return 0
        return self.lib.itt_get_timestamp()

    def resolve_pointer(self, module, addr):
        if sys.platform == 'win32':
            if not self.lib:
                return
            return self.lib.resolve_pointer(module, addr)

    def time_sync(self):
        if not self.lib:
            return
        self.lib.itt_write_time_sync_markers()

    def parse_standard_source(self, path, reader):
        if not hasattr(self.lib, 'parse_standard_source'):
            return None
        receivers = []

        def receive(receiver, time, count, names, values, progress):  # typedef bool (*receive_t)(void* receiver, uint64_t time, uint16_t count, const wchar_t** names, const wchar_t** values, double progress);
            receiver = receivers[receiver - 1]  # Should be: receiver = cast(receiver, POINTER(py_object)).contents.value, but it doesn't work so we use index of the array
            args = {}
            for i in range(0, count):
                args[names[i]] = values[i]
            reader.set_progress(progress)
            receiver.receive(time, args)
            return True

        def get_receiver(provider, opcode, taskName):  # typedef void* (*get_receiver_t)(const wchar_t* provider, const wchar_t* opcode, const wchar_t* taskName);
            receiver = reader.get_receiver(provider, opcode, taskName)
            if not receiver:
                return 0
            receivers.append(receiver)
            return len(receivers)  # Should be: cast(pointer(py_object(receiver)), c_void_p).value, but it doesn't work, so we return index of the array

        return self.lib.parse_standard_source(bytes(path, encoding='utf-8'), self.get_receiver_t(get_receiver), self.receive_t(receive))

    def can_parse_standard_source(self):
        return hasattr(self.lib, 'parse_standard_source')

    def get_gpa_version(self):
        if not self.lib:
            return ""
        return self.lib.get_gpa_version()
