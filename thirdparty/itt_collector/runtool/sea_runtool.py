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
import imp
import sys
import sea
import copy
import time
import shutil
import struct
import signal
import fnmatch
import tempfile
import binascii
import platform
import traceback
import threading
import subprocess # nosec
from python_compat import *
from glob import glob
from datetime import datetime, timedelta

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), 'decoders')))

try:
    sys.setdefaultencoding("utf-8")
except:
    pass

ProgressConst = 20000

TIME_SHIFT_FOR_GT = 1000
# on OSX an Application launched from Launchpad has nothing in PATH
if sys.platform == 'darwin':
    if '/usr/bin' not in os.environ['PATH']:
        os.environ['PATH'] += os.pathsep + '/usr/bin'
    if '/usr/sbin' not in os.environ['PATH']:
        os.environ['PATH'] += os.pathsep + '/usr/sbin'


def global_storage(name, default={}):
    if isinstance(__builtins__, dict):
        seapi = __builtins__.setdefault('SEAPI', {})
    else:  # pypy
        if not hasattr(__builtins__, 'SEAPI'):
            setattr(__builtins__, 'SEAPI', {})
        seapi = getattr(__builtins__, 'SEAPI', None)
    return seapi.setdefault(name, copy.deepcopy(default)) if name else seapi  # FIXME put copy.deepcopy under condition


def reset_global(name, value):
    global_storage(None)[name] = value


def format_bytes(num):
    for unit in ['', 'K', 'M', 'G']:
        if abs(num) < 1024.0:
            return "%3.1f %sB" % (num, unit)
        num /= 1024.0
    return str(num) + 'B'


class DummyWith():  # for conditional with statements
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        return False


def get_extensions(name, multiple=False):
    big_name = (name + 's').upper()
    this_module = sys.modules[__name__]
    if big_name in dir(this_module):
        return getattr(this_module, big_name)
    extensions = {}
    root = os.path.join(os.path.dirname(os.path.realpath(__file__)), name + 's')
    for extension in glob(os.path.join(root, '*.py')):
        module_name = name + '.' + os.path.splitext(os.path.basename(extension))[0]
        if name not in sys.modules:
            sys.modules[name] = imp.new_module(name)
        if module_name in sys.modules:
            module = sys.modules[module_name]
        else:
            module = imp.load_source(module_name, extension)
        for desc in getattr(module, name.upper() + '_DESCRIPTORS', []):
            if desc['available']:
                if multiple:
                    extensions.setdefault(desc['format'], []).append(desc[name])
                else:
                    extensions[desc['format']] = desc[name]
    setattr(this_module, big_name, extensions)
    return extensions


def get_exporters():
    return get_extensions('exporter')


verbose_choices = ['fatal', 'error', 'warning', 'info']


def parse_args(args):
    import argparse
    parser = argparse.ArgumentParser(epilog="After this command line add ! followed by command line of your program")
    format_choices = list(get_exporters().keys())
    if sys.platform == 'win32':
        format_choices.append("etw")
    elif sys.platform == 'darwin':
        format_choices.append("xcode")
    elif sys.platform == 'linux':
        format_choices.append("kernelshark")
    parser.add_argument("-o", "--output", help='Output folder pattern -<pid> will be added to it')
    parser.add_argument("-b", "--bindir", help='If you run script not from its location')
    parser.add_argument("-i", "--input", help='Provide input folder for transformation (<the one you passed to -o>-<pid>)')
    parser.add_argument("-t", "--trace", nargs='*', help='Additional trace file in one of supported formats')
    parser.add_argument("-d", "--dir", help='Working directory for target (your program)')
    parser.add_argument("-v", "--verbose", default="warning", choices=verbose_choices)
    parser.add_argument("-c", "--cuts", nargs='*', help='Set "all" to merge all cuts in one trace')
    parser.add_argument("-r", "--ring", type=int, const='5', default=None, action='store', nargs='?', help='Makes trace to cycle inside ring buffer of given length in seconds')
    parser.add_argument("--target", help='Pid of target')
    parser.add_argument("-s", "--app_status", action="store_true", help='Script returns the application status')

    separators = ['!', '?', '%']
    separator = None
    for sep in separators:
        if sep in args:
            separator =  args.index(sep)
            break
    # separator = args.index("!") if "!" in args else args.index("?") if "?" in args else None
    if separator is not None:
        parsed_args = parser.parse_args(args[:separator])
        if parsed_args.input:
            parser.print_help()
            print("Error: Input argument (-i) contradicts launch mode")
            sys.exit(-1)
        victim = args[separator + 1:]
        victim[-1] = victim[-1].strip()  # removal of trailing '\r' - when launched from .sh
        handle_args(parsed_args)
        return parsed_args, victim
    else:  # nothing to launch, transformation mode
        if args:
            args[-1] = args[-1].strip()  # removal of trailing '\r' - when launched from .sh
        parsed_args = parser.parse_args(args)
        handle_args(parsed_args)
        setattr(parsed_args, 'user_input', parsed_args.input)
        if not parsed_args.output:
            parsed_args.output = parsed_args.input
        return parsed_args, None


def handle_args(args):
    if args.input:
        args.input = subst_env_vars(args.input)
    if args.output:
        args.output = subst_env_vars(args.output)
    if args.dir:
        args.dir = subst_env_vars(args.dir)
    if not args.bindir:
        args.bindir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../bin')
    args.bindir = os.path.abspath(args.bindir)


def get_args():
    return global_storage('arguments')


def get_original_env():
    return global_storage('environ')


def verbose_level(level=None, statics={}):
    if not statics:
        args = get_args()
        if not args:
            return verbose_choices.index(level) if level else 'warning'
        statics['level'] = verbose_choices.index(get_args().verbose)
    return verbose_choices.index(level) if level else statics['level']


def message(level, txt, statics={}):
    if not isinstance(statics, dict):
        return False
    if level and verbose_level(level) > verbose_level():  # see default in "parse_args"
        return False

    # in python2 type(parent_frame) is tuple with length == 4
    # in python3 type(parent_frame) is FrameSummary
    parent_frame = traceback.extract_stack()[-2]

    # slice operation returns tuple
    history = statics.setdefault(parent_frame[:4], {'count': 0, 'heap': []})

    history['count'] += 1
    if history['count'] < 5 or not level:
        print('\n', (level.upper() + ':') if level else '', '%s' % txt)
        print('\tFile "%s", line %d, in %s' % parent_frame[:3])
        Collector.log("\n%s:\t%s\n" % (level.upper() if level else 'RUNTIME', txt), stack=(verbose_level(level) <= verbose_level('warning')))
    elif history['count'] == 5:
        print('\n', level.upper(), 'Stopping pollution from', parent_frame[:3])
    return True


def main():
    reset_global('environ', os.environ.copy())
    (args, victim) = parse_args(sys.argv[1:])  # skipping the script name
    reset_global('arguments', args)

    if args.output:
        ensure_dir(args.output, clean=True)
    ret_code = launch(args, victim)
    Collector.log('Started with arguments: %s' % str(sys.argv))
    if ret_code != 0 and not args.app_status:
        ret_code = 0
    return ret_code


def os_lib_ext():
    if sys.platform == 'win32':
        return '.dll'
    elif sys.platform == 'darwin':
        return '.dylib'
    elif 'linux' in sys.platform:
        return '.so'
    raise "Unsupported platform"


def launch(args, victim):
    ret_code = 0
    sea.prepare_environ(args)
    sea_itf = sea.ITT('tools')

    global_storage('collection').setdefault('time', {'start': time.time(), 'itt_start': sea_itf.get_timestamp()})

    env = {}
    paths = []
    macosx = sys.platform == 'darwin'
    win32 = sys.platform == 'win32'
    bits_array = [''] if macosx else ['32', '64']
    for bits in bits_array:
        search = os.path.sep.join([args.bindir, "*IntelSEAPI" + os_lib_ext()])
        files = glob(search)
        if not len(files):
            message('warning', "didn't find any files for: %s" % search)
            continue
        paths.append((bits, files[0]))
    if not len(paths):
        print("Error: didn't find any *IntelSEAPI%s files. Please check that you run from bin directory, or use --bindir." % os_lib_ext())
        sys.exit(-1)
    if macosx:
        env["DYLD_INSERT_LIBRARIES"] = paths[0][1]
    else:
        paths = dict(paths)
        if '32' in paths:
            env["INTEL_LIBITTNOTIFY32"] = paths['32']
            env["INTEL_JIT_PROFILER32"] = paths['32']
        if '64' in paths:
            env["INTEL_LIBITTNOTIFY64"] = paths['64']
            env["INTEL_JIT_PROFILER64"] = paths['64']

    env["INTEL_SEA_FEATURES"] = os.environ['INTEL_SEA_FEATURES'] if 'INTEL_SEA_FEATURES' in os.environ else ""
    env["INTEL_SEA_FEATURES"] += (" stat")

    if args.verbose == 'info':
        env['INTEL_SEA_VERBOSE'] = '1'

    if args.ring:
        env["INTEL_SEA_RING"] = str(args.ring)

    if args.output:
        env["INTEL_SEA_SAVE_TO"] = os.path.join(args.output, 'pid')

    # vulkan support
    os_name = 'WIN' if win32 else 'OSX' if macosx else 'LIN'
    var_name = os.pathsep.join(['VK_LAYER_INTEL_SEA_%s%s' % (os_name, bits) for bits in bits_array])

    env['VK_INSTANCE_LAYERS'] = (os.environ['VK_INSTANCE_LAYERS'] + os.pathsep + var_name) if 'VK_INSTANCE_LAYERS' in os.environ else var_name
    env['VK_LAYER_PATH'] = (os.environ['VK_LAYER_PATH'] + os.pathsep + args.bindir) if 'VK_LAYER_PATH' in os.environ else args.bindir

    message('info', "Running: " + str(victim))
    message('info', "Environment: " + str(env))

    environ = global_storage('sea_env')
    for key, val in env.items():
        if key in environ and val != environ[key]:
            if key not in ['LD_PRELOAD', 'DYLD_INSERT_LIBRARIES']:
                raise key + ' wasn\'t found!'
            environ[key] += ':' + val
        else:
            environ[key] = val

    if args.dir:
        full_victim = os.path.join(args.dir, victim[0])
        if os.path.exists(full_victim):
            victim[0] = full_victim

    setattr(args, 'victim', victim[0])

    if '!' in sys.argv[1:]:
        proc = subprocess.Popen(victim, env=environ, shell=False, cwd=args.dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if sys.platform != 'win32':  # FIXME: implement suspended start on Windows!
            if not args.ring:
                proc.send_signal(signal.SIGSTOP)

        args.target = proc.pid

        if sys.platform != 'win32':  # collector start may be long, so we freeze victim during this time
            print("PID:", proc.pid)
            if not args.ring:
                proc.send_signal(signal.SIGCONT)


        print("Waiting application to exit...")
        global_storage('collection')['time']['before'] = time.time()

        try:
            proc.wait()
        except KeyboardInterrupt:
            print("Stopping all...")
            proc.send_signal(signal.SIGABRT)
            ret_code = -1
        out, err = proc.communicate()
        if out or err:
            print("\n\n -= Target output =- {\n")
            print(out.decode().strip())
            print("\n", "-" * 50, "\n")
            print(err.decode().strip())
            print("\n}\n\n")
        ret_code = proc.returncode
    else:
        message('error', 'unsupported separator')
        return -1

    global_storage('collection')['time']['after'] = time.time()
    print("Stopping collectors...")

    if not args.output:
        return ret_code

    args.input = args.output

    times = global_storage('collection')['time']
    times['end'] = time.time()
    times['itt_end'] = sea_itf.get_timestamp()

    if args.target:
        if isinstance(args.target, list):
            allowed_pids = args.target
        else:
            allowed_pids = [args.target]
        global_storage('collection').setdefault('targets', allowed_pids)

    transform_all(args)

    return ret_code


def subst_env_vars(path):
    return os.path.expandvars(path) if sys.platform == 'win32' else os.path.expanduser(path)


UserProfile = subst_env_vars('%USERPROFILE%' if sys.platform == 'win32' else '~')
PermanentCache = os.path.join(UserProfile, '.isea_cache.dict')


def ensure_dir(path, clean, statics={}):
    if path in statics:
        return
    statics[path] = clean
    if os.path.exists(path):
        if clean:
            shutil.rmtree(path)
        else:
            return
    os.makedirs(path)


def transform_all(args):
    setattr(args, 'user_input', args.input)
    path = os.path.join(args.user_input, 'transform')
    ensure_dir(path, True)

    output = []
    saved_output = args.output
    sea_folders = [folder for folder in glob(os.path.join(args.input, 'pid-*')) if os.path.isdir(folder)]
    if sea_folders:
        for folder in sea_folders:
            args.input = folder
            args.output = saved_output + '.' + os.path.basename(folder)
            output += transform(args)

    args.output = saved_output

    replacement = ('/', '\\') if sys.platform == 'win32' else ('\\', '/')
    for path in output:
        print('Result:', os.path.abspath(path).replace(*replacement), format_bytes(os.path.getsize(path)))

    return output


def split_filename(path):
    (dir, name) = os.path.split(path)
    (name, ext) = os.path.splitext(name)
    ring = None
    cut = None
    if '-' in name:
        (name, ring) = name.split("-")
    if '!' in name:
        (name, cut) = name.split("!")
    return {'dir': dir, 'name': name, 'cut': cut, 'ring':ring, 'ext': ext}


def default_tree(args):
    tree = {"strings": {}, "domains": {}, "threads": {}, "groups": {}, "modules": {}, "ring_buffer": False, "cuts": set()}
    if os.path.isdir(args.input):
        for filename in glob(os.path.join(args.input, '*.mdl')):
            with open(filename, 'r') as file:
                parts = file.readline().split()
                tree["modules"][int(os.path.basename(filename).replace(".mdl", ""))] = [' '.join(parts[0:-1]), parts[-1]]
    return tree


def build_tid_map(args, path):
    tid_map = {}

    def parse_process(src):
        if not os.path.isdir(src):
            return
        pid = src.rsplit('-', 1)[1]
        if not pid.isdigit():
            return
        pid = int(pid)
        for folder in glob(os.path.join(src, '*', '*.sea')):
            tid = int(os.path.basename(folder).split('!')[0].split('-')[0].split('.')[0])
            tid_map[tid] = pid
        if pid not in tid_map:
            tid_map[pid] = pid

    for folder in glob(os.path.join(path, '*-*')):
        parse_process(folder)
    return tid_map


def sea_reader(args):  # reads the structure of .sea format folder into dictionary
    folder = args.input
    if not os.path.exists(folder):
        print("""Error: folder "%s" doesn't exist""" % folder)
    tree = default_tree(args)
    pos = folder.rfind("-")  # pid of the process is encoded right in the name of the folder
    tree["pid"] = int(folder[pos + 1:])
    folder = folder.replace("\\", "/").rstrip("/")
    toplevel = next(os.walk(folder))
    for filename in toplevel[2]:
        with open("/".join([folder, filename]), "r") as file:
            if filename.endswith(".str"):  # each string_handle_create writes separate file, name is the handle, content is the value
                tree["strings"][int(filename.replace(".str", ""))] = file.readline()
            elif filename.endswith(".tid"):  # named thread makes record: name is the handle and content is the value
                tree["threads"][filename.replace(".tid", "")] = file.readline()
            elif filename.endswith(".pid"):  # named groups (pseudo pids) makes record: group is the handle and content is the value
                tree["groups"][filename.replace(".pid", "")] = file.readline()
    for domain in toplevel[1]:  # data from every domain gets recorded into separate folder which is named after the domain name
        tree["domains"][domain] = {"files": []}
        for file in next(os.walk("/".join([folder, domain])))[2]:  # each thread of this domain has separate file with data
            if not file.endswith(".sea"):
                print("Warning: weird file found:", file)
                continue
            filename = file[:-4]

            tree["ring_buffer"] = tree["ring_buffer"] or ('-' in filename)
            tid = int(filename.split("!")[0].split("-")[0])
            tree["cuts"].add(split_filename(filename)['cut'])

            tree["domains"][domain]["files"].append((tid, "/".join([folder, domain, file])))

        def time_sort(item):
            with open(item[1], "rb") as file:
                tuple = read_chunk_header(file)
                return tuple[0]

        tree["domains"][domain]["files"].sort(key=time_sort)
    return tree


g_progress_interceptor = None
verbose_progress = True

# FIXME: doesn't belong this file, move to 'utils'


class Progress:
    def __init__(self, total, steps, message=""):
        self.total = total
        self.steps = steps
        self.shown_steps = -1
        self.message = message
        self.last_tick = None

    def __enter__(self):
        return self

    def time_to_tick(self, interval=1):
        return (datetime.now() - self.last_tick).total_seconds() > interval if self.last_tick else True

    def tick(self, current):
        self.last_tick = datetime.now()
        if g_progress_interceptor:
            g_progress_interceptor(self.message, current, self.total)
        if self.total:
            self.show_progress(int(self.steps * current / self.total), float(current) / self.total)

    def show_progress(self, show_steps, percentage):
        if self.shown_steps < show_steps:
            if verbose_progress:
                print('\r%s: %d%%' % (self.message, int(100*percentage)), end='')
                sys.stdout.flush()
            self.shown_steps = show_steps

    def __exit__(self, type, value, traceback):
        if g_progress_interceptor:
            g_progress_interceptor(self.message, self.total, self.total)
        self.show_progress(self.steps, 1)
        if verbose_progress:
            print('\r%s: %d%%\n' % (self.message, 100))
        return False

    @staticmethod
    def set_interceptor(interceptor, verbose_mode=False):
        global g_progress_interceptor
        global verbose_progress
        g_progress_interceptor = interceptor
        verbose_progress = verbose_mode


class PseudoProgress(Progress):

    def profiler(self, frame, event, arg):
        if 'return' not in event:
            return
        cur_time = time.time()
        if cur_time - self.time > 1:
            self.time = cur_time
            self.tick(cur_time)

    def __init__(self, message=""):
        self.time = None
        Progress.__init__(self, 0, 0, message)
        self.old_profiler = sys.getprofile()

    def __enter__(self):
        self.time = time.time()
        sys.setprofile(self.profiler)
        return self

    def __exit__(self, type, value, traceback):
        sys.setprofile(self.old_profiler)
        return Progress.__exit__(self, type, value, traceback)


def read_chunk_header(file):
    chunk = file.read(10)  # header of the record, see STinyRecord in Recorder.cpp
    if not chunk:
        return 0, 0, 0
    return struct.unpack('Qbb', chunk)


def transform(args):
    message('info', "Transform: " + str(args))
    tree = sea_reader(args)  # parse the structure
    if args.cuts and args.cuts == ['all'] or not args.cuts:
        return transform2(args, tree)
    else:
        result = []
        output = args.output[:]  # deep copy
        for current_cut in tree['cuts']:
            if args.cuts and current_cut not in args.cuts:
                continue
            args.output = (output + "!" + current_cut) if current_cut else output
            print("Cut #", current_cut if current_cut else "<None>")

            def skip_fn(path):
                filename = os.path.split(path)[1]
                if current_cut:  # read only those having this cut name in filename
                    if current_cut != split_filename(filename)['cut']:
                        return True
                else:  # reading those having not cut name in filename
                    if "!" in filename:
                        return True
                return False

            result += transform2(args, tree, skip_fn)
        args.output = output
        return result


# FIXME: doesn't belong this file, move to Combiners or something

TaskTypes = [
    "task_begin", "task_end",
    "task_begin_overlapped", "task_end_overlapped",
    "metadata_add",
    "marker",
    "counter",
    "frame_begin", "frame_end",
    "object_new", "object_snapshot", "object_delete",
    "relation"
]


class TaskCombinerCommon:
    def __init__(self, args, tree):
        self.no_begin = []  # for the ring buffer case when we get task end but no task begin
        self.time_bounds = [2 ** 64, 0]  # left and right time bounds
        self.tree = tree
        self.args = args
        self.domains = {}
        self.prev_sample = 0
        self.total_memory = 0
        self.prev_memory = None
        self.memcounters = {}

    def finish(self):
        pass

    def __call__(self, fn, data):
        domain = self.domains.setdefault(data['domain'], {'tasks': {}, 'counters': {}})
        thread = domain['tasks'].setdefault(data['tid'], {'byid': {}, 'stack': [], 'args': {}})

        def get_tasks(id):
            if not id:
                return thread['stack']
            return thread['byid'].setdefault(id, [])

        def get_task(id):
            if id:
                tasks = get_tasks(id)
                if not tasks:  # they can be stacked
                    tasks = get_tasks(None)
                    if not tasks or ('id' not in tasks[-1]) or tasks[-1]['id'] != id:
                        return None
            else:
                tasks = get_tasks(None)
            if tasks:
                return tasks[-1]
            else:
                return None

        def find_task(id):
            for thread_stacks in domain['tasks'].values():  # look in all threads
                if (id in thread_stacks['byid']) and thread_stacks['byid'][id]:
                    return thread_stacks['byid'][id][-1]
                else:
                    for item in thread_stacks['stack']:
                        if ('id' in item) and item['id'] == id:
                            return item

        def get_stack(tid):
            stack = []
            for domain in self.domains.values():
                if tid not in domain['tasks']:
                    continue
                thread = domain['tasks'][tid]
                for byid in thread['byid'].values():
                    stack += byid
                if thread['stack']:
                    stack += thread['stack']
            stack.sort(key=lambda item: item['time'])
            return stack

        def get_last_index(tasks, type):
            if not len(tasks):
                return None
            index = len(tasks) - 1
            while index > -1 and tasks[index]['type'] != type:
                index -= 1
            if index > -1:
                return index
            return None

        if fn == "task_begin" or fn == "task_begin_overlapped":
            if not (('str' in data) or ('pointer' in data)):
                data['str'] = 'Unknown'
            self.time_bounds[0] = min(self.time_bounds[0], data['time'])
            if 'delta' in data and data['delta']:  # turbo mode, only begins are written
                end = data.copy()
                end['time'] = data['time'] + int(data['delta'])
                self.time_bounds[1] = max(self.time_bounds[1], end['time'])
                self.complete_task('task', data, end)  # for now arguments are not supported in turbo tasks. Once argument is passed, task gets converted to normal.
            else:
                get_tasks(None if fn == "task_begin" else data['id']).append(data)
        elif fn == "task_end" or fn == "task_end_overlapped":
            self.time_bounds[1] = max(self.time_bounds[1], data['time'])
            tasks = get_tasks(None if fn == "task_end" else data['id'])
            index = get_last_index(tasks, data['type'] - 1)
            if index is not None:
                item = tasks.pop(index)
                if self.task_postprocessor:
                    self.task_postprocessor.postprocess('task', item, data)
                if not self.handle_special('task', item, data):
                    if data['time'] > item['time']:
                        self.complete_task('task', item, data)
                    else:
                        message('warning', 'Negative length task: %s => %s' % (str(item), str(data)))
            else:
                if 'str' in data:  # nothing to show without name
                    self.no_begin.append(data)
        elif fn == "frame_begin":
            get_tasks(data['id'] if 'id' in data else None).append(data)
        elif fn == "frame_end":
            frames = get_tasks(data['id'] if 'id' in data else None)
            index = get_last_index(frames, 7)
            if index is not None:
                item = frames.pop(index)
                self.complete_task("frame", item, data)
        elif fn == "metadata_add":
            if 'id' in data:
                task = get_task(data['id'])
                if task:
                    args = task.setdefault('args', {})
                else:
                    args = thread['args'].setdefault(data['id'], {})

                args[data['str']] = data['delta'] if 'delta' in data else '0x0'
            else:  # global metadata
                if not self.handle_special('meta', data, None):
                    self.global_metadata(data)
        elif fn == "object_snapshot":
            if 'args' in data:
                args = data['args'].copy()
            else:
                args = {'snapshot': {}}
            if 'data' in data:
                state = data['data']
                for pair in state.split(","):
                    (key, value) = tuple(pair.split("="))
                    args['snapshot'][key] = value
            data['args'] = args
            self.complete_task(fn, data, data)
        elif fn in ["marker", "counter", "object_new", "object_delete"]:
            if fn == "marker" and data['data'] == 'task':
                markers = get_tasks("marker_" + (data['id'] if 'id' in data else ""))
                if markers:
                    item = markers.pop()
                    item['type'] = 7  # frame_begin
                    item['domain'] += ".continuous_markers"
                    item['time'] += 1
                    self.complete_task("frame", item, data)
                markers.append(data)
            else:
                if ('id' in data) and (data['id'] in thread['args']):
                    data['args'] = thread['args'][data['id']]
                    del thread['args'][data['id']]
                self.complete_task(fn, data, data)
        elif fn == "relation":
            self.relation(
                data,
                get_task(data['id'] if 'id' in data else None),
                get_task(data['parent']) or find_task(data['parent'])
            )
        else:
            raise "Unsupported type:" + fn

    def compress_counter(self, cache, data):
        values = cache['values']
        if values and not data:
            length = len(values)
            avg_value = sum([value['delta'] for value in values]) / length
            if cache['last'] != avg_value:
                avg_time = int(sum([value['time'] for value in values]) / length)
                self.process(values[0]['pid']).thread(values[0]['tid']).counter(values[0]['str']).set_value(avg_time, avg_value)
                cache['last'] = avg_value
            cache['values'] = []

    def handle_special(self, kind, begin, end):
        if self.sea_decoders:
            for decoder in self.sea_decoders:
                if decoder.handle_special(kind, begin, end):
                    return True
        return False

    def flush_counters(self, domain, data):
        for name, counter in domain['counters'].items():
            common_data = data.copy()
            common_data['time'] = counter['begin'] + (counter['end'] - counter['begin']) / 2
            common_data['str'] = name
            common_data['delta'] = sum(counter['values']) / len(counter['values'])
            self.complete_task('counter', common_data, common_data)

    def flush_compressed_counters(self):
        for pid, threads in self.memcounters.items():
            for tid, counters in threads.items():
                for name, counter in counters.items():
                    self.compress_counter(counter, None)


def default_event_filer(cls, type, begin, end):
    if begin['domain'] == 'Metal':
        if 'FailureType' in begin['str']:
            return None, None, None
    return type, begin, end


class Callbacks(TaskCombinerCommon):
    event_filter = default_event_filer
    task_postprocessor = None

    def __init__(self, args, tree):
        TaskCombinerCommon.__init__(self, args, tree)
        self.callbacks = []  # while parsing we might have one to many 'listeners' - output format writers
        self.stack_sniffers = [] # only stack listeners
        self.allowed_pids = set()
        self.processes = {}
        self.tasks_from_samples = {}
        self.on_finalize_callbacks = []

        collection = global_storage('collection')
        if 'targets' in collection:
            self.allowed_pids = set(collection['targets'])
        else:
            self.allowed_pids = set()
        self.tid_map = self.get_globals()['tid_map']
        if hasattr(self.args, 'user_input') and os.path.isdir(self.args.user_input):
            tid_map = build_tid_map(self.args, self.args.user_input)
            self.tid_map.update(tid_map)
            self.allowed_pids |= set(tid_map.values())

        self.callbacks.append(get_exporters()['stat'](args, tree))

        if args.target:
            if isinstance(args.target, list):
                self.allowed_pids += args.target
            else:
                self.allowed_pids.add(int(args.target))

        self.sea_decoders = []

        self.globals = self.get_globals()
        self.cpus = set()
        self.all_cpus_started = os.path.isfile(self.args.user_input) or None
        self.proc_names = {}

    @classmethod
    def get_globals(cls):
        return global_storage('Callbacks', {
            'starts': {}, 'ends': {}, 'dtrace': {'finished': False}, 'tid_map': {}
        })

    def add_stack_sniffer(self, sniffer):
        self.stack_sniffers.append(sniffer)

    @classmethod
    def set_event_filter(cls, filter):
        prev = cls.event_filter
        cls.event_filter = filter
        return prev

    @classmethod
    def set_task_postprocessor(cls, postprocessor):
        cls.task_postprocessor = postprocessor

    def on_finalize(self, function):  # will be called with callbacks(self) as the only argument
        self.on_finalize_callbacks.append(function)

    def is_empty(self):
        return 0 == len(self.callbacks)

    def __enter__(self):
        [callback.__enter__() for callback in self.callbacks]
        return self

    def __exit__(self, type, value, traceback):
        self.finalize()
        [callback.__exit__(type, value, traceback) for callback in self.callbacks]  # emulating 'with' statement
        return False

    def finalize(self):
        for decoder in self.sea_decoders:
            decoder.finalize()
        for kind, data in self.tasks_from_samples.items():
            for pid, threads in data.items():
                for tid, tasks in threads.items():
                    self.handle_stack(pid, tid, tasks.last_stack_time + TIME_SHIFT_FOR_GT * len(tasks) + 1, [], kind)
        for function in self.on_finalize_callbacks:
            function(self)

        if self.allowed_pids:
            global_storage('collection').setdefault('targets', self.allowed_pids)

        self.finish()

    def on_event(self, type, data):
        if self.event_filter:
            type, data, end = self.event_filter(type, data, None)
            if not type:
                return False

        if data.get('internal_name', None):
            return False

        self.__call__(type, data)
        return True

    def complete_task(self, type, begin, end):
        if self.event_filter:
            type, begin, end = self.event_filter(type, begin, end)
            if not type:
                return False
        if self.handle_special(type, begin, end):  # returns True if event is consumed and doesn't require processing
            return True

        if end:
            # copy here as handler can change the data for own good - this shall not affect other handlers
            [callback.complete_task(type, begin.copy(), end.copy() if end else end) for callback in self.callbacks]
            return True
        else:
            return False

    def global_metadata(self, data):
        [callback.global_metadata(data.copy()) for callback in self.callbacks]

    def relation(self, data, head, tail):
        for callback in self.callbacks:
            callback.relation(data, head, tail)

    def get_result(self):
        res = []
        for callback in self.callbacks:
            res += callback.get_targets()
        return res

    def check_time_in_cs_bounds(self, timestamp, statics={}):
        if not statics:
            globals = self.get_globals()
            if not globals['dtrace']['finished'] or 'context_switch' not in self.globals['ends']:
                return None
            statics['start'] = globals['starts']['context_switch']
            statics['end'] = globals['ends']['context_switch']

        return statics['start'] <= timestamp <= statics['end']

    def get_pid(self, tid):
        if tid in self.tid_map:
            return self.tid_map[tid]
        return None

    class Process:
        def __init__(self, callbacks, pid, name):
            self.callbacks = callbacks
            self.pid = int(pid)
            self.threads = {}
            if name:
                self.set_name(name)

        def set_name(self, name):
            self.callbacks.set_process_name(self.pid, name)

        class Thread:
            def __init__(self, process, tid, name):
                self.process = process
                self.tid = int(tid)
                tid_map = self.process.callbacks.tid_map
                if process.pid > 0 and self.tid > 0:
                    if self.tid not in tid_map:
                        tid_map[self.tid] = process.pid
                    elif tid_map[self.tid] != process.pid:
                        message('error', 'TID %d was part of PID %d and now PID %d... How come?' % (self.tid, tid_map[self.tid], process.pid))
                self.overlapped = {}
                self.to_overlap = {}
                self.task_stack = []
                self.task_pool = {}
                self.snapshots = {}
                self.lanes = {}
                if name:
                    self.set_name(name)
                self.process.callbacks.on_finalize(self.finalize)

            def auto_break_overlapped(self, call_data, begin):
                id = call_data['id']
                if begin:
                    call_data['realtime'] = call_data['time']  # as we gonna change 'time'
                    call_data['lost'] = 0
                    self.overlapped[id] = call_data
                else:
                    if id in self.overlapped:
                        real_time = self.overlapped[id]['realtime']
                        to_remove = []
                        del self.overlapped[id]  # the task has ended, removing it from the pipeline
                        time_shift = 0
                        for begin_data in sorted(self.overlapped.values(), key=lambda data: data['realtime']):  # finish all and start again to form melting task queue
                            time_shift += 1  # making sure the order of tasks on timeline, probably has to be done in Chrome code rather
                            end_data = begin_data.copy()  # the end of previous part of task is also here
                            end_data['time'] = call_data['time'] - time_shift  # new begin for every task is here
                            end_data['type'] = call_data['type']
                            self.process.callbacks.on_event('task_end_overlapped', end_data)  # finish it
                            if begin_data['realtime'] < real_time:
                                begin_data['lost'] += 1
                            if begin_data['lost'] > 10:  # we seem lost the end ETW call
                                to_remove.append(begin_data['id'])  # main candidate is the event that started earlier but nor finished when finished the one started later
                            else:
                                begin_data['time'] = call_data['time'] + time_shift  # new begin for every task is here
                                self.process.callbacks.on_event('task_begin_overlapped', begin_data)  # and start again
                        for id in to_remove:  # FIXME: but it's better somehow to detect never ending tasks and not show them at all or mark somehow
                            if id in self.overlapped:
                                del self.overlapped[id]  # the task end was probably lost
                            else:
                                message('error', '???')

            def process_overlapped(self, threshold=100):
                if not threshold or 0 != (len(self.to_overlap) % threshold):
                    return
                keys = sorted(self.to_overlap)[0:threshold//2]
                to_del = set()
                for key in keys:
                    task = self.to_overlap[key]
                    if task.overlap_begin:
                        self.auto_break_overlapped(task.data, True)
                        self.process.callbacks.on_event("task_begin_overlapped", task.data)
                        task.overlap_begin = False
                    else:
                        end_data = task.data.copy()
                        end_data['time'] = key
                        end_data['type'] += 1
                        self.auto_break_overlapped(end_data, False)
                        self.process.callbacks.on_event("task_end_overlapped", end_data)
                    to_del.add(key)
                for key in to_del:
                    del self.to_overlap[key]

            def finalize(self, _):
                self.process_overlapped(0)

            def set_name(self, name):
                self.process.callbacks.set_thread_name(self.process.pid, self.tid, name)

            class EventBase:
                def __init__(self, thread, name, domain, internal_name=None):
                    self.thread = thread
                    self.name = name
                    self.domain = domain
                    self.internal_name = internal_name

            class Counter(EventBase):
                def __init__(self, *args):
                    Callbacks.Process.Thread.EventBase.__init__(self, *args)

                def set_value(self, time_stamp, value):
                    data = {
                        'pid': self.thread.process.pid, 'tid': self.thread.tid,
                        'domain': self.domain, 'str': self.name,
                        'time': time_stamp, 'delta': value, 'type': 6,
                        'internal_name': self.internal_name
                    }
                    self.thread.process.callbacks.on_event('counter', data)

                def set_multi_value(self, time_stamp, values_dict):  # values_dict is name:value dictionary
                    data = {
                        'pid': self.thread.process.pid, 'tid': self.thread.tid,
                        'domain': self.domain, 'str': self.name,
                        'time': time_stamp, 'args': values_dict, 'type': 6
                    }
                    self.thread.process.callbacks.on_event('counter', data)

            def counter(self, name, domain='sea', internal_name=None):
                return Callbacks.Process.Thread.Counter(self, name, domain, internal_name)

            class Marker(EventBase):
                def __init__(self, thread, scope, name, domain):
                    Callbacks.Process.Thread.EventBase.__init__(self, thread, name, domain)
                    self.scope = scope

                def set(self, time_stamp, args=None):
                    data = {
                        'pid': self.thread.process.pid, 'tid': self.thread.tid,
                        'domain': self.domain, 'str': self.name,
                        'time': time_stamp, 'type': 5, 'data': self.scope
                    }
                    if args is not None:
                        data.update({'args': args})

                    return self.thread.process.callbacks.on_event('marker', data)

            def marker(self, scope, name, domain='sea'):  # scope is one of 'task', 'global', 'process', 'thread'
                scopes = {'task': 'task', 'global': 'global', 'process': 'track_group', 'thread': 'track'}
                return Callbacks.Process.Thread.Marker(self, scopes[scope], name, domain)

            class TaskBase(EventBase):
                def __init__(self, type_id, type_name, thread, name, domain):
                    Callbacks.Process.Thread.EventBase.__init__(self, thread, name, domain)
                    self.data = None
                    self.args = {}
                    self.meta = {}
                    # These must be set in descendants!
                    self.event_type = type_id  # first of types
                    self.event_name = type_name
                    self.overlap_begin = True

                def __begin(self, time_stamp, task_id, args, meta):
                    data = {
                        'pid': self.thread.process.pid, 'tid': self.thread.tid,
                        'domain': self.domain, 'str': self.name,
                        'time': time_stamp, 'str': self.name, 'type': self.event_type
                    }
                    if task_id is not None:
                        data.update({'id': task_id})
                    if args:
                        data.update({'args': args})
                    if meta:
                        data.update(meta)
                    return data

                def begin(self, time_stamp, task_id=None, args={}, meta={}):
                    self.data = self.__begin(time_stamp, task_id, args, meta)

                    if self.event_type == 2:  # overlapped task
                        self.thread.auto_break_overlapped(self.data, True)
                        self.thread.process.callbacks.on_event("task_begin_overlapped", self.data)
                    return self

                def add_args(self, args):  # dictionary is expected
                    self.args.update(args)
                    return self

                def add_meta(self, meta):  # dictionary is expected
                    self.meta.update(meta)
                    return self

                def get_data(self):
                    return self.data

                def get_args(self):
                    args = self.data['args'].copy()
                    args.update(self.args)
                    return args

                def end(self, time_stamp):
                    if not self.data:
                        return
                    if time_stamp:
                        end_data = self.data.copy()
                        end_data.update({'time': time_stamp, 'type': self.event_type + 1})
                        if self.args:
                            if 'args' in end_data:
                                end_data['args'].update(self.args)
                            else:
                                end_data['args'] = self.args
                        if self.meta:
                            end_data.update(self.meta)
                    else:
                        end_data = None  # special case when end is unknown and has to be calculated by viewer

                    if self.event_type == 2:  # overlapped task
                        self.thread.auto_break_overlapped(end_data, False)
                        self.thread.process.callbacks.on_event("task_end_overlapped", end_data)
                    else:
                        self.thread.process.callbacks.complete_task(self.event_name, self.data, end_data)
                    self.data = None
                    self.args = {}
                    self.meta = {}

                def complete(self, start_time, duration, task_id=None, args={}, meta={}):
                    begin_data = self.__begin(start_time, task_id, args, meta)
                    end_data = begin_data.copy()
                    end_data['time'] = start_time + duration
                    end_data['type'] = self.event_type + 1
                    self.thread.process.callbacks.complete_task(self.event_name, begin_data, end_data)
                    return begin_data

                def end_overlap(self, time_stamp):
                    while self.data['time'] in self.thread.to_overlap:
                        self.data['time'] += 1
                    self.thread.to_overlap[self.data['time']] = self
                    while time_stamp in self.thread.to_overlap:
                        time_stamp -= 1
                    self.thread.to_overlap[time_stamp] = self
                    self.data['id'] = time_stamp
                    self.data['type'] = self.event_type = 2
                    self.thread.process_overlapped()

            class Task(TaskBase):
                def __init__(self, thread, name, domain, overlapped):
                    Callbacks.Process.Thread.TaskBase.__init__(
                        self,
                        2 if overlapped else 0,
                        'task',
                        thread,
                        name, domain
                    )
                    self.relation = None
                    self.related_begin = None

                def end(self, time_stamp):
                    begin_data = self.data.copy()  # expected to be initialized in self.begin call
                    Callbacks.Process.Thread.TaskBase.end(self, time_stamp)
                    self.__check_relation(begin_data)

                def __check_relation(self, begin):
                    if not self.relation:
                        return
                    if self.related_begin:  # it's the later task, let's emit the relation
                        self.__emit_relation(begin, self.related_begin)
                        self.related_begin = None
                    else:  # we store our begin in the related task and it will emit the relation on its end
                        self.relation.related_begin = begin
                    self.relation = None

                def __emit_relation(self, left, right):
                    relation = (left.copy(), right.copy(), left)
                    if 'realtime' in relation[1]:
                        relation[1]['time'] = relation[1]['realtime']
                    if 'realtime' in relation[2]:
                        relation[2]['time'] = relation[2]['realtime']
                    relation[0]['parent'] = left['id'] if 'id' in left else id(left)
                    self.thread.process.callbacks.relation(*relation)

                def complete(self, start_time, duration, task_id=None, args={}, meta={}):
                    begin_data = Callbacks.Process.Thread.TaskBase.complete(self, start_time, duration, task_id, args, meta)
                    self.__check_relation(begin_data)

                def relate(self, task):  # relation is being written when last of two related tasks was fully emitted
                    if self.relation != task:
                        self.relation = task
                        task.relate(self)

                def end_overlap(self, time_stamp):
                    Callbacks.Process.Thread.TaskBase.end_overlap(self, time_stamp)
                    if self.relation:
                        self.__emit_relation(self.data, self.relation.data)

            def task(self, name, domain='sea', overlapped=False):
                return Callbacks.Process.Thread.Task(self, name, domain, overlapped)

            class Frame(TaskBase):
                def __init__(self, thread, name, domain):
                    Callbacks.Process.Thread.TaskBase.__init__(self, 7, 'frame', thread, name, domain)

            def frame(self, name, domain='sea'):
                return Callbacks.Process.Thread.Frame(self, name, domain)

            class Lane:
                def __init__(self, thread, name, domain):
                    self.thread, self.domain = thread, domain
                    self.name = '%s (%d):' % (name, thread.tid)
                    self.first_frame = None
                    self.id = hex(hash(self))
                    self.thread.process.callbacks.on_finalize(self.finalize)

                def finalize(self, _):
                    if self.first_frame:
                        Callbacks.Process.Thread\
                            .TaskBase(7, 'frame', self.thread, self.name, self.domain) \
                            .begin(self.first_frame - 1000, self.id).end(None)  # the open-ended frame (automatically closed by viewer)

                def frame_begin(self, time_stamp, name, args={}, meta={}):
                    if not self.first_frame or time_stamp < self.first_frame:
                        self.first_frame = time_stamp
                    return Callbacks.Process.Thread.TaskBase(7, 'frame', self.thread, name, self.domain).begin(time_stamp, self.id, args, meta)

            def lane(self, name, domain='sea'):
                if name not in self.lanes:
                    self.lanes[name] = Callbacks.Process.Thread.Lane(self, name, domain)
                return self.lanes[name]

            class Object(EventBase):
                def __init__(self, thread, id, name, domain):
                    Callbacks.Process.Thread.EventBase.__init__(self, thread, name, domain)
                    self.id = id
                    if not self.thread.snapshots:
                        self.thread.snapshots = {'last_time': 0}

                def create(self, time_stamp):
                    data = {
                        'pid': self.thread.process.pid, 'tid': self.thread.tid,
                        'domain': self.domain, 'str': self.name,
                        'time': time_stamp, 'type': 9, 'id': self.id
                    }
                    self.thread.process.callbacks.on_event("object_new", data)
                    return self

                def snapshot(self, time_stamp, args):
                    if time_stamp is None or time_stamp <= self.thread.snapshots['last_time']:
                        time_stamp = self.thread.snapshots['last_time'] + 1
                    self.thread.snapshots['last_time'] = time_stamp
                    data = {
                        'pid': self.thread.process.pid, 'tid': self.thread.tid,
                        'domain': self.domain, 'str': self.name,
                        'time': time_stamp, 'type': 10, 'id': self.id,
                        'args': {'snapshot': args}
                    }
                    self.thread.process.callbacks.on_event("object_snapshot", data)
                    return self

                @staticmethod  # use to prepare argument for 'snapshot' call, only png in base64 string is supported by chrome
                def create_screenshot_arg(png_base64):
                    return {'screenshot': png_base64}

                def destroy(self, time_stamp):
                    data = {
                        'pid': self.thread.process.pid, 'tid': self.thread.tid,
                        'domain': self.domain, 'str': self.name,
                        'time': time_stamp, 'type': 11, 'id': self.id
                    }
                    self.thread.process.callbacks.on_event("object_delete", data)

            def object(self, id, name, domain='sea'):
                return Callbacks.Process.Thread.Object(self, id, name, domain)

        def thread(self, tid, name=None):
            if tid not in self.threads:
                self.threads[tid] = Callbacks.Process.Thread(self, tid, name)
            return self.threads[tid]

    def process(self, pid, name=None):
        if pid not in self.processes:
            self.processes[pid] = Callbacks.Process(self, pid, name)
        return self.processes[pid]

    def vsync(self, time_stamp, args={}, statics={}):
        if not statics:
            statics['marker'] = self.process(-1).thread(-1, 'VSYNC').marker('thread', 'vblank', 'gpu')
        args.update({'AbsTime': time_stamp})
        statics['marker'].set(time_stamp, args)

    def context_switch(self, time_stamp, cpu, prev_tid, next_tid, prev_name='', next_name='', prev_state='S', prev_prio=0, next_prio=0):
        if cpu not in self.cpus:
            self.cpus.add(cpu)
            all_cpus_started = max(self.cpus) + 1 == len(self.cpus)
            if self.all_cpus_started != all_cpus_started:
                self.globals['starts']['context_switch'] = time_stamp
                self.all_cpus_started = all_cpus_started
        if not self.all_cpus_started:
            return
        self.globals['ends']['context_switch'] = time_stamp
        for callback in self.callbacks:
            callback.context_switch(
                time_stamp, cpu,
                {
                    'tid': prev_tid,
                    'name': prev_name.replace(' ', '_'),
                    'state': prev_state,
                    'prio': prev_prio,
                },
                {
                    'tid': next_tid,
                    'prio': next_prio,
                    'name': next_name.replace(' ', '_')
                }
            )

    def wakeup(self, time_stamp, cpu, prev_pid, prev_tid, next_pid, next_tid, prev_name='', next_name='', sync_prim='', sync_prim_addr=None):
        if prev_pid not in self.allowed_pids and next_pid not in self.allowed_pids:
            return False

        args = {'target': next_tid, 'type': sync_prim, 'addr': sync_prim_addr} if sync_prim_addr else {}
        args.update({'target': next_tid, 'by': prev_tid})
        event_width = 2000
        from_task = self.process(prev_pid).thread(prev_tid).task('wakes').begin(time_stamp - event_width, args=args)
        to_task = self.process(next_pid).thread(next_tid).task('woken').begin(time_stamp, args=args)
        from_task.relate(to_task)
        from_task.end(time_stamp - event_width/2)
        to_task.end(time_stamp + event_width/2)

        for callback in self.callbacks:
            callback.wakeup(
                time_stamp, cpu,
                {
                    'pid': prev_pid,
                    'tid': prev_tid,
                    'name': prev_name.replace(' ', '_')
                },
                {
                    'pid': next_pid,
                    'tid': next_tid,
                    'name': next_name.replace(' ', '_')
                }
            )

    def get_process_name(self, pid):
        return self.proc_names[pid] if pid in self.proc_names else None

    def set_process_name(self, pid, name, labels=[]):
        order = -1 if pid in self.allowed_pids else pid
        if pid not in self.proc_names:
            self.proc_names[pid] = [name]
            self.__call__("metadata_add", {'domain': 'IntelSEAPI', 'str': '__process__', 'pid': pid, 'tid': -1, 'delta': order, 'data': name, 'labels': labels})
        elif name not in self.proc_names[pid]:
            self.proc_names[pid].append(name)
            full_name = '->'.join(self.proc_names[pid])
            self.__call__("metadata_add", {'domain': 'IntelSEAPI', 'str': '__process__', 'pid': pid, 'tid': -1, 'delta': order, 'data': full_name, 'labels': labels})
            message('warning', 'Pid %d name changed: %s' % (pid, full_name))

    def set_thread_name(self, pid, tid, name):
        self.__call__("metadata_add", {'domain': 'IntelSEAPI', 'str': '__thread__', 'pid': pid, 'tid': tid, 'data': '%s (%d)' % (name, tid), 'delta': tid})

    def add_metadata(self, name, data):
        self.__call__("metadata_add", {'domain': 'IntelSEAPI', 'data': data, 'str': name, 'tid': None})

    class AttrDict(dict):
        pass  # native dict() refuses setattr call

    def handle_stack(self, pid, tid, time, stack, kind='sampling'):
        use_lanes = False

        tasks = self.tasks_from_samples.setdefault(kind, {}).setdefault(pid, {}).setdefault(tid, self.AttrDict())
        tasks.last_stack_time = time
        to_remove = []

        if not use_lanes:
            pid = -pid if pid > 100 else pid
            tid = -tid

        # Find currently present tasks:
        present = set()
        for frame in stack:
            ptr = frame['ptr']
            if not frame['str']:
                frame['str'] = '0x%x' % ptr
            else:
                frame['str'] = '%s(0x%x)' % (frame['str'], ptr)
            present.add(ptr)

        # Remove currently absent tasks (they are finished):
        for ptr in tasks:
            if ptr not in present:
                to_remove.append(ptr)

        to_add = []
        # Find affected tasks, those to the right of most recent of removed. These affected are to be 'restarted'
        if to_remove:
            leftmost_time = min(tasks[ptr]['begin'] for ptr in to_remove)
            for ptr, task in tasks.items():
                if task['begin'] > leftmost_time and ptr not in to_remove:
                    to_remove.append(ptr)
                    to_add.append(task.copy())

            # Actual removal of the tasks with flushing them to timeline:
            to_remove.sort(key=lambda ptr: tasks[ptr]['begin'])
            shift = 1
            if use_lanes:
                lane = self.process(pid).thread(tid).lane(kind)  #TODO: implement proper lane frames
            else:
                thread = self.process(pid).thread(tid)
            for ptr in to_remove:
                task = tasks[ptr]
                end_time = time - TIME_SHIFT_FOR_GT * shift
                if end_time <= task['begin']:  # this might happen on finalization and with very deep stack
                    continue
                args = {'module': task['module'].replace('\\', '/')}
                if '__file__' in task and '__line__' in task:
                    args.update({
                        'pos': '%s(%d)' % (task['__file__'], int(task['__line__']))
                    })
                if use_lanes:
                    lane.frame_begin(
                        task['begin'], task['str'], args=args, meta={'sampled': True}
                    ).end(end_time)
                else:
                    if kind in ['sampling', 'ustack'] or (pid == 0 and kind == 'kstack'):  # temporary workaround for OSX case where there are three stacks
                        thread.task(task['str']).begin(task['begin'], args=args, meta={'sampled': True}).end(end_time)
                del tasks[ptr]
                shift += 1

        # pre-sort restarted tasks by their initial time to keep natural order
        to_add.sort(key=lambda task: task['begin'])

        # Add new tasks to the end of the list
        for frame in reversed(stack):  # Frames originally come in reverse order [bar, foo, main]
            if frame['ptr'] not in tasks:
                to_add.append(frame.copy())

        # Actual adding of tasks:
        shift = 1
        for task in to_add:
            task['begin'] = time + TIME_SHIFT_FOR_GT * shift
            tasks[task['ptr']] = task
            shift += 1

        for callback in self.callbacks + self.stack_sniffers:
            callback.handle_stack({'pid': pid, 'tid': tid, 'time': time}, stack, kind)



# example:
#
# the_thread = callbacks.process(-1).thread(-1)
# counter = the_thread.counter(domain='mydomain', name='countername')
# for i in range(5):
#   counter.set_value(time_stamp=%timestamp%, value=i)
# task = the_thread.task('MY_TASK')  # same with frames
# for i in range(7):
#   task.begin(%timestamp%)
#   task.add_args({'a':1, 'b':'2'})
#   task.end(%timestamp%)

# FIXME: doesn't belong this file, move to 'SEA reader' or something


class FileWrapper:
    def __init__(self, path, args, tree, domain, tid):
        self.args = args
        self.tree = tree
        self.domain = domain
        self.tid = tid
        self.next_wrapper = None
        self.file = open(path, "rb")
        self.record = self.read()

    def __del__(self):
        self.file.close()

    def next(self):
        self.record = self.read()

    def get_record(self):
        return self.record

    def get_pos(self):
        return self.file.tell()

    def get_size(self):
        return os.path.getsize(self.file.name)

    def get_path(self):
        return self.file.name

    def read(self):
        call = {"tid": self.tid, "pid": self.tree["pid"], "domain": self.domain}

        tuple = read_chunk_header(self.file)
        if tuple == (0, 0, 0):  # mem mapping wasn't trimmed on close, zero padding goes further
            return None
        call["time"] = tuple[0]

        if tuple[1] >= len(TaskTypes):
            return None
        call["type"] = tuple[1]

        flags = tuple[2]
        if flags & 0x1:  # has id
            chunk = self.file.read(2 * 8)
            call["id"] = struct.unpack('QQ', chunk)[0]
        if flags & 0x2:  # has parent
            chunk = self.file.read(2 * 8)
            call["parent"] = struct.unpack('QQ', chunk)[0]
        if flags & 0x4:  # has string
            chunk = self.file.read(8)
            str_id = struct.unpack('Q', chunk)[0]  # string handle
            call["str"] = self.tree["strings"][str_id]
        if flags & 0x8:  # has tid, that differs from the calling thread (virtual tracks)
            chunk = self.file.read(8)
            call["tid"] = int(struct.unpack('q', chunk)[0])

        if flags & 0x10:  # has data
            chunk = self.file.read(8)
            length = struct.unpack('Q', chunk)[0]
            call["data"] = self.file.read(length).decode()

        if flags & 0x20:  # has delta
            chunk = self.file.read(8)
            call["delta"] = struct.unpack('d', chunk)[0]

        if flags & 0x40:  # has pointer
            chunk = self.file.read(8)
            ptr = struct.unpack('Q', chunk)[0]
            if not resolve_pointer(self.args, self.tree, ptr, call):
                call["pointer"] = ptr

        if flags & 0x80:  # has pseudo pid
            chunk = self.file.read(8)
            call["pid"] = struct.unpack('q', chunk)[0]

        return call

    def set_next(self, wrapper):
        self.next_wrapper = wrapper

    def get_next(self):
        return self.next_wrapper


def transform2(args, tree, skip_fn=None):
    with Callbacks(args, tree) as callbacks:
        if callbacks.is_empty():
            return callbacks.get_result()

        wrappers = {}
        for domain, content in tree["domains"].items():  # go thru domains
            for tid, path in content["files"]:  # go thru per thread files
                parts = split_filename(path)

                file_wrapper = FileWrapper(path, args, tree, domain, tid)
                if file_wrapper.get_record():  # record is None if something wrong with file reading
                    wrappers.setdefault(parts['dir'] + '/' + parts['name'], []).append(file_wrapper)

        for unordered in wrappers.values():  # chain wrappers by time
            ordered = sorted(unordered, key=lambda wrapper: wrapper.get_record()['time'])
            prev = None
            for wrapper in ordered:
                if prev:
                    prev.set_next(wrapper)
                prev = wrapper

        files = []
        for unordered in wrappers.values():
            for wrapper in unordered:
                next = wrapper.get_next()
                if skip_fn and skip_fn(wrapper.get_path()):  # for "cut" support
                    continue
                files.append(wrapper)

        if verbose_level() > verbose_level('warning'):
            progress = DummyWith()
        else:
            size = sum([file.get_size() for file in files])
            progress = Progress(size, 50, 'Converting: %s (%s)' % (os.path.basename(args.input), format_bytes(size)))

        with progress:
            count = 0
            while True:  # records iteration
                record = None
                earliest = None
                for file in files:
                    rec = file.get_record()
                    if not rec:  # finished
                        continue
                    if not record or rec['time'] < record['time']:
                        record = rec
                        earliest = file
                if not record:  # all finished
                    break
                earliest.next()

                if message('info', "%d\t%s\t%s" % (count, TaskTypes[record['type']], record)):
                    pass
                elif count % ProgressConst == 0:
                    progress.tick(sum([file.get_pos() for file in files]))
                callbacks.on_event(TaskTypes[record['type']], record)
                count += 1

        callbacks("metadata_add", {'domain': 'IntelSEAPI', 'str': '__process__', 'pid': tree["pid"], 'tid': -1, 'delta': -1})
        for pid, name in tree['groups'].items():
            callbacks.set_process_name(tree["pid"], name)

    return callbacks.get_result()


# FIXME: doesn't belong this file, move to 'utils'

def get_module_by_ptr(tree, ptr):
    keys = list(tree['modules'].keys())
    keys.sort()  # looking for first bigger the address, previous is the module we search for
    item = keys[0]
    for key in keys[1:]:
        if key > ptr:
            break
        item = key
    module = tree['modules'][item]
    if item < ptr < item + int(module[1]):
        return item, module[0]
    else:
        return None, None


def win_parse_symbols(symbols):
    sym = []
    for line in symbols.split('\n'):
        line = line.strip()
        if not line:
            continue
        if '\t' in line:
            parts = line.strip().split('\t')
            addr, size, name = parts[:3]
            if int(size):
                sym.append({'addr': int(addr), 'size': int(size), 'name': name})
                if len(parts) == 4:
                    sym[-1].update({'pos': parts[3]})
    sym.sort(key=lambda data: data['addr'])
    return sym


def win_resolve(symbols, addr):
    idx = bisect_right(symbols, addr, lambda data: data['addr']) - 1
    if idx > -1:
        sym = symbols[idx]
        if sym['addr'] <= addr <= (sym['addr'] + sym['size']):
            return (sym['pos'] + '\n' + sym['name']) if 'pos' in sym else sym['name']
    return ''


def resolve_cmd(args, path, load_addr, ptr, cache={}):
    if sys.platform == 'win32':
        if path.startswith('\\'):
            path = 'c:' + path
        if path.lower() in cache:
            return win_resolve(cache[path.lower()], ptr - load_addr)
        bitness = '32' if '32' in platform.architecture()[0] else '64'
        executable = os.path.sep.join([args.bindir, 'TestIntelSEAPI%s.exe' % bitness])
        cmd = '"%s" "%s"' % (executable, path)
    elif sys.platform == 'darwin':
        cmd = 'atos -o "%s" -l %s %s' % (path, to_hex(load_addr), to_hex(ptr))
    elif 'linux' in sys.platform:
        cmd = 'addr2line %s -e "%s" -i -p -f -C' % (to_hex(ptr), path)
    else:
        raise "Unsupported platform!"

    env = dict(os.environ)
    if "INTEL_SEA_VERBOSE" in env:
        del env["INTEL_SEA_VERBOSE"]

    try:
        proc = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        (symbol, err) = proc.communicate()
    except IOError:
        err = traceback.format_exc()
        import gc
        gc.collect()
        print("gc.collect()")
    except:
        err = traceback.format_exc()
    if err:
        print(cmd)
        print(err)
        return ''

    if sys.platform == 'win32':
        cache[path.lower()] = win_parse_symbols(symbol.decode())
        return win_resolve(cache[path.lower()], ptr - load_addr)
    return symbol


# finds first bigger
def bisect_right(array, value, key=lambda item: item):  #upper_bound, dichotomy, binary search
    lo = 0
    hi = len(array)
    while lo < hi:
        mid = (lo + hi) // 2
        if value < key(array[mid]):
            hi = mid
        else:
            lo = mid + 1
    return lo


def resolve_jit(tree, ptr, cache):
    if 'jit' not in tree:
        return False
    jit = tree['jit']
    if jit['start'] <= ptr <= jit['end']:
        jit_data = jit['data']
        idx = bisect_right(jit_data, ptr, lambda item: item['addr']) - 1
        if idx > -1:
            offset = ptr - jit_data[idx]['addr']
            if offset > jit_data[idx]['size']:
                return False
            cache[ptr] = {'module': 'jit'}
            cache[ptr]['str'] = jit_data[idx]['name']
            if not cache[ptr]['str']:
                cache[ptr]['str'] = 'jit_method_%d' % jit_data[idx]['id']
            cache[ptr]['__file__'] = jit_data[idx]['file']
            lines = jit_data[idx]['lines']
            idx = bisect_right(lines, offset, lambda item: item[0]) - 1
            if idx > -1:
                cache[ptr]['__line__'] = lines[idx][1]
        return True
    else:
        return False


def resolve_pointer(args, tree, ptr, call, cache={}):
    if ptr not in cache:
        if not resolve_jit(tree, ptr, cache):
            (load_addr, path) = get_module_by_ptr(tree, ptr)
            if path is None or not os.path.exists(path):
                cache[ptr] = None
            else:
                symbol = resolve_cmd(args, path, load_addr, ptr)
                cache[ptr] = {'module': path}
                lines = symbol.splitlines()
                if lines:
                    if sys.platform == 'win32':
                        if len(lines) == 1:
                            cache[ptr]['str'] = lines[0]
                        elif len(lines) == 2:
                            cache[ptr]['str'] = lines[1]
                            (cache[ptr]['__file__'], cache[ptr]['__line__']) = lines[0].rstrip(")").rsplit("(", 1)
                    elif sys.platform == 'darwin':
                        if '(in' in lines[0]:
                            parts = lines[0].split(" (in ")
                            cache[ptr]['str'] = parts[0]
                            if ') (' in parts[1]:
                                (cache[ptr]['__file__'], cache[ptr]['__line__']) = parts[1].split(') (')[1].split(':')
                                cache[ptr]['__line__'] = cache[ptr]['__line__'].strip(')')
                    else:
                        if ' at ' in lines[0]:
                            (cache[ptr]['str'], fileline) = lines[0].split(' at ')
                            (cache[ptr]['__file__'], cache[ptr]['__line__']) = fileline.strip().split(':')
    if not cache[ptr] or 'str' not in cache[ptr]:
        return False
    call.update(cache[ptr])
    return True


def resolve_stack(args, tree, data):
    if tree['process']['bits'] == 64:
        frames = struct.unpack('Q' * (len(data) / 8), data)
    else:
        frames = struct.unpack('I' * (len(data) / 4), data)
    stack = []
    for frame in frames:
        res = {'ptr': frame}
        if resolve_pointer(args, tree, frame, res):
            stack.append(res)
    return stack


def attachme():
    print("Attach me!")
    while not sys.gettrace():
        pass
    import time
    time.sleep(1)


class TaskCombiner:
    not_implemented_err_string = 'You must implement this method in the TaskCombiner derived class!'

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.finish()
        return False

    def __init__(self, args, tree):
        self.tree = tree
        self.args = args
        self.event_map = {}
        self.events = []

        (self.source_scale_start, self.target_scale_start, self.ratio) = tuple([0, 0, 1. / 1000])  # nanoseconds to microseconds

    def get_targets(self):
        """Returns list with the paths to output files."""
        raise NotImplementedError(TaskCombiner.not_implemented_err_string)

    def complete_task(self, type, begin, end):
        """
        Handles task to the derived class output format.

        Args:
            type: Task type.
            begin: Dictionary with task begin data.
            end: Dictionary with task end data.
        """
        raise NotImplementedError(TaskCombiner.not_implemented_err_string)

    def finish(self):
        """Called to finalize a derived class."""
        raise NotImplementedError(TaskCombiner.not_implemented_err_string)

    def convert_time(self, time):
        return (time - self.source_scale_start) * self.ratio + self.target_scale_start

    def global_metadata(self, data):
        pass

    def relation(self, data, head, tail):
        pass

    def handle_stack(self, task, stack, name='stack'):
        pass

    def context_switch(self, time, cpu, prev, next):
        """
        Called to process context switch events on CPU.
        :param cpu: CPU number (int)
        :param prev: previous task description (dict. Example: {'tid': 2935135, 'state': 'S', 'name': u'vtplay', 'prio': 31})
        :param next: next task description (dict. see above)
        """
        pass

    def wakeup(self, time, cpu, prev, next):
        """
        Called to process thread wakup events on CPU.
        :param cpu: CPU on which the event occurred
        :param prev: currently running process description for CPU (dict. Example: {'tid': 123, 'name': 'kuku', 'pid': 12})
        :param next: thread being woken up (dict. see above)
        """
        pass


def to_hex(value):
    return "0x" + hex(value).rstrip('L').replace("0x", "").upper()


def get_name(begin):
    if 'str' in begin:
        return begin['str']
    elif 'pointer' in begin:
        return "func<" + to_hex(begin['pointer']) + ">"
    else:
        return "<unknown>"


class GraphCombiner(TaskCombiner):
    def __init__(self, args, tree):
        TaskCombiner.__init__(self, args, tree)
        self.args = args
        self.per_domain = {}
        self.relations = {}
        self.threads = set()
        self.per_thread = {}

    @staticmethod
    def get_name_ex(begin):
        name = get_name(begin)
        if ':' in name:
            parts = name.split(':')
            if parts[1].isdigit():
                return parts[0]
        return name

    def get_per_domain(self, domain):
        return self.per_domain.setdefault(domain, {
            'counters': {}, 'objects': {}, 'frames': {}, 'tasks': {}, 'markers': {}, 'threads': {}
        })

    def complete_task(self, type, begin, end):
        if 'sampled' in begin and begin['sampled']:
            return
        tid = begin['tid'] if 'tid' in begin else None
        self.threads.add(tid)
        domain = self.get_per_domain(begin['domain'])
        if type == 'task':
            task = domain['tasks'].setdefault(self.get_name_ex(begin), {'time': []})
            task['time'].append(end['time'] - begin['time'])
            if '__file__' in begin:
                task['src'] = begin['__file__'] + ":" + begin['__line__']

            if begin['type'] == 0:  # non-overlapped only
                # We expect parents to be reported in the end order (when the end time becomes known)
                orphans = self.per_thread.setdefault(begin['tid'], [])
                left_index = bisect_right(orphans, begin['time'], lambda orphan: orphan[0]['time'])  # first possible child
                right_index = bisect_right(orphans, end['time'], lambda orphan: orphan[0]['time']) - 1  # last possible child
                for i in range(right_index, left_index - 1, -1):  # right to left to be able deleting from array
                    orphan = orphans[i]
                    if orphan[1]['time'] < end['time']:  # a parent is found!
                        self.add_relation({
                            'label': 'calls', 'from': self.make_id(begin['domain'], self.get_name_ex(begin)),
                            'to': self.make_id(orphan[0]['domain'], self.get_name_ex(orphan[0]))})
                        del orphans[i]
                orphans.insert(left_index, (begin, end))
            else:
                self.add_relation({'label': 'executes', 'from': self.make_id("threads", str(tid)),
                                   'to': self.make_id(begin['domain'], self.get_name_ex(begin)), 'color': 'gray'})
        elif type == 'marker':
            domain['markers'].setdefault(begin['str'], [])
        elif type == 'frame':
            pass
        elif type == 'counter':
            if 'delta' in begin:
                domain['counters'].setdefault(begin['str'], []).append(begin['delta'])
            else:
                return  # TODO: add multi-value support
        elif 'object' in type:
            if 'snapshot' in type:
                return
            objects = domain['objects'].setdefault(begin['str'], {})
            object = objects.setdefault(begin['id'], {})
            if 'new' in type:
                object['create'] = begin['time']
            elif 'delete' in type:
                object['destroy'] = begin['time']
        else:
            message('message', "Unhandled: " + type)

    def finish(self):
        for tid, orphans in self.per_thread.items():
            last_time = 0
            for orphan in orphans:
                if (orphan[1]['time'] < last_time):
                    print("FIXME: orphan[1]['time'] < last_time")
                last_time = orphan[1]['time']
                begin = orphan[0]
                self.add_relation({'label': 'executes', 'from': self.make_id("threads", str(tid)),
                                   'to': self.make_id(begin['domain'], self.get_name_ex(begin)), 'color': 'gray'})

    @staticmethod
    def make_id(domain, name):
        import re
        res = "%s_%s" % (domain, name)
        return re.sub("[^a-z0-9]", "_", res.lower())

    def relation(self, data, head, tail):
        if head and tail:
            self.add_relation({'label': self.get_name_ex(data), 'from': self.make_id(head['domain'], self.get_name_ex(head)), 'to': self.make_id(tail['domain'], self.get_name_ex(tail)), 'color': 'red'})

    def add_relation(self, relation):
        key = frozenset(relation.items())
        if key in self.relations:
            return
        self.relations[key] = relation

    def handle_stack(self, task, stack, name='stack'):
        tid = abs(task['tid']) if 'tid' in task else None
        self.threads.add(tid)
        parent = None
        for frame in reversed(stack):
            domain = self.get_per_domain(frame['module'])
            name = frame['str'].split('+')[0]
            domain['tasks'].setdefault(name, {'time': [0]})
            if parent:
                self.add_relation({'label': 'calls', 'from': self.make_id(parent['module'], parent['name']), 'to': self.make_id(frame['module'], name)})
            else:
                self.add_relation({'label': 'executes', 'from': self.make_id("threads", str(tid)), 'to': self.make_id(frame['module'], name), 'color': 'gray'})
            parent = frame.copy()
            parent.update({'name': name})


class Collector:
    def __init__(self, args):
        self.args = args

    @classmethod
    def set_output(cls, output): # has to be object supporting 'write' method
        global_storage('log')['file'] = output

    @classmethod
    def get_output(cls, statics = {}):
        log = global_storage('log')
        if not log:
            args = get_args()
            log_name = datetime.now().strftime('sea_%Y_%m_%d__%H_%M_%S.log')
            if args:
                log_path = subst_env_vars(args.output)
                if os.path.isfile(log_path):
                    log_path = os.path.dirname(log_path)
                ensure_dir(log_path, False)
                if 'tempfile' in statics:
                    statics['tempfile'].close()
                    if os.path.dirname(statics['tempfile'].name) != log_path:
                        shutil.copy(statics['tempfile'].name, log_path)
                        del statics['tempfile']
            else:
                if 'tempfile' in statics:
                    return statics['tempfile']
                log_path = (tempfile.gettempdir() if sys.platform == 'win32' else '/tmp')
            log_file = os.path.join(
                log_path,
                log_name
            )
            print("For execution details see:", log_file)
            if args:
                cls.set_output(open(log_file, 'a'))
            else:
                statics['tempfile'] = open(log_file, 'a')
                return statics['tempfile']
        return log['file']

    @classmethod
    def log(cls, msg, stack=False):
        if not type(stack) is bool:
            stack = False
        msg = msg.strip()
        cut = '\n' + '-' * 100 + '\n'
        msg = cut + msg + '\n\n' + (''.join(traceback.format_stack()[:-1]) if stack else '') + cut
        output = cls.get_output()
        output.write(msg + '\n')
        output.flush()

    @classmethod
    def execute(cls, cmd, log=True, **kwargs):
        start_time = time.time()
        if 'stdout' not in kwargs:
            kwargs['stdout'] = subprocess.PIPE
        if 'stderr' not in kwargs:
            kwargs['stderr'] = subprocess.PIPE
        if 'env' not in kwargs:
            kwargs['env'] = get_original_env()
        if sys.version[0] == '3':
            kwargs['encoding'] = 'utf8'

        (out, err) = subprocess.Popen(cmd, shell=False, **kwargs).communicate()
        if log:
            cls.log("\ncmd:\t%s:\nout:\t%s\nerr:\t%s\ntime: %s" % (cmd, str(out).strip(), str(err).strip(), str(timedelta(seconds=(time.time() - start_time)))), stack=True if err else False)
        if verbose_level() == verbose_level('info'):
            print("\n\n -= '%s' output =- {\n" % cmd)
            print(out.strip() if out else '')
            print("\n", "-" * 50, "\n")
            print(err.strip() if err else '')
            print("\n}\n\n")
        return out, err

    @classmethod
    def execute_detached(cls, cmd, **kwargs):
        cls.log("\nDetached:\t%s" % cmd)
        if sys.platform == 'win32':
            DETACHED_PROCESS = 0x00000008
            CREATE_NEW_PROCESS_GROUP = 0x00000200
            CREATE_NO_WINDOW = 0x08000000
            info = subprocess.STARTUPINFO()
            info.dwFlags = subprocess.STARTF_USESHOWWINDOW
            info.wShowWindow = 0  # SW_HIDE
            subprocess.Popen(cmd, shell=False, startupinfo=info, stdin=None, stdout=None, stderr=None, creationflags=(CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP), **kwargs)
        else:
            subprocess.Popen(cmd, shell=False, stdin=None, stdout=None, stderr=None, **kwargs)

    def start(self):
        raise NotImplementedError('Collector.start is not implemented!')

    def stop(self, wait=True):
        raise NotImplementedError('Collector.stop is not implemented!')

    @classmethod
    def detect_instances(cls, what):
        instances = []
        cmd = 'where' if sys.platform == 'win32' else 'which'
        (out, err) = cls.execute('%s %s' % (cmd, what))
        out = out.decode() if hasattr(out, 'decode') else out
        if err:
            return instances
        for line in out.split('\n'):
            line = line.strip()
            if line:
                instances.append(line)
        return instances


if __name__ == "__main__":
    start_time = time.time()
    ret_code = main()
    elapsed = time.time() - start_time
    print("Time Elapsed:", str(timedelta(seconds=elapsed)).split('.')[0])
    exit(ret_code)
