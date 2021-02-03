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
import re
import os
import sys
import time
from datetime import datetime, timedelta
import shutil
import subprocess
import threading
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))
from sea_runtool import Collector, is_domain_enabled, message, get_original_env

"""
    sudo dtrace -l | perl -pe 's/^.*?\S+\s+(\S+?)([0-9]|\s).*/\1/' | sort | uniq > /tmp/dtrace_providers.txt
    sudo dtrace -l > /tmp/dtrace_list.txt
    dtrace -n 'fbt:::entry { @[probefunc] = count(); }' -c 'ping host'
    http://www.brendangregg.com/DTrace/DTrace-cheatsheet.pdf

    https://docs.oracle.com/cd/E19253-01/817-6223/chp-variables-5/index.html
    TODO: printf("%s: called from %a\n", probefunc, caller);
    objc_runtime$target::: { ustack(); } /*objc_exception_throw*/
    pid$target::objc_msgSend:entry
    sudo dtrace -qn 'fbt::*vent13k*:entry/arg3/{printf("%d\n",arg2)}'  # keylogger
"""

DSCRIPT_HEADER = r"""
#pragma D option nolibs
#define GREEDY_ON ++self->greedy_enabled
#define GREEDY_OFF self->greedy_enabled = (self->greedy_enabled > 0) ? (self->greedy_enabled - 1) : self->greedy_enabled

BEGIN
{
    self->greedy_enabled = 0;
}


"""

dtrace_context_switch = r"""

/*
off-cpu

Probe that fires when the current CPU is about to end execution of a thread.
The curcpu variable indicates the current CPU.
The curlwpsinfo variable indicates the thread that is ending execution.
The curpsinfo variable describes the process containing the current thread.
The lwpsinfo_t structure of the thread that the current CPU will next execute is pointed to by args[0].
The psinfo_t of the process containing the next thread is pointed to by args[1].
*/

sched:::off-cpu
{
    printf(
        "%x\toff\t%x\t%x\t%x\t%s\t%x\t%x\t%s\n", machtimestamp, curcpu->cpu_id,
        curlwpsinfo->pr_lwpid, curlwpsinfo->pr_pri, curpsinfo->pr_fname,
        args[0]->pr_lwpid, args[0]->pr_pri, args[1]->pr_fname
    );
}
"""

OFF_CPU_STACKS = r"""
sched:::off-cpu
/pid == $target/
{
    printf("%x\tkstack\t%x\t%x:", machtimestamp, pid, tid);
    stack();
    printf("\n%x\tustack\t%x\t%x:", machtimestamp, pid, tid);
    ustack();
    /*
    printf("\n%x\tjstack\t%x\t%x:", machtimestamp, pid, tid);
    jstack(); //TODO: enable better support for jstack-s
    */
    printf("\n");
}
"""

dtrace_wakeup = r"""
sched:::wakeup
/curpsinfo->pr_pid == $target || args[1]->pr_pid == $target/
{
    printf("%x\twkp\t%x\t%x\t%s\t%x\t%s\t%x\t%x\t%x\t%x\n", machtimestamp,
        curpsinfo->pr_pid, curlwpsinfo->pr_lwpid,
        execname, cpu,
        stringof(args[1]->pr_fname),
        args[1]->pr_pid, args[0]->pr_lwpid,
        args[0]->pr_stype, args[0]->pr_wchan
    );
}

"""


osxaskpass = r"""#!/bin/bash
osascript -e 'Tell application "System Events" to display dialog "Password:" default answer "" with hidden answer with title "DTrace requires root priveledges"' -e 'text returned of result' 2>/dev/null
"""


pid_dtrace_hooks = [r"""
pid$target::*dtSEAHookScope*:entry /*{UMD_STACKS}*/
{
    printf(
        "%x\te\t%x\t%x\t%s\t%s\n", machtimestamp, pid, tid, copyinstr(arg1), copyinstr(arg2)
    );
}
""", r"""
pid$target::*dtSEAHookEndScope*:entry
{
    printf(
        "%x\tr\t%x\t%x\t%s\t%s\n", machtimestamp, pid, tid, copyinstr(arg0), copyinstr(arg1)
    );
}
""", r"""
/*
pid$target::*dtSEAHookArgStr*:entry
{
    printf(
        "%x\targ\t%x\t%x\t%s\t%s\n",
        machtimestamp, pid, tid, copyinstr(arg0), copyinstr(arg1)
    );
}

pid$target::*dtSEAHookArgInt*:entry
{
    printf(
        "%x\targ\t%x\t%x\t%s\t%d\n",
        machtimestamp, pid, tid, copyinstr(arg0), arg1
    );
}
*/

"""

]

pid_dtrace_hooks += [  # XXX
r"""
objc$target:::entry
/*{CONDITIONS}*/
{
    printf(
        "%x\te\t%x\t%x\tobjc\t%s%s\n", machtimestamp, pid, tid, probemod, probefunc
    );
    /*{ARGUMENTS}*/
}
""", r"""
objc$target:::return
/*{CONDITIONS}*/
{
    printf(
        "%x\tr\t%x\t%x\tobjc\t%s%s\n", machtimestamp, pid, tid, probemod, probefunc
    );
}
"""
] if 0 else []


FOLLOW_CHILD = r"""
//https://www.synack.com/2015/11/17/monitoring-process-creation-via-the-kernel-part-i/
proc:::exec-success /$target == curpsinfo->pr_ppid/{
    printf("%x\tfollowchild\t%x\t%x\t%s\t%x\t%x\t%s\n", machtimestamp, pid, tid, probename, curpsinfo->pr_ppid, curpsinfo->pr_pid, curpsinfo->pr_psargs);
    system("printf \"%d\n\" >> /*{FOLLOW_CHILD}*/", curpsinfo->pr_pid);
}

proc:::exec /$target == curpsinfo->pr_ppid/{
    printf("%x\tfollowchild\t%x\t%x\t%s\t%x\t%x\t%s\n", machtimestamp, pid, tid, probename, curpsinfo->pr_ppid, curpsinfo->pr_pid, curpsinfo->pr_psargs);
    system("printf \"%d\n\" >> /*{FOLLOW_CHILD}*/", curpsinfo->pr_pid);
}

syscall::exec*:return /$target == curpsinfo->pr_ppid/
{
    printf(
        "%x\tfollowchild\t%x\t%x\t%s\t%s\t%s\n", machtimestamp, pid, tid, probemod, probefunc, probename
    );
}

syscall::fork:return /$target == curpsinfo->pr_ppid/
{
    printf(
        "%x\tfollowchild\t%x\t%x\t%s\t%s\t%s\n", machtimestamp, pid, tid, probemod, probefunc, probename
    );
    system("printf \"%d\n\" >> /*{FOLLOW_CHILD}*/", curpsinfo->pr_pid);
}

"""

BRACKET_FUNC = r"""
pid$target:/*{M:F}*/:entry /*{UMD_STACKS}*/
/*{CONDITIONS}*/
{
    printf(
        "%x\te\t%x\t%x\t%s\t%s\n", machtimestamp, pid, tid, probemod, probefunc
    );
    /*{ARGUMENTS}*/
    GREEDY_ON;
}
""", r"""
pid$target:/*{M:F}*/:return
/*{CONDITIONS}*/
{
    GREEDY_OFF;
    
    printf(
        "%x\tr\t%x\t%x\t%s\t%s\n", machtimestamp, pid, tid, probemod, probefunc
    );
}
"""


def bracket_hook(module='', function=''):
    res = []
    for item in BRACKET_FUNC:
        res.append(item.replace('/*{M:F}*/', '%s:%s' % (module, function)))
    return res


for mask in ['JavaScriptCore', '*GL*:*gl*', '*GL*:*GL*', 'Metal', '*MTLDriver', '*GLDriver']:  # ,'mdtest', 'libigdmd.dylib'
    pid_dtrace_hooks += bracket_hook(*mask.split(':'))


# TODO: add opencl_api & opencl_cpu providers

IO_HOOKS = r"""

fsinfo:::open,fsinfo:::close
{
    printf(
        "%x\tio\t%x\t%x\t%s\t%s\n", machtimestamp, pid, tid, probename, stringof(args[0]->fi_pathname)
    );
}

"""


OPEN_CL = r"""
opencl_api$target:::, opencl_cpu$target:::
{
    printf(
        "%x\tocl\t%x\t%x\t%s\t%s\t%s\n", machtimestamp, pid, tid, probemod, probefunc, probename
    );
}
"""

# FIXME: extract Interrupt handling and do conditional
fbt_dtrace_hooks = [r"""

//void kernel_debug_enter(uint32_t coreid, uint32_t debugid, uint64_t timestamp, uintptr_t arg1, uintptr_t arg2, uintptr_t arg3, uintptr_t arg4, uintptr_t threadid);
fbt::kernel_debug_enter:entry
{
    printf(
        "%x\tkd\t%x\t%x\t%s\t%x\t%x\t%x\t%x\t%x\t%x\t%x\t%x\n", machtimestamp, pid, tid, probefunc, arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7
    );
}

""", r"""

fbt::*debugid*enabled*:entry
{
    printf(
        "%x\tkd\t%x\t%x\t%s\t%x\t%x\t%x\t%x\t%x\t%x\n", machtimestamp, pid, tid, probefunc, arg0, arg1, arg2, arg3, arg4, arg5
    );
}


""" if not "DISABLED XXX" else '', r"""
fbt::*dtSEAHookScope*:entry /*{KMD_STACKS}*/
{
    printf(
        "%x\te\t%x\t%x\t%s_%s\t%s\t%d\n",
        machtimestamp, pid, tid, stringof(probemod), stringof(arg1), stringof(arg2), arg0
    );
}
""", r"""
fbt::*dtSEAHookEndScope*:entry
{
    printf(
        "%x\tr\t%x\t%x\t%s_%s\t%s\n",
        machtimestamp, pid, tid, stringof(probemod), stringof(arg0), stringof(arg1)
    );
}

fbt::*dtSEAHookArgStr*:entry
{
    printf(
        "%x\targ\t%x\t%x\t%s\t%s\n",
        machtimestamp, pid, tid, stringof(arg0), stringof(arg1)
    );
}

fbt::*dtSEAHookArgInt*:entry
{
    printf(
        "%x\targ\t%x\t%x\t%s\t%d\n",
        machtimestamp, pid, tid, stringof(arg0), arg1
    );
}

fbt::*dtSEAHookArgBlobStart*:entry
{
    printf(
        "%x\tbs\t%x\t%x\t%x\t%s\t%d\n",
        machtimestamp, pid, tid, arg1, stringof(arg1), arg0
    );
    trace(stringof(arg0));
    tracemem(arg0, 10);
}

fbt::*dtSEAHookArgBlob1024*:entry
{
    printf(
        "%x\tblb\t%x\t%x\n",
        machtimestamp, pid, tid
    );
    tracemem(arg0, 1024);
}

fbt::*dtSEAHookArgBlobEnd*:entry
{
    printf(
        "%x\tbe\t%x\t%x\n",
        machtimestamp, pid, tid
    );
}

""" if not "DISABLED XXX" else '', r"""

/*
  Interrupt handling.
  The list of interrupts was obtained by running 'dtrace -l | grep handleInterrupt'
*/
fbt:com.apple.driver.AppleAPIC:_ZN28AppleAPICInterruptController15handleInterruptEPvP9IOServicei:entry,
fbt:com.apple.iokit.IOPCIFamily:_ZN8AppleVTD15handleInterruptEP22IOInterruptEventSourcei:entry,
fbt:com.apple.iokit.IOPCIFamily:_ZN32IOPCIMessagedInterruptController15handleInterruptEPvP9IOServicei:entry,
fbt:com.apple.driver.AppleSMBusController:_ZN23AppleSMBusControllerMCP15handleInterruptEP22IOInterruptEventSourcei:entry,
fbt:com.apple.driver.AppleThunderboltNHI:_ZN19AppleThunderboltNHI15handleInterruptEv:entry
{
    printf("%x\tie\t%x\t%x\t%s\t%x\t%s\t%s\n", machtimestamp,
        curpsinfo->pr_pid, curlwpsinfo->pr_lwpid,
        execname, cpu, probemod, probefunc
    );
}

fbt:com.apple.driver.AppleAPIC:_ZN28AppleAPICInterruptController15handleInterruptEPvP9IOServicei:return,
fbt:com.apple.iokit.IOPCIFamily:_ZN8AppleVTD15handleInterruptEP22IOInterruptEventSourcei:return,
fbt:com.apple.iokit.IOPCIFamily:_ZN32IOPCIMessagedInterruptController15handleInterruptEPvP9IOServicei:return,
fbt:com.apple.driver.AppleSMBusController:_ZN23AppleSMBusControllerMCP15handleInterruptEP22IOInterruptEventSourcei:return,
fbt:com.apple.driver.AppleThunderboltNHI:_ZN19AppleThunderboltNHI15handleInterruptEv:return
{
    printf("%x\tir\t%x\t%x\t%s\t%x\t%s\t%s\n", machtimestamp,
        curpsinfo->pr_pid, curlwpsinfo->pr_lwpid,
        execname, cpu, probemod, probefunc
    );
}

"""]


STACKS = {

'UMD': r"""
{
    printf("%x\tustack\t%x\t%x:", machtimestamp, pid, tid);
    ustack();
    printf("\n");
}
""",

'KMD': r"""
{
    printf("%x\tkstack\t%x\t%x:", machtimestamp, pid, tid);
    stack();
    printf("\n");
}
"""
}


def mach_absolute_time(static={}):
    if not static:
        import ctypes
        libc = ctypes.CDLL('libc.dylib', use_errno=True)
        static['mach_absolute_time'] = libc.mach_absolute_time
        static['mach_absolute_time'].restype = ctypes.c_uint64
    return static['mach_absolute_time']()


class FifoReader(threading.Thread):
    def __init__(self, collector, path):
        threading.Thread.__init__(self)
        self.collector = collector
        self.pipe = path
        if os.path.exists(self.pipe):
            os.remove(self.pipe)
        os.mkfifo(self.pipe)
        self.file = os.open(self.pipe, os.O_RDWR)

    def run(self):
        print('Started reading', self.pipe)
        while self.file:
            chunks = os.read(self.file, 1024).strip()
            for chunk in chunks.split('\n'):
                if chunk != 'close':
                    self.collector.attach(int(chunk))
                print('Read:', chunk)
        print('Stopped reading', self.pipe)

    def stop(self):
        os.write(self.file, 'close\n')
        os.close(self.file)
        self.file = None
        os.unlink(self.pipe)


class DTraceCollector(Collector):
    class Subcollector:
        @staticmethod
        def get_hooks(args):
            return None

        @staticmethod
        def collect(collector, on):
            pass

    def __init__(self, args):
        Collector.__init__(self, args)

        self.processes = {}

        self.files = []
        self.subcollectors = set()

        self.attached = set()

        if 'SUDO_ASKPASS' not in os.environ:
            get_original_env()['SUDO_ASKPASS'] = self.create_ask_pass()
        assert 'DYLD_INSERT_LIBRARIES' not in os.environ

        self.sudo_execute('pkill dtrace')
        self.script = None
        self.prepare()
        self.times = []
        self.attach_by_pid = True

    @staticmethod
    def create_ask_pass():
        path = '/tmp/osxaskpass.sh'
        if os.path.exists(path):
            return path
        with open(path, 'w') as file:
            file.write(osxaskpass)
        os.chmod(path, 0o700)
        return path

    @staticmethod
    def gen_options(options):
        return '\n'.join('#pragma D option %s=%s' % (key, str(value)) for key, value in options) + '\n'

    def prepare(self):
        self.files = [os.path.join(self.args.output, 'data-%s.dtrace' % (self.args.cuts[0] if self.args.cuts else '0'))]
        assert not os.path.exists(self.files[0])  # TODO: remove if not asserts, or return back: was if ... os.remove(self.files[0])

        dtrace_script = [DSCRIPT_HEADER]
        options = [
            ('bufresize', 'auto'),
        ]
        if self.args.ring:  # https://docs.oracle.com/cd/E19253-01/817-6223/chp-buf/index.html
            options += [
                ('bufpolicy', 'ring'),
                ('bufsize', '64m')  # 64 is maximum, system crashes on any bigger, even 65m
            ]
        else:
            options += [
                ('switchrate', '10hz'),  # print at 10Hz (instead of 1Hz) - brendangregg
                ('bufsize', '4g')
            ]

        dtrace_script.append(self.gen_options(options))

        if is_domain_enabled('context_switches'):
            dtrace_script.append(dtrace_context_switch)

        if is_domain_enabled('fbt_hooks'):
            dtrace_script += fbt_dtrace_hooks

        for subcollector in self.subcollectors:
            hooks = subcollector.get_hooks(self.args)  # support multi pid for subcollectors
            if hooks:
                dtrace_script += hooks
            subcollector.collect(self, True)

        self.script = dtrace_script

    def sudo_execute(self, cmd):
        return self.execute('sudo -E -A ' + cmd)

    def prepare_per_pid(self):
        dtrace_script = []
        if is_domain_enabled('pid_hooks'):
            dtrace_script += pid_dtrace_hooks  # TODO: add %app_name% hotspots unconditionally

        if is_domain_enabled('instrument_target'):
            if self.args.victim:
                mod_name = os.path.basename(self.args.victim)
            elif self.args.target:
                (out, err) = DTraceCollector.execute('ps -p %d -o args' % self.args.target, log=False)
                if not err:
                    lines = out.strip().split('\n')
                    if len(lines) > 1:
                        executable = lines[1].split()[0]
                        mod_name = executable.split('/')[-1]

            print('Auto-instrumented module:', mod_name)
            dtrace_script += bracket_hook(mod_name.replace(' ', '*'))

        if is_domain_enabled('opencl'):
            dtrace_script.append(OPEN_CL)

        return dtrace_script

    def patch_per_pid(self, pids, items):
        result = []
        for item in items:
            if '$target:' in item:
                for pid in pids:
                    result.append(item.replace('$target:', '%s:' % str(pid)))
            else:
                result.append(item)

        return result

    def get_cmd(self, out, script, pids=[]):
        # -C Run the C preprocessor
        # -Z Permit probe descriptions that match zero probes
        # -w Permit destructive actions in D programs
        cmd = 'sudo -E -A dtrace -C -Z -w -o "%s" -s "%s"' % (out, script)  # FIXME: sudo_execute
        if self.args.verbose != 'info':
            cmd += ' -q'  # Set quiet mode
        else:
            # -S Show D compiler intermediate code
            # -v Print an interface stability report
            # -V Report the highest D programming interface version
            # -e Exit after compiling
            # -l List probes instead of enabling them
            cmd += ' '

        for pid in pids:
            cmd += " -p %s" % pid
        return cmd

    def launch_victim(self, victim, env):
        proc = self.run_dtrace(victim=victim, env=env)
        if not proc:
            return None

        class PopenWrapper:
            def __init__(self, parent, victim):
                self.parent = parent
                cmd = 'pgrep -n "%s"' % os.path.basename(victim[0])
                while True:
                    data, err = parent.execute(cmd)
                    if data:
                        self.pid = int(data)
                        break
                    time.sleep(1)

            def send_signal(self, sig):
                self.parent.sudo_execute('kill -%d %d' % (sig, self.pid))

            def wait(self, sec=10):
                proc['proc'].wait()
                """ XXX
                for x in range(0, sec):
                    proc['proc'].poll()
                    time.sleep(1)
                """
                return proc['proc'].returncode

            def communicate(self):
                self.wait()
                return None, None

        return PopenWrapper(self, victim)

    def run_dtrace(self, attach_by_name=False, victim=None, env=None):
        self.attach_by_pid = False
        # spawn dtrace tracers and exit, all means to stop it must be saved to self members:
        # launch command line with dtrace script and remember pid
        script = os.path.join(self.args.output, 'script.d')
        cmd = self.get_cmd(self.files[0], script)
        dtrace_script = self.script[:]

        hooks = []
        dtrace_script += hooks
        # The target is known only when start is called, so doing part of preparation here
        if attach_by_name:
            dtrace_script += self.prepare_per_pid()
            cmd += " -W %s" % os.path.basename(self.args.victim)
        elif victim:
            dtrace_script += self.prepare_per_pid()
            cmd += ' -c "%s"' % ' '.join(victim)
        else:
            assert not any('$target' in item for item in dtrace_script)
            pids = self.args.target if isinstance(self.args.target, list) else [self.args.target]
            for pid in pids:
                cmd += " -p %s" % pid
            print("Attaching PIDs:", pids)
            items = self.prepare_per_pid()
            dtrace_script += self.patch_per_pid(pids, items)
            for pid in pids:
                if self.args.stacks:
                    dtrace_script.append(OFF_CPU_STACKS.replace('$target', str(pid)))
                if is_domain_enabled('wakeups'):
                    dtrace_script.append(dtrace_wakeup.replace('$target', str(pid)))
                for hook in hooks:
                    dtrace_script.append(hook.replace('/*{CONDITIONS}*/', '/pid == %s/' % str(pid)))

            if self.args.stacks:
                dtrace_script = self.patch_stacks(pids, dtrace_script)

        # remove duplicates from the list:
        dtrace_script = [item for n, item in enumerate(dtrace_script) if item not in dtrace_script[:n]]
        dtrace_script = '\n'.join(dtrace_script)

        with open(script, 'w') as file:
            file.write(dtrace_script)

        return self.run_parallel(cmd, env)

    def start(self):  # FIXME: see man dtrace -W option for proper attach
        self.times.append(datetime.now())
        if self.attach_by_pid:
            self.run_dtrace()

    def run_parallel(self, cmd, env=None):
        self.log(cmd)
        proc = self.processes.setdefault(cmd, {})
        proc['proc'] = subprocess.Popen(cmd, shell=True, stdin=None, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env = env or os.environ)
        proc['pid'] = proc['proc'].pid
        self.log("%s -> pid: %d" % (cmd, proc['proc'].pid))
        return proc

    @staticmethod
    def patch_stacks(pids, items):
        reg_exp = re.compile(r"""(.*)\/\*\{(.*)_STACKS\}\*\/.*""", re.IGNORECASE | re.DOTALL)
        result = []
        for item in items:
            result.append(item)
            if '_STACKS}*/' in item:
                lines = item.strip().split('\n')
                assert lines[-1].strip().endswith('}')
                res = reg_exp.search(lines[0])
                what, where = res.groups()
                condition = '/$target == pid/' if pids else ''
                code = '\n%s %s %s' % (what, condition, STACKS[where])
                if pids:
                    for pid in pids:
                        result.append(code.replace('$target', str(pid)))
                else:
                    result.append(code)
        return result

    def attach(self, pid):
        if pid in self.attached:
            return
        pids = [pid] + list(self.get_pid_children(pid))
        self.attached |= set(pids)
        items = self.prepare_per_pid()
        dtrace_script = [DSCRIPT_HEADER] + [item for item in self.script if '$target' in item]
        dtrace_script += self.patch_per_pid(pids, items)
        script = os.path.join(self.args.output, 'script_%d.d' % pid)
        dtrace_script = '\n'.join(dtrace_script)
        self.files.append(os.path.join(self.args.output, 'data-%d-%s.dtrace' % (pid, self.args.cuts[0] if self.args.cuts else '0')))
        cmd = self.get_cmd(self.files[-1], script, pids)
        with open(script, 'w') as file:
            file.write(dtrace_script)
        self.run_parallel(cmd)

    @staticmethod
    def get_pid_children(parent):
        (out, err) = DTraceCollector.execute('ps -o pid,ppid -ax', log=False)
        if err:
            print(err)
            return
        for line in out.split('\n'):
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            pid, ppid = line.split()
            if str(parent) == ppid:
                yield int(pid)

    @staticmethod
    def locate(what, statics={}):
        try:
            if not statics:
                res = subprocess.check_output(['locate', '-S']).decode("utf-8")
                statics['locate'] = 'WARNING' not in res
                if not statics['locate']:
                    print(res)
            if statics['locate']:
                return subprocess.check_output(['locate', what]).decode("utf-8").split('\n')
        except Exception:
            pass
        return []

    def collect_codes(self):
        items = self.locate("*.codes")
        files = [item.strip() for item in items if not item.startswith('/Volumes')]

        items = self.locate("kdebug.h")
        files += [item.strip() for item in items if not item.startswith('/Volumes')]

        filtered = {}
        for file in files:
            if not file or not os.path.exists(file):
                continue
            name = os.path.basename(file)
            size = os.path.getsize(file)
            if size and (name not in filtered or os.path.getsize(filtered[name]) < size):
                filtered[name] = file
        for file in filtered.values():
            shutil.copy(file, self.args.output)

        items = self.locate("IntelGPUSignposts.plist")
        plists = []
        for line in items:
            line = line.strip()
            if line:

                plists.append((os.path.getmtime(line), line))  # finding newest
        if plists:
            plist = sorted(plists)[-1][1]
            shutil.copy(plist, self.args.output)

    def collect_system_info(self):
        with open(os.path.join(self.args.output, 'sysinfo.txt'), 'w') as file:
            (probes, err) = self.execute('sysctl -a', stdout=file)

    def stop(self, wait=True):
        for name, data in self.processes.items():
            print('Stopping:', name)
            pids = [data['pid']] + list(self.get_pid_children(data['pid']))
            for pid in pids:
                self.sudo_execute("kill -2 %d" % pid)
            for pid in pids:
                try:
                    os.waitpid(pid, 0)
                except:
                    pass

            if not data['proc']:
                continue
            out, err = data['proc'].communicate()
            message(None, "\n\n -= Target %s output =- {\n" % name)
            if out:
                self.log("Trace %s out:\n%s" % (name, out.decode()))
                message(None, out.strip())
            message(None, "-" * 50)
            if err:
                self.log("Trace %s err:\n%s" % (name, err.decode()), True)
                message(None, err.strip())
            message(None, "}\n\n")

            if data['proc'].returncode != 0:
                message('error', '%s(%d) has exited with error code %d check logs for details' % (name, data['pid'], data['proc'].returncode))

        for subcollector in self.subcollectors:
            print('Stopping:', subcollector)
            subcollector.collect(self, False)

        self.times.append(datetime.now())

        sys_log = os.path.join(self.args.output, 'sys_log.json')
        with open(sys_log, 'w') as file:
            cmd = 'log show --source --style json --debug --signpost'  # --last 1m --start, --end 'YYYY-MM-DD HH:MM:SS'
            cmd += self.times[1].strftime(" --end '%Y-%m-%d %H:%M:%S'")
            if self.args.ring:
                cmd += (self.times[1] - timedelta(seconds=self.args.ring)).strftime(" --start '%Y-%m-%d %H:%M:%S'")
            else:
                cmd += self.times[0].strftime(" --start '%Y-%m-%d %H:%M:%S'")
            self.execute(cmd, stdout=file)  # FIXME: get time of collection or ring size

        self.collect_system_info()
        self.collect_codes() 

        res = self.files + [sys_log]

        return res

    @classmethod
    def available(cls):
        if 'darwin' not in sys.platform:
            return False
        (out, err) = cls.execute('csrutil status')
        if 'disabled' not in out:
            print('Please do: "csrutil disable" from Recovery OS terminal to be able using dtrace...')
            return False
        return True


COLLECTOR_DESCRIPTORS = [{
    'format': 'dtrace',
    'available': DTraceCollector.available(),
    'collector': DTraceCollector
}]

if __name__ == "__main__":
    print(mach_absolute_time())
    DTraceCollector.check_graphics_firmware('com.apple.driver.AppleIntelSKLGraphics')

