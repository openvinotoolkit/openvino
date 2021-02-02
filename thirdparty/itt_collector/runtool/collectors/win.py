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
import time
from datetime import datetime
import shutil
import tempfile
import platform
import traceback
import subprocess
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))
from sea_runtool import Collector, is_domain_enabled
import sea


def async_exec(cmd, title=None, env=None):
    cmd = 'start "%s" /MIN /LOW %s' % (title if title else cmd, cmd)
    subprocess.Popen(cmd, shell=True, stdin=None, stdout=None, stderr=None, creationflags=0x00000008, env=env)  # DETACHED_PROCESS


class WPRCollector(Collector):
    def __init__(self, args):
        Collector.__init__(self, args)
        self.wpr = self.detect()
        self.started = False
        if self.args.cuts:
            self.file = os.path.join(args.output, "wpa-%s.etl" % (self.args.cuts[0] if self.args.cuts else '0'))
        else:
            self.file = os.path.join(args.output, "wpa.etl")

    @classmethod
    def detect(cls, statics={}):
        if 'res' in statics:
            return statics['res']
        wprs = cls.detect_instances('wpr')
        res = []
        for wpr in wprs:
            proc = subprocess.Popen('"%s" /?' % wpr, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (out, err) = proc.communicate()
            out = out.decode()
            if err:
                return None
            for line in out.split('\n'):
                pos = line.find('Version')
                if -1 != pos:
                    version = line[pos + len('Version '):].strip()
                    if int(version.split('.')[0]) >= 10:
                        res.append((wpr, version.split()[0]))
                    break
        if not res:
            return None
        statics['res'] = sorted(res, key=lambda __ver: [int(item) for item in __ver[1].split('.')], reverse=True)[0][0]
        return statics['res']

    @staticmethod
    def get_options():
        wpr = WPRCollector.detect()
        if not wpr:
            return
        proc = subprocess.Popen('"%s" -profiles' % wpr, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (out, err) = proc.communicate()
        if err:
            return
        for line in out.split('\n'):
            if not line.startswith('\t'):
                continue
            parts = line.strip().split()
            yield parts[0], parts[0] in ['DiskIO', 'FileIO', 'GPU', 'GeneralProfile', 'Handle', 'Heap', 'Network', 'Power', 'Video', 'VirtualAllocation']

    def start(self):
        if not self.wpr:
            print("Failed to start without WPR...")
            return
        if self.is_recording():
            self.cancel()
        profile = os.path.normpath(os.path.join(self.args.bindir, '..', 'ETW', 'IntelSEAPI.wprp'))
        profiles = ['-start %s' % option for option, _ in WPRCollector.get_options() if is_domain_enabled('wpa.' + option)]
        cmd = '"%s" -start "%s" %s %s' % (self.wpr, profile, ' '.join(profiles), ('' if self.args.ring else '-filemode'))
        (out, err) = self.execute(cmd)
        if err:
            return
        self.started = True
        return self

    def cancel(self):
        return self.execute('"%s" -cancel' % self.wpr)

    @classmethod
    def is_recording(cls, statics={}):
        if not statics:
            statics['wpr'] = cls.detect()
            statics['xperf'] = os.path.normpath(os.path.join(os.path.dirname(statics['wpr']), 'xperf.exe'))
        if os.path.exists(statics['xperf']):
            (out, err) = cls.execute('"%s" -Loggers | find "WPR_"' % statics['xperf'])
            return any('WPR_' in line for line in out.split('\n'))
        else:
            (out, err) = cls.execute('"%s" -status' % statics['wpr'])
            return err or not any('WPR is not recording' in line for line in out.split('\n'))

    def stop(self, wait=True):
        if not self.started:
            return []

        self.log("Stop wait=%s" % str(wait))
        if not wait:
            cmd = 'start "WPR stop" /MIN /LOW "%s" "%s" wpa "%s" "%s"' % (sys.executable, os.path.realpath(__file__), self.file, self.args.output)
            self.log(cmd)
            subprocess.Popen(cmd, shell=False, stdin=None, stdout=None, stderr=None, creationflags=0x00000008, env=sea.prepare_environ(self.args))  # DETACHED_PROCESS
            while self.is_recording():
                self.log("is_recording")
                time.sleep(1)
            return [self.file]
        else:
            env = sea.prepare_environ(self.args)
            self.stop_wpr(self.wpr, self.file, self.args.output, env)
            return [self.file]

    @classmethod
    def stop_wpr(cls, wpr, file, output, env=None):
        (out, err) = cls.execute('"%s" -stop "%s"' % (wpr, file), env=env)
        if err:
            return []
        assert(file in out)

    @classmethod
    def launch(cls, args):
        cls.stop_wpr(cls.detect(), args[0], args[1])


class ETWTrace(Collector):
    def __init__(self, args):
        Collector.__init__(self, args)
        wpr = WPRCollector.detect()
        self.xperf = os.path.normpath(os.path.join(os.path.dirname(wpr), 'xperf')) if wpr else None
        if not self.xperf or not os.path.exists(self.xperf):
            variants = self.detect_instances('xperf')
            if variants:
                self.xperf = variants[0]  # TODO: select by higher version
            else:
                self.xperf = None
        self.files = []
        self.start()

    def start(self):
        self.stop()
        cmd = None

        if self.args.cuts:
            self.files.append('%s\\etw-%s.etl' % (self.args.output, (self.args.cuts[0] if self.args.cuts else '0')))
            self.files.append('%s\\kernel-%s.etl' % (self.args.output, (self.args.cuts[0] if self.args.cuts else '0')))
        else:
            self.files.append('%s\\etw.etl' % self.args.output)
            self.files.append('%s\\kernel.etl' % self.args.output)

        logman_pf = os.path.join(tempfile.gettempdir(), 'gpa_logman.pf')
        count = 0
        with open(logman_pf, 'w') as file:
            if is_domain_enabled('Microsoft-Windows-DxgKrnl'):
                file.write('"Microsoft-Windows-DxgKrnl" (Base,GPUScheduler,Profiler,Resource,References,0x4000000000000001)\n')
                count += 1
            if is_domain_enabled('Microsoft-Windows-Dwm-Core'):
                file.write('"Microsoft-Windows-Dwm-Core" (DetailedFrameInformation)\n')
                count += 1
            if is_domain_enabled('Microsoft-Windows-DXGI'):
                file.write('"Microsoft-Windows-DXGI" (Events)\n')
                count += 1
            if is_domain_enabled('SteamVR'):
                file.write('"{8C8F13B1-60EB-4B6A-A433-DE86104115AC}"\n')
                count += 1
            if is_domain_enabled('OculusVR'):
                file.write('"{553787FC-D3D7-4F5E-ACB2-1597C7209B3C}"\n')
                count += 1
            if is_domain_enabled('Intel_Graphics_D3D10'):
                file.write('"{AD367E62-97EF-4B20-8235-E8AB49DB0C23}"\n')
                count += 1

        if count:
            cmd = 'logman start GPA_SEA -ct perf -bs 1024 -nb 120 480'
            cmd += ' -pf "%s" -o "%s" %s -ets' % (logman_pf, self.files[0], (('-max %d -f bincirc' % (self.args.ring * 15)) if self.args.ring else ''))
        else:
            del self.files[0]

        if cmd:
            (out, err) = self.execute(cmd)
            if err:
                return None

        if self.xperf:
            time_multiplier = 0
            kernel_logger = []  # logman query providers "Windows Kernel Trace"
            complimentary = ''
            if is_domain_enabled('Kernel::ContextSwitches'):
                time_multiplier += 10
                kernel_logger += ['PROC_THREAD', 'CSWITCH']
            if is_domain_enabled('Kernel::Stacks', self.args.stacks):
                time_multiplier += 20
                kernel_logger += ['LOADER', 'PROFILE']
                complimentary += ' -stackwalk PROFILE+CSWITCH -SetProfInt 1000000'
            if is_domain_enabled('Kernel::IO'):
                time_multiplier += 5
                kernel_logger += ['FILE_IO', 'FILE_IO_INIT', 'DISK_IO', 'DISK_IO_INIT', 'FILENAME', 'OPTICAL_IO', 'OPTICAL_IO_INIT']
            if is_domain_enabled('Kernel::Network', False):
                time_multiplier += 5
                kernel_logger += ['NETWORKTRACE']
            if is_domain_enabled('Kernel::Memory', False):
                time_multiplier += 5
                kernel_logger += ['VIRT_ALLOC', 'MEMINFO', 'VAMAP', 'POOL', 'MEMINFO_WS']  # 'FOOTPRINT', 'MEMORY'
            if is_domain_enabled('Kernel::PageFaults', False):
                time_multiplier += 5
                kernel_logger += ['ALL_FAULTS', 'HARD_FAULTS']
            if kernel_logger:
                cmd = '"%s" -on %s %s -f "%s" -ClockType PerfCounter -BufferSize 1024 -MinBuffers 120 -MaxBuffers 480' % (self.xperf, '+'.join(kernel_logger), complimentary, self.files[-1])
                if self.args.ring:
                    cmd += ' -MaxFile %d -FileMode Circular' % (self.args.ring * time_multiplier)  # turning seconds into megabytes...
                (out, err) = self.execute(cmd)
                if err or 'Error:' in out:
                    del self.files[-1]
                    return self
            else:
                del self.files[-1]
        else:
            time_multiplier = 0
            kernel_logger = []  # logman query providers "Windows Kernel Trace"
            if is_domain_enabled('Kernel::ContextSwitches'):
                time_multiplier += 10
                kernel_logger += ['process', 'thread', 'cswitch']
            if is_domain_enabled('Kernel::Stacks', self.args.stacks):
                time_multiplier += 10
                kernel_logger += ['img', 'profile']
            if is_domain_enabled('Kernel::IO'):
                time_multiplier += 5
                kernel_logger += ['fileio', 'disk']
            if is_domain_enabled('Kernel::Network', False):
                time_multiplier += 5
                kernel_logger += ['net']
            if is_domain_enabled('Kernel::Memory', False):
                time_multiplier += 5
                kernel_logger += ['virtalloc']
            if is_domain_enabled('Kernel::PageFaults', False):
                time_multiplier += 5
                kernel_logger += ['pf', 'hf']
            if kernel_logger:
                cmd = 'logman start "NT Kernel Logger" -p "Windows Kernel Trace" (%s) -ct perf -bs 1024 -nb 120 480' % ','.join(kernel_logger)
                cmd += ' -o "%s" %s -ets' % (self.files[-1], (('-max %d -f bincirc' % (self.args.ring * time_multiplier)) if self.args.ring else ''))
                (out, err) = self.execute(cmd)
                if err or 'Error:' in out:
                    del self.files[-1]
                    return self
            else:
                del self.files[-1]

        self.files.append('%s/etw_profilers.logman' % self.args.output)
        cmd = 'cmd /c logman query providers ^> "%s"' % self.files[-1]
        async_exec(cmd, 'Collecting ETW providers')

        return self

    def stop(self, wait=True):  # TODO: stop without waits
        if self.xperf:
            proc = subprocess.Popen('xperf -stop', shell=False)
            if wait:
                proc.wait()
        else:
            proc = subprocess.Popen('logman stop "NT Kernel Logger" -ets', shell=False)
            if wait:
                proc.wait()
        proc = subprocess.Popen('logman stop "GPA_SEA" -ets', shell=False)
        if wait:
            proc.wait()

        return self.files

COLLECTOR_DESCRIPTORS = [
    {
        'available': sys.platform == 'win32' and WPRCollector.detect(),
        'collector': WPRCollector,
        'format': 'wpa'
    },
    {
        'available': sys.platform == 'win32',
        'collector': ETWTrace,
        'format': 'etw'
    }
]

if __name__ == "__main__":
    with open(os.path.join(tempfile.gettempdir(), datetime.now().strftime('sea_%H_%M_%S__%d_%m_%Y.log')), 'a') as log:
        log.write(str(sys.argv) + '\n')
        try:
            name = sys.argv[1]
            for desc in COLLECTOR_DESCRIPTORS:
                if desc['format'] == name:
                    cls = desc['collector']
                    cls.set_output(log)
                    cls.launch(sys.argv[2:])
                    break
        except:
            log.write(traceback.format_exc())
