from __future__ import print_function
import os
import subprocess
from sea_runtool import Collector, subst_env_vars


class Android(Collector):
    def __init__(self, args):
        Collector.__init__(self, args)
        self.adb = self.detect()
        self.file = None

    def is_root(self, statics={}):
        if statics:
            return statics['root']
        out, err = self.execute(self.adb + ' shell id')
        if err:
            return False

        statics['root'] = 'root' in out
        return statics['root']

    @classmethod
    def detect(cls):
        adbs = cls.detect_instances('adb')
        systraces = []
        for adb in adbs:
            out, err = cls.execute('"%s" version' % adb)
            if err:
                continue
            parts = out.split()
            version = parts[parts.index('version') + 1]
            systraces.append((version, adb))
        if systraces:
            sorted_by_version = sorted(systraces, key=lambda ver__: [int(item) for item in ver__[0].split('.')], reverse=True)
            return '"%s"' % sorted_by_version[0][1]
        else:
            return None

    def echo(self, what, where):
        out, err = self.execute(self.adb + ' shell "echo %s > %s"' % (what, where))
        if err:
            return out, err
        if 'no such file or directory' in str(out).lower():
            return out, out
        if 'Permission denied' in out:
            proc = subprocess.Popen(self.adb + ' shell su', shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if proc.pid:
                proc.stdin.write('echo %s > %s\n' % (what, where))
                proc.stdin.write('exit\n')
                proc.stdin.close()
                out_lines = list(proc.stdout)
                out = '\n'.join(out_lines[1:-1])
                err = '\n'.join(list(proc.stderr))
        return out, err

    def start(self):
        if not self.adb:
            print("Failed to run without adb...")
            return
        self.file = os.path.join(subst_env_vars(self.args.input), 'atrace-%s.ftrace' % (self.args.cuts[0] if self.args.cuts else '0'))
        self.echo('0', '/sys/kernel/debug/tracing/tracing_on')
        self.echo('', '/sys/kernel/debug/tracing/trace')
        self.echo('1', '/sys/kernel/debug/tracing/events/i915/enable')
        self.echo('1', '/sys/kernel/debug/tracing/events/kgsl/enable')

        if self.is_root():
            out, err = self.execute(self.adb + ' shell atrace --list_categories')
            if err:
                return False
            features = []
            for line in out.split('\n'):
                parts = line.split()
                if not parts:
                    continue
                features.append(parts[0])

            cmd = self.adb + ' shell atrace'
            if self.args.ring:
                cmd += ' -b %d -c' % (self.args.ring * 1000)
            cmd += ' --async_start %s' % ' '.join(features)
            self.execute_detached(cmd)
        else:  # non roots sometimes have broken atrace, so we won't use it
            out, err = self.execute(self.adb + ' shell setprop debug.atrace.tags.enableflags 0xFFFFFFFF')
            if err:
                return None
            for event in self.enum_switchable_events():
                self.echo('1', event)
            if self.args.ring:
                self.echo("%d" % (self.args.ring * 1024), '/sys/kernel/debug/tracing/buffer_size_kb')
            out, err = self.echo('1', '/sys/kernel/debug/tracing/tracing_on')
            if err:
                return None
        return self

    def enum_switchable_events(self):
        out, err = self.execute(self.adb + ' shell ls -l -R /sys/kernel/debug/tracing/events')
        if err:
            return
        root_dir = None
        for line in out.split('\n'):
            line = line.strip()
            if not line:
                root_dir = None
            else:
                if root_dir:
                    if 'shell' in line and line.endswith('enable'):
                        yield root_dir + '/enable'
                else:
                    root_dir = line.strip(':')

    def stop(self, wait=True):
        out, err = self.echo('0', '/sys/kernel/debug/tracing/tracing_on')
        if err:
            return []
        self.execute(self.adb + ' shell setprop debug.atrace.tags.enableflags 0')
        out, err = self.execute('%s pull /sys/kernel/debug/tracing/trace "%s"' % (self.adb, self.file))
        if err or 'error' in out:
            with open(self.file, 'w') as file:
                self.execute(self.adb + ' shell cat /sys/kernel/debug/tracing/trace', stdout=file)
        return [self.file]

COLLECTOR_DESCRIPTORS = [{
    'format': 'android',
    'available': True,
    'collector': Android
}]
