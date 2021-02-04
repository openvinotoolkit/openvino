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
import os
import sys
import glob
import shutil
import traceback
import subprocess

# http://www.brendangregg.com/perf.html
# sudo perf probe --funcs

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))
import sea

from sea_runtool import Collector, Progress, format_bytes


def time_sync():
    sea.ITT('lin').time_sync()

supported_events = [
    "binder_locked",
    "binder_unlock",
    "binder_lock",
    "binder_transaction",
    "binder_transaction_received",
    "memory_bus_usage",
    "clock_set_rate",
    "cpufreq_interactive_up",
    "cpufreq_interactive_down",
    "cpufreq_interactive_already",
    "cpufreq_interactive_notyet",
    "cpufreq_interactive_setspeed",
    "cpufreq_interactive_target",
    "cpufreq_interactive_boost",
    "cpufreq_interactive_unboost",
    "f2fs_write_begin",
    "f2fs_write_end",
    "f2fs_sync_file_enter",
    "f2fs_sync_file_exit",
    "ext4_sync_file_enter",
    "ext4_sync_file_exit",
    "ext4_da_write_begin",
    "ext4_da_write_end",
    "block_rq_issue",
    "block_rq_complete",
    "drm_vblank_event",
    "exynos_busfreq_target_int",
    "exynos_busfreq_target_mif",
    "exynos_page_flip_state",
    "i915_gem_object_create",
    "i915_gem_object_bind",
    "i915_gem_object_unbind",
    "i915_gem_object_change_domain",
    "i915_gem_object_pread",
    "i915_gem_object_pwrite",
    "i915_gem_object_fault",
    "i915_gem_object_clflush",
    "i915_gem_object_destroy",
    "i915_gem_ring_dispatch",
    "i915_gem_ring_flush",
    "i915_gem_request",
    "i915_gem_request_add",
    "i915_gem_request_complete",
    "i915_gem_request_retire",
    "i915_gem_request_wait_begin",
    "i915_gem_request_wait_end",
    "i915_gem_ring_wait_begin",
    "i915_gem_ring_wait_end",
    "i915_mvp_read_req",
    "i915_reg_rw",
    "i915_flip_request",
    "i915_flip_complete",
    "intel_gpu_freq_change",
    "irq_handler_entry",
    "irq_handler_exit",
    "softirq_raise",
    "softirq_entry",
    "softirq_exit",
    "ipi_entry",
    "ipi_exit",
    "graph_ent",
    "graph_ret",
    "mali_dvfs_event",
    "mali_dvfs_set_clock",
    "mali_dvfs_set_voltage",
    "tracing_mark_write:mali_driver",
    "mm_vmscan_kswapd_wake",
    "mm_vmscan_kswapd_sleep",
    "mm_vmscan_direct_reclaim_begin",
    "mm_vmscan_direct_reclaim_end",
    "workqueue_execute_start",
    "workqueue_execute_end",
    "power_start",
    "power_frequency",
    "cpu_frequency",
    "cpu_idle",
    "regulator_enable",
    "regulator_enable_delay",
    "regulator_enable_complete",
    "regulator_disable",
    "regulator_disable_complete",
    "regulator_set_voltage",
    "regulator_set_voltage_complete",
    "sched_switch",
    "sched_wakeup",
    "workqueue_execute_start",
    "workqueue_execute_end",
    "workqueue_queue_work",
    "workqueue_activate_work",
]


class FTrace(Collector):
    def __init__(self, args, remote=False):
        Collector.__init__(self, args)
        self.remote = remote
        self.event_list = []
        self.file = None
        self.perf_file = None
        self.perf_proc = None
        for event in supported_events:
            for path in glob.glob('/sys/kernel/debug/tracing/events/*/%s/enable' % event):
                self.event_list.append(path)

    def echo(self, what, where):
        self.log("echo %s > %s" % (what, where))
        try:
            if self.remote:
                self.remote.execute('echo %s > %s' % (what, where))
            else:
                with open(where, "w") as file:
                    file.write(what)
        except:
            self.log("Failed: " + traceback.format_exc())
            return False
        return True

    def start(self):
        if not self.echo("nop", "/sys/kernel/debug/tracing/current_tracer"):
            self.log("Warning: failed to access ftrace subsystem")
            return
        self.file = os.path.join(self.args.output, 'nop-%s.ftrace' % (self.args.cuts[0] if self.args.cuts else '0'))

        self.echo("0", "/sys/kernel/debug/tracing/tracing_on")
        self.echo("nop", "/sys/kernel/debug/tracing/current_tracer")  # google chrome understands this format
        self.echo("", "/sys/kernel/debug/tracing/set_event")  # disabling all events
        self.echo("", "/sys/kernel/debug/tracing/trace")  # cleansing ring buffer (we need it's header only)
        if self.args.ring:
            self.echo("%d" % (self.args.ring * 1024), "/sys/kernel/debug/tracing/buffer_size_kb")

        # best is to write sync markers here
        self.echo("1", "/sys/kernel/debug/tracing/tracing_on")  # activate tracing
        time_sync()
        self.echo("0", "/sys/kernel/debug/tracing/tracing_on")  # deactivate tracing
        # saving first part of synchronization as it will be wiped out in ring
        self.copy_from_target("/sys/kernel/debug/tracing/trace", self.file)
        self.echo("", "/sys/kernel/debug/tracing/trace")  # cleansing ring buffer again

        for event in self.event_list:  # enabling only supported
            self.echo("1", event)

        for path in glob.glob('/sys/kernel/debug/dri/*/i915_mvp_enable'):  # special case for Intel GPU events
            self.echo("1", path)
        self.echo("1", "/sys/kernel/debug/tracing/tracing_on")
        if self.args.stacks and self.args.target:
            self.perf_file = os.path.join(self.args.output, 'perf-%s.data' % (self.args.cuts[0] if self.args.cuts else '0'))
            if os.path.exists(self.perf_file):
                os.remove(self.perf_file)
            cmd = 'perf record -a -g -o "%s" --pid=%s' % (self.perf_file, self.args.target)
            self.log(cmd)
            self.perf_proc = subprocess.Popen(cmd, shell=True, stdout=self.get_output(), stderr=self.get_output(), preexec_fn=os.setpgrp)

    def copy_from_target(self, what, where):
        self.log("copy %s > %s" % (what, where))
        if self.remote:
            self.remote.copy('%s:%s' % (self.args.ssh, what), where)
        else:
            shutil.copy(what, where)

    def stop(self, wait=True):
        results = []
        if self.perf_proc:
            self.perf_proc.wait()
            if os.path.exists(self.perf_file):
                results.append(self.perf_file + '.perf')
                with open(results[-1], 'wb') as file:
                    self.execute('perf script -F comm,tid,pid,time,ip,sym,dso,symoff --show-kernel-path --demangle-kernel --full-source-path -i "%s"' % self.perf_file, stdout=file)
                os.remove(self.perf_file)

        if not self.file:
            return results
        time_sync()
        self.echo("0", "/sys/kernel/debug/tracing/tracing_on")
        for path in glob.glob('/sys/kernel/debug/dri/*/i915_mvp_enable'):  # special case for Intel GPU events
            self.echo("0", path)
        file_name = os.path.join(self.args.output, "tmp.ftrace")
        self.copy_from_target("/sys/kernel/debug/tracing/trace", file_name)
        self.log("append %s > %s" % (file_name, self.file))
        with open(file_name) as file_from, open(self.file, 'a') as file_to:
            shutil.copyfileobj(file_from, file_to)
        os.remove(file_name)
        results.append(self.file)
        self.execute('chmod -R a+rwX "%s"' % self.args.output)
        return results


COLLECTOR_DESCRIPTORS = [{
    'format': 'ftrace',
    'available': True,
    'collector': FTrace
}]
