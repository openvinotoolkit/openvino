import math
import multiprocessing
import os

from conan.internal.util.files import load


def cpu_count():
    try:
        try:
            # This is necessary to deduce docker cpu_count
            cfs_quota_us = cfs_period_us = 0
            # cgroup2
            if os.path.exists("/sys/fs/cgroup/cgroup.controllers"):
                cpu_max = load("/sys/fs/cgroup/cpu.max").split()
                if cpu_max[0] != "max":
                    if len(cpu_max) == 1:
                        cfs_quota_us, cfs_period_us = int(cpu_max[0]), 100_000
                    else:
                        cfs_quota_us, cfs_period_us = map(int, cpu_max)
            else:  # cgroup1
                cfs_quota_us = int(load("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"))
                cfs_period_us = int(load("/sys/fs/cgroup/cpu/cpu.cfs_period_us"))
            if cfs_quota_us > 0 and cfs_period_us > 0:
                return int(math.ceil(cfs_quota_us / cfs_period_us))
        except (EnvironmentError, TypeError):
            pass
        return multiprocessing.cpu_count()
    except NotImplementedError:
        # print("multiprocessing.cpu_count() not implemented. Defaulting to 1 cpu")
        return 1  # Safe guess
