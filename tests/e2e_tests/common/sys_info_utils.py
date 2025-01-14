# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=logging-fstring-interpolation,fixme

"""
Functions for getting system information.
"""

import os
import sys
import contextlib
import logging
import multiprocessing
import pathlib
import platform
import re
import subprocess
from enum import Enum

import cpuinfo
import distro
import yaml

if sys.hexversion < 0x3060000:
    raise Exception('Python version must be >= 3.6')


with open(os.path.join(os.path.dirname(__file__), 'platforms.yml'), 'r') as f:
    platforms = yaml.safe_load(f)


# Host name

def get_host_name():
    """ Get hostname """
    return platform.node()


# OS info

class UnsupportedOsError(Exception):
    """ Exception for unsupported OS type
    Originally taken from https://gitlab-icv.toolbox.iotg.sclab.intel.com/inference-engine/infrastructure/blob/master/common/system_info.py  # pylint: disable=line-too-long
    All changes shall be done in the original location first (inference-engine/infrastructure repo)
    """
    def __init__(self, *args, **kwargs):
        error_message = f'OS type "{platform.system()}" is not currently supported'
        if args or kwargs:
            super().__init__(*args, **kwargs)
        else:
            super().__init__(error_message)


class OsType(Enum):
    """ Container for supported os types
    Originally taken from https://gitlab-icv.toolbox.iotg.sclab.intel.com/inference-engine/infrastructure/blob/master/common/system_info.py  # pylint: disable=line-too-long
    All changes shall be done in the original location first (inference-engine/infrastructure repo)
    """
    WINDOWS = 'Windows'
    LINUX = 'Linux'
    DARWIN = 'Darwin'


def get_os_type():
    """ Return OS type """
    return platform.system()


def os_type_is_windows():
    """ Returns True if OS type is Windows. Otherwise returns False"""
    return platform.system() == OsType.WINDOWS.value


def os_type_is_linux():
    """ Returns True if OS type is Linux. Otherwise returns False"""
    return platform.system() == OsType.LINUX.value


def os_type_is_darwin():
    """ Returns True if OS type is Darwin. Otherwise returns False"""
    return platform.system() == OsType.DARWIN.value


def get_os_name():
    """ Check OS type and return OS name
    Originally taken from https://gitlab-icv.toolbox.iotg.sclab.intel.com/inference-engine/infrastructure/blob/master/common/system_info.py  # pylint: disable=line-too-long
    All changes shall be done in the original location first (inference-engine/infrastructure repo)

    :return: OS name
    :rtype: String | Exception if it is not supported
    """
    if os_type_is_linux():
        return distro.id().lower()
    if os_type_is_windows() or os_type_is_darwin():
        return platform.system().lower()
    raise UnsupportedOsError()


def get_os_version():
    """ Check OS version and return it
    Originally taken from https://gitlab-icv.toolbox.iotg.sclab.intel.com/inference-engine/infrastructure/blob/master/common/system_info.py  # pylint: disable=line-too-long
    All changes shall be done in the original location first (inference-engine/infrastructure repo)

    :return: OS version
    :rtype: tuple of strings | Exception if it is not supported
    """
    if os_type_is_linux():
        return distro.major_version(), distro.minor_version()
    if os_type_is_windows():
        return str(sys.getwindowsversion().major), str(sys.getwindowsversion().minor)
    if os_type_is_darwin():
        return tuple(platform.mac_ver()[0].split(".")[:2])
    raise UnsupportedOsError()


def get_os():
    """ Get OS """
    if os_type_is_linux():
        # distro.linux_distribution() => ('Ubuntu', '16.04', 'xenial')
        _os = ''.join(distro.linux_distribution()[:2])
    elif os_type_is_windows():
        # platform.win32_ver() => ('10', '10.0.17763', 'SP0', 'Multiprocessor Free')
        _os = 'Windows{}'.format(str(platform.win32_ver()[0]))
    elif os_type_is_darwin():
        # platform.mac_ver() => ('10.5.8', ('', '', ''), 'i386')
        _os = 'MacOS{}'.format(str(platform.mac_ver()[0]))
    else:
        raise UnsupportedOsError()
    return _os


# Platform info

def get_platform(env):
    """ Get platform """
    platform_info = {'alias': '', 'info': ''}
    alias = env.get('platform')
    if alias:
        platform_info.update({'alias': alias})
        platform_info.update({'info': get_platform_info(alias)})
    return platform_info


def get_platform_info(platform_alias):
    """  Get platform info """
    platform_info = {
        # CPU/GPU
        'apl': 'ApolloLake',
        'cfl': 'CoffeeLake',
        'clx': 'CascadeLake',
        'clx-ap': 'CascadeLake',
        'cslx': 'CascadeLake',
        'cpx': 'CooperLake',
        'halo': 'Skylake',
        'iclu': 'IceLake',
        'skl': 'Skylake',
        'sklx': 'Skylake',
        'skx-avx512': 'Skylake',
        'skl-e': 'Skylake',
        'tglu': 'TigerLake',
        'whl': 'WhiskyLake',
        'epyc': 'AMD EPYC 7601',

        # Myriad/HDDL
        'myriad': 'Myriad 2 Stick',
        'myriadx': 'Myriad X Stick',
        'myriadx-evm': 'Myriad X Board',
        'myriadx-pc': 'Myriad X Board 2085',
        'hddl': 'HDDL-R',

        # FPGA
        'hddlf': 'PyramidLake',
        'hddlf_SG2': 'PyramidLake SG2',

        # VCAA
        'vcaa': 'Hiker Hights PCI-e board CPU/GPU/HDDL',
    }

    return platform_info.get(platform_alias, '')


# CPU info

def get_cpu_name():
    """  Get CPU name """
    # cpuinfo.get_cpu_info().get('brand', '') => Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz
    return cpuinfo.get_cpu_info().get('brand', '')


def get_device_description(platform_selector, device):
    """Get device detailed info
    :param platform_selector: platform identifier (Jenkins label)
    :param device: device
    :return: string with detailed info
    """
    return platforms.get(platform_selector, {}).get(device, {}).get('description', '')


def get_cpu_count():
    """
    Originally taken from https://gitlab-icv.toolbox.iotg.sclab.intel.com/inference-engine/infrastructure/blob/master/common/system_info.py#L138  # pylint: disable=line-too-long
    All changes shall be done in the original location first (inference-engine/infrastructure repo).

    Custom `cpu_count` calculates the number of CPUs as minimum of:
     * System CPUs count by ``multiprocessing.cpu_count()``
     * CPU affinity settings of the current process
     * CFS scheduler CPU bandwidth limit

    :return: The number of CPUs available to be used by the current process, it is >= 1
    :rtype: int
    """

    cpu_counts = []
    cpu_counts.append(multiprocessing.cpu_count())

    # Number of available CPUs given affinity settings
    # More info: http://man7.org/linux/man-pages/man2/sched_setaffinity.2.html
    if hasattr(os, "sched_getaffinity"):
        with contextlib.suppress(NotImplementedError):
            cpu_counts.append(len(os.sched_getaffinity(0))) # pylint: disable=no-member

    if os_type_is_linux():
        # CFS scheduler CPU bandwidth limit
        # More info: https://www.kernel.org/doc/Documentation/scheduler/sched-bwc.txt
        with contextlib.suppress(OSError, ValueError):
            # CPU clock time allocated within a period (in microseconds)
            cfs_quota = int(pathlib.Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").
                            read_text(errors="strict"))
            # Real world time length of a period (in microseconds)
            cfs_period = int(pathlib.Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").
                             read_text(errors="strict"))
            if cfs_quota > 0 and cfs_period > 0:
                cpu_counts.append(int(cfs_quota / cfs_period))
    elif os_type_is_windows():
        # Workaround for Python bug with some pre-production CPU
        try:
            env_cpu_count = os.getenv('NUMBER_OF_PROCESSORS')
            if env_cpu_count and env_cpu_count != cpu_counts[0]:
                proc = subprocess.run(
                    'powershell "$cs=Get-WmiObject -class Win32_ComputerSystem; '\
                    '$cs.numberoflogicalprocessors"',
                    stdout=subprocess.PIPE, encoding='utf-8', shell=True, timeout=5, check=True)
                cpu_counts[0] = int(proc.stdout)
        except Exception:  # pylint: disable=broad-except
            pass

    return max(min(cpu_counts), 1)


class CoreInfo:
    """Wrapper for getting cpu info"""

    def __init__(self):
        self._log = logging.getLogger("sys_info.coreinfo")

    def _run_tool(self, cmd):
        """ Run tool, return stdout or None if running is not successful. """

        # pylint: disable=subprocess-run-check

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        if result.returncode:
            self._log.warning(f"{cmd} running failed")
            return None
        return result.stdout.decode("utf-8")

    def _get_lscpu_info(self, cpu_property_name, regex):
        """ Linux specific method. Run lscpu tool and parse its output.
        Return extracted CPU property value in case of successful run, None otherwise.

        Refer https://man7.org/linux/man-pages/man1/lscpu.1.html for tool manual.
        """
        cpu_prop_count = None

        stdout = self._run_tool([f"lscpu | grep '{cpu_property_name}'"])
        if stdout:
            match = re.search(regex, stdout.rstrip())
            if match:
                cpu_prop_count = int(match.group(1))

        return cpu_prop_count

    def _get_coreinfo_info(self, cpu_property_opt, regex):
        """ Windows specific method. Run coreinfo tool and parse its output.
        Return extracted CPU property value in case of successful run, None otherwise.

        Refer https://docs.microsoft.com/en-us/sysinternals/downloads/coreinfo for tool manual.
        """
        cpu_prop_count = 0

        stdout = self._run_tool(["Coreinfo.exe", cpu_property_opt])
        if stdout:
            for line in stdout.split("\n"):
                if re.search(regex, line.rstrip()):
                    cpu_prop_count += 1
        return cpu_prop_count or None

    def get_cpu_cores(self):
        """ Return the number of CPU cores """
        if os_type_is_linux():
            return self._get_lscpu_info(cpu_property_name="Core(s) per socket", regex=r"(\d+)$")

        if os_type_is_windows():
            return self._get_coreinfo_info(cpu_property_opt="-c", regex=r"Physical Processor (\d+)")

        self._log.warning(f"OS type '{get_os_type()}' is not currently supported")
        return None

    def get_cpu_sockets(self):
        """ Return the number of CPU sockets """
        if os_type_is_linux():
            return self._get_lscpu_info(cpu_property_name="Socket(s)", regex=r"(\d+)$")

        if os_type_is_windows():
            return self._get_coreinfo_info(cpu_property_opt="-s", regex=r"Socket (\d+)")

        self._log.warning(f"OS type '{get_os_type()}' is not currently supported")
        return None

    def get_cpu_numa_nodes(self):
        """ Return the number of CPU numa nodes """
        if os_type_is_linux():
            return self._get_lscpu_info(cpu_property_name="NUMA node(s)", regex=r"(\d+)$")

        if os_type_is_windows():
            return self._get_coreinfo_info(cpu_property_opt="-n", regex=r"NUMA Node (\d+)")

        self._log.warning(f"OS type '{get_os_type()}' is not currently supported")
        return None


def get_cpu_max_instructions_set():
    """  Get CPU max instructions set """
    look_for = ['avx512vnni', 'avx512_vnni', 'avx512', 'avx2', 'sse4_2']
    for item in look_for:
        for instruction in cpuinfo.get_cpu_info().get('flags', []):
            if item in instruction:
                return item

    return ''


def get_default_bf16_settings():
    """ Get default BF16 settings
    We suppose that BF16 is enabled by default if platform supports BF16 (in other words if
    avx512_bf16 is in instructions set)
    :return: boolean, True if BF16 is enabled by default (e.g. CPX), False - otherwise
    """
    for instruction in cpuinfo.get_cpu_info().get('flags', []):
        if 'avx512_bf16' in instruction:
            return True
    return False


def get_sys_info():
    """Return dictionary with system information"""
    return {
        "hostname": get_host_name(),
        "os": get_os(),
        "os_name": get_os_name(),
        "os_version": get_os_version(),
        "cpu_info": get_cpu_name(),
        "cpu_count": get_cpu_count(),
        "cpu_cores": CoreInfo().get_cpu_cores(),
        "cpu_sockets": CoreInfo().get_cpu_sockets(),
        "cpu_numa_nodes": CoreInfo().get_cpu_numa_nodes(),
        "cpu_max_instructions_set": get_cpu_max_instructions_set(),
        "bf16_support": get_default_bf16_settings(),
    }

# Jenkins-related utils


def is_running_under_jenkins():
    """  Checks if running under Jenkins """
    return 'JENKINS_URL' in os.environ


def get_jenkins_url():
    """  Get Jenkins URL of the current job"""
    return os.environ.get('BUILD_URL', '').rstrip('/')


def get_parent_jenkins_url():
    """  Get Jenkins URL of the parent job"""
    return os.environ.get('PARENT_BUILD_URL', '').rstrip('/')


def get_mc_entrypoint_url():
    """Get Jenkins URL of the MC entrypoint job"""
    return os.environ.get('MC_ROOT_JOB_URL', '').rstrip('/')


def get_jenkins_info():
    """Return dictionary with Jenkins information"""
    return {
        "jenkins_run": is_running_under_jenkins(),
        "jenkins_url": get_jenkins_url(),
        "parent_jenkins_url": get_parent_jenkins_url(),
    }


def get_mc_jenkins_info():
    """Return dictionary with MC specific Jenkins information"""
    return {
        "jenkins_run": is_running_under_jenkins(),
        "mc_task_url": get_jenkins_url(),
        "mc_entrypoint_url": get_mc_entrypoint_url(),
    }


def path_to_url(artifact_path, test_folder):
    # TODO:  USED BY OLD ACCURACY TESTS - TO REMOVE LOOKING FORWARD
    """  Converts Jenkins artifact path to URL """
    if is_running_under_jenkins():
        work_dir = os.path.join(os.environ["WORKSPACE"], os.path.join('tests', test_folder))
        return os.path.join(
            os.environ["BUILD_URL"], 'artifact', 'tests', test_folder,
            os.path.relpath(artifact_path, work_dir)).replace(
            '\\', '/')
    return None


def path_to_url_new(log_name, log_path=None):
    """  Converts Jenkins artifact path to URL - new infrastructure"""
    if is_running_under_jenkins():
        log_path = os.environ["LOG_PATH"] if not log_path else log_path
        return os.path.join(
            os.environ["BUILD_URL"],
            'artifact/b/logs',
            os.path.relpath(log_name, log_path)
        ).replace('\\', '/')
    return None
