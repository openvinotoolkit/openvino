# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import platform
import argparse
import glob
import os
import re
import itertools
import subprocess
import json
import time
import sys

from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict

try:
    import requests
except ImportError:
    requests = None


INTEL_FAMILIES = {
    (0x06, 0x5E): "Sky Lake",
    (0x06, 0x55): "Sky Lake",
    (0x06, 0x8E): "Kaby Lake",
    (0x06, 0x9E): "Kaby Lake",
    (0x06, 0xA5): "Comet Lake",
    (0x06, 0xA6): "Comet Lake",
    (0x06, 0x66): "Cannon Lake",
    (0x06, 0x6A): "Ice Lake",
    (0x06, 0x6C): "Ice Lake",
    (0x06, 0x7D): "Ice Lake",
    (0x06, 0x7E): "Ice Lake",
    (0x06, 0x9D): "Ice Lake NNPI",
    (0x06, 0xA7): "Rocket Lake",
    (0x06, 0x8C): "Tiger Lake",
    (0x06, 0x8D): "Tiger Lake",
    (0x06, 0x8F): "Sapphire Rapids",
    (0x06, 0xCF): "Emerald Rapids",
    (0x06, 0xAD): "Granite Rapids",
    (0x06, 0xAE): "Granite Rapids",
    (0x13, 0x01): "Diamond Rapids",
    (0x06, 0xD7): "Bartlett Lake",
    (0x06, 0x8A): "Lakefield",
    (0x06, 0x97): "Alder Lake",
    (0x06, 0x9A): "Alder Lake",
    (0x06, 0xB7): "Raptor Lake",
    (0x06, 0xBA): "Raptor Lake P",
    (0x06, 0xBF): "Raptor Lake",
    (0x06, 0xAC): "Meteor Lake",
    (0x06, 0xAA): "Meteor Lake",
    (0x06, 0xC5): "Arrow Lake",
    (0x06, 0xC6): "Arrow Lake",
    (0x06, 0xB5): "Arrow Lake U",
    (0x06, 0xBD): "Lunar Lake M",
    (0x06, 0xCC): "Panther Lake",
    (0x06, 0xD5): "Wildcat Lake",
    (0x12, 0x01): "Nova Lake",
    (0x12, 0x03): "Nova Lake"
}


def get_cpu_family():
    arch = platform.machine().lower()
    system = platform.system()
    if arch in ("arm64", "aarch64"):
        if system == "Darwin":
            return subprocess.check_output(
                ["sysctl", "machdep.cpu.brand_string"]).decode().strip()
        else:
            # not implemented
            return "Unknown arm64"
    elif arch in ("x86_64", "amd64"):
        if system == "Windows":
            cpuinfo = subprocess.check_output(
                ["powershell", "(Get-WmiObject -Class Win32_Processor).Caption"]).decode().strip()
            infomatch = re.match(r".* Family (\d+) Model (\d+) .*",
                cpuinfo.splitlines()[-1])
            if not infomatch:
                return "Unknown x86_64"
            try:
                family, model = map(int, infomatch.groups())
            except ValueError:
                return "Unknown x86_64"
        elif system == "Linux":
            with open("/proc/cpuinfo") as cpuinfofile:
                cpuinfo = cpuinfofile.read().strip().split("\n\n")[0]
            cpuinfo = dict([
                map(str.strip, line.split(":", 1))
                for line in cpuinfo.split("\n")
            ])
            try:
                family = int(cpuinfo.get("cpu family"))
                model = int(cpuinfo.get("model"))
            except ValueError:
                return "Unknown x86_64"
        elif system == "Darwin":
            family = subprocess.check_output(
                ["sysctl", "machdep.cpu.family"]).decode().strip()
            model = subprocess.check_output(
                ["sysctl", "machdep.cpu.model"]).decode().strip()
            try:
                family = int(family)
                model = int(model)
            except ValueError:
                return "Unknown x86_64"
        else:
            return "Unknown x86_64"
        return INTEL_FAMILIES.get((family, model), "Unknown x86_64")
    else:
        return "Unknown"


CPU_FAMILY = get_cpu_family()


def value_diff(value, reference):
    difference = value - reference
    diff_ratio = difference / reference
    return (value, reference, difference, diff_ratio)


def attempt(func, *args, **kwargs):
    tries = 3
    while tries:
        tries -= 1
        try:
            return func(*args, **kwargs)
        except Exception as ex:
            print(f"Error: {ex}; tries left: {tries}")
        time.sleep(5)
    print("No more attempts will be made")


@dataclass
class MemSample:
    system_size: int
    system_peak: int
    system_rss: int
    system_hwm: int
    threads: int
    gpu_local_used: int = -1
    gpu_local_total: int = -1
    gpu_nonlocal_used: int = -1
    gpu_nonlocal_total: int = -1

    def as_dict(self):
        def to_camel_case(s: str):
            first, *others = s.split("_")
            return "".join([first, *[x.capitalize() for x in others]])

        return {
            to_camel_case(k): v
            for k, v in asdict(self).items()
        }

    @classmethod
    def from_dict(cls, values: dict):
        class_fields = MemSample.__dataclass_fields__.keys()
        selected_values = {
            k: int(values[k])
            for k in class_fields
        }
        return MemSample(**selected_values)

    def __repr__(self):
        return "; ".join(f"{k} {v:>10}" for k, v in self.as_dict().items())


def run_test_executable_extract_result(command):
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    run_out = proc.stdout.decode()
    if run_out.startswith("TEST_RESULTS: "):
        results_json = run_out.splitlines()[0].removeprefix("TEST_RESULTS: ")
        results = json.loads(results_json)
        if "samples" in results:
            results["samples"] = {
                sname: MemSample.from_dict(sample)
                for sname, sample in results["samples"].items()
            }
        return results
    elif run_out.startswith("TEST_INFO: "):
        return json.loads(run_out.splitlines()[0].removeprefix("TEST_INFO: "))
    return {
        "error": "Test did not run correctly",
        "returncode": proc.returncode,
        "stdout": proc.stdout.decode(),
        "stderr": proc.stderr.decode()
    }


MODELID_RE = re.compile(r"\/?([^\/]+)\/(\w+)\/?[^\/]*\/((OV_)?(FP|INT)\d+(\/INT\d+)?[^\/]*)(\/\d+\/ov)?\/(.+).xml")


def modelid_assume_info(modelid):
    match = MODELID_RE.match(modelid.replace("\\", "/"))
    if not match:
        return None
    model, fw, prec, _n1, _n2, _n3, _n4, model2 = match.groups()
    if model == model2:
        modelname = model
    else:
        modelname = f"{model}/{model2}"
    framework = fw
    precision = prec.replace("/", "-")
    return modelname, framework, precision


class TestSession:
    def __init__(self, executable, ir_cache_dirs, devices, api=None, report_reference=False):
        self.executable = executable
        self.test_name = executable.rsplit("/", 1)[-1].removesuffix(".exe").removeprefix("test_")
        self.ir_cache_dirs = ir_cache_dirs
        self.devices = devices
        self.report_api = api
        self.report_reference = report_reference
        self.test_info = self.get_test_info()

        self.report_metadata = None
        if self.report_api:
            self.detect_report_metadata()

    def get_test_info(self):
        result = run_test_executable_extract_result([self.executable, "--info"])
        if "error" in result:
            raise Exception(f"Test executable does not behave correctly: {result}")
        if "samples" not in result:
            raise Exception(f"Unexpected test info: {result}")
        return result

    def api(self, method, data=None, **kwargs):
        extra_args = {"timeout": 30}
        extra_args.update(kwargs)
        if self.report_api is None:
            raise Exception("Report API was not specified")
        if requests is None:
            raise Exception("`requests` need to be installed to contact Report API")
        endpoint = f"{self.report_api}/api/{method}"
        response = requests.post(endpoint, json=data, **extra_args)
        if not response.ok:
            print(f"API Error: {response.text}")
        return response.json()

    def api_push_test_result(self, model_path, modelid, weights_size, device, result):
        if not self.report_metadata:
            print("No job metadata found, no report will be made.")
            return
        model_assumptions = modelid_assume_info(modelid)
        modelname, framework, precision = model_assumptions or (modelid, "unknown", "unknown")
        test_report = []
        sample_names = result.get("samples", {}).keys() or self.test_info["samples"]
        for sname in sample_names:
            sample = result.get("samples", {}).get(sname, MemSample(-1, -1, -1, -1, -1))
            sample_report = self.report_metadata.copy()
            sample_report.update({
                "test_name": f"{result.get('test', self.test_name)}:{sname}",
                "status": "failed" if "error" in result else "passed",
                "source": model_path,
                "log": result.get("stderr", ""),
                "model_name": modelname,
                "model": modelid,
                "device": result.get("device") or device,
                "framework": framework,
                "precision": precision,
                "metrics": sample.as_dict(),
                "familyCpu": CPU_FAMILY,
                "model_size": weights_size
            })
            test_report.append(sample_report)
        response = attempt(self.api, "v2/memory/push-2-db-facade", {"data": test_report})
        if response:
            print(f"Push result to API: {response}")

    def detect_report_metadata(self):
        try:
            build_number = os.environ["TT_PRODUCT_BUILD_NUMBER"]
            if self.report_reference:
                build_number = f"reference-{build_number}"
            self.report_metadata = {
                "build_url": os.environ["BUILD_URL"],
                "os": os.environ.get("os", "unknown"),
                "commit_date": os.environ.get("commitDate", "2030-12-22T22:22:22.000Z"),
                "branch": os.environ.get("sourceBranch", "unknown"),
                "target_branch": os.environ.get("targetBranch", "unknown"),
                "log_path": os.environ.get("SHARED_LOG_PATH", ""),
                "version": build_number,
                "ext": {}
            }
        except KeyError as err:
            if self.report_api:
                print(f"Environment lacks {err} to upload results to API")

    def scan_directory(self, directory: Path):
        matching_dirs = sorted(
            glob.glob(str(directory)),
            key=lambda item: os.lstat(item).st_ctime
        )
        if not matching_dirs:
            raise Exception(f"{directory} not found")
        cache_dir = os.path.abspath(os.path.normpath(matching_dirs[-1]))
        print(f"Scanning requested {directory} -> {cache_dir}")

        found_models = set()

        while True:
            fileset = set(glob.glob(f"{cache_dir}/**/*.xml", recursive=True))
            new_files = fileset - found_models
            if not new_files:
                # no new tests found, we're finished here
                return
            found_models.update(new_files)
            new_files = sorted(new_files)
            yield from ((path.removeprefix(cache_dir).replace("\\", "/"), path) for path in new_files)

    def generate_test_cases(self):
        def _with_filesize(paths):
            for (modelid, path) in paths:
                weights_path, _ = os.path.splitext(path)
                weights_path = f"{weights_path}.bin"
                weights_size = os.path.getsize(weights_path)
                yield modelid, path, weights_size
        for ir_cache_dir in self.ir_cache_dirs:
            yield from itertools.product(
                _with_filesize(self.scan_directory(ir_cache_dir)),
                self.devices
            )

    def run_test_case(self, model_path, device):
        try:
            return run_test_executable_extract_result([self.executable, model_path, device])
        except Exception as ex:
            print(f"  When running test an unexpected error happened: {ex}")
            return {"error": "unexpected error", "exception": ex}

    def handle_test_result(self, modelid, weights_size, device, result):
        base2_suffixes = ["bytes", "KiB", "MiB", "GiB", "TiB", "PiB"]

        def _base2_human_readable(number):
            order_of_magnitude = 0
            one_order_higher = 1 << 10
            while number > one_order_higher and order_of_magnitude < len(base2_suffixes) - 1:
                number >>= 10
                order_of_magnitude += 1
            suffix = base2_suffixes[order_of_magnitude]
            return f"{number} {suffix}"

        status = "error" if "error" in result else "ok"
        weights_size_human_read = _base2_human_readable(weights_size)
        print(f"TEST {modelid} ({weights_size_human_read}) x {device}: {status}")
        if status == "error":
            error = result.get("error")
            stdout = result.get("stdout")
            stderr = result.get("stderr")
            exception = result.get("exception")
            print(f"  Error: {error}")
            if stdout:
                print(f"  stdout: {stdout.strip()}\n  === END OF STDOUT ===")
            if stderr:
                print(f"  stderr: {stderr.strip()}\n  === END OF STDERR ===")
            if exception:
                print(f"  {repr(exception)}")
        else:
            for sname, sample in result["samples"].items():
                print(f"  {sname:>15}: {sample}")
        sys.stdout.flush()
        sys.stderr.flush()

    def run(self):
        for (modelid, model_path, weights_size), device in self.generate_test_cases():
            result = self.run_test_case(model_path, device)
            test_name = result.get("test", self.test_name)
            self.api_push_test_result(model_path, modelid, weights_size, device, result)
            self.handle_test_result(modelid, weights_size, device, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("test_executable")
    parser.add_argument("--ir-cache", "--ir_cache", type=Path, nargs='*', required=True,
                        help='Path to directory with *.xml model files; '
                        'can be a wildcard expression, among all directories '
                        'that match the expression the newest is taken; '
                        'inside the directory all *.xml files are listed;'
                        'can be used more than once, but directories content'
                        'must not intersect')
    parser.add_argument("--devices", default="CPU")

    parser.add_argument("--api", help="API endpoint for results to upload")
    parser.add_argument("--upload-reference", "--upload_reference",
                        "--upload-references", "--upload_references", action="store_true",
                        help="This run will make new reference values")

    args = parser.parse_args()

    if args.upload_reference and args.api is None:
        raise Exception("To upload reference values --api must be specified")

    TestSession(
        args.test_executable,
        args.ir_cache,
        [device.upper() for device in args.devices.split(",")],
        args.api,
        args.upload_reference
    ).run()
