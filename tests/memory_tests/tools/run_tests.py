# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import argparse
import glob
import os
import re
import itertools
import subprocess
import json
import time

from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict

try:
    import requests
except ImportError:
    requests = None


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
    vmsize: int
    vmpeak: int
    vmrss: int
    vmhwm: int
    threads: int

    as_dict = asdict

    @staticmethod
    def from_dict(values: dict):
        class_fields = MemSample.__dataclass_fields__.keys()
        selected_values = {k: int(values[k]) for k in class_fields}
        return MemSample(**selected_values)

    def compare(self, reference: "MemSample"):
        ref_dict = reference.as_dict()
        for key, value in self.as_dict().items():
            refval = ref_dict[key]
            yield (key, *value_diff(value, refval))

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
                sname: MemSample(**sample)
                for sname, sample in results["samples"].items()
            }
        return results
    return {
        "error": "Test did not run correctly",
        "returncode": proc.returncode,
        "stdout": proc.stdout.decode(),
        "stderr": proc.stderr.decode()
    }


MODELID_RE = re.compile(r"\/?([^\/]+)\/(\w+)\/?[^\/]*\/((OV_)?(FP|INT)\d+(\/INT\d+)?[^\/]*)(\/\d+\/ov)?\/(.+).xml")


def modelid_assume_info(modelid):
    match = MODELID_RE.match(modelid)
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

        self.report_metadata = None
        self.reference_values = {}
        if self.report_api:
            self.detect_report_metadata()
            self.reference_values = self.api_get_reference_values()

    def api(self, method, data=None, **kwargs):
        extra_args = {"timeout": 30}
        extra_args.update(kwargs)
        if self.report_api is None:
            raise Exception("Report API was not specified")
        if requests is None:
            raise Exception("`requests` need to be installed to contact Report API")
        endpoint = f"{self.report_api}/api/{method}"
        return requests.post(endpoint, json=data, **extra_args).json()

    def api_get_reference_values(self):
        api_response = self.api("v1/reports/memory/root-ref-metrics", [], timeout=5)
        reference_test_values = defaultdict(lambda: defaultdict(dict))
        for ref_item in api_response:
            try:
                test_name, sample = ref_item["test"].split(":")
                network = ref_item["network"]
            except ValueError:
                continue
            reference_test_values[test_name][network][sample] = MemSample.from_dict(ref_item)
        return reference_test_values

    def api_push_reference_values(self, modelid, device, result):
        if "error" in result:
            print("Failed result is not going to be uploaded as reference.")
            return
        test_report = []
        test_name = result.get("test", self.test_name)
        model_assumptions = modelid_assume_info(modelid)
        modelname, framework, precision = model_assumptions or (modelid, "unknown", "unknown")
        for sname, sample in result["samples"].items():
            sample_report = {
                "test": f"{test_name}:{sname}",
                "network": modelname,
                "device": result.get("device", device),
                "framework": framework,
                "precision": precision
            }
            sample_report.update(sample.as_dict())
            test_report.append(sample_report)
        response = attempt(self.api, "v1/memory/push-2-db-facade/root-ref-metrics", test_report)
        if response:
            print(f"Push reference to API: {response}")

    def api_push_test_result(self, source, modelid, device, result, refsamples=None):
        if refsamples is None:
            refsamples = {}
        if "error" in result and not refsamples:
            print("Failed result is not going to be uploaded since no reference results were found.")
            return
        if not self.report_metadata:
            print("No job metadata found, no report will be made.")
            return
        model_assumptions = modelid_assume_info(modelid)
        modelname, framework, precision = model_assumptions or (modelid, "unknown", "unknown")
        test_report = []
        sample_names = result.get("samples", refsamples).keys()
        for sname in sample_names:
            sample = result.get("samples", {}).get(sname, MemSample(-1, -1, -1, -1, -1))
            sample_report = self.report_metadata.copy()
            sample_report.update({
                "test_name": f"{result['test']}:{sname}",
                "status": "failed" if "error" in result else "passed",
                "source": source,
                "log": result.get("stderr", ""),
                "model_name": modelname,
                "model": modelid,
                "device": result.get("device") or device,
                "framework": framework,
                "precision": precision,
                "metrics": sample.as_dict(),
                "ref_metrics": (refsamples.get(sname) or sample).as_dict()
            })
            test_report.append(sample_report)
        response = attempt(self.api, "v1/memory/push-2-db-facade", {"data": test_report})
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
                "dldt_version": build_number,
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
            yield from ((path.removeprefix(cache_dir), path) for path in new_files)

    def generate_test_cases(self):
        for ir_cache_dir in self.ir_cache_dirs:
            yield from itertools.product(
                self.scan_directory(ir_cache_dir), self.devices)

    def run_test_case(self, model_path, device):
        try:
            return run_test_executable_extract_result([self.executable, model_path, device])
        except Exception as ex:
            print(f"  When running test an unexpected error happened: {ex}")
            return {"error": "unexpected error", "exception": ex}

    def handle_test_result(self, modelid, device, result, refsamples=None):
        status = "error" if "error" in result else "ok"
        print(f"TEST {modelid} x {device}: {status}")
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
            return
        for sname, sample in result["samples"].items():
            refsample = None
            if refsamples:
                refsample = refsamples.get(sname)
            if not refsample:
                print(f"  {sname:>15}: {sample}")
                continue
            print(f"  sample {sname}:")
            for key, val, ref, diff, diffp in sample.compare(refsample):
                print(f"    {key:>8}: {val:>10} (ref: {ref:>10}) {diff:>+8} ({diffp:+.2f}%)")

    def run(self):
        for (modelid, model_path), device in self.generate_test_cases():
            result = self.run_test_case(model_path, device)
            test_name = result.get("test", self.test_name)
            refsamples = self.reference_values.get(test_name, {}).get(modelid)
            self.api_push_test_result(model_path, modelid, device, result, refsamples)
            if self.report_reference:
                self.api_push_reference_values(modelid, device, result)
            self.handle_test_result(modelid, device, result, refsamples)


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

    parser.add_argument("--api", help="API endpoint for results to compare and upload")
    parser.add_argument("--upload-reference", "--upload_reference",
                        "--upload-references", "--upload_references", action="store_true",
                        help="This run will make new reference values")

    args = parser.parse_args()

    if args.upload_reference and args.api is None:
        raise Exception("To upload reference values --api must be specified")

    TestSession(
        args.test_executable,
        args.ir_cache,
        args.devices.split(","),
        args.api,
        args.upload_reference
    ).run()
