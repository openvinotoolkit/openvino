import argparse
import glob
import os
import re
import itertools
import subprocess
import json

from pathlib import Path


class MemSample:
    keys = ("vmsize", "vmpeak", "vmrss", "vmhwm", "threads")

    def __init__(self, **kwargs):
        self._val = tuple((kwargs[k] for k in self.keys))

    def get(self, key):
        try:
            return self._val[self.keys.index(key)]
        except ValueError:
            return None

    def items(self):
        return zip(self.keys, self._val)

    def as_dict(self):
        return {
            key: value
            for key, value
            in self.items()
        }

    def per_value_diff(self, ref):
        for key in self.keys:
            val = self.get(key)
            ref = self.get(key)
            diff = val - ref
            diffp = diff / ref
            yield (key, val, ref, diff, diffp)

    def __repr__(self):
        return "; ".join(f"{k} {v:>10}" for k, v in self.items())


datatype_re = re.compile(r"((FP|INT)\d+|(\d+BIT))")

def map_datatype(t: str) -> str:
    return {"4BIT": "INT4"}.get(t, t)


def testname_safestring(s: str) -> str:
    def translate_char(c:str) -> str:
        if c.isalnum():
            return c
        return '_'
    return ''.join(map(translate_char, s))


def run_extract_test_result(command):
    run_out = subprocess.check_output(command).decode()
    if not run_out.startswith("TEST_RESULTS: "):
        print(f"Test failed or test results format is wrong: {run_out}")
    results_json = run_out.splitlines()[0].removeprefix("TEST_RESULTS: ")
    results = json.loads(results_json)
    results["samples"] = {
        sname: MemSample(**sample)
        for sname, sample in results["samples"].items()
    }
    return results


def run_single_test(executable, model_path, device):
    comand = [executable, model_path, device]
    result = None
    try:
        result = run_extract_test_result(comand)
    except Exception as ex:
        print(f"When running test an unexpected error happened: {ex}")
    if result is not None:
        return result
    print("Trying to re-run test once")
    try:
        return run_extract_test_result(comand)
    except Exception as ex:
        print(f"Test failed to run twice, ignoring this test case.")


# def combine_samples(run_samples: list[dict]):
#     snames = tuple(run_samples[0].keys())
#     keys = tuple(next(run_samples[0].values()).keys())
#     assert all([tuple(r.keys()) == snames for r in run_samples])
#     return {
#         sname: {
#             key: max([r[sname][key] for r in run_samples])
#             for key in keys
#         }
#         for sname in snames
#     }


def run_testcase(executable, model_path, device, niter=1):
    assert niter == 1
    # TODO: support niter>1
    return run_single_test(executable, model_path, device)


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

    found_models: dict[str, dict] = {}

    for path in args.ir_cache:
        matching_dirs = sorted(
            glob.glob(str(path)),
            key=lambda item: os.lstat(item).st_ctime
        )
        if not matching_dirs:
            raise Exception("Path {path} not found")
        cache_dir = os.path.abspath(os.path.normpath(matching_dirs[-1]))
        print(f"Scanning requested {path} -> {cache_dir}")

        for modelfullpath in glob.glob(f"{cache_dir}/**/*.xml", recursive=True):
            modelidpath = modelfullpath.removeprefix(cache_dir)
            if modelidpath in found_models:
                raise Exception(f"Specified directories have repeating models: {modelidpath}")
            found_models[modelidpath] = {
                "precision": tuple([
                    map_datatype(m[0].upper()) 
                    for m in datatype_re.findall(modelidpath)
                ]),
                "model_path": modelfullpath,
                "model": modelidpath,
            }
    
    print(f"Found {len(found_models)} models.")

    test_cases = list(itertools.product(found_models, args.devices.split(",")))

    print(f"Generated {len(test_cases)} test cases.")

    reference_values = {}
    if args.api:
        # TODO: move to a separate function
        import requests
        print(f"Getting reference values from API.")
        endpoint = f"{args.api}/api/v1/reports/memory/root-ref-metrics"
        # try:
        response = requests.post(endpoint, json=[], timeout=10).json()
        print(f"Fetched {len(response)} reference samples.")
        for item in response:
            try:
                test_name, sample = item["test"].split(":")
                network = item["network"]
            except ValueError:
                continue
            if test_name not in reference_values:
                reference_values[test_name] = {}
            test_references = reference_values[test_name]
            if network not in test_references:
                test_references[network] = {}
            memsample = MemSample(**item)
            # memsample._from_api = item
            test_references[network][sample] = memsample
        # except Exception as ex:
        #     print(f"Failed to collect reference values: {ex}; {response[0]}")

    for model, device in test_cases:
        model_path = found_models[model]["model_path"]
        print(f"network: {model}")
        print(f"path: {model_path}")
        print(f"device: {device}")
        result = run_testcase(args.test_executable,
            model_path, device, niter=1)
        if result is None:
            continue
        test_name = result['test']
        print(f"test: {test_name}")
        refs = {}
        if reference_values:
            try:
                refs = reference_values[test_name][model]
            except KeyError:
                print("No reference found to compare to")

        for sname, sample in result["samples"].items():
            ref = refs.get(sname)
            if not ref:
                print(f"  {sname:>15}: {sample}")
                continue
            print(f"  sample {sname}:")
            for key, val, ref, diff, diffp in sample.per_value_diff(ref):
                print(f"    {key:>8}: {val:>10} (ref: {ref:>10}) {diff:>+6} ({diffp:+.2f}%)")
        # if args.api:
        #     # upload test result
        #     pass
        if args.upload_reference:
            import requests
            endpoint = f"{args.api}/api/v1/memory/push-2-db-facade/root-ref-metrics"
            test_report = []
            for sname, sample in result["samples"].items():
                sample_report = {
                    "test": f"{result['test']}:{sname}",
                    "network": model,
                    "device": result["device"],
                    "framework": "unknown",
                    "precision": "unknown"
                }
                sample_report.update(sample.as_dict())
                test_report.append(sample_report)
            response = requests.post(endpoint, json=test_report)
            print(response.json())
