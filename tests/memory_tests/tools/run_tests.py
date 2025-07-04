import argparse
import glob
import os
import re
import itertools
import subprocess
import json

from pathlib import Path


datatype_re = re.compile(r"((FP|INT)\d+|(\d+BIT))")

def map_datatype(t: str) -> str:
    return {"4BIT": "INT4"}.get(t, t)


def testname_safestring(s: str) -> str:
    def translate_char(c:str) -> str:
        if c.isalnum():
            return c
        return '_'
    return ''.join(map(translate_char, s))


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

    args = parser.parse_args()

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

    for model, device in test_cases:
        model_path = found_models[model]["model_path"]
        run_out = subprocess.check_output([args.test_executable, model_path, device]).decode()
        if not run_out.startswith("TEST_RESULTS: "):
            print(f"Test failed or test results format is wrong: {run_out}")
        results_json = run_out.splitlines()[0].removeprefix("TEST_RESULTS: ")
        results = json.loads(results_json)
        print(f"testid: {model}")
        print(f"path: {results['model_path']}")
        print(f"device: {results['device']}")
        print("samples:")
        for sample in results["samples"]:
            print(f"  {sample['name']:>20}: vmsize {sample['vmsize']:>10}; "
                  f"vmpeak {sample['vmpeak']:>10}; rssize {sample['rssize']:>10}; "
                  f"rspeak {sample['rspeak']:>10}; threads {sample['thr']};")
        print()

