import re
import argparse
from pathlib import Path
import subprocess
import requests
from packaging import version
import pkg_resources


def determine_python_version(file_path):
    # Define the regex pattern
    pattern = r"version_name\s*=\s*['\"]([^'\"]+)['\"]"
    
    with open(file_path, 'r') as file:
        content = file.read()
    
    match = re.search(pattern, content)
    
    if match:
        return match.group(1)
    else:
        return None


def get_latest_version(major_version):
    url = f"https://pypi.org/pypi/openvino/json"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        versions = data['releases'].keys()
        
        # Filter versions by the major version prefix
        matching_versions = [v for v in versions if v.startswith(major_version)]
        
        # Sort the matching versions and return the latest one
        if matching_versions:
            matching_versions.sort(key=version.parse)
            return matching_versions[-1]
    
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ov_dir', type=Path, help='OpenVINO docs directory')
    parser.add_argument('--python', type=Path, help='Python executable')
    parser.add_argument('--build_dir', type=Path, help='Build directory')
    args = parser.parse_args()
    ov_dir = args.ov_dir
    python_executable = args.python
    build_dir = args.build_dir
    version_name = determine_python_version(ov_dir.joinpath("conf.py"))

    if version_name is None:
        ov_version = "openvino"
    elif version_name == "nightly":
        ov_version = "openvino-nightly"
    else:
        latest_version = get_latest_version(version_name)
        if latest_version:
            ov_version = f"openvino=={latest_version}"
        else:
            ov_version = f"openvino=={version_name}"
    subprocess.check_call([f'{python_executable}', '-m', 'pip', 'install', '-U', ov_version, '--no-cache-dir'])
    # openvino_path = Path(pkg_resources.get_distribution("openvino").location).joinpath("openvino")
    # subprocess.check_call(['cp', '-r', openvino_path, build_dir])


if __name__ == "__main__":
    main()