import re
import argparse
import subprocess
import requests
from packaging import version
from pathlib import Path


def determine_openvino_version(file_path):
    pattern = r"version_name\s*=\s*['\"]([^'\"]+)['\"]"
    with open(file_path, 'r') as file:
        content = file.read()
    match = re.search(pattern, content)
    return match.group(1) if match else None


def get_latest_version(package, major_version):
    url = f"https://pypi.org/pypi/{package}/json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        versions = data['releases'].keys()
        matching_versions = [v for v in versions if v.startswith(major_version)]
        if matching_versions:
            matching_versions.sort(key=version.parse)
            return matching_versions[-1]
    return None


def install_package(python_executable, package):
    subprocess.check_call([f'{python_executable}', '-m', 'pip', 'install', '-U', package, '--no-cache-dir'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ov_dir', type=Path, help='OpenVINO docs directory')
    parser.add_argument('--python', type=Path, help='Python executable')
    parser.add_argument('--enable_genai', type=str, choices=['ON', 'OFF'], default='OFF', help='Enable GenAI API installation')
    args = parser.parse_args()
    
    version_name = determine_openvino_version(args.ov_dir.joinpath("conf.py"))

    if version_name == "nightly":
        install_package(args.python, "openvino-nightly")
        print("OpenVINO nightly version installed. OpenVINO GenAI nightly version is not available.")
    elif version_name is None or version_name == "latest":
        install_package(args.python, "openvino")
        if args.enable_genai == 'ON':
            install_package(args.python, "openvino-genai")
    else:
        ov_version = get_latest_version("openvino", version_name)
        if ov_version:
            install_package(args.python, f"openvino=={ov_version}")
        else:
            print(f"No matching OpenVINO version found for {version_name}")
        
        if args.enable_genai == 'ON':
            ov_genai_version = get_latest_version("openvino-genai", version_name)
            if ov_genai_version:
                install_package(args.python, f"openvino-genai=={ov_genai_version}")
            else:
                print(f"No matching OpenVINO GenAI version found for {version_name}")


if __name__ == "__main__":
    main()