import requests
import re
import argparse
import logging
import sys

from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))
from common import action_utils


# Function to download the JSON file
def load_json_file(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to download the file, status code: {response.status_code}")


# Function to recursively gather all file paths from the JSON structure
def gather_all_files(node, base_path=''):
    all_files = []
    if 'children' in node:
        for child in node['children']:
            new_base = f"{base_path}/{child['name']}"
            if 'children' in child:
                all_files += gather_all_files(child, new_base)
            else:
                all_files.append(new_base)
    return all_files


# Function to filter files based on the product, version, platform, architecture, and folder
def filter_files_by_criteria(files, product, version_pattern, platform, arch, folder):
    matching_files = []
    for file in files:
        if re.search(
                fr"{product}/packages/{folder}(/?.*/)?{version_pattern}/.*{platform}.*{arch}.*\.(tar\.gz|tgz|zip)$",
                file):
            matching_files.append(file)
    return matching_files


# Main function to load the JSON, gather file paths, and filter based on criteria
def main(product, version_pattern, platform, arch, folder):
    action_utils.init_logger()
    logger = logging.getLogger(__name__)

    url = 'https://storage.openvinotoolkit.org/filetree.json'
    filetree = load_json_file(url)
    all_files = gather_all_files(filetree)
    matching_files = filter_files_by_criteria(all_files, product, version_pattern, platform, arch, folder)
    if matching_files:
        logger.info(f"Matching packages: {sorted(matching_files)}")
        if len(matching_files) > 1:
            custom_release_build_pattern = fr".*/{version_pattern}/(linux_|windows_|macos_).*/.*"
            # Exclude custom release builds, if any, from matches
            matching_files = [file for file in matching_files if not re.search(custom_release_build_pattern, file)]
        package_url = f"https://storage.openvinotoolkit.org{sorted(matching_files)[-1]}"
        logger.info(f"Returning package URL: {package_url}")
        action_utils.set_github_output("package_url", package_url)
    else:
        logger.error("No matching files found.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Search OpenVINO archives based on product, version, platform, architecture, and folder.')
    parser.add_argument('--product', required=True, choices=['openvino', 'openvino_genai', 'openvino_tokenizers'],
                        help='Product name')
    parser.add_argument('--version', required=True, help='Version pattern (supports regex)')
    parser.add_argument('--platform', default='ubuntu22', help='Platform (default: ubuntu22)')
    parser.add_argument('--arch', default='x86_64', help='Architecture (default: x86_64)')
    parser.add_argument('--folder', default='(.*/)?', help='Folder type (e.g., pre-release, nightly, default: (.*)?')

    args = parser.parse_args()

    main(args.product, args.version, args.platform, args.arch, args.folder)
