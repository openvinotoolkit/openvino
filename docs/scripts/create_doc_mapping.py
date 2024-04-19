import re
import os
import json
from pathlib import Path
import argparse
import logging
# from github import Github

def create_mapping(output_dir: Path, doc_dir_name: str):
    mapping = {}
    output_dir = output_dir.resolve()
    script_dir = os.path.dirname(os.path.realpath(__file__))
    directory = os.path.abspath(os.path.join(script_dir, '..', doc_dir_name))
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.rst'):
                file_path = os.path.join(root, file)
                file_path = os.path.relpath(file_path, start=os.path.join(directory, '..', '..', '..'))
                file_path = file_path.replace('\\', '/').replace('OpenVINO', 'openvino').replace(' ', '-')
                file_name_without_extension = os.path.basename(os.path.splitext(file_path)[0])
                mapping[file_name_without_extension] = file_path
    with open(output_dir.joinpath('doc_mapping.json'), 'w') as f:
        json.dump(mapping, f)


def connect_jsons(output_dir: Path):
    output_dir = output_dir.resolve()
    try:
        with open(output_dir.joinpath('mapping.json'), 'r') as f:
            data1 = json.load(f)
        with open(output_dir.joinpath('doc_mapping.json'), 'r') as f:
            data2 = json.load(f)
        merged_data = {**data1, **data2}
        with open(output_dir.joinpath('mapping.json'), 'w') as f:
            json.dump(merged_data, f, indent=4) 
    except Exception as e:
        logging.error(f"Couldn't connect .json files. Error: {repr(e)}")


# def get_file_list_from_github_repo() -> dict:
#     username = 'openvinotoolkit'
#     repository_name = 'model_server'
#     path = 'docs'
#     g = Github()
#     mapping = {}
#     pattern = r'{#([^#]+?)}'
#     repo = g.get_repo(f"{username}/{repository_name}")
#     contents = repo.get_contents(path)
#     for content in contents:
#         if content.type == 'file':
#             try:
#                 decoded_content = content.decoded_content.decode('utf-8')[:200]
#                 match = re.search(pattern, decoded_content[:200])
#                 if match:
#                     html_name = match.group(1)
#                     mapping[html_name] = 'ovms/' + content.path
#             except UnicodeDecodeError:
#                 pass
#     return mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', type=Path, help='Path to the output folder')
    parser.add_argument('doc_dir_name', type=Path, help='Path to the documentation folder')
    args = parser.parse_args()
    output_dir = args.output_dir
    doc_dir_name = args.doc_dir_name
    create_mapping(output_dir, doc_dir_name)
    connect_jsons(output_dir)


if __name__ == '__main__':
    main()
