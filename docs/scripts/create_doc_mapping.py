import os
import json
from pathlib import Path
import argparse
import logging

def create_mapping(output_dir: Path):
    mapping = {}
    output_dir = output_dir.resolve()
    script_dir = os.path.dirname(os.path.realpath(__file__))
    directory = os.path.abspath(os.path.join(script_dir, '..', 'articles_en'))
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


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('output_dir', type=Path, help='Path to the output folder')
    # args = parser.parse_args()
    # output_dir = args.output_dir
    output_dir = Path(r'C:\Users\bbielawx\OneDrive - Intel Corporation\Desktop\OpenVINO\build')
    create_mapping(output_dir)
    connect_jsons(output_dir)


if __name__ == '__main__':
    main()
