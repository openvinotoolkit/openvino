# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import glob
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()

    skip_config_help = "Paths to folder with skip_config_files"
    input_folders_help = "Paths to folders with logs"
    extend_file_help = "Extend exiting file"

    parser.add_argument("-s", "--skip_config_folders", help=skip_config_help, nargs='*', required=True)
    parser.add_argument("-i", "--input_logs", help=input_folders_help, nargs='*', required=True)
    parser.add_argument("-e", "--extend_file", help=extend_file_help, default=False, required=False)

    return parser.parse_args()


def is_conformance(content: str):
    if 'conformance' in content:
        return True
    return False


def is_hung_test(content: str):
    if content == '' or \
        "SKIPPED" in content or \
        "FAILED" in content or \
        "Unexpected application crash!" in content or \
        "PASSED" in content:
        return False
    return True


def get_device_name(content: str):
    target_device_str = 'TargetDevice='
    pos_start = content.find(target_device_str)
    pos_end = content.find('\n')
    return f'{content[pos_start + len(target_device_str):pos_end]}'.lower()


def get_regex(content: str):
    ir_name_str = 'IR_name='
    pos_start = content.find(ir_name_str)
    pos_end = content.find('.xml_')
    return f'.*{content[pos_start + len(ir_name_str):pos_end]}.*\n'


def get_conformance_hung_test(test_log_dirs: list):
    regexp = dict()
    for test_log_dir in test_log_dirs:
        if not os.path.isdir(test_log_dir):
            continue
        for log_file in glob.glob(os.path.join(test_log_dir, '*/*')):
            with open(log_file) as log:
                content = log.read()
                if not (is_hung_test(content) and is_conformance(content)):
                    continue
                device = get_device_name(content)
                if 'arm' in content or 'arm' in log_file:
                    device = 'arm'
                if not device in regexp.keys():
                    regexp.update({device: []})
                if get_regex(content) in regexp[device]:
                    continue
                regexp[device].append(get_regex(content))
    for device, re_list in regexp.items():
        re_list.sort()
    return regexp


def save_to_file(skip_folder_paths: list, regexps: dict, extend_file: str):
    for skip_folder_path in skip_folder_paths:
        if not os.path.isdir(skip_folder_path):
            continue
        skip_files_paths = glob.glob(os.path.join(skip_folder_path, 'skip_config_*.lst'))
        for skip_files_path in skip_files_paths:
            for device, re_list in regexps.items():
                if device in skip_files_path:
                    if extend_file:
                        with open(skip_files_path, 'r') as file:
                            content = file.readlines()
                    with open(skip_files_path, 'w') as file:
                        if extend_file:
                            file.writelines(content)
                        file.writelines(re_list)


if __name__ == "__main__":
    args = parse_arguments()
    save_to_file(args.skip_config_folders, get_conformance_hung_test(args.input_logs), args.extend_file)
