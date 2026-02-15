# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

from argparse import ArgumentParser
from pathlib import Path
import sys
import re

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.file_utils import prepare_filelist, find_latest_dir
from utils.conformance_utils import get_logger
from utils.constants import SUPPORTED_MODEL_EXTENSION

logger = get_logger("prepare_model_list")

def get_default_re_path(is_take_all_models = False):
    SCRIPT_DIR_PATH, _ = os.path.split(os.path.abspath(__file__))
    return os.path.join(SCRIPT_DIR_PATH, "..", "data", "custom_re_to_find_models.lst") if is_take_all_models else os.path.join(SCRIPT_DIR_PATH, "..", "data", "default_re_to_find_models.lst")

def get_default_re_exclude_model_path(is_take_all_models = False):
    SCRIPT_DIR_PATH, _ = os.path.split(os.path.abspath(__file__))
    return '' if is_take_all_models else os.path.join(SCRIPT_DIR_PATH, "..", "data", "default_re_to_exclude_modes.lst")

def parse_arguments():
    parser = ArgumentParser()

    model_help = "Path to model directories path file to prepare filelist. Separator is `,`"
    output_help = "Path to output dir to save model list file"
    filename_help = "Output filename to save model list file"
    latest_only_help = "Use only latest directory matched reg exp. In other case all directories will be taken from the dir"
    exclude_models_help = "Use only latest directory matched reg exp. In other case all directories will be taken from the dir"

    parser.add_argument("-m", "--model_dirs", type=str, help=model_help, required=True)
    parser.add_argument("-o", "--output_dir", type=str, help=output_help, required=False, default=".")
    parser.add_argument("-f", "--filename", type=str, help=filename_help, required=False, default="model_filelist")
    parser.add_argument("-l", "--latest_only", type=bool, help=latest_only_help, required=False, default=False)
    parser.add_argument("-e", "--exclude_models", type=str, help=exclude_models_help, required=False, default=Path(Path(__file__).resolve().parent, "..", "data", "default_re_to_exclude_modes.lst"))

    return parser.parse_args()


def str_to_dir_list(input_str: str):
    dir_path_list = []
    while True:
        separator_pos = input_str.find(',')
        dir_path = ""
        if separator_pos == -1:
            if len(input_str) > 0:
                dir_path = input_str
                input_str = ""
            else:
                break
        else:
            dir_path = input_str[:separator_pos:]
            input_str = input_str[separator_pos+1::]
            separator_pos = input_str.find(',')
        if os.path.isdir(dir_path):
            dir_path_list.append(dir_path)
    logger.info(f"Model dir list: {dir_path_list}")
    return dir_path_list


def read_re_exp(re_exp_file_path: str):
    re_exps = []
    if os.path.isfile(re_exp_file_path):
        with open(re_exp_file_path, "r") as re_exp_file:
            for line in re_exp_file.readlines():
                if "#" in line:
                    continue
                re_exps.append(line.replace('\n', ''))
    return re_exps


def read_dir_re_exp(re_exp_file_path: str):
    dir_re_exps = read_re_exp(re_exp_file_path)
    if len(dir_re_exps) == 0:
        dir_re_exps.append('*')
    return dir_re_exps


def generate_model_list_file(input_str: str, re_exp_file_path: str, output_file_path: os.path, is_latest_only: bool, re_exclude_model_file: str):
    with open(output_file_path, 'w', newline='\n') as output_file:
        model_dir_paths = str_to_dir_list(input_str)
        dir_re_exps = read_dir_re_exp(re_exp_file_path)
        logger.info(f"Model dir re exp list: {dir_re_exps}")
        model_list = list()
        for model_dir_path in model_dir_paths:
            for dir_re_exp in dir_re_exps:
                dirs = [model_dir_path]
                if dir_re_exp != "*":
                    if is_latest_only:
                        try:
                            dirs = [find_latest_dir(model_dir_path, dir_re_exp)]
                        except:
                            dirs = []
                    else:
                        dirs = Path(model_dir_path).glob(dir_re_exp)
                for dir in dirs:
                    try:
                        logger.info(f"Processing dir: {dir}")
                        model_list.extend(prepare_filelist(str(dir), SUPPORTED_MODEL_EXTENSION, is_save_to_file=False))
                        if is_latest_only:
                            break
                    except:
                        pass
        exclude_re_exps = read_re_exp(re_exclude_model_file)
        logger.info(f"Model exclude re exp list: {exclude_re_exps}")
        for line in model_list:
            str_line = str(line)
            exclude = False
            for r in exclude_re_exps:
                if re.match(r, str_line):
                    exclude = True
            if "tfhub_module.pb" in str_line or "_metadata.pb" in str_line or exclude:
                continue
            output_file.write(f"{line}\n")
        output_file.close()

if __name__ == "__main__":
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"[ ARGUMENTS ] --model_dirs={args.model_dirs}")
    logger.info(f"[ ARGUMENTS ] --output_dir={args.output_dir}")
    logger.info(f"[ ARGUMENTS ] --filename={args.filename}")
    logger.info(f"[ ARGUMENTS ] --latest_only={args.latest_only}")
    logger.info(f"[ ARGUMENTS ] --exclude_models={args.exclude_models}")
    re_file = get_default_re_path(not args.latest_only)
    if not args.latest_only:
        logger.warning(f"{re_file} will be taken to get all models from the dirs")
    output_model_list_file = os.path.join(args.output_dir, f"{args.filename}.lst")
    generate_model_list_file(args.model_dirs, re_file, output_model_list_file, args.latest_only, args.exclude_models.as_posix())
    logger.info(f"Model file list is saved to {output_model_list_file}")
