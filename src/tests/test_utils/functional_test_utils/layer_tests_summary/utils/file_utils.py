# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import tarfile

from pathlib import Path, PurePath
from shutil import copyfile
from zipfile import ZipFile, is_zipfile
from urllib.parse import urlparse

from . import constants
from . import conformance_utils

# generates file list file inside directory. Returns path to saved filelist
def prepare_filelist(input_dir: os.path, patterns: list, is_save_to_file = True):
    filelist_path = input_dir
    if os.path.isdir(filelist_path):
        filelist_path = os.path.join(input_dir, "conformance_ir_files.lst")
    elif os.path.isfile(filelist_path):
        head, _ = os.path.split(filelist_path)
        input_dir = head
    if os.path.isfile(filelist_path):
        conformance_utils.UTILS_LOGGER.info(f"{filelist_path} is exists! The script will update it!")
    model_list = list()
    for pattern in patterns:
        model_list.extend(Path(input_dir).rglob(pattern))
    if is_save_to_file:
        try:
            with open(filelist_path, 'w') as file:
                for xml in model_list:
                    file.write(str(xml) + '\n')
                file.close()
        except:
            conformance_utils.UTILS_LOGGER.warning(f"Impossible to update {filelist_path}! Something going is wrong!")
    else:
        return model_list
    return filelist_path

def is_archieve(input_path: os.path):
    return tarfile.is_tarfile(input_path) or is_zipfile(input_path)

def is_url(url: str):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def unzip_archieve(zip_path: os.path, dst_path: os.path):
    _, tail = os.path.split(zip_path)
    dst_path = os.path.join(dst_path, tail)
    if zip_path != dst_path:
        copyfile(zip_path, dst_path)
    conformance_utils.UTILS_LOGGER.info(f"Archieve {zip_path} was copied to {dst_path}")
    dst_dir, _ = os.path.splitext(dst_path)
    if tarfile.is_tarfile(zip_path):
        file = tarfile.open(dst_path)
        file.extractall(dst_dir)
        file.close()
    elif is_zipfile(zip_path):
        with ZipFile(dst_path, 'r') as zObject:
            zObject.extractall(path=dst_dir)
    else:
        conformance_utils.UTILS_LOGGER.error(f"Impossible to extract {zip_path}")
        exit(-1)
    conformance_utils.UTILS_LOGGER.info(f"Archieve {dst_path} was extacted to {dst_dir}")
    os.remove(dst_path)
    conformance_utils.UTILS_LOGGER.info(f"Archieve {dst_path} was removed")
    return dst_dir

# find latest changed directory
def find_latest_dir(in_dir: Path, pattern = "*"):
    get_latest_dir = lambda path: sorted(Path(path).glob(pattern), key=os.path.getmtime)
    entities = get_latest_dir(in_dir)
    entities.reverse()

    for entity in entities:
        if entity.is_dir():
            return entity
    conformance_utils.UTILS_LOGGER.error(f"{in_dir} does not contain applicable directories to pattern: {pattern}")
    exit(-1)

def get_ov_path(script_dir_path: os.path, ov_dir=None, is_bin=False):
    if ov_dir is None or not os.path.isdir(ov_dir):
        ov_dir = os.path.abspath(script_dir_path)[:os.path.abspath(script_dir_path).find(constants.OPENVINO_NAME) + len(constants.OPENVINO_NAME)]
    if is_bin:
        ov_dir = os.path.join(ov_dir, find_latest_dir(ov_dir, 'bin'))
        ov_dir = os.path.join(ov_dir, find_latest_dir(ov_dir))
        ov_dir = os.path.join(ov_dir, find_latest_dir(ov_dir))
    return ov_dir
