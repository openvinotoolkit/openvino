# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from shutil import rmtree, copyfile
from zipfile import ZipFile, is_zipfile

import tarfile

from shutil import rmtree, copyfile
import sys
from pathlib import Path, PurePath

from urllib.parse import urlparse

from . import constants
from . import utils

# generates file list file inside directory. Returns path to saved filelist
def prepare_filelist(input_dir: os.path, pattern: str):
    filelist_path = input_dir
    if os.path.isdir(filelist_path):
        filelist_path = os.path.join(input_dir, "conformance_ir_files.lst")
    elif os.path.isfile(filelist_path):
        head, _ = os.path.split(filelist_path)
        input_dir = head
    if os.path.isfile(filelist_path):
        utils.utils_logger.info(f"{filelist_path} is exists! The script will update it!")
    xmls = Path(input_dir).rglob(pattern)
    try:
        with open(filelist_path, 'w') as file:
            for xml in xmls:
                file.write(str(xml) + '\n')
            file.close()
    except:
        utils.utils_logger.warning(f"Impossible to update {filelist_path}! Something going is wrong!")
    return filelist_path

def is_archieve(input_path: os.path):
    return tarfile.is_tarfile(input_path) or is_zipfile(input_path)

def unzip_archieve(zip_path: os.path, dst_path: os.path):
    _, tail = os.path.split(zip_path)
    dst_path = os.path.join(dst_path, tail)
    if zip_path != dst_path:
        copyfile(zip_path, dst_path)
    utils.utils_logger.info(f"Archieve {zip_path} was copied to {dst_path}")
    dst_dir, _ = os.path.splitext(dst_path)
    if tarfile.is_tarfile(zip_path):
        file = tarfile.open(dst_path)
        file.extractall(dst_dir)
        file.close()
    elif is_zipfile(zip_path):
        with ZipFile(dst_path, 'r') as zObject:
            zObject.extractall(path=dst_dir)
    else:
        utils.utils_logger.error(f"Impossible to extract {zip_path}")
        exit(-1)
    utils.utils_logger.info(f"Archieve {dst_path} was extacted to {dst_dir}")
    os.remove(dst_path)
    utils.utils_logger.info(f"Archieve {dst_path} was removed")
    return dst_dir

# find latest changed directory
def find_latest_dir(in_dir: Path, pattern_list = list()):
    get_latest_dir = lambda path: sorted(Path(path).iterdir(), key=os.path.getmtime)
    entities = get_latest_dir(in_dir)
    entities.reverse()

    for entity in entities:
        if entity.is_dir():
            if not pattern_list:
                return entity
            else:
                for pattern in pattern_list: 
                    if pattern in str(os.fspath(PurePath(entity))):
                        return entity
    utils.utils_logger.error(f"{in_dir} does not contain applicable directories to patterns: {pattern_list}")
    exit(-1)
