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

TEST_STATUS = {
    'passed': ["[       OK ]"],
    'failed': ["[  FAILED  ]"],
    'hanged': ["Test finished by timeout"],
    'crashed': ["Unexpected application crash with code", "Segmentation fault", "Crash happens", "core dumped"],
    'skipped': ["[  SKIPPED ]"],
    'interapted': ["interapted", "Killed"]}
RUN = "[ RUN      ]"
GTEST_FILTER = "Google Test filter = "
DISABLED_PREFIX = "DISABLED_"

IS_WIN = "windows" in sys.platform or "win32" in sys.platform

OS_SCRIPT_EXT = ".bat" if IS_WIN else ""
OS_BIN_FILE_EXT = ".exe" if IS_WIN else ""

NO_MODEL_CONSTANT = "http://ov-share-03.sclab.intel.com/Shares/conformance_ir/dlb/master/2022.3.0-8953-8c3425ff698.tar"

ENV_SEPARATOR = ";" if IS_WIN else ":"

PYTHON_NAME = "python" if IS_WIN else "python3"
PIP_NAME = "pip" if IS_WIN else "pip3"

def get_logger(app_name: str):
    logging.basicConfig()
    logger = logging.getLogger(app_name)
    logger.setLevel(logging.INFO)
    return logger

utils_logger = get_logger('conformance_utilities')


def update_passrates(results: ET.SubElement):
    for device in results:
        for op in device:
            passed_tests = 0
            total_tests = 0
            for attrib in op.attrib:
                if attrib == "passrate":
                    continue
                if attrib == "implemented":
                    continue
                if attrib == "passed":
                    passed_tests = int(op.attrib.get(attrib))
                total_tests += int(op.attrib.get(attrib))
            if total_tests == 0:
                passrate = 0
            else:
                passrate = float(passed_tests * 100 / total_tests) if passed_tests < total_tests else 100
            op.set("passrate", str(round(passrate, 1)))


def update_conformance_test_counters(results: ET.SubElement):
    max_test_cnt = dict()
    incorrect_ops = set()
    for device in results.find("results"):
        for op in device:
            op_test_count = 0
            for attr_name in op.attrib:
                if attr_name == "passrate" or attr_name == "implemented":
                    continue
                op_test_count += int(op.attrib.get(attr_name))
            if not op.tag in max_test_cnt.keys():
                max_test_cnt.update({op.tag: op_test_count})
            if op_test_count != max_test_cnt[op.tag]:
                incorrect_ops.add(op.tag)
                max_test_cnt[op.tag] = max(op_test_count, max_test_cnt[op.tag])

    for device in results.find("results"):
        for op in device:
            if op.tag in incorrect_ops:
                test_cnt = 0
                for attr_name in op.attrib:
                    if attr_name == "passrate" or attr_name == "implemented":
                        continue
                    test_cnt += int(op.attrib[attr_name])
                if test_cnt != max_test_cnt[op.tag]:
                    diff = max_test_cnt[op.tag] - test_cnt
                    op.set("skipped", str(int(op.attrib["skipped"]) + diff))
                    utils_logger.warning(f'{device.tag}: added {diff} skipped tests for {op.tag}')
    update_passrates(results)

def prepare_filelist(input_dir: os.path, pattern: str):
    filelist_path = input_dir
    if os.path.isdir(filelist_path):
        filelist_path = os.path.join(input_dir, "conformance_ir_files.lst")
    elif os.path.isfile(filelist_path):
        head, _ = os.path.split(filelist_path)
        input_dir = head
    if os.path.isfile(filelist_path):
        prepare_filelist.info(f"{filelist_path} is exists! The script will update it!")
    xmls = Path(input_dir).rglob(pattern)
    try:
        with open(filelist_path, 'w') as file:
            for xml in xmls:
                file.write(str(xml) + '\n')
            file.close()
    except:
        prepare_filelist.warning(f"Impossible to update {filelist_path}! Something going is wrong!")
    return filelist_path

def is_archieve(input_path: os.path):
    return tarfile.is_tarfile(input_path) or is_zipfile(input_path)

def unzip_archieve(zip_path: os.path, dst_path: os.path):
    _, tail = os.path.split(zip_path)
    dst_path = os.path.join(dst_path, tail)
    if zip_path != dst_path:
        copyfile(zip_path, dst_path)
    utils_logger.info(f"Archieve {zip_path} was copied to {dst_path}")
    dst_dir, _ = os.path.splitext(dst_path)
    if tarfile.is_tarfile(zip_path):
        file = tarfile.open(dst_path)
        file.extractall(dst_dir)
        file.close()
    elif is_zipfile(zip_path):
        with ZipFile(dst_path, 'r') as zObject:
            zObject.extractall(path=dst_dir)
    else:
        utils_logger.error(f"Impossible to extract {zip_path}")
        exit(-1)
    utils_logger.info(f"Archieve {dst_path} was extacted to {dst_dir}")
    os.remove(dst_path)
    utils_logger.info(f"Archieve {dst_path} was removed")
    return dst_dir

    

def progressbar(it_num, message="", progress_bar_size=60, out=sys.stdout):
    max_len = len(it_num)
    if max_len == 0:
        return
    def show(sym_pos):
        x = int(progress_bar_size * sym_pos / max_len)
        print("{}[{}{}] {}/{}".format(message, "#"*x, "."*(progress_bar_size-x), sym_pos, max_len), 
                end='\r', file=out, flush=True)
    show(0)
    for i, item in enumerate(it_num):
        yield item
        show(i+1)
    print("", flush=True, file=out)

def set_env_variable(env: os.environ, var_name: str, var_value: str):
    if var_name in env:
        env[var_name] = var_value + ENV_SEPARATOR + env[var_name]
    else:
        env[var_name] = var_value
    return env

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
    utils_logger.error(f"{in_dir} does not contain applicable directories to patterns: {pattern_list}")
    exit(-1)

    

OPENVINO_NAME = 'openvino'


DEBUG_DIR = "Debug"
RELEASE_DIR = "Release"

def get_ov_path(script_dir_path: os.path, ov_dir=None, is_bin=False):
    if ov_dir is None or not os.path.isdir(ov_dir):
        ov_dir = os.path.abspath(script_dir_path)[:os.path.abspath(script_dir_path).find(OPENVINO_NAME) + len(OPENVINO_NAME)]
    if is_bin:
        ov_dir = os.path.join(ov_dir, find_latest_dir(ov_dir, ['bin']))
        ov_dir = os.path.join(ov_dir, find_latest_dir(ov_dir))
        ov_dir = os.path.join(ov_dir, find_latest_dir(ov_dir, [DEBUG_DIR, RELEASE_DIR]))
    return ov_dir

def is_url(url: str):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False