# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from sys import platform

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
REF_COEF = "[ CONFORMANCE ] Influence coefficient: "

IS_WIN = "windows" in platform or "win32" in platform

OS_SCRIPT_EXT = ".bat" if IS_WIN else ""
OS_BIN_FILE_EXT = ".exe" if IS_WIN else ""
ENV_SEPARATOR = ";" if IS_WIN else ":"
PYTHON_NAME = "python" if IS_WIN else "python3"
PIP_NAME = "pip" if IS_WIN else "pip3"
LD_LIB_PATH_NAME = "PATH" if IS_WIN or platform == "darwin" else "LD_LIBRARY_PATH"

OPENVINO_NAME = 'openvino'
PY_OPENVINO = "python_api"

DEBUG_DIR = "Debug"
RELEASE_DIR = "Release"

OP_CONFORMANCE = "OP"
API_CONFORMANCE = "API"
