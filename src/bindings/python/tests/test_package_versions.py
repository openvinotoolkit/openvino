# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

import openvino                         as ov
import openvino.runtime                 as ov_run
import openvino.frontend                as ov_front
import openvino.preprocess              as ov_pre
import openvino.offline_transformations as ov_off_transf
import openvino.inference_engine        as ov_ie
import openvino.pyopenvino              as ov_py
import openvino.utils                   as ov_utils
import openvino.tools                   as ov_tools

### == Version rules for all packages ==
# Refer to ../docs/code_examples.md
# openvino                         - namespace for packages - no version provided
# openvino.utils                   - not a package - no version provided
# openvino.tools                   - namespace - no version provided 
# openvino.pyopenvino              - due to being a pybind, provides only get_version method
# openvino.inference_engine        - 
# openvino.offline_transformations - 
# openvino.frontend                -  Remaining entries are proper packages with
# openvino.preprocess              -  __version__ and get_version method provided
# openvino.runtime                 -
### ====================================
    

def test_get_version_exists_and_match():
    packages = [ov_run,ov_front,ov_pre,ov_off_transf,ov_ie,ov_py]
    versions = set()

    for package in packages:
        versions.add(package.get_version())

    assert len(versions) == 1

def test_dunder_version_exists_and_match():
    packages = [ov_run,ov_front,ov_pre,ov_off_transf,ov_ie]
    versions = set()

    for package in packages:
        versions.add(package.__version__)

    assert len(versions) == 1
