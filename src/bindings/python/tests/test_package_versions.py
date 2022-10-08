# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino.preprocess as ov_pre
import openvino.runtime as ov_run
import openvino.frontend as ov_front
import openvino._offline_transformations as ov_off_transf
import openvino.pyopenvino as ov_py


def test_get_version_match():
    packages = [ov_run, ov_front, ov_pre, ov_off_transf, ov_py]
    versions = set()

    for package in packages:
        versions.add(package.get_version())

    assert len(versions) == 1


def test_dunder_version_match():
    packages = [ov_run, ov_front, ov_pre, ov_off_transf]
    versions = set()

    for package in packages:
        versions.add(package.__version__)

    assert len(versions) == 1
