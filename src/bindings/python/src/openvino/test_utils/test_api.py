# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .test_utils_api import compare_functions as compare_func
from openvino._pyopenvino import Model

def compare_functions(lhs: Model, rhs: Model) -> tuple:
    return compare_func(lhs._Model__model, rhs._Model__model)