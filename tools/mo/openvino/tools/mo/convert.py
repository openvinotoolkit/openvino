# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
from openvino.tools.mo.convert_impl import _convert

InputCutInfo = namedtuple("InputInfo", ["name", "shape", "type", "value"])
LayoutMap = namedtuple("LayoutMap", ["source_layout", "target_layout"])


def convert(**args):
    """
    Converts the model from original framework to nGraph function.
    Run convert() to list available parameter.
    """
    return _convert(**args)
