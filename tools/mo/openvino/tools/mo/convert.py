# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
from convert_impl import convert_

InputCutInfo = namedtuple("InputInfo", ["name", "shape", "type", "value"])
LayoutMap = namedtuple("LayoutMap", ["source_layout", "target_layout"])


def convert(**args):
    """
    Converts the model from original framework to nGraph function.
    """
    return convert_(**args)
