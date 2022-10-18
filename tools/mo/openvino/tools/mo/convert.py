# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
from openvino.tools.mo.convert_impl import _convert

InputCutInfo = namedtuple("InputInfo", ["name", "shape", "type", "value"])
LayoutMap = namedtuple("LayoutMap", ["source_layout", "target_layout"])


def convert(input_model=None, **args):
    """
    Converts the model from original framework to OpenVino Model.

    Args:
        input_model:
            Tensorflow*: a file with a pre-trained model (binary or text .pb file after freezing).
            Caffe*: a model proto file with model weights

    Run convert(help=true) to list all available parameters.

    Returns:
        openvino.runtime.Model
    """
    args.update({'input_model': input_model})
    return _convert(**args)
