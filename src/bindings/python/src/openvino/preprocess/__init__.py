# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Package: openvino
Low level wrappers for the PrePostProcessing C++ API.
"""

# flake8: noqa

from openvino.utils import add_openvino_libs_to_path
from pkg_resources import get_distribution, DistributionNotFound


try:
    __version__ = get_distribution("openvino-core").version
except DistributionNotFound:
    __version__ = "0.0.0.dev0"

add_openvino_libs_to_path()

# main classes
from openvino.pyopenvino.preprocess import InputInfo
from openvino.pyopenvino.preprocess import OutputInfo
from openvino.pyopenvino.preprocess import InputTensorInfo
from openvino.pyopenvino.preprocess import OutputTensorInfo
from openvino.pyopenvino.preprocess import InputModelInfo
from openvino.pyopenvino.preprocess import OutputModelInfo
from openvino.pyopenvino.preprocess import PrePostProcessor
from openvino.pyopenvino.preprocess import PreProcessSteps
from openvino.pyopenvino.preprocess import PostProcessSteps
from openvino.pyopenvino.preprocess import ColorFormat
from openvino.pyopenvino.preprocess import ResizeAlgorithm

# version
from openvino.pyopenvino import get_version
