# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa

import warnings

warnings.warn(message="The module is private and following namespace "
                      "`offline_transformations` will be removed in "
                      "the future, use `openvino.runtime.passes` instead!",
              category=FutureWarning)

from openvino.utils import add_openvino_libs_to_path, deprecated
add_openvino_libs_to_path()

from openvino.pyopenvino import get_version
from openvino.pyopenvino import serialize as _base_serialize
import openvino.pyopenvino._offline_transformations as _base

__version__ = get_version()


@deprecated(version="2023.1",
            message="The module is private and following namespace "
                    "`offline_transformations` will be removed in "
                    "the future, use `openvino.runtime.passes` instead!")
def serialize(model, xml_path, bin_path, version):  # type: ignore
    _base_serialize(model, xml_path, bin_path, version)


@deprecated(version="2023.1",
            message="The module is private and following namespace "
                    "`offline_transformations` will be removed in "
                    "the future, use `openvino.runtime.passes` instead!")
def apply_moc_transformations(model, cf):  # type: ignore
    _base.apply_moc_transformations(model, cf)


@deprecated(version="2023.1",
            message="The module is private and following namespace "
                    "`offline_transformations` will be removed in "
                    "the future, use `openvino.runtime.passes` instead!")
def apply_moc_legacy_transformations(model, params_with_custom_types):
    _base.apply_moc_legacy_transformations(model, params_with_custom_types)


@deprecated(version="2023.1",
            message="The module is private and following namespace "
                    "`offline_transformations` will be removed in "
                    "the future, use `openvino.runtime.passes` instead!")
def apply_pot_transformations(model, device):  # type: ignore
    _base.apply_pot_transformations(model, device)


@deprecated(version="2023.1",
            message="The module is private and following namespace "
                    "`offline_transformations` will be removed in "
                    "the future, use `openvino.runtime.passes` instead!")
def apply_low_latency_transformation(model, use_const_initializer):  # type: ignore
    _base.apply_low_latency_transformation(model, use_const_initializer)


@deprecated(version="2023.1",
            message="The module is private and following namespace "
                    "`offline_transformations` will be removed in "
                    "the future, use `openvino.runtime.passes` instead!")
def apply_pruning_transformation(model):  # type: ignore
    _base.apply_pruning_transformation(model)


@deprecated(version="2023.1",
            message="The module is private and following namespace "
                    "`offline_transformations` will be removed in "
                    "the future, use `openvino.runtime.passes` instead!")
def generate_mapping_file(model, path, extract_names):  # type: ignore
    _base.generate_mapping_file(model, path, extract_names)


@deprecated(version="2023.1",
            message="The module is private and following namespace "
                    "`offline_transformations` will be removed in "
                    "the future, use `openvino.runtime.passes` instead!")
def apply_make_stateful_transformation(model, param_res_names):
    _base.apply_make_stateful_transformation(model, param_res_names)


@deprecated(version="2023.1",
            message="The module is private and following namespace "
                    "`offline_transformations` will be removed in "
                    "the future, use `openvino.runtime.passes` instead!")
def compress_model_transformation(model):  # type: ignore
    _base.compress_model_transformation(model)


@deprecated(version="2023.1",
            message="The module is private and following namespace "
                    "`offline_transformations` will be removed in "
                    "the future, use `openvino.runtime.passes` instead!")
def compress_quantize_weights_transformation(model):  # type: ignore
    _base.compress_quantize_weights_transformation(model)


@deprecated(version="2023.1",
            message="The module is private and following namespace "
                    "`offline_transformations` will be removed in "
                    "the future, use `openvino.runtime.passes` instead!")
def convert_sequence_to_tensor_iterator_transformation(model):  # type: ignore
    _base.convert_sequence_to_tensor_iterator_transformation(model)
