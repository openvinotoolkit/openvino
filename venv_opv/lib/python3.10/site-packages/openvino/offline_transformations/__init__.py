# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# type: ignore

import warnings

warnings.warn(
    message="The module is private and following namespace "
    "`offline_transformations` will be removed in the future.",
    category=FutureWarning,
)

from openvino.utils import _add_openvino_libs_to_search_path, deprecated

_add_openvino_libs_to_search_path()

from openvino._pyopenvino import get_version
from openvino._pyopenvino import serialize as _base_serialize
import openvino._pyopenvino._offline_transformations as _base

__version__ = get_version()


@deprecated(
    version="2023.1",
    message="The module is private and following namespace "
    "`offline_transformations` will be removed in "
    "the future, use `openvino.runtime.passes` instead!",
)
def serialize(model, xml_path, bin_path, version):
    _base_serialize(model, xml_path, bin_path, version)


@deprecated(version="2023.1", message="The module is private and following namespace " "`offline_transformations` will be removed in " "the future.")
def apply_fused_names_cleanup(model):
    _base.apply_fused_names_cleanup(model)


@deprecated(version="2023.1", message="The module is private and following namespace " "`offline_transformations` will be removed in " "the future.")
def apply_moc_transformations(model, cf, smart_reshape=False):
    _base.apply_moc_transformations(model, cf, smart_reshape)


@deprecated(version="2023.1", message="The module is private and following namespace " "`offline_transformations` will be removed in " "the future.")
def apply_moc_legacy_transformations(model, params_with_custom_types):
    _base.apply_moc_legacy_transformations(model, params_with_custom_types)


@deprecated(version="2023.1", message="The module is private and following namespace " "`offline_transformations` will be removed in " "the future.")
def apply_pot_transformations(model, device):
    _base.apply_pot_transformations(model, device)


@deprecated(version="2023.1", message="The module is private and following namespace " "`offline_transformations` will be removed in " "the future.")
def apply_low_latency_transformation(model, use_const_initializer):
    _base.apply_low_latency_transformation(model, use_const_initializer)


@deprecated(version="2023.1", message="The module is private and following namespace " "`offline_transformations` will be removed in " "the future.")
def apply_pruning_transformation(model):
    _base.apply_pruning_transformation(model)


@deprecated(version="2023.1", message="The module is private and following namespace " "`offline_transformations` will be removed in " "the future.")
def apply_make_stateful_transformation(model, param_res_names):
    _base.apply_make_stateful_transformation(model, param_res_names)


@deprecated(version="2023.1", message="The module is private and following namespace " "`offline_transformations` will be removed in " "the future.")
def compress_model_transformation(model):
    _base.compress_model_transformation(model)


@deprecated(version="2023.1", 
            message="The module is private and following namespace " 
            "`offline_transformations` will be removed in the future. " 
            "This transformation will be enabled as a part of read_model method of ov::Core "
            "and convert method of ov::Frontend classes.")
def compress_quantize_weights_transformation(model):
    _base.compress_quantize_weights_transformation(model)


@deprecated(version="2023.1", message="The module is private and following namespace " "`offline_transformations` will be removed in " "the future.")
def convert_sequence_to_tensor_iterator_transformation(model):
    _base.convert_sequence_to_tensor_iterator_transformation(model)
