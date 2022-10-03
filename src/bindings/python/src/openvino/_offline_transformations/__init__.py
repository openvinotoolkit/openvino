# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa

from openvino.utils import add_openvino_libs_to_path
add_openvino_libs_to_path()

from openvino.pyopenvino import get_version
__version__ = get_version()

from openvino.pyopenvino._offline_transformations import apply_fused_names_cleanup
from openvino.pyopenvino._offline_transformations import apply_moc_transformations
from openvino.pyopenvino._offline_transformations import apply_moc_legacy_transformations
from openvino.pyopenvino._offline_transformations import apply_pot_transformations
from openvino.pyopenvino._offline_transformations import apply_low_latency_transformation
from openvino.pyopenvino._offline_transformations import apply_pruning_transformation
from openvino.pyopenvino._offline_transformations import generate_mapping_file
from openvino.pyopenvino._offline_transformations import apply_make_stateful_transformation
from openvino.pyopenvino._offline_transformations import compress_model_transformation
from openvino.pyopenvino._offline_transformations import compress_quantize_weights_transformation
from openvino.pyopenvino._offline_transformations import convert_sequence_to_tensor_iterator_transformation
