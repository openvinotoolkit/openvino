# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa

from openvino._pyopenvino import get_version

__version__ = get_version()

from openvino._pyopenvino._offline_transformations import (
    apply_fused_names_cleanup,
    apply_low_latency_transformation,
    apply_make_stateful_transformation,
    apply_moc_legacy_transformations,
    apply_moc_transformations,
    apply_pruning_transformation,
    compress_model_transformation,
    compress_quantize_weights_transformation,
    convert_sequence_to_tensor_iterator_transformation,
    paged_attention_transformation,
)
