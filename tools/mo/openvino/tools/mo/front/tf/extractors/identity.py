# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.elemental import copy_shape_infer


def tf_identity_ext(pb):
    return {
        'infer': copy_shape_infer
    }
