# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.concat import concat_infer


def concat_ext(attrs):
    node_attrs = {
        'type': 'Concat',
        'axis': attrs.int("dim", 1),
        'infer': concat_infer
    }
    return node_attrs
