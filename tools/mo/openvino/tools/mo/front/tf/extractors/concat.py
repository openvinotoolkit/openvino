# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.concat import concat_infer


def tf_concat_ext(pb):
    return {
        'type': 'Concat',
        'N': pb.attr["N"].i,
        'infer': concat_infer
    }
