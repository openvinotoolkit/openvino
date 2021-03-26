# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.common.partial_infer.elemental import copy_shape_infer


def tf_identity_ext(pb):
    return {
        'infer': copy_shape_infer
    }
