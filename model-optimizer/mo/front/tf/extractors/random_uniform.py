# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.common.partial_infer.random_uniform import tf_random_uniform_infer


def tf_random_uniform_ext(pb):
    return {
        'infer': tf_random_uniform_infer
    }
