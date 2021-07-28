# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.common.partial_infer.concat import tf_pack_infer


def tf_pack_ext(pb):
    assert (pb.attr["N"].i == len(pb.input))
    return {
        'axis': pb.attr["axis"].i,
        'N': pb.attr["N"].i,
        'infer': tf_pack_infer
    }
