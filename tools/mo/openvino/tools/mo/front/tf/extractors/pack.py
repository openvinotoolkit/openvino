# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


def tf_pack_ext(pb):
    assert (pb.attr["N"].i == len(pb.input))
    return {
        'axis': pb.attr["axis"].i,
        'N': pb.attr["N"].i,
        'infer': None
    }
