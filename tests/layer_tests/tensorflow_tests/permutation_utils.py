# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from mo.ops.op import PermuteAttrs


def permute_nhwc_to_nchw(shape):
    perm = PermuteAttrs.get_nhwc_to_nchw_permutation(len(shape)).perm
    new_shape = np.array(shape)[perm]
    return new_shape


def permute_nchw_to_nhwc(shape):
    perm = PermuteAttrs.get_nchw_to_nhwc_permutation(len(shape)).perm
    new_shape = np.array(shape)[perm]
    return new_shape


def permute_axis(axis, permutation_inv):
    return permutation_inv[axis]
