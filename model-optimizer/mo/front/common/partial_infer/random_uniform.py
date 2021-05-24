# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np


def tf_random_uniform_infer(node):
    if node.in_node(0).value is None:
        return

    output_shape = node.in_node(0).value
    node.out_node().shape = np.array(output_shape)
