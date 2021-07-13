# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


def tf_random_uniform_infer(node):
    node.out_port(0).data.set_shape(node.in_port(0).data.get_value())
