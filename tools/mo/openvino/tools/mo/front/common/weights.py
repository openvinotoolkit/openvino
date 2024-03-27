# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.utils.graph import Graph


def swap_weights_xy(graph: Graph, nodes: list):
    from openvino.tools.mo.front.tf.ObjectDetectionAPI import swap_weights_xy as new_swap_weights_xy
    new_swap_weights_xy(graph, nodes)
