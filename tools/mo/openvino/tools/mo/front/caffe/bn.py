# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.caffe.extractors.utils import input_as_const
from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.front.common.replacement import FrontReplacementOp
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.scale_shift import ScaleShiftOp
from openvino.tools.mo.utils.error import Error


class BNToScaleShift(FrontReplacementOp):
    """
    Replaces BN layer with ScaleShift.
    """
    op = "BN"
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        attrs = {'name': node.id + "/ScaleShift_"}

        param = graph.node[node.id]['pb'].bn_param
        pb_model = graph.node[node.id]['model_pb']

        blobs = pb_model.blobs

        if len(blobs) != 4:
            raise Error("Incorrect number of blobs in BN layer {}".format(node.id))

        mean = mo_array(blobs[0].data)
        var = mo_array(blobs[1].data)
        betta = mo_array(blobs[2].data)
        gamma = mo_array(blobs[3].data)

        gamma = gamma + np.repeat(param.eps, gamma.shape)

        scale = 1.0 / np.sqrt(gamma) * mean
        shift = var - betta * scale

        ss = ScaleShiftOp(graph, attrs)
        scale_shift = ss.create_node([node.in_node(0)])
        input_as_const(scale_shift, attrs, 1, 'weights', scale)
        input_as_const(scale_shift, attrs, 2, 'biases', shift)

        return [scale_shift.id]
