"""
 Copyright (C) 2018-2021 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import numpy as np

from mo.front.caffe.extractors.utils import input_as_const
from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Node, Graph
from mo.ops.scale_shift import ScaleShiftOp
from mo.utils.error import Error


class BNToScaleShift(FrontReplacementOp):
    """
    Replaces BN layer with ScaleShift.
    """
    op = 'BatchNormInference'
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        attrs = {'name': node.id + "/ScaleShift_"}

        param = graph.node[node.id]['pb'].bn_param
        pb_model = graph.node[node.id]['model_pb']
        blobs = pb_model.blobs

        if len(blobs) != 4:
            raise Error("Incorrect number of blobs in BN layer {}".format(node.id))

        mean = np.array(blobs[0].data)
        var = np.array(blobs[1].data)
        betta = np.array(blobs[2].data)
        gamma = np.array(blobs[3].data)

        gamma = gamma + np.repeat(param.eps, gamma.shape)

        scale = 1.0 / np.sqrt(gamma) * mean
        shift = var - betta * scale

        ss = ScaleShiftOp(graph, attrs)
        scale_shift = ss.create_node([node.in_node(0)])
        input_as_const(scale_shift, attrs, 1, 'weights', scale)
        input_as_const(scale_shift, attrs, 2, 'biases', shift)

        return [scale_shift.id]
