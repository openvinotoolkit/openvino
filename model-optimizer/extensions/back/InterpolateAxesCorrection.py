"""
 Copyright (c) 2020 Intel Corporation

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

from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph
from extensions.ops.interpolate import Interpolate


class InterpolateV4AxesCorrection(BackReplacementPattern):
    """
    This transformation converts data from 'axes' input of Interpolate-4 from NHWC to NCHW layout.
    """
    enabled = True
    graph_condition = [lambda graph: graph.graph['fw'] == 'tf' and not graph.graph['cmd_params'].disable_nhwc_to_nchw]

    def run_before(self):
        from extensions.back.InterpolateReshape import InterpolateConcat
        return [InterpolateConcat]

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(type='Interpolate', version='opset4'):
            input_shape = node.in_port(0).data.get_shape()
            assert input_shape is not None, \
                'Interpolate-4 node with name {} has None as input shape'.format(node.soft_get('name', node.id))
            input_rank = len(input_shape)
            assert input_rank in [4, 5], \
                'The transformation InterpolateAxesCorrection supports only 4D and 5D inputs'
            axes = Interpolate.get_axes(node)
            if sorted(axes) != list(range(0, input_rank)):
                corrected_axes = get_correct_axes(axes, input_rank)
                node.in_port(3).data.set_value(corrected_axes)


def get_correct_axes(axes: np.array, input_rank) -> np.array:
    """
    Conversion of axes of Interpolate-4 from NHWC to NCHW layout
    :param axes: axes to convert
    :param input_rank: rank of input data of Interpolate-4
    :return: axes after conversion
    """
    axes_rank = len(axes)
    if axes_rank == input_rank:
        return axes

    corrected_axes = np.zeros(axes_rank).astype(np.int64)
    for i, axis in enumerate(axes):
        if axis == 0:
            corrected_axis = 0
        elif axis == input_rank - 1:
            corrected_axis = 1
        else:
            corrected_axis = axis + 1

        corrected_axes[i] = corrected_axis

    return corrected_axes

