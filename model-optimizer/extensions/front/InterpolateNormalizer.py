"""
 Copyright (c) 2019 Intel Corporation

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
import inspect
import logging as log

import numpy as np

from extensions.ops.elementwise import Mul, Add
from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Graph
from mo.ops.const import Const
from mo.ops.shape import Shape
from mo.ops.strided_slice import StridedSlice
from mo.utils.utils import refer_to_faq_msg


class InterpolateNormalizer(FrontReplacementOp):
    op = 'Interpolate'
    enabled = True

    def replace_sub_graph(self, graph: Graph, match: dict):
        node = match['op']

        if 1 not in node.in_ports() or node.in_port(1).disconnected():

            if node.has_valid('factor') and not node.has_valid('width') and not node.has_valid('height'):
                factor = Const(graph, {'value': np.array(node.factor)}).create_node()

                shape = Shape(graph, {'name': node.name + '/shape'}).create_node()

                begin = Const(graph, {'value': np.array([2])}).create_node()
                end = Const(graph, {'value': np.array([4])}).create_node()
                stride = Const(graph, {'value': np.array([1])}).create_node()
                ss = StridedSlice(graph, {'name': node.name + '/ss_0_port', 'begin_mask': np.array([1]),
                                          'end_mask': np.array([0]), 'new_axis_mask': np.array([0]),
                                          'shrink_axis_mask': np.array([0]),
                                          'ellipsis_mask': np.array([0])}).create_node()

                mul = Mul(graph, {'name': node.name + '/factor_mul_'}).create_node()

                source = node.in_port(0).get_connection().get_source()
                source.connect(shape.in_port(0))
                shape.out_port(0).connect(ss.in_port(0))
                begin.out_port(0).connect(ss.in_port(1))
                end.out_port(0).connect(ss.in_port(2))
                stride.out_port(0).connect(ss.in_port(3))
                ss.out_port(0).connect(mul.in_port(0))
                factor.out_port(0).connect(mul.in_port(1))

                node.add_input_port(1, skip_if_exist=True)
                assert node.in_port(1).disconnected()
                mul.out_port(0).connect(node.in_port(1))

            else:
                shape = Shape(graph, {'name': node.name + '/shape'}).create_node()

                begin = Const(graph, {'value': np.array([2])}).create_node()
                end = Const(graph, {'value': np.array([4])}).create_node()
                stride = Const(graph, {'value': np.array([1])}).create_node()
                ss = StridedSlice(graph, {'name': node.name + '/ss_0_port', 'begin_mask': np.array([1]),
                                          'end_mask': np.array([0]), 'new_axis_mask': np.array([0]),
                                          'shrink_axis_mask': np.array([0]),
                                          'ellipsis_mask': np.array([0])}).create_node()

                source = node.in_port(0).get_connection().get_source()
                source.connect(shape.in_port(0))
                shape.out_port(0).connect(ss.in_port(0))
                begin.out_port(0).connect(ss.in_port(1))
                end.out_port(0).connect(ss.in_port(2))
                stride.out_port(0).connect(ss.in_port(3))

                pads_value = node.pads_begin + node.pads_end
                pads_const = Const(graph, {'value': np.array(pads_value)}).create_node()
                add = Add(graph, {'name': node.name + '/pad_add'}).create_node()
                ss.out_port(0).connect(add.in_port(0))
                add.in_port(1).connect(pads_const.out_port(0))

                if node.soft_get('shrink_factor') != 1 and node.soft_get('zoom_factor') == 1:
                    shrink_factor = node.shrink_factor
                    if shrink_factor < 1:
                        log.error('Shrink factor should be positive in node {}'.format(node.id))
                        return None

                    const = Const(graph, {'name': node.name + '/pre_shrink_sub_const',
                                          'value': np.array(-1)}).create_node()
                    sub = Add(graph, {'name': node.name + '/pre_shrink_sub'}).create_node()
                    add.out_port(0).connect(sub.in_port(0))
                    sub.in_port(1).connect(const.out_port(0))

                    const = Const(graph, {'value': np.array(1 / shrink_factor),
                                          'name': node.name + 'shrink_factor_div_const'}).create_node()
                    div = Mul(graph, {'name': node.name + 'shrink_factor_div'}).create_node()
                    sub.out_port(0).connect(div.in_port(0))
                    div.in_port(1).connect(const.out_port(0))

                    const = Const(graph, {'name': node.name + '/shrink_factor_add_one_const', 'value': np.array(1)
                                          }).create_node()
                    add = Add(graph, {'name': node.name + '/shrink_factor_add_one'}).create_node()
                    div.out_port(0).connect(add.in_port(0))
                    const.out_port(0).connect(add.in_port(1))

                    node.add_input_port(1, skip_if_exist=True)
                    assert node.in_port(1).disconnected()
                    add.out_port(0).connect(node.in_port(1))

                elif node.soft_get('shrink_factor') == 1 and node.soft_get('zoom_factor') != 1:
                    zoom_factor = node.zoom_factor
                    if zoom_factor < 1:
                        log.error('Zoom factor should be positive in node {}'.format(node.id))
                        return None

                    node['debug_message'] = 'Interpolate layer replacer may be wrong, please, try to update it in the' \
                                            ' file (extensions/front/InterpolateNormalizer.py at the line {}).' \
                                            ''.format(inspect.currentframe().f_lineno) + refer_to_faq_msg(100)

                    # Reshape methods can be different in some cases
                    # Commented out section represents reshape that used in deeplab-caffe
                    # Uncomment the following lines, if your model was trained with deeplab-caffe
                    # or have the same reshape method
                    # const = Const(graph, {'value': np.array(-1),
                    #                       'name': node.name + 'zoom_factor_deeplab-caffe_sub_const'}).create_node()
                    # sub = Add(graph, {'name': node.name + 'zoom_factor_deeplab-caffe_sub'}).create_node()
                    # add.out_port(0).connect(sub.in_port(0))
                    # const.out_port(0).connect(sub.in_port(1))
                    #
                    # const = Const(graph, {'value': np.array(zoom_factor - 1),
                    #                       'name': node.name + 'zoom_factor_deeplab-caffe_mul_const'}).create_node()
                    # mul = Mul(graph, {'name': node.name + 'zoom_factor_deeplab-caffe_mul'}).create_node()
                    # sub.out_port(0).connect(mul.in_port(0))
                    # const.out_port(0).connect(mul.in_port(1))
                    #
                    # sum = Add(graph, {'name': node.name + 'zoom_factor_deeplab-caffe_sum'}).create_node()
                    # add.out_port(0).connect(sum.in_port(0))
                    # mul.out_port(0).connect(sum.in_port(1))
                    #
                    # node.add_input_port(1, skip_if_exist=True)
                    # assert node.in_port(1).disconnected()
                    # sum.out_port(0).connect(node.in_port(1))

                    # Comment out the following lines if you use the reshape method from previous section
                    const = Const(graph, {'value': np.array(zoom_factor),
                                          'name': node.name + '/zoom_factor_mul_const'}).create_node()
                    mul = Mul(graph, {'name': node.name + '/zoom_factor_mul'}).create_node()

                    add.out_port(0).connect(mul.in_port(0))
                    const.out_port(0).connect(mul.in_port(1))

                    node.add_input_port(1, skip_if_exist=True)
                    assert node.in_port(1).disconnected()
                    mul.out_port(0).connect(node.in_port(1))

                elif node.soft_get('width') != 0 and node.soft_get('height') != 0:
                    const = Const(graph, {'value': np.array([node.height, node.width])}).create_node()
                    node.add_input_port(1, skip_if_exist=True)
                    assert node.in_port(1).disconnected()
                    const.out_port(0).connect(node.in_port(1))

                elif node.soft_get('shrink_factor') != 1 and node.soft_get('zoom_factor') != 1:
                    shrink_factor = node.shrink_factor
                    zoom_factor = node.zoom_factor
                    if shrink_factor < 1:
                        log.error('Shrink factor should be positive in node {}'.format(node.id))
                        return None
                    if zoom_factor < 1:
                        log.error('Zoom factor should be positive in node {}'.format(node.id))
                        return None

                    const = Const(graph, {'value': np.array(-1)}).create_node()
                    sub = Add(graph, {'name': node.name + '/shrink_zoom_factor_sub'}).create_node()
                    add.out_port(0).connect(sub.in_port(0))
                    const.out_port(0).connect(sub.in_port(1))

                    const = Const(graph, {'value': np.array(1 / (shrink_factor + 1))}).create_node()
                    div = Mul(graph, {'name': node.name + '/shrink_factor_div'}).create_node()
                    sub.out_port(0).connect(div.in_port(0))
                    const.out_port(0).connect(div.in_port(1))

                    const = Const(graph, {'value': np.array(-1),
                                          'name': node.name + 'shrink_zoom_factor_sum_const'}).create_node()
                    sum = Add(graph, {'name': node.name + '/shrink_zoom_factor_sum'}).create_node()
                    div.out_port(0).connect(sum.in_port(0))
                    const.out_port(0).connect(sum.in_port(1))

                    const = Const(graph, {'value': np.array(zoom_factor - 1)}).create_node()
                    mul = Mul(graph, {'name': node.name + '/zoom_factor_mul'}).create_node()
                    sum.out_port(0).connect(mul.in_port(0))
                    const.out_port(0).connect(mul.in_port(1))

                    sum = Add(graph, {'name': node.name + '/final_shrink_zoom_factor_sum'}).create_node()
                    div.out_port(0).connect(sum.in_port(0))
                    mul.out_port(0).connect(sum.in_port(1))

                    node.add_input_port(1, skip_if_exist=True)
                    assert node.in_port(1).disconnected()
                    sum.out_port(0).connect(node.in_port(1))
        else:
            if node.soft_get('fw') == 'caffe':
                shape = Shape(graph, {'name': node.name + '/shape'}).create_node()

                begin = Const(graph, {'value': np.array([2])}).create_node()
                end = Const(graph, {'value': np.array([4])}).create_node()
                stride = Const(graph, {'value': np.array([1])}).create_node()
                ss = StridedSlice(graph, {'name': node.name + '/ss_0_port', 'begin_mask': np.array([1]),
                                          'end_mask': np.array([0]), 'new_axis_mask': np.array([0]),
                                          'shrink_axis_mask': np.array([0]),
                                          'ellipsis_mask': np.array([0])}).create_node()

                source = node.in_port(1).get_connection().get_source()
                node.in_port(1).disconnect()
                source.connect(shape.in_port(0))
                shape.out_port(0).connect(ss.in_port(0))
                begin.out_port(0).connect(ss.in_port(1))
                end.out_port(0).connect(ss.in_port(2))
                stride.out_port(0).connect(ss.in_port(3))
                ss.out_port(0).connect(node.in_port(1))
