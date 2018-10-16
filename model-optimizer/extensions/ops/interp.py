"""
 Copyright (c) 2017-2018 Intel Corporation

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

import networkx as nx
import numpy as np

from extensions.ops.resize_factor_utils import factor_update
from mo.graph.graph import Node
from mo.ops.op import Op
from mo.utils.utils import refer_to_faq_msg


class InterpOp(Op):
    op = 'Interp'

    def __init__(self, graph: nx.MultiDiGraph, attrs: dict):
        mandatory_props = {
            'type': __class__.op,
            'op': __class__.op,
            'factor': None,
            'infer': InterpOp.interp_infer
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return [
            'height',
            'width',
            'zoom_factor',
            'shrink_factor',
            'factor',  # float factor required by IE shape inference
            'pad_beg',
            'pad_end'
        ]

    @staticmethod
    def interp_infer(node: Node):
        if len(node.in_nodes()) == 2:
            src_shape = node.in_node(0).shape
            dst_shape = node.in_node(1).value
            if src_shape is None or dst_shape is None or len(src_shape) != 4 or len(dst_shape) != 2:
                log.error(
                    'Node {} with op {} cannot be converted to Resample layer because there is no enough info about src/dst shapes:' +
                    ', src_shape = {}, dst_shape = {}'.format(node.name, node.op, src_shape, dst_shape))
                node.type = None  # prevent translation to a valid IE layer
                return
            out_shape = src_shape.copy()
            log.warning('This works only for NHWC layout. If real layout is different, result will be incorrect.')
            out_shape[1] = dst_shape[0]
            out_shape[2] = dst_shape[1]
            real_factor = [float(out_shape[1])/src_shape[1], float(out_shape[2])/src_shape[2]]
            node.factor = factor_update(
                node.factor,
                real_factor,
                [src_shape[1], src_shape[2]],
                [out_shape[1], out_shape[2]],
                node.soft_get('name')
            )
            node.out_node().shape = out_shape
            node.graph.remove_edge(node.in_node(1).id, node.id)
        else:
            outn = node.out_node(0)

            in_shape = node.in_node(0)
            num_ = in_shape.shape[0]
            channels_ = in_shape.shape[1]
            height_in_ = in_shape.shape[2]
            width_in_ = in_shape.shape[3]

            height_out_ = height_in_ + node.pad_beg + node.pad_end
            width_out_ = width_in_ + node.pad_beg + node.pad_end

            if node.shrink_factor != 1 and node.zoom_factor == 1:
                shrink_factor = node.shrink_factor
                if shrink_factor < 1:
                    log.error('Shrink factor should be positive in node {}'.format(node.id))
                    return None
                height_out_ = (height_out_ - 1) / shrink_factor + 1
                width_out_ = (width_out_ - 1) / shrink_factor + 1
            elif node.shrink_factor == 1 and node.zoom_factor != 1:
                zoom_factor = node.zoom_factor
                if zoom_factor < 1:
                    log.error('Zoom factor should be positive in node {}'.format(node.id))
                    return None

                node['debug_message'] = 'Interp layer shape inference function may be wrong, please, try to update ' \
                                        'layer shape inference function in the file (extensions/ops/interp.op at the ' \
                                        'line {}).'.format(inspect.currentframe().f_lineno) + refer_to_faq_msg(100)
                # Reshape methods can be different in some cases
                # Commented out section represents reshape that used in deeplab-caffe
                # Uncomment the following lines, if your model was trained with deeplab-caffe
                # or have the same reshape method
                # height_out_ = height_out_ + (height_out_ - 1) * (zoom_factor - 1)
                # width_out_ = width_out_ + (width_out_ - 1) * (zoom_factor - 1)

                # Comment out the following lines if you use the reshape method from previous section
                height_out_ = height_out_ * zoom_factor
                width_out_ = width_out_ * zoom_factor
            elif node.width != 0 and node.height != 0:
                height_out_ = node.height
                width_out_ = node.width
            elif node.shrink_factor != 1 and node.zoom_factor != 1:
                shrink_factor = node.shrink_factor
                zoom_factor = node.zoom_factor
                if shrink_factor < 1:
                    log.error('Shrink factor should be positive in node {}'.format(node.id))
                    return None
                if zoom_factor < 1:
                    log.error('Zoom factor should be positive in node {}'.format(node.id))
                    return None
                height_out_ = (height_out_ - 1) / shrink_factor + 1
                width_out_ = (width_out_ - 1) / shrink_factor + 1
                height_out_ = height_out_ + (height_out_ - 1) * (zoom_factor - 1)
                width_out_ = width_out_ + (width_out_ - 1) * (zoom_factor - 1)

            outn.shape = np.array([num_, channels_, height_out_, width_out_], dtype=np.int64)
