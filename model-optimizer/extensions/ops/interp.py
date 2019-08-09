"""
 Copyright (c) 2017-2019 Intel Corporation

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

from extensions.ops.resize_factor_utils import factor_update
from mo.front.common.layout import get_batch_dim, get_features_dim, get_height_dim, get_width_dim, shape_for_layout
from mo.graph.graph import Node, Graph
from mo.ops.op import Op
from mo.utils.utils import refer_to_faq_msg


class InterpOp(Op):
    op = 'Interp'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': __class__.op,
            'op': __class__.op,
            'factor': None,
            'align_corners': 1,
            'parse_2nd_input': 'value',
            'in_ports_count': 2,
            'out_ports_count': 1,
            'infer': None
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
            'pad_end',
            'align_corners'
        ]

    @staticmethod
    def interp_infer(node: Node):
        layout = node.graph.graph['layout']
        assert len(layout) == 4
        if len(node.in_nodes()) == 2:
            src_shape = node.in_node(0).shape
            dst_shape = node.in_node(1).shape

            # in Caffe can be 2 inputs too, but shape should be got from shape of the second input
            if node.parse_2nd_input == 'shape':
                dst_shape = [dst_shape[get_height_dim(layout, 4)], dst_shape[get_width_dim(layout, 4)]]
            else:
                # it is TF case
                dst_shape = node.in_node(1).value

            if src_shape is None or dst_shape is None or len(src_shape) != 4 or len(dst_shape) != 2:
                log.error(
                    'Node {} with op {} cannot be converted to Resample layer because there is no enough info about '
                    'src/dst shapes: src_shape = {}, dst_shape = {}'.format(node.name, node.op, src_shape, dst_shape))
                node.type = None  # prevent translation to a valid IE layer
                return
            in_height = src_shape[get_height_dim(layout, 4)]
            in_width = src_shape[get_width_dim(layout, 4)]
            out_height = dst_shape[0]
            out_width = dst_shape[1]

            node.factor = factor_update(
                node.factor,
                [float(out_height) / in_height, float(out_width) / in_width],
                [in_height, in_width],
                [out_height, out_width],
                node.soft_get('name')
            )

            if node.factor is None:
                node['width'] = out_width
                node['height'] = out_height

            node.out_node().shape = shape_for_layout(layout,
                                                     batch=src_shape[get_batch_dim(layout, 4)],
                                                     features=src_shape[get_features_dim(layout, 4)],
                                                     height=out_height,
                                                     width=out_width)
            node.graph.remove_edge(node.in_node(1).id, node.id)
        else:
            outn = node.out_node(0)

            in_shape = node.in_node(0)
            num_ = in_shape.shape[get_batch_dim(layout, 4)]
            channels_ = in_shape.shape[get_features_dim(layout, 4)]
            height_in_ = in_shape.shape[get_height_dim(layout, 4)]
            width_in_ = in_shape.shape[get_width_dim(layout, 4)]

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

            outn.shape = shape_for_layout(layout,
                                          batch=num_,
                                          features=channels_,
                                          height=height_out_,
                                          width=width_out_)
