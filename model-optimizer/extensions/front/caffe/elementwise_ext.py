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
import numpy as np

from mo.front.caffe.collect_attributes import merge_attrs
from mo.front.caffe.extractors.utils import embed_input
from mo.front.common.extractors.utils import layout_attrs
from mo.front.extractor import FrontExtractorOp
from mo.graph.graph import Node
from extensions.ops.elementwise import Add, Mul, Maximum
from mo.ops.power import Power


class BiasToAdd(FrontExtractorOp):
    """
    Replaces Bias layer with Add.
    """
    op = "Bias"
    enabled = True

    @staticmethod
    def extract(node: Node):
        attrs = {'axis': node.pb.bias_param.axis}
        embed_input(attrs, 1, 'bias', node.model_pb.blobs[0].data, 'biases')

        Add.update_node_stat(node, attrs)

        return __class__.enabled


class EltwiseExtractor(FrontExtractorOp):
    op = 'Eltwise'
    enabled = True

    @staticmethod
    def extract(node):
        proto_layer = node.pb
        param = proto_layer.eltwise_param

        eltwise_caffe_map = {
            0: Mul,
            1: Add,
            2: Maximum,
        }

        operation = int(param.operation)
        if operation not in eltwise_caffe_map:
            raise Exception('Unsupported type of operation in Eltwise layer: ' + node.name)

        lin_op_class = eltwise_caffe_map[operation]

        mapping_rule = merge_attrs(param, {'coeff': np.array(param.coeff)})
        mapping_rule.update(layout_attrs())

        assert len(param.coeff) <= len(node.in_edges())

        lin_op_class.update_node_stat(node, mapping_rule)
        return __class__.enabled


class PowerExtractor(FrontExtractorOp):
    op = 'power'
    enabled = True

    @staticmethod
    def extract(node: Node):
        pb = node.pb
        assert pb, 'Protobuf layer can not be empty'
        param = pb.power_param
        attrs = {
            'output_spatial_shape': None,
            'power': param.power,
            'scale': param.scale,
            'shift': param.shift,
        }
        Power.update_node_stat(node, attrs)
        return __class__.enabled
