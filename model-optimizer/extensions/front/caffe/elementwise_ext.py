# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from extensions.ops.elementwise import Add, Mul, Maximum
from mo.front.caffe.collect_attributes import merge_attrs
from mo.front.caffe.extractors.utils import embed_input
from mo.front.common.extractors.utils import layout_attrs
from mo.front.extractor import FrontExtractorOp
from mo.graph.graph import Node
from mo.ops.eltwise_n import EltwiseNMul, EltwiseNAdd, EltwiseNMax
from mo.ops.power import AttributedPower


class BiasToAdd(FrontExtractorOp):
    """
    Replaces Bias layer with Add.
    """
    op = "Bias"
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        attrs = {'axis': node.pb.bias_param.axis}
        embed_input(attrs, 1, 'bias', node.model_pb.blobs[0].data, 'biases')

        Add.update_node_stat(node, attrs)

        return cls.enabled


class EltwiseExtractor(FrontExtractorOp):
    op = 'Eltwise'
    enabled = True

    @classmethod
    def extract(cls, node):
        proto_layer = node.pb
        param = proto_layer.eltwise_param

        input_len = len(node.in_edges())

        eltwise_caffe_map = {
            0: EltwiseNMul if input_len > 2 else Mul,
            1: EltwiseNAdd if input_len > 2 else Add,
            2: EltwiseNMax if input_len > 2 else Maximum,
        }

        operation = int(param.operation)
        if operation not in eltwise_caffe_map:
            raise Exception('Unsupported type of operation in Eltwise layer: ' + node.name)

        lin_op_class = eltwise_caffe_map[operation]

        mapping_rule = merge_attrs(param, {'coeff': np.array(param.coeff)})
        mapping_rule.update(layout_attrs())

        assert len(param.coeff) <= input_len

        lin_op_class.update_node_stat(node, mapping_rule)
        return cls.enabled


class PowerExtractor(FrontExtractorOp):
    op = 'power'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        pb = node.pb
        assert pb, 'Protobuf layer can not be empty'
        param = pb.power_param
        attrs = {
            'output_spatial_shape': None,
            'power': param.power,
            'scale': param.scale,
            'shift': param.shift,
        }
        AttributedPower.update_node_stat(node, attrs)
        return cls.enabled
