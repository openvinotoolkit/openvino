# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.caffe.conv_ext import conv_create_attrs, conv_set_params
from openvino.tools.mo.front.caffe.extractors.utils import weights_biases
from openvino.tools.mo.front.common.extractors.utils import layout_attrs
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.ops.convolution import Convolution
from openvino.tools.mo.utils.error import Error


class ConvFrontExtractor(FrontExtractorOp):
    op = 'ConvolutionBinary'
    enabled = True

    @classmethod
    def extract(cls, node):
        proto_layer, model_layer = node.pb, node.model_pb

        if not proto_layer:
            raise Error('Protobuf layer can not be empty')

        conv_param = proto_layer.convolution_param
        conv_type = 'ConvND' if len(proto_layer.bottom) > 1 else 'Conv2D'

        params = conv_set_params(conv_param, conv_type)
        attrs = conv_create_attrs(params)
        attrs.update({'op': __class__.op,
                      'get_group': lambda node: node.group,
                      'get_output_feature_dim': lambda node: node.output,
                      'weights_index': 1 if conv_type == 'Conv2D' else 2
                      })

        # Embed weights and biases as attributes
        # It will be moved to a separate nodes in special pass
        attrs.update(
            weights_biases(conv_param.bias_term, model_layer, start_index=len(proto_layer.bottom), proto=conv_param))
        attrs.update(layout_attrs())

        # update the attributes of the node
        Convolution.update_node_stat(node, attrs)
        return cls.enabled

