# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.caffe.extractors.utils import embed_input
from openvino.tools.mo.front.common.extractors.utils import layout_attrs
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.kaldi.loader.utils import read_token_value, collect_until_whitespace, find_next_tag
from openvino.tools.mo.front.kaldi.utils import read_learning_info, read_binary_matrix, read_binary_vector
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.ops.convolution import Convolution
from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.utils import refer_to_faq_msg


class ConvolutionalComponentFrontExtractor(FrontExtractorOp):
    op = 'convolutional1dcomponent'  # Naming like in Kaldi
    enabled = True
    @classmethod
    def extract(cls, node: Node) -> bool:
        """
        Extract conv parameters from node.parameters.
        node.parameters like file descriptor object.
        :param node: Convolution node
        :return:
        """
        pb = node.parameters
        read_learning_info(pb)

        kernel = read_token_value(pb, b'<PatchDim>')
        stride = read_token_value(pb, b'<PatchStep>')
        patch_stride = read_token_value(pb, b'<PatchStride>')

        token = find_next_tag(pb)
        if token == '<AppendedConv>':
            appended_conv = True
            token = find_next_tag(pb)
        if token != '<FilterParams>':
            raise Error('Can not load token {} from Kaldi model'.format(token) +
                        refer_to_faq_msg(94))
        collect_until_whitespace(pb)
        weights, weights_shape = read_binary_matrix(pb)

        collect_until_whitespace(pb)
        biases = read_binary_vector(pb)

        if (patch_stride - kernel) % stride != 0:
            raise Error(
                'Kernel size and stride does not correspond to `patch_stride` attribute of Convolution layer. ' +
                refer_to_faq_msg(93))

        output = biases.shape[0]
        if weights_shape[0] != output:
            raise Error('Weights shape does not correspond to the `output` attribute of Convolution layer. ' +
                        refer_to_faq_msg(93))

        mapping_rule = {
            'output': output,
            'patch_stride': patch_stride,
            'bias_term': None,
            'pad': int64_array([[0, 0], [0, 0], [0, 0], [0, 0]]),
            'pad_spatial_shape': int64_array([[0, 0], [0, 0]]),
            'dilation': int64_array([1, 1, 1, 1]),
            'kernel': int64_array([weights_shape[0], weights_shape[1] // kernel, 1, kernel]),
            'stride': int64_array([1, 1, 1, stride]),
            'kernel_spatial': int64_array([1, kernel]),
            'input_feature_channel': 1,
            'output_feature_channel': 0,
            'kernel_spatial_idx': [2, 3],
            'group': 1,
            'reshape_kernel': True,
            'appended_conv': appended_conv  # pylint: disable=possibly-used-before-assignment
        }

        mapping_rule.update(layout_attrs())
        embed_input(mapping_rule, 1, 'weights', weights)
        embed_input(mapping_rule, 2, 'biases', biases)

        mapping_rule['bias_addable'] = len(biases) > 0

        Convolution.update_node_stat(node, mapping_rule)
        return cls.enabled
