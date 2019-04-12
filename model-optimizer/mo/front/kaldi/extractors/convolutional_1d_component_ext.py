"""
 Copyright (c) 2018-2019 Intel Corporation

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

from mo.front.caffe.extractors.utils import embed_input
from mo.front.common.extractors.utils import layout_attrs
from mo.front.extractor import FrontExtractorOp
from mo.front.kaldi.loader.utils import read_token_value, collect_until_whitespace, find_next_tag
from mo.front.kaldi.utils import read_learning_info, read_binary_matrix, read_binary_vector
from mo.graph.graph import Node
from mo.ops.convolution import Convolution
from mo.utils.error import Error
from mo.utils.utils import refer_to_faq_msg


class ConvolutionalComponentFrontExtractor(FrontExtractorOp):
    op = 'convolutional1dcomponent'  # Naming like in Kaldi
    enabled = True

    @staticmethod
    def extract(node: Node) -> bool:
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
            'pad': np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.int64),
            'pad_spatial_shape': np.array([[0, 0], [0, 0]], dtype=np.int64),
            'dilation': np.array([1, 1, 1, 1], dtype=np.int64),
            'kernel': np.array([1, 1, 1, kernel], dtype=np.int64),
            'stride': np.array([1, 1, 1, stride], dtype=np.int64),
            'kernel_spatial': np.array([1, kernel], dtype=np.int64),
            'input_feature_channel': 1,
            'output_feature_channel': 0,
            'kernel_spatial_idx': [2, 3],
            'group': 1,
            'reshape_kernel': True,
            'appended_conv': appended_conv
        }

        mapping_rule.update(layout_attrs())
        embed_input(mapping_rule, 1, 'weights', weights)
        embed_input(mapping_rule, 2, 'biases', biases)

        mapping_rule['bias_addable'] = len(biases) > 0

        Convolution.update_node_stat(node, mapping_rule)
        return __class__.enabled
