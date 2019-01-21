"""
 Copyright (c) 2018 Intel Corporation

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

from mo.graph.graph import Node
from mo.front.common.partial_infer.elemental import copy_shape_infer, single_output_infer
from mo.utils.error import Error
from mo.utils.utils import refer_to_faq_msg


def node_pb_arg(pb_extractor):
    return lambda node: pb_extractor(node.parameters)


kaldi_type_extractors = {}


def common_kaldi_fields(node: Node) -> dict:
    layer_type = node.op
    return {
        'kind': 'op',
        'name': node.id,
        'op': layer_type,
        # generic code relies on op; it should be overridden by specific op extractor
        'infer': None,
        'precision': 'FP32'
    }


def kaldi_extractor(node: Node) -> (bool, dict):
    result = common_kaldi_fields(node)
    layer_type = result['op']
    if layer_type not in kaldi_type_extractors:
        raise Error('Found unsupported layer {}. '.format(node.id) +
                    'Model Optimizer does not support this layer type: {}. '.format(layer_type) +
                    'Please, implement extension. ' +
                    refer_to_faq_msg(45))

    result.update(kaldi_type_extractors[layer_type](node))
    supported = True

    return supported, result
