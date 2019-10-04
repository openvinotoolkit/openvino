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

from mo.front.mxnet.extractors.batchnorm import batch_norm_ext
from mo.front.mxnet.extractors.concat import concat_ext
from mo.front.mxnet.extractors.crop import crop_ext
from mo.front.mxnet.extractors.fully_connected import fully_connected_ext
from mo.front.mxnet.extractors.l2_normalization import l2_normalization_ext
from mo.front.mxnet.extractors.lrn import lrn_ext
from mo.front.mxnet.extractors.multibox_detection import multi_box_detection_ext
from mo.front.mxnet.extractors.multibox_prior import multi_box_prior_ext
from mo.front.mxnet.extractors.null import null_ext
from mo.front.mxnet.extractors.scaleshift import scale_shift_ext
from mo.front.mxnet.extractors.slice_axis import slice_axis_ext
from mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from mo.graph.graph import Node
from mo.utils.error import Error
from mo.utils.utils import refer_to_faq_msg


def extractor_wrapper(mxnet_extractor):
    return lambda node: mxnet_extractor(get_mxnet_layer_attrs(node.symbol_dict))


mxnet_op_extractors = {
    'BatchNorm': extractor_wrapper(batch_norm_ext),
    'ScaleShift': extractor_wrapper(scale_shift_ext),
    'slice_axis': extractor_wrapper(slice_axis_ext),
    'null': lambda node: null_ext(node.symbol_dict),
    'Concat': extractor_wrapper(concat_ext),
    'FullyConnected': extractor_wrapper(fully_connected_ext),
    'LRN': extractor_wrapper(lrn_ext),
    'L2Normalization': extractor_wrapper(l2_normalization_ext),
    '_contrib_MultiBoxPrior': extractor_wrapper(multi_box_prior_ext),
    '_contrib_MultiBoxDetection': extractor_wrapper(multi_box_detection_ext),
}


def common_mxnet_fields(node: Node):
    return {
        'kind': 'op',
        'name': node.id,
        'type': node['symbol_dict']['op'],
        'op': node['symbol_dict']['op'],
        'infer': None,
        'precision': 'FP32'
    }


def mxnet_op_extractor(node: Node):
    result = common_mxnet_fields(node)
    op = result['op']
    if op not in mxnet_op_extractors:
        raise Error(
            "Operation '{}' not supported. Please register it as custom op. " +
            refer_to_faq_msg(86),
            op)
    result_attr = mxnet_op_extractors[op](node)

    if result_attr is None:
        raise Error('Model Optimizer does not support layer "{}". Please, implement extension. '.format(node.name) +
                    refer_to_faq_msg(45))

    result.update(result_attr)
    supported = bool(result_attr)
    return supported, result
