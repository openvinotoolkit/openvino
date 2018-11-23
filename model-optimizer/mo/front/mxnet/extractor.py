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
from mo.front.common.partial_infer.elemental import copy_shape_infer
from mo.front.mxnet.extractors.batchnorm import batch_norm_ext
from mo.front.mxnet.extractors.concat import concat_ext
from mo.front.mxnet.extractors.crop import crop_ext
from mo.front.mxnet.extractors.eltwise import eltwise_ext
from mo.front.mxnet.extractors.flatten import flatten_ext
from mo.front.mxnet.extractors.fully_connected import fully_connected_ext
from mo.front.mxnet.extractors.l2_normalization import l2_normalization_ext
from mo.front.mxnet.extractors.lrn import lrn_ext
from mo.front.mxnet.extractors.multibox_detection import multi_box_detection_ext
from mo.front.mxnet.extractors.multibox_prior import multi_box_prior_ext
from mo.front.mxnet.extractors.null import null_ext
from mo.front.mxnet.extractors.reshape import reshape_ext
from mo.front.mxnet.extractors.scaleshift import scale_shift_ext
from mo.front.mxnet.extractors.transpose import transpose_ext
from mo.front.mxnet.extractors.up_sampling import up_sampling_ext
from mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from mo.front.mxnet.extractors.slice_axis import slice_axis_ext
from mo.utils.error import Error
from mo.graph.graph import Node
from mo.utils.utils import refer_to_faq_msg

def extractor_wrapper(mxnet_extractor):
    return lambda node: mxnet_extractor(get_mxnet_layer_attrs(node.symbol_dict))


mxnet_op_extractors = {
    'BatchNorm': extractor_wrapper(batch_norm_ext),
    'Crop': extractor_wrapper(crop_ext),
    'ScaleShift': extractor_wrapper(scale_shift_ext),
    'slice_axis': extractor_wrapper(slice_axis_ext),
    'null': lambda node: null_ext(node.symbol_dict),
    'Concat': extractor_wrapper(concat_ext),
    'elemwise_add': extractor_wrapper(lambda attrs: eltwise_ext(attrs, infer=lambda a, b: a + b, op_type="sum")),
    'elemwise_mul': extractor_wrapper(lambda attrs: eltwise_ext(attrs, infer=lambda a, b: a * b, op_type="mul")),
    '_Plus': extractor_wrapper(lambda attrs: eltwise_ext(attrs, infer=lambda a, b: a + b, op_type="sum")),
    'Flatten': extractor_wrapper(flatten_ext),
    'FullyConnected': extractor_wrapper(fully_connected_ext),
    'Reshape': extractor_wrapper(reshape_ext),
    'UpSampling': extractor_wrapper(up_sampling_ext),
    'transpose': extractor_wrapper(transpose_ext),
    'LRN': extractor_wrapper(lrn_ext),
    'L2Normalization': extractor_wrapper(l2_normalization_ext),
    'Dropout': extractor_wrapper(lambda _: dict(infer=copy_shape_infer)),
    '_copy': extractor_wrapper(lambda _: dict(infer=copy_shape_infer)),
    '_contrib_MultiBoxPrior': extractor_wrapper(multi_box_prior_ext),
    '_contrib_MultiBoxDetection': extractor_wrapper(multi_box_detection_ext),
    'broadcast_add': extractor_wrapper(lambda attrs: eltwise_ext(attrs, infer=lambda a, b: a + b, op_type="sum")),
}


def common_mxnet_fields(node: Node):
    return {
        'kind': 'op',
        'name': node['symbol_dict']['name'],
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
    result.update(result_attr)
    supported = bool(result_attr)
    return supported, result
