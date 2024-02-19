# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.mxnet.extractors.batchnorm import batch_norm_ext
from openvino.tools.mo.front.mxnet.extractors.concat import concat_ext
from openvino.tools.mo.front.mxnet.extractors.l2_normalization import l2_normalization_ext
from openvino.tools.mo.front.mxnet.extractors.multibox_prior import multi_box_prior_ext
from openvino.tools.mo.front.mxnet.extractors.scaleshift import scale_shift_ext
from openvino.tools.mo.front.mxnet.extractors.slice_axis import slice_axis_ext
from openvino.tools.mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.utils import refer_to_faq_msg


def extractor_wrapper(mxnet_extractor):
    return lambda node: mxnet_extractor(get_mxnet_layer_attrs(node.symbol_dict))


mxnet_op_extractors = {
    'BatchNorm': extractor_wrapper(batch_norm_ext),
    'ScaleShift': extractor_wrapper(scale_shift_ext),
    'slice_axis': extractor_wrapper(slice_axis_ext),
    'Concat': extractor_wrapper(concat_ext),
    'L2Normalization': extractor_wrapper(l2_normalization_ext),
    '_contrib_MultiBoxPrior': extractor_wrapper(multi_box_prior_ext),
}


def common_mxnet_fields(node: Node):
    return {
        'kind': 'op',
        'name': node.id,
        'type': node['symbol_dict']['op'],
        'op': node['symbol_dict']['op'],
        'infer': None,
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
