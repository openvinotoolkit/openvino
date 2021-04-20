# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.mxnet.extractors.batchnorm import batch_norm_ext
from mo.front.mxnet.extractors.concat import concat_ext
from mo.front.mxnet.extractors.l2_normalization import l2_normalization_ext
from mo.front.mxnet.extractors.multibox_prior import multi_box_prior_ext
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


def mxnet_op_extractor(node: Node, name_to_extractor_map: dict):
    result = common_mxnet_fields(node)
    node.graph.node[node.id].update(result)

    supported = False
    op = result['op']
    if op in name_to_extractor_map:
        result_attr = name_to_extractor_map[op](node)
        if result_attr is not None:
            result.update(result_attr)
        supported = True
    return supported, result
