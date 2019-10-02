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

from mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from mo.front.extractor import FrontExtractorOp
from mo.ops.op import Op


class ProposalFrontExtractor(FrontExtractorOp):
    op = '_contrib_Proposal'
    enabled = True

    @staticmethod
    def extract(node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        pre_nms_topn = attrs.int('rpn_pre_nms_top_n', 6000)
        post_nms_topn = attrs.int('rpn_post_nms_top_n', 300)
        nms_thresh = attrs.float('threshold', 0.7)
        min_size = attrs.int('rpn_min_size', 16)
        scale = attrs.tuple("scales", float, (4, 8, 16, 32))
        ratio = attrs.tuple("ratios", float, (0.5, 1, 2))
        feat_stride = attrs.int('feature_stride', 16)

        update_attrs = {
            'feat_stride': feat_stride,
            'ratio': np.array(ratio),
            'min_size': min_size,
            'scale': np.array(scale),
            'pre_nms_topn': pre_nms_topn,
            'post_nms_topn': post_nms_topn,
            'nms_thresh': nms_thresh,
            'base_size': feat_stride
        }

        # update the attributes of the node
        Op.get_op_class_by_name('Proposal').update_node_stat(node, update_attrs)
        return __class__.enabled
