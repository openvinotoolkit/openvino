"""
 Copyright (c) 2019 Intel Corporation

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

from extensions.ops.proposal import ProposalOp
from mo.front.extractor import MXNetCustomFrontExtractorOp
from mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs


class RPNProposalMXNetFrontExtractor(MXNetCustomFrontExtractorOp):
    op = 'proposal'
    enabled = True

    def extract(self, node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        feat_stride = attrs.int("feat_stride", 16)
        ratio = attrs.tuple("ratios", float, (0.5, 1, 2))
        scale = attrs.tuple("scales", int, (4, 8, 16, 32))
        min_size = attrs.int("rpn_min_size", 16)
        pre_nms_topn = attrs.int("rpn_pre_nms_top_n", 6000)
        post_nms_topn = attrs.int("rpn_post_nms_top_n", 300)
        nms_thresh = attrs.float("threshold", 0.7)

        node_attrs = {
            'feat_stride': feat_stride,
            'base_size': 0,
            'min_size': min_size,
            'ratio': np.array(ratio),
            'scale': np.array(scale),
            'pre_nms_topn': pre_nms_topn,
            'post_nms_topn': post_nms_topn,
            'nms_thresh': nms_thresh,
            'for_deformable': 1,
        }

        ProposalOp.update_node_stat(node, node_attrs)
        return (True, node_attrs)
