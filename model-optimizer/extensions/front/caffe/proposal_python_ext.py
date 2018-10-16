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

from mo.front.extractor import CaffePythonFrontExtractorOp
from mo.ops.op import Op
from mo.utils.error import Error


class ProposalPythonFrontExtractor(CaffePythonFrontExtractorOp):
    op = 'rpn.proposal_layer.ProposalLayer'
    enabled = True

    @staticmethod
    def extract(node):
        param = node.pb.python_param
        attrs = CaffePythonFrontExtractorOp.parse_param_str(param.param_str)
        update_attrs = {
            'feat_stride': 16,
            'base_size': 16,
            'min_size': 16,
            'ratio': [0.5, 1, 2],
            'scale': [8, 16, 32],
            'pre_nms_topn': 6000,
            'post_nms_topn': 300,
            'nms_thresh': 0.7
        }
        update_attrs.update(attrs)
        Op.get_op_class_by_name('Proposal').update_node_stat(node, update_attrs)
        return __class__.enabled
