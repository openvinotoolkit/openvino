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
import logging as log

from mo.front.extractor import CaffePythonFrontExtractorOp
from mo.ops.op import Op


class ProposalPythonFrontExtractor(CaffePythonFrontExtractorOp):
    op = 'rpn.proposal_layer.ProposalLayer'
    enabled = True

    @staticmethod
    def extract_proposal_params(node, defaults):
        param = node.pb.python_param
        attrs = CaffePythonFrontExtractorOp.parse_param_str(param.param_str)
        update_attrs = defaults
        if 'ratios' in attrs and 'ratio' in attrs:
            log.error('Both ratios and ratio found, value of ratios will be used', extra={'is_warning': True})
        if 'scales' in attrs and 'scale' in attrs:
            log.error('Both scales and scale found, value of scales will be used', extra={'is_warning': True})

        if 'ratios' in attrs:
            attrs['ratio'] = attrs['ratios']
            del attrs['ratios']
        if 'scales' in attrs:
            attrs['scale'] = attrs['scales']
            del attrs['scales']

        update_attrs.update(attrs)
        CaffePythonFrontExtractorOp.check_param(Op.get_op_class_by_name('Proposal'), update_attrs)
        Op.get_op_class_by_name('Proposal').update_node_stat(node, update_attrs)

    @staticmethod
    def extract(node):
        defaults = {
            'feat_stride': 16,
            'base_size': 16,
            'min_size': 16,
            'ratio': [0.5, 1, 2],
            'scale': [8, 16, 32],
            'pre_nms_topn': 6000,
            'post_nms_topn': 300,
            'nms_thresh': 0.7
        }
        __class__.extract_proposal_params(node, defaults)
        return __class__.enabled


class SSHProposalPythonFrontExtractor(CaffePythonFrontExtractorOp):
    op = 'SSH.layers.proposal_layer.ProposalLayer'
    enabled = True

    @staticmethod
    def extract(node):
        defaults = {
            'feat_stride': 16,
            'base_size': 16,
            'min_size': 16,
            'ratio': [0.5, 1, 2],
            'scale': [8, 16, 32],
            'pre_nms_topn': 1000,
            'post_nms_topn': 1000,
            'nms_thresh': 1.0
        }
        ProposalPythonFrontExtractor.extract_proposal_params(node, defaults)
        return __class__.enabled
