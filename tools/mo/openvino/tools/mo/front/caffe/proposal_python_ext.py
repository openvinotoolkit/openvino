# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

from openvino.tools.mo.ops.proposal import ProposalOp
from openvino.tools.mo.front.extractor import CaffePythonFrontExtractorOp


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
        CaffePythonFrontExtractorOp.check_param(ProposalOp, update_attrs)
        ProposalOp.update_node_stat(node, update_attrs)

    @classmethod
    def extract(cls, node):
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
        cls.extract_proposal_params(node, defaults)
        return cls.enabled


class SSHProposalPythonFrontExtractor(CaffePythonFrontExtractorOp):
    op = 'SSH.layers.proposal_layer.ProposalLayer'
    enabled = True

    @classmethod
    def extract(cls, node):
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
        return cls.enabled
