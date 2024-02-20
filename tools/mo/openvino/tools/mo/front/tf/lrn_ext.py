# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.ops.lrn import AttributedLRN


class LRNExtractor(FrontExtractorOp):
    """
        TF and IE(CAFFE) parameters in LRN differs in several places :
            region (IE) : in TF there is no such parameter, they just use last dimension (feature dimension in case of NHWC)
            local-size (IE) : it's the size of 1D vector in Caffe. In TF they have 'depth_radius' that eq
            '(local-size * 2) + 1'
            alpha (IE) : in Caffe 'alpha' divides on local-size, so we should multiply alpha on local-size

        Caffe ref : http://caffe.berkeleyvision.org/tutorial/layers/lrn.html
        TF ref : https://www.tensorflow.org/api_docs/python/tf/nn/local_response_normalization
    """
    op = 'LRN'
    enabled = True

    @classmethod
    def extract(cls, node):
        pb = node.pb
        AttributedLRN.update_node_stat(node, {
            'alpha': pb.attr['alpha'].f * (2. * pb.attr['depth_radius'].i + 1.),
            'beta': pb.attr['beta'].f,
            'bias': pb.attr['bias'].f,
            'local_size': (2 * pb.attr['depth_radius'].i + 1),
        })
        return cls.enabled
