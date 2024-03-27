# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.ops.BatchNormInference import BatchNormInference
from openvino.tools.mo.front.caffe.extractors.utils import embed_input
from openvino.tools.mo.front.extractor import FrontExtractorOp


class BatchNormalizationExtractor(FrontExtractorOp):
    op = 'batchnorm'
    enabled = True

    @classmethod
    def extract(cls, node):
        eps = node.pb.batch_norm_param.eps
        attrs = {
           'eps': eps
        }
        pb_model = None if not node.soft_get('model_pb', None) else node.model_pb
        if pb_model:
            blobs = pb_model.blobs
            assert len(blobs) >= 2, 'BatchNorm accepts not less then two input blobs'
            mean = mo_array(blobs[0].data)
            variance = mo_array(blobs[1].data)

            if len(blobs) == 3:
                scale = blobs[2].data[0]
                if scale != 0:
                    scale = 1.0 / scale
                mean *= scale
                variance *= scale

            embed_input(attrs, 1, 'gamma', np.ones(mean.shape), 'gamma')
            embed_input(attrs, 2, 'beta', np.zeros(variance.shape), 'beta')
            embed_input(attrs, 3, 'mean', mean, 'biases')
            embed_input(attrs, 4, 'variance', variance, 'weights')

        BatchNormInference.update_node_stat(node, attrs)
        return cls.enabled
