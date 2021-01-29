"""
 Copyright (C) 2018-2021 Intel Corporation

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

from extensions.ops.BatchNormInference import BatchNormInference
from mo.front.caffe.extractors.utils import embed_input
from mo.front.extractor import FrontExtractorOp


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
            mean = np.array(blobs[0].data)
            variance = np.array(blobs[1].data)

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
