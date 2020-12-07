"""
 Copyright (C) 2018-2020 Intel Corporation

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

from extensions.ops.BatchNormInference import BatchNormInference
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr



class BatchNormalizationExtractor(FrontExtractorOp):
    op = 'BatchNormalization'
    enabled = True

    @classmethod
    def extract(cls, node):
        attr_dict = {
           'data_format': 'NCHW',
           'eps': onnx_attr(node, 'epsilon', 'f', 1e-5),
        }
        BatchNormInference.update_node_stat(node, attr_dict)
        return cls.enabled
