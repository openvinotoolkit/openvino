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


import numpy as np

from extensions.ops.lstm_sequence import LSTMSequence
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr


class LSTMFrontExtractor(FrontExtractorOp):
    op = 'LSTM'
    enabled = True

    @staticmethod
    def extract(node):

        attrs = {
            'hidden_size': np.array(onnx_attr(node, 'hidden_size', 'i'), dtype=np.int64),
            'batch_dim': 1,
            'sequence_dim': 0,
            'blobs_wrb': True,
        }

        LSTMSequence.update_node_stat(node, attrs)
        return __class__.enabled
