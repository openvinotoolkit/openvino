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
import numpy as np

from extensions.ops.argmax import ArgMaxOp
from mo.front.extractor import FrontExtractorOp
from mo.front.tf.extractors.utils import tf_dtype_extractor


class ArgMaxFrontExtractor(FrontExtractorOp):
    op = 'ArgMax'
    enabled = True

    @classmethod
    def extract(cls, node):
        ArgMaxOp.update_node_stat(node, {'out_max_val': 0, 'top_k': 1, 'axis': None,
                                         'dim_attrs': ['axis'], 'keepdims': 0, 'remove_values_output': True,
                                         'output_type': tf_dtype_extractor(node.pb.attr['out_type'].type, np.int64),
                                         })
        return cls.enabled
