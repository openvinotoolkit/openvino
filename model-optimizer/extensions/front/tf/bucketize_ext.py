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

from extensions.ops.bucketize import Bucketize
from mo.front.extractor import FrontExtractorOp


class BucketizeFrontExtractor(FrontExtractorOp):
    op = 'Bucketize'
    enabled = True

    @classmethod
    def extract(cls, node):
        boundaries = np.array(node.pb.attr['boundaries'].list.f, dtype=np.float)
        Bucketize.update_node_stat(node, {'boundaries': boundaries, 'with_right_bound': False, 'output_type': np.int32})
        return cls.enabled
