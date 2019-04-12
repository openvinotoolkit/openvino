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

import numpy as np


def tf_const_infer(node):
    output = node.out_node()
    output.shape = np.array(node.shape, np.int64)
    # no broadcast, copy as-is (tensor or scalar) or apply broadcast depending on value and shape
    output.value = node.value if isinstance(node.value, np.ndarray) or len(node.shape) == 0 else np.full(node.shape,
                                                                                                         node.value)
