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

from mo.front.caffe.extractors.utils import embed_input, weights_biases
from mo.front.common.partial_infer.elemental import copy_shape_infer
from mo.front.extractor import FrontExtractorOp
from mo.ops.scale_shift import ScaleShiftOp
from mo.utils.utils import NamedAttrsClass


class ScaleFrontExtractor(FrontExtractorOp):
    op = 'scale'
    enabled = True

    @classmethod
    def extract(cls, node):
        pb = node.pb
        model = node.model_pb
        param = pb.scale_param
        attrs = {
            'axis': param.axis,
        }
        
        if model is None and len(pb.bottom) == 1:
            # default weights and biases for scale layer if the caffemodel file doesn't contain them
            model = NamedAttrsClass({'blobs': np.array([NamedAttrsClass({'data': np.array([1])}),
                                                 NamedAttrsClass({'data': np.array([0])})])})
        # scale with 1 input and 1 or 2 blobs
        if model and len(model.blobs) != 0 and len(pb.bottom) == 1:
            attrs.update(weights_biases(param.bias_term, model))
        # 2 inputs + bias
        elif len(pb.bottom) == 2 and param.bias_term:
            if model is None or len(model.blobs) == 0:
                # default bias for scale layer with 2 inputs if the caffemodel file doesn't contain them
                model = NamedAttrsClass({'blobs': np.array([NamedAttrsClass({'data': np.array([0])})])})

            embed_input(attrs, 1, 'biases', model.blobs[0].data)
        ScaleShiftOp.update_node_stat(node, attrs)
        return cls.enabled

