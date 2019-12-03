"""
 Copyright (c) 2019 Intel Corporation

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

from extensions.ops.random_uniform_ops import RandomUniform
from mo.front.extractor import FrontExtractorOp
from tensorflow.core.framework import types_pb2 as tf_types
from tensorflow.python.framework import tensor_util

class RandomUniformExtractor(FrontExtractorOp):
    op = 'RandomUniform'
    enabled = True

    @staticmethod
    def extract(node):
        pb = node.pb
        dtype = tf_types.DT_FLOAT
        attrs = {}
        if "T" in pb.attr.keys():
            attrs['T'] = pb.attr["T"].type
        if "dtype" in pb.attr.keys():
            dtype = pb.attr["dtype"].type
            attrs['dtype'] = dtype
        if "seed" in pb.attr.keys():
            attrs['seed'] = pb.attr["seed"].i
        if "seed2" in pb.attr.keys():
            attrs['seed2'] = pb.attr["seed2"].i
        if "minval" in pb.attr.keys():
            minval = pb.attr["minval"].f \
                if (dtype == tf_types.DT_FLOAT or dtype == tf_types.DT_DOUBLE) \
                else pb.attr["minval"].i
            attrs['minval'] = minval
        if "maxval" in pb.attr.keys():
            maxval = pb.attr["maxval"].f \
                if (dtype == tf_types.DT_FLOAT or dtype == tf_types.DT_DOUBLE) \
                else pb.attr["maxval"].i
            attrs['maxval'] = maxval
        RandomUniform.update_node_stat(node, attrs)
        return __class__.enabled
