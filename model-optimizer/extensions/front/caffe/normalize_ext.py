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
from extensions.ops.normalize import NormalizeOp
from mo.front.caffe.collect_attributes import collect_attributes
from mo.front.caffe.extractors.utils import weights_biases
from mo.front.extractor import FrontExtractorOp


class NormalizeFrontExtractor(FrontExtractorOp):
    op = 'Normalize'
    enabled = True

    @classmethod
    def extract(cls, node):
        proto_layer = node.pb
        param = proto_layer.norm_param

        attrs = collect_attributes(param, enable_flattening_nested_params=True)
        attrs.update(weights_biases(False, node.model_pb))
        # update the attributes of the node
        NormalizeOp.update_node_stat(node, attrs)
        return cls.enabled
