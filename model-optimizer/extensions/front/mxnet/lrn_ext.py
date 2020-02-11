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
from mo.front.extractor import FrontExtractorOp
from mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from mo.ops.lrn import AttributedLRN


class LRNExtractor(FrontExtractorOp):
    op = 'LRN'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)

        alpha = attrs.float("alpha", 0.0001)
        beta = attrs.float("beta", 0.75)
        knorm = attrs.float("knorm", 2.0)
        nsize = attrs.int("nsize", None)

        AttributedLRN.update_node_stat(node, {
            'alpha': alpha,
            'beta': beta,
            'bias': knorm,
            'local_size': nsize,
        })
        return cls.enabled
