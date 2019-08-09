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

from extensions.ops.reorgyolo import ReorgYoloOp
from mo.front.extractor import FrontExtractorOp


class ExtractImagePatchesExtractor(FrontExtractorOp):
    op = 'ExtractImagePatches'
    enabled = True

    @staticmethod
    def extract(node):
        node['batch_dims'] = 0
        node['channel_dims'] = 3
        node['spatial_dims'] = [1, 2]
        ReorgYoloOp.update_node_stat(node, {'stride': np.array(node.pb.attr['strides'].list.i[1])})
        return __class__.enabled
