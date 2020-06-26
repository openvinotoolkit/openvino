"""
 Copyright (C) 2020 Intel Corporation

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

from extensions.ops.depth_to_space import DepthToSpaceOp
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr


class DepthToSpaceFrontExtractor(FrontExtractorOp):
    op = 'DepthToSpace'
    enabled = True

    @classmethod
    def extract(cls, node):
        # update the attributes of the node
        node_name = node.soft_get('name', node.id)
        block_size = onnx_attr(node, 'blocksize', 'i', default=None)
        onnx_mode = onnx_attr(node, 'mode', 's', default=b'DCR').decode()
        assert onnx_mode in ['DCR', 'CRD'], 'Unrecognized mode provided for DepthToSpace node {}'.format(node_name)
        if onnx_mode == 'DCR':
            mode = 'depth_first'
        else:
            mode = 'blocks_first'

        DepthToSpaceOp.update_node_stat(node, {'block_size': block_size, 'mode': mode})
        return cls.enabled
