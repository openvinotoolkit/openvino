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

import logging as log

from mo.front.extractor import FrontExtractorOp
from mo.ops.roipooling import ROIPooling


class CropAndResizeFrontExtractor(FrontExtractorOp):
    op = 'CropAndResize'
    enabled = True

    @staticmethod
    def extract(node):
        # update the attributes of the node and force 'op' to be 'CropAndResize' so extension that merges two of its
        # inputs would be called
        method = node.pb.attr['method'].s.decode('utf-8')
        if method != 'bilinear':
            log.warning(
                'The crop and resize method "{}" for node "{}" is not supported.'.format(method, node.soft_get('name')))
            return False
        ROIPooling.update_node_stat(node, {'spatial_scale': 1, 'op': 'CropAndResize', 'method': method})
        return __class__.enabled
