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
import logging as log

from mo.front.extractor import FrontExtractorOp
from mo.ops.reshape import Reshape


class ReshapeFrontExtractor(FrontExtractorOp):
    op = 'reshape'
    enabled = True

    @classmethod
    def extract(cls, node):
        param = node.pb.reshape_param

        if param.axis != 0:
            log.error('The operation "Reshape" has attribute "axis" with unsupported value "{}"'.format(param['axis']))
            return False

        if param.num_axes != -1:
            log.error('The operation "Reshape" has attribute "num_axes" with unsupported value "{}"'.format(
                param['num_axes']))
            return False

        Reshape.update_node_stat(node, {
            'dim': list(param.shape.dim),
        })
        return cls.enabled
