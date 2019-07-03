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

from mo.front.extractor import FrontExtractorOp
from mo.graph.graph import Node
from extensions.ops.stop_gradient import StopGradientOp


class StopGradientExtractor(FrontExtractorOp):
    op = 'StopGradient'
    enabled = True

    @staticmethod
    def extract(node: Node):
        StopGradientOp.update_node_stat(node, {})
        return __class__.enabled