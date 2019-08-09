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
from extensions.ops.transpose import Transpose
from mo.front.extractor import FrontExtractorOp


class TransposeFrontExtractorTF(FrontExtractorOp):
    op = 'Transpose'
    enabled = True

    @staticmethod
    def extract(node):
        Transpose.update_node_stat(node, {'order': None})
        return __class__.enabled
