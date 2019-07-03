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
from mo.ops.pad import Pad


class PadFrontExtractor(FrontExtractorOp):
    op = 'Pad'
    enabled = True

    @staticmethod
    def extract(node):
        Pad.update_node_stat(node)
        return __class__.enabled


class PadV2FrontExtractor(FrontExtractorOp):
    op = 'PadV2'
    enabled = True

    @staticmethod
    def extract(node):
        return __class__.enabled


class MirrorPadFrontExtractor(FrontExtractorOp):
    op = 'MirrorPad'
    enabled = True

    @staticmethod
    def extract(node):
        Pad.update_node_stat(node, {'mode': node.pb.attr['mode'].s.decode('utf-8').lower()})
        return __class__.enabled
