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

from extensions.ops.LookupTableInsert import LookupTableInsert
from mo.front.extractor import FrontExtractorOp


class LookupTableInsertFrontExtractor(FrontExtractorOp):
    op = 'LookupTableInsert'
    enabled = True

    @classmethod
    def extract(cls, node):
        LookupTableInsert.update_node_stat(node, {})
        return cls.enabled


class LookupTableInsertV2FrontExtractor(FrontExtractorOp):
    op = 'LookupTableInsertV2'
    enabled = True

    @classmethod
    def extract(cls, node):
        LookupTableInsert.update_node_stat(node, {})
        return cls.enabled
