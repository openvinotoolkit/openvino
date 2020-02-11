"""
 Copyright (c) 2020 Intel Corporation

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

from mo.utils.ir_reader.extender import Extender
from mo.utils.graph import Node


class TopK_extender(Extender):
    op = 'TopK'

    @staticmethod
    def extend(op: Node):
        if op.graph.graph['cmd_params'].framework in ('tf', 'caffe'):
            op['remove_values_output'] = True
