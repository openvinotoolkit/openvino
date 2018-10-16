"""
 Copyright (c) 2018 Intel Corporation

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

from mo.front.common.partial_infer.eltwise import eltwise_infer


def eltwise_ext(attrs, infer: callable, op_type: str):
    node_attrs = {
        'type': 'Eltwise',
        'operation': op_type,
        'infer': lambda node: eltwise_infer(node, infer)
    }
    return node_attrs
