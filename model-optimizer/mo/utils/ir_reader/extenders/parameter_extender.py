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
from mo.front.common.partial_infer.utils import int64_array
from mo.middle.passes.convert_data_type import destination_type_to_np_data_type
from mo.utils.graph import Node
from mo.utils.ir_reader.extender import Extender


class Parameter_extender(Extender):
    op = 'Parameter'

    @staticmethod
    def extend(op: Node):
        assert op.has_valid('element_type'), 'Parameter node {} has missed element_type attr!'.format(op.name)
        op['data_type'] = destination_type_to_np_data_type(op.element_type)
        if op.shape == '':
            op.shape = int64_array([])
        else:
            Extender.attr_to_list(op, 'shape')
