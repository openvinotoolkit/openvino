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
from mo.middle.passes.convert_data_type import destination_type_to_np_data_type

from mo.utils.graph import Node
from mo.utils.ir_reader.extender import Extender


class BucketizeExtender(Extender):
    op = 'Bucketize'

    @staticmethod
    def extend(op: Node):
        if op.get_opset() != "extension":
            op['output_type'] = destination_type_to_np_data_type(op.output_type)
