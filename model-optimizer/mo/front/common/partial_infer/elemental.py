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


def single_output_infer(node, shape_infer, value_infer=None):
    node.out_node(0).shape = shape_infer(node)

    if value_infer is not None and \
            'value' in node.in_node() and \
            node.in_node().value is not None:
        node.out_node(0).value = value_infer(node)


def copy_shape_infer(node, value_infer=None):
    """
    Sets output dimensions of node equal to input ones
    Args:
        node: graph node
    """
    single_output_infer(node, lambda n: n.in_node().shape, value_infer)


def copy_value(node):
    return None if node.in_node().value is None else node.in_node().value.copy()
