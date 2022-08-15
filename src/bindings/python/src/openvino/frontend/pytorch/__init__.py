# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Package: openvino
Low level wrappers for the FrontEnd C++ API.
"""

# flake8: noqa

from openvino.utils import add_openvino_libs_to_path
from openvino.runtime import PartialShape, Type, OVAny

add_openvino_libs_to_path()


try:
    from openvino.frontend.pytorch.py_pytorch_frontend import _FrontEndPytorchDecoder as Decoder

    class TorchScriptPythonDecoder (Decoder):
        def __init__ (self, pt_module):
            Decoder.__init__(self)
            self.pt_module = pt_module

        def inputs (self):
            return [x.unique() for x in self.pt_module.inputs()]

        def input (self, index):
            return self.inputs()[index] # TODO: find specialized method

        def get_input_shape (self, index):
            return PartialShape.dynamic()
            pass

        def get_input_type (self, index):
            return OVAny(Type.undefined)

        def get_input_transpose_order (self, index):
            return []

        def get_subgraph_size (self):
            return len(self.pt_module.blocks()) if hasattr(self.pt_module, 'blocks') else 1

        def visit_subgraph (self, index, node_visitor):
            # make sure topological order is satisfied
            if index < self.get_subgraph_size():
                if hasattr(self.pt_module, 'blocks'):
                    for node in self.pt_module.blocks()[index].nodes():
                        print('inside 1')
                        decoder = TorchScriptPythonDecoder(node)
                        node_visitor(decoder)
                else:
                    for node in self.pt_module.nodes():
                        print('inside 2')
                        decoder = TorchScriptPythonDecoder(node)
                        node_visitor(decoder)
            else:
                raise Exception(f'Index {index} of block is out of range, total number of blocks is {self.get_subgraph_size()}')

        def get_op_type (self):
            return self.pt_module.kind()

        def outputs (self):
            return [x.unique() for x in self.pt_module.outputs()]

        def num_of_outputs (self):
            return len(self.outputs())

        def output (self, index):
            return self.outputs()[index]

        def mark_node (self, node):
            return node

except ImportError as err:
    raise ImportError("OpenVINO Pytorch frontend is not available, please make sure the frontend is built."
                      "{}".format(err))
