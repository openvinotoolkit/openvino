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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging as log

import torch
from torch.autograd import Variable

from extensions.load.loader import Loader
from mo.front.common.register_custom_ops import update_extractors_with_extensions, check_for_duplicates
from mo.front.extractor import extract_node_attrs
from mo.front.pytorch.extractor import pytorch_op_extractor, pytorch_op_extractors
from mo.graph.graph import Graph

tensors_map = {}

class PyTorchLoader(Loader):
    enabled = True

    def load(self, graph: Graph):
        argv = graph.graph['cmd_params']
        graph.graph['fw'] = 'pytorch'
        graph.graph['layout'] = 'NCHW'

        update_extractors_with_extensions(pytorch_op_extractors)

        # Create a dummy input
        inp = Variable(torch.randn(list(argv.placeholder_shapes)))

        model = argv.input_model

        def forward_hook(self, inputs, output):
            layer_type = self.__class__.__name__

            # Create a unique name
            name = graph.unique_id(prefix=layer_type + '_')

            graph.add_node(name, kind='op', op=layer_type, name=name, module=self)

            # Find all inputs
            for inp in inputs:
                h = hash(inp)
                src_id = tensors_map[h]

                edge_attrs = {
                    'out': 0,
                    'in': 0,
                    'name': src_id,
                    'fw_tensor_debug_info': [(src_id, src_id)],
                    'in_attrs': ['in', 'name'],
                    'out_attrs': ['out', 'name'],
                    'data_attrs': ['fw_tensor_debug_info']
                }
                graph.add_edge(src_id, name, **edge_attrs)

            for key, value in self.state_dict().items():
                assert key == 'weight' or key == 'bias', 'Unknown parameter: ' + key

                param_name = name + '/' + key
                graph.add_node(param_name, kind='op', op='Const', value=value.numpy())
                edge_attrs = {
                    'out': 0,
                    'in': 1 if key == 'weight' else 2,
                    'name': param_name,
                    'fw_tensor_debug_info': [(param_name, param_name)],
                    'in_attrs': ['in', 'name'],
                    'out_attrs': ['out', 'name'],
                    'data_attrs': ['fw_tensor_debug_info']
                }
                graph.add_edge(param_name, name, **edge_attrs)


            out_hash = hash(output)
            tensors_map[out_hash] = name

        for module in model.modules():
            if len([m for m in module.modules()]) != 1:
                continue
            module.register_forward_hook(forward_hook)

        tensors_map = {hash(inp): 'input'}
        graph.add_node('input', kind='op', op='Parameter', name='input', shape=list(inp.shape))
        outs = model(inp)

        # Add output nodes
        for out in outs if isinstance(outs, tuple) else (outs):
            name = tensors_map[hash(out)]
            graph.add_node('output', kind='op', op='Result')
            edge_attrs = {
                'out': 0,
                'in': 0,
                'name': name,
                'fw_tensor_debug_info': [(name, name)],
                'in_attrs': ['in', 'name'],
                'out_attrs': ['out', 'name'],
                'data_attrs': ['fw_tensor_debug_info']
            }
            graph.add_edge(name, 'output', **edge_attrs)

        extract_node_attrs(graph, lambda node: pytorch_op_extractor(node, check_for_duplicates(pytorch_op_extractors)))
