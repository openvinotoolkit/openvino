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

from extensions.load.loader import Loader
from mo.front.common.register_custom_ops import update_extractors_with_extensions, check_for_duplicates
from mo.front.extractor import extract_node_attrs
from mo.front.pytorch.extractor import pytorch_op_extractor, pytorch_op_extractors
from mo.graph.graph import Graph

tensors_map = {}
layer_id = 0

class PyTorchLoader(Loader):
    enabled = True

    def load(self, graph: Graph):
        graph.graph['fw'] = 'pytorch'

        update_extractors_with_extensions(pytorch_op_extractors)

        import torch
        print(torch.__version__)
        import torch.nn as nn
        from torch.autograd import Variable
        import torchvision.models as models

        model = models.alexnet(pretrained=True)

        inp = Variable(torch.randn([1, 3, 227, 227]))

        def myhook(self, inputs, output):
            global layer_id
            layer_type = self.__class__.__name__

            # Create a unique name
            name = '{}_{}'.format(layer_type, layer_id)
            layer_id += 1

            graph.add_node(name, kind='op', op=layer_type, name=name, in_ports_count=1, shape=list(output.shape))

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
                print(src_id, name)
                graph.add_edge(src_id, name, **edge_attrs)


            out_hash = hash(output)
            tensors_map[out_hash] = name

        for module in model.modules():
            if len([m for m in module.modules()]) != 1:
                continue
            module.register_forward_hook(myhook)

        tensors_map = {hash(inp): 'input'}
        graph.add_node('input', kind='op', op='Parameter', name='input', shape=list(inp.shape))
        out = model(inp)

        graph.add_node('output', kind='op', op='Result', type='Result', shape=list(out.shape))
        edge_attrs = {
            'out': 0,
            'in': 0,
            'name': 'Linear_20',
            'fw_tensor_debug_info': [('Linear_20', 'Linear_20')],
            'in_attrs': ['in', 'name'],
            'out_attrs': ['out', 'name'],
            'data_attrs': ['fw_tensor_debug_info']
        }
        graph.add_edge('Linear_20', 'output', **edge_attrs)

        extract_node_attrs(graph, lambda node: pytorch_op_extractor(node, check_for_duplicates(pytorch_op_extractors)))
