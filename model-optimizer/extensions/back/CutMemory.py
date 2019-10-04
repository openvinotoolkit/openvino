"""
 Copyright (c) 2019 Intel Corporation

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
import numpy as np

from extensions.back.ParameterToPlaceholder import ParameterToInput
from extensions.ops.parameter import Parameter
from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph
from mo.ops.crop import Crop
from mo.utils.logger import log


class CutMemory(BackReplacementPattern):
    """
    Cut Memory layers and have inputs/outputs in graph instead of them
    """
    enabled = False

    def run_before(self):
        return [ParameterToInput]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('op', dict(kind='op', op='Memory'))],
            edges=[]
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        node = match['op']
        node_id = node['id']

        if node.in_port(0).disconnected():
            i = 0
            for dest in node.out_port(0).get_destinations():
                new_in = Parameter(graph, {'name': "Parameter_"+str(i)+"_for_"+node_id,
                                           'shape': dest.data.get_shape()}).create_node()
                i += 1
                dest.disconnect()
                new_in.out_port(0).connect(dest)
                log.error("Add input/output mapped {} -> {} ".format(new_in.name, "Result_for_"+node_id),
                          extra={'is_warning': True})
        else:
            out_node_port = node.out_port(0).get_destination()
            in_node_port = node.in_port(0).get_source()
            node.in_port(0).disconnect()
            node.out_port(0).disconnect()
            crop = Crop(graph, {'name': 'Result_for_'+node_id, 'dim': np.array([1]), 'offset': np.array([0]), 'axis': np.array([0])}).create_node()
            in_node_port.connect(crop.in_port(0))
            crop.out_port(0).connect(out_node_port)
