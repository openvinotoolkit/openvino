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

import networkx as nx
import logging as log

from mo.graph.graph import Node
from mo.middle.replacement import MiddleReplacementPattern


class ConcatOptimization(MiddleReplacementPattern):
    # This optimization reduces number of edges between Concat operations
    # that significantly reduce memory consumption

    enabled = False

    def run_after(self):
        return []

    def find_and_replace_pattern(self, graph: nx.MultiDiGraph):
        mp = {}
        used = {}
        for node in graph.nodes():
            node = Node(graph, node)
            if node.kind == 'op' and node.soft_get('type') == 'Concat':
                in_nodes = tuple([node.in_node(idx).id for idx in range(len(node.in_nodes()))])
                out_node = (node.id, node.out_node().id)
                if in_nodes in mp:
                    log.warning("Something is weird! {} and {}".format(node.id, mp[in_nodes]))
                else:
                    mp.update({in_nodes: out_node})
                    used.update({node.id: {x: False for x in in_nodes}})

        for key in mp.keys():
            replacers = []
            for i in range(len(key)):
                for j in range(i + 1, len(key)):
                    arr = tuple(key[i:j + 1])
                    if arr in mp.keys() and arr != key:
                        # print("Output of {} can be used as input for {} ({})".format(mp[arr][0], mp[key][0], len(arr)))
                        replacers.append((len(arr), arr))

            replacers.sort(reverse=True)

            concat_id = mp[key][0]
            for ln, arr in replacers:
                # Check that we can do it!!!
                we_can = True
                for x in arr:
                    if used[concat_id][x]:
                        # print("Sorry but {} input was already removed from {}".format(x, concat_id))
                        we_can = False
                        break

                if not we_can:
                    continue

                for x in arr:
                    used[concat_id][x] = True

                edge_attrs = graph.get_edge_data(arr[0], concat_id)[0]
                for in_node in arr:
                    graph.remove_edge(in_node, concat_id)

                new_input = mp[arr][1]
                out_port = len(Node(graph, new_input).out_nodes()) + 1
                edge_attrs['out'] = out_port
                graph.add_edge(new_input, concat_id, **edge_attrs)

                # Renumber 'in' attrs
                concat_node = Node(graph, concat_id)
                ln = len(concat_node.in_nodes())
                ports = [x for x in concat_node.in_nodes().keys()]
                ports.sort()

                p_id = 0
                for p in ports:
                    in_node = concat_node.in_nodes()[p]
                    graph[in_node.id][concat_id][0]['in'] = p_id
                    p_id += 1
