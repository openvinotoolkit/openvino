# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

from extensions.middle.fusings import Fusing
from extensions.middle.pass_separator import PostMiddleStart
from mo.graph.graph import Node, Graph
from mo.middle.replacement import MiddleReplacementPattern


class ConcatOptimization(MiddleReplacementPattern):
    # This optimization reduces number of edges between Concat operations
    # that significantly reduce memory consumption

    enabled = True
    graph_condition = [lambda graph: graph.graph['cmd_params'].enable_concat_optimization]

    def run_after(self):
        return [Fusing]

    def run_before(self):
        return [PostMiddleStart]

    def find_and_replace_pattern(self, graph: Graph):
        mp = {}
        used = {}
        for node in graph.get_op_nodes(type='Concat'):
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
                        replacers.append((len(arr), arr))

            replacers.sort(reverse=True)

            concat_id = mp[key][0]
            for ln, arr in replacers:
                # Check that we can do it!!!
                we_can = True
                for x in arr:
                    if used[concat_id][x]:
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


class ConcatOdInputEraserAndPortsReconnect(MiddleReplacementPattern):
    """
    The transformation performs two actions with Concat operations:
    1. Disconnects empty inputs (input tensor has at least one input dimension equal to 0)
    2. Renumber Concat inputs to be 0, 1, 2,...
    """
    enabled = True
    force_clean_up = True

    def find_and_replace_pattern(self, graph: Graph):
        for concat in graph.get_op_nodes(type='Concat'):
            for in_port in concat.in_ports().values():
                if not in_port.disconnected():
                    shape = in_port.data.get_shape()
                    assert shape is not None
                    if 0 in shape:
                        concat.delete_input_port(in_port.idx)

            connected_ports = [port for port_idx, port in sorted(concat.in_ports().items()) if not port.disconnected()]
            assert len(connected_ports), 'Concat "{}" have no inputs after removing inputs with 0 dimensions' \
                                         ''.format(concat.soft_get('name', concat.id))

            max_port_index = max([port_idx for port_idx in concat.in_ports().keys()])
            # re-connect input ports sequentially and remove all not used
            port_idx_to_connect = 0
            for port_idx in range(max_port_index + 1):
                if concat.is_in_port_connected(port_idx):
                    if port_idx != port_idx_to_connect:
                        concat.add_input_port(port_idx_to_connect, skip_if_exist=True)
                        concat.in_port(port_idx).get_connection().set_destination(concat.in_port(port_idx_to_connect))
                    port_idx_to_connect += 1
                elif port_idx in concat.in_ports():
                    concat.delete_input_port(port_idx)
