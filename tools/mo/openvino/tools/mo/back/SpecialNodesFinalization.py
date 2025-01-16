# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
import re
from collections import defaultdict

import numpy as np

from openvino.tools.mo.back.pass_separator import BackFinish
from openvino.tools.mo.back.replacement import BackReplacementPattern
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.const import Const
from openvino.tools.mo.ops.tensor_iterator import TensorIterator
from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.runtime_info import RTInfo
from openvino.tools.mo.utils.utils import refer_to_faq_msg


class RemoveConstOps(BackReplacementPattern):
    enabled = False

    def run_after(self):
        return [BackFinish]

    def run_before(self):
        return []

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(type='Const'):
            graph.remove_edge(node.id, node.out_node().id)
            graph.remove_node(node.id)


class CreateConstNodesReplacement(BackReplacementPattern):
    enabled = False

    def run_before(self):
        return []

    def run_after(self):
        return [RemoveConstOps]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('data', dict(kind='data'))
            ],
            edges=[]
        )

    @staticmethod
    def _check_bin_attrs(node):
        """Check that at least one output edge from node without 'bin' attribute."""
        out_edges = node.out_edges()
        bin_in_out_ports = ['bin' in edge for edge in out_edges]
        out_node = [node.has('op') and node.op == 'Result' for node in node.out_nodes()]
        return np.any(out_node) or not np.all(bin_in_out_ports)

    @staticmethod
    def _check_that_node_from_body(node):
        """Check that all output edges from node have 'internal_port_id'
        (that shows that this node is from TI body)"""
        n_ports = len(node.out_edges())
        internal_port_in_out_ports = ['internal_port_id' in edge for edge in node.out_edges()]
        return np.any(internal_port_in_out_ports) and n_ports

    def replace_pattern(self, graph: Graph, match: dict):
        """
            Adds layers with type 'Const' that produce blob from 'bin' file. The pass finds data nodes with one output which
            doesn't have edge with 'bin' attribute (or with two outputs and at least one output doesn't have 'bin' attr)
            and generate Const op node before the node and data node before the Const node. The data node before 'Const'
            node is needed because the op node dumps input tensors to bin file.
        """
        node = match['data']
        if len(node.in_nodes()) > 0:
            return

        if self._check_bin_attrs(node):
            if node.has_valid('value'):
                const_node_name = node.soft_get('name', node.id)
                const_node_name = graph.unique_id(re.sub(r'\/Output_\d+\/Data_(.?)+', '', const_node_name))
                log.debug("Added Const node '{}'".format(const_node_name))
                const_node = Const(graph, {'name': const_node_name, 'value': node.value,
                                           'force_shape': node.soft_get('force_shape', None),
                                           'override_output_shape': node.has_valid('force_shape'),
                                           'force_type': node.soft_get('force_type', None),
                                           'correct_data_type': node.soft_get('correct_data_type', None),
                                           'rt_info': node.soft_get('rt_info', RTInfo()),
                                           }).create_node()
                const_node.add_input_port(0)
                graph.add_edges_from([(const_node_name, node.id, {'out': 0})])

                node_copy = node.copy_node()
                const_node.type_infer(const_node)
                graph.add_edges_from([(node_copy.id, const_node_name, {'in': 0, 'bin': 'custom'})])
            elif not self._check_that_node_from_body(node):
                log.debug('node = {}'.format(node.graph.node[node.id]))
                raise Error(
                    'Discovered data node without inputs and value, node.name = {}, consumer.name = {}. ' +
                    refer_to_faq_msg(23),
                    node.soft_get('name'),
                    node.out_node().soft_get('name') if len(node.out_nodes()) else "<no consumer>"
                )


class NormalizeTI(BackReplacementPattern):
    """
    Transformation changes linking mechanism of TensorIterator outer graph with inner graph
        from linking outer graph node ports with inner Parameter and Result operations
        to linking outer graph node ports with functional operations and their input/output ports

    1. Updating `input_port_map`, `output_port_map` and `back_edges` maps
    2. Removing Parameter/Input operation nodes

    NOTE: Result operation will be removed by a separate transformation
    """
    enabled = False

    @staticmethod
    def maps_uniqueization(ti):
        assert ti.has_valid('input_port_map')
        assert ti.has_valid('output_port_map')
        assert ti.has_valid('back_edges')

        ti.input_port_map = [dict(unique_r) for unique_r in set([tuple(rec.items()) for rec in ti.input_port_map])]
        ti.output_port_map = [dict(unique_r) for unique_r in set([tuple(rec.items()) for rec in ti.output_port_map])]
        ti.back_edges = [dict(unique_rec) for unique_rec in set([tuple(rec.items()) for rec in ti.back_edges])]

    @staticmethod
    def external_nodes_normalization(ti):
        """
        TensorIterator external ports may have several internal layer connections.

        Current transformation does the following:
            - normalizes port maps (eliminating duplicated records)
            - replicates external input/output port for each internal Parameter/Result it is connected to
            - updates input and output port maps according to previous step replications
        """

        def update_external_port_id(ti, port_type, old_external_port_id, new_external_port_id, internal_layer_id):
            assert port_type in ['in', 'out']

            port_map = ti.input_port_map if port_type == 'in' else ti.output_port_map
            for record in port_map:
                if record['external_port_id'] == old_external_port_id and \
                        record['internal_layer_id'] == internal_layer_id:
                    record['external_port_id'] = new_external_port_id

        NormalizeTI.maps_uniqueization(ti)

        body = ti.body

        external_input_ports = defaultdict(list)
        for record in ti.input_port_map:
            assert 'external_port_id' in record
            external_input_ports[record['external_port_id']].append(record)

        for external_port_id, record_list in external_input_ports.items():
            if len(record_list) == 1:
                continue

            real_external_port_id = TensorIterator.special_port_to_real_port(ti, external_port_id, 'in')
            source = ti.in_port(real_external_port_id).get_source()

            for record in record_list[1:]:
                assert 'internal_layer_id' in record

                new_real_input_port_id = max(map(int, ti.in_ports().keys())) + 1
                new_external_port_id = max([int(d['external_port_id']) for d in
                                            list(ti.in_edges().values()) + list(ti.out_edges().values())]) + 1

                ti.add_input_port(new_real_input_port_id)
                source.connect(ti.in_port(new_real_input_port_id))

                ti.in_edge(new_real_input_port_id)['external_port_id'] = new_external_port_id
                update_external_port_id(ti, 'in', external_port_id, new_external_port_id, record['internal_layer_id'])

        external_output_ports = defaultdict(list)
        for record in ti.output_port_map:
            assert 'external_port_id' in record
            external_output_ports[record['external_port_id']].append(record)

        for external_port_id, record_list in external_output_ports.items():
            if len(record_list) == 1:
                continue

            real_external_port_id = TensorIterator.special_port_to_real_port(ti, external_port_id, 'out')
            dsts = ti.out_port(real_external_port_id).get_destinations()

            for record in record_list[1:]:
                assert 'internal_layer_id' in record

                new_real_output_port_id = max(map(int, ti.out_ports().keys())) + 1
                new_external_port_id = max([int(d['external_port_id']) for d in
                                            list(ti.in_edges().values()) + list(ti.out_edges().values())]) + 1

                ti.add_output_port(new_real_output_port_id)
                for dst in dsts:
                    ti.out_port(new_real_output_port_id).connect(dst)

                update_external_port_id(ti, 'out', external_port_id, new_external_port_id, record['internal_layer_id'])

        body.clean_up()

    def find_and_replace_pattern(self, graph: Graph):
        for ti in graph.get_op_nodes(type='TensorIterator'):
            self.external_nodes_normalization(ti)

            if len([record for record in ti.input_port_map if record.get('axis') is not None]) == 0:
                for record in ti.output_port_map:
                    if record.get('axis') is not None:
                        record['start'] = 0
                        real_output_port = TensorIterator.special_port_to_real_port(ti, record['external_port_id'], 'out')
                        output_shape = ti.out_port(real_output_port).data.get_shape()
                        assert output_shape is not None
                        record['end'] = output_shape[record['axis']]
