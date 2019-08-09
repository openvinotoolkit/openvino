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

import logging as log

import numpy as np

from extensions.ops.elementwise import Mul, Add, Pow
from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Node, Graph
from mo.middle.passes.fusing.helpers import get_value_in_port, get_tensor_in_port
from mo.middle.pattern_match import check_node_usages_out_of_match
from mo.ops.const import Const
from mo.ops.eltwise import Eltwise
from mo.ops.power import Power

simple_eltwise_ops = ['Pow', 'Add', 'Multiply', 'Maximum', 'LogicalAnd', 'LogicalOr', 'Less', 'LessEqual',
                      'Greater', 'GreaterEqual', 'Equal', 'NotEqual']

op_to_operation_map = {
    'Pow': 'pow',
    'Add': 'sum',
    'Multiply': 'mul',
    'Maximum': 'max',
    'LogicalAnd': 'logical_and',
    'LogicalOr': 'logical_or',
    'Less': 'less',
    'LessEqual': 'less_equal',
    'Greater': 'greater',
    'GreaterEqual': 'greater_equal',
    'Equal': 'equal',
    'NotEqual': 'not_equal',
}


class SimpleEltwiseToEltwiseOp(BackReplacementPattern):
    enabled = True

    @staticmethod
    def pattern():
        return dict(
            nodes=[('op', {'type': lambda t: t in simple_eltwise_ops})],
            edges=[],
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: [str, Node]):
        Eltwise.update_node_stat(match['op'], {'operation': op_to_operation_map[match['op'].type]})


class DivideToEltwises(BackReplacementPattern):
    enabled = True
    graph_condition = [lambda graph: not graph.graph['cmd_params'].generate_experimental_IR_V10]

    def run_before(self):
        return [SimpleEltwiseToEltwiseOp]

    @staticmethod
    def pattern():
        return dict(
            nodes=[('div', {'type': 'Divide'})],
            edges=[],
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: [str, Node]):
        node = match['div']
        power_of_exponent = Const(graph, {'value': np.float64(-1)}).create_node()
        reciprocal = Pow(graph, {'name': node.name + '/reciprocal_'}).create_node()
        mul = Mul(graph, {'name': node.name + '/mul_'}).create_node()

        # Connect nodes
        node.in_port(1).get_connection().set_destination(reciprocal.in_port(0))
        power_of_exponent.out_port(0).connect(reciprocal.in_port(1))
        node.in_port(0).get_connection().set_destination(mul.in_port(1))
        reciprocal.out_port(0).connect(mul.in_port(0))

        node.out_port(0).get_connection().set_source(mul.out_port(0))


class SubtractToEltwises(BackReplacementPattern):
    enabled = True
    force_clean_up = True
    graph_condition = [lambda graph: not graph.graph['cmd_params'].generate_experimental_IR_V10]

    def run_before(self):
        return [SimpleEltwiseToEltwiseOp]

    @staticmethod
    def pattern():
        return dict(
            nodes=[('sub', {'type': 'Subtract'})],
            edges=[],
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: [str, Node]):
        node = match['sub']

        # Add new nodes
        negate_const = Const(graph, dict(name=node.name + '/negate_const', value=np.array(-1))).create_node()
        negate = Mul(graph, {'name': node.name + '/negate_'}).create_node()
        add = Add(graph, {'name': node.name + '/add_'}).create_node()

        # Connect nodes
        node.in_port(1).get_connection().set_destination(negate.in_port(0))
        negate_const.out_port(0).connect(add.in_port(1))
        node.in_port(0).get_connection().set_destination(add.in_port(1))
        negate.out_port(0).connect(add.in_port(0))

        node.out_port(0).get_connection().set_source(add.out_port(0))


class EltwisesWithScalarInputToPower(BackReplacementPattern):
    enabled = True
    graph_condition = [lambda graph: not graph.graph['cmd_params'].generate_experimental_IR_V10]
    force_clean_up = True
    eltw_types = ['Add', 'Multiply', 'Pow']

    def run_before(self):
        return [SimpleEltwiseToEltwiseOp]

    @staticmethod
    def pattern():
        return dict(
            nodes=[('op', {'type': lambda type: type in EltwisesWithScalarInputToPower.eltw_types})],
            edges=[],
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: [str, Node]):
        op = match['op']
        op_type = op.type

        const_port, tensor_port = get_value_in_port(op), get_tensor_in_port(op)
        if const_port is None or tensor_port is None:
            return
        value = const_port.data.get_value()
        assert value is not None
        if value.size != 1:
            return
        value = value.item(0)

        assert op_type in EltwisesWithScalarInputToPower.eltw_types
        if op_type == 'Add':
            delete_node = value == 0
            Power.update_node_stat(op, {'shift': value})
        elif op_type == 'Multiply':
            delete_node = value == 1
            Power.update_node_stat(op, {'scale': value})
        elif op_type == 'Pow':
            delete_node = value == 1
            Power.update_node_stat(op, {'power': value})

        const_port.disconnect()
        if tensor_port.idx != 0:
            tensor_port.get_connection().set_destination(op.in_port(0))

        # TODO: uncomment this lines in future to allow useless operations deleting
        # if delete_node:
        #     op.out_port(0).get_connection().set_source(op.in_port(0).get_connection().get_source())


class MulAddPowerMerge(BackReplacementPattern):
    enabled = True
    graph_condition = [lambda graph: not graph.graph['cmd_params'].generate_experimental_IR_V10]
    force_clean_up = True

    def run_after(self):
        return [EltwisesWithScalarInputToPower]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('mul', dict(type='Power', shift=lambda x: np.all(x == 0), power=lambda x: np.all(x == 1))),
                ('mul_d', dict()),
                ('add', dict(type='Power', scale=lambda x: np.all(x == 1), power=lambda x: np.all(x == 1))),
                ('add_d', dict())
            ],
            edges=[
                ('mul', 'mul_d'),
                ('mul_d', 'add'),
                ('add', 'add_d'),
            ],
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: [str, Node]):
        consumers = [n for n in match if n not in ['mul', 'add_d'] and not check_node_usages_out_of_match(match, n)]
        if consumers:
            log.warning('Power(mul,add) pattern was detected. Non pattern consumers of nodes: "{}" were found.'
                        ' Won\'t replace'.format(', '.join([match[n].id for n in consumers])))
            return
        mul = match['mul']
        add = match['add']
        new_power = Power(graph, {'name': mul.name + '/fused_power', 'scale': mul.scale,
                                  'shift': add.shift}).create_node()

        source = mul.in_port(0).get_connection().get_source()
        mul.in_port(0).disconnect()
        new_power.in_port(0).connect(source)
        add.out_port(0).get_connection().set_source(new_power.out_port(0))

        log.debug('Power nodes {} and {} were fused to single Power node {}'.format(mul.name, add.name, new_power.name))


class MulPowPowerMerge(BackReplacementPattern):
    enabled = True
    graph_condition = [lambda graph: not graph.graph['cmd_params'].generate_experimental_IR_V10]
    force_clean_up = True

    def run_after(self):
        return [MulAddPowerMerge]

    def run_before(self):
        return [SimpleEltwiseToEltwiseOp]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('mul', dict(type='Power', shift=lambda x: np.all(x == 0), power=lambda x: np.all(x == 1))),
                ('mul_d', dict()),
                ('pow', dict(type='Power', scale=lambda x: np.all(x == 1), shift=lambda x: np.all(x == 0))),
                ('pow_d', dict())
            ],
            edges=[
                ('mul', 'mul_d'),
                ('mul_d', 'pow'),
                ('pow', 'pow_d'),
            ],
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: [str, Node]):
        consumers = [n for n in match if n not in ['mul', 'pow_d'] and not check_node_usages_out_of_match(match, n)]
        if consumers:
            log.warning('Power(mul,pow) pattern was detected. Non pattern consumers of nodes: "{}" were found.'
                        ' Won\'t replace'.format(', '.join([match[n].id for n in consumers])))
            return
        mul = match['mul']
        pow = match['pow']
        new_power = Power(graph, {'name': mul.name + '/fused_power', 'scale': mul.scale,
                                  'power': pow.power}).create_node()

        source = mul.in_port(0).get_connection().get_source()
        mul.in_port(0).disconnect()
        new_power.in_port(0).connect(source)
        pow.out_port(0).get_connection().set_source(new_power.out_port(0))

        log.debug('Power nodes {} and {} were fused to single Power node {}'.format(mul.name, pow.name, new_power.name))


class AddPowPowerMerge(BackReplacementPattern):
    enabled = True
    graph_condition = [lambda graph: not graph.graph['cmd_params'].generate_experimental_IR_V10]
    force_clean_up = True

    def run_after(self):
        return [MulAddPowerMerge]

    def run_before(self):
        return [SimpleEltwiseToEltwiseOp]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('add', dict(type='Power', scale=lambda x: np.all(x == 1), power=lambda x: np.all(x == 1))),
                ('add_d', dict()),
                ('pow', dict(type='Power', scale=lambda x: np.all(x == 1), shift=lambda x: np.all(x == 0))),
                ('pow_d', dict())
            ],
            edges=[
                ('add', 'add_d'),
                ('add_d', 'pow'),
                ('pow', 'pow_d'),
            ],
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: [str, Node]):
        consumers = [n for n in match if n not in ['add', 'pow_d'] and not check_node_usages_out_of_match(match, n)]
        if consumers:
            log.warning('Power(add,pow) pattern was detected. Non pattern consumers of nodes: "{}" were found.'
                        ' Won\'t replace'.format(', '.join([match[n].id for n in consumers])))
            return
        add = match['add']
        pow = match['pow']
        new_power = Power(graph, {'name': add.name + '/fused_power', 'shift': add.shift,
                                  'power': pow.power}).create_node()

        source = add.in_port(0).get_connection().get_source()
        add.in_port(0).disconnect()
        new_power.in_port(0).connect(source)
        pow.out_port(0).get_connection().set_source(new_power.out_port(0))

        log.debug('Power nodes {} and {} were fused to single Power node {}'.format(add.name, pow.name, new_power.name))


class MulAddPowPowerMerge(BackReplacementPattern):
    enabled = True
    graph_condition = [lambda graph: not graph.graph['cmd_params'].generate_experimental_IR_V10]
    force_clean_up = True

    def run_after(self):
        return [MulAddPowerMerge, MulPowPowerMerge, AddPowPowerMerge]

    def run_before(self):
        return [SimpleEltwiseToEltwiseOp]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('mul_add', dict(type='Power', power=lambda x: np.all(x == 1))),
                ('mul_add_d', dict()),
                ('pow', dict(type='Power', scale=lambda x: np.all(x == 1), shift=lambda x: np.all(x == 0))),
                ('pow_d', dict())
            ],
            edges=[
                ('mul_add', 'mul_add_d'),
                ('mul_add_d', 'pow'),
                ('pow', 'pow_d'),
            ],
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: [str, Node]):
        consumers = [n for n in match if n not in ['mul_add', 'pow_d'] and not check_node_usages_out_of_match(match, n)]
        if consumers:
            log.warning('Power(mul_add,pow) pattern was detected. Non pattern consumers of nodes: "{}" were found.'
                        ' Won\'t replace'.format(', '.join([match[n].id for n in consumers])))
            return
        mul_add = match['mul_add']
        pow = match['pow']
        new_power = Power(graph, {'name': mul_add.name + '/fused_power', 'shift': mul_add.shift, 'scale': mul_add.scale,
                                  'power': pow.power}).create_node()

        source = mul_add.in_port(0).get_connection().get_source()
        mul_add.in_port(0).disconnect()
        new_power.in_port(0).connect(source)
        pow.out_port(0).get_connection().set_source(new_power.out_port(0))

        log.debug('Power nodes {} and {} were fused to single Power node {}'.format(mul_add.name, pow.name,
                                                                                    new_power.name))
