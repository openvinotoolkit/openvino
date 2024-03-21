# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.op import Op
from openvino.tools.mo.utils.error import Error


class EltwiseN(Op):
    """
    The elementwise operation that has more than 2 inputs. This operation is replaced in a front phase with a number of
    simple elementwise operations with 2 inputs. Refer to EltwiseNFrontReplacer for a list of supported operations.
    """
    op = 'EltwiseN'

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'type': None,  # type is None because this operation should not appear in IR
            'infer': None,
            'out_ports_count': 1,
        }, attrs)
        if 'operation' not in self.attrs:
            raise Error('"operation" attribute is not set for operation "{}".'.format(self.op))


class EltwiseNMul(EltwiseN):
    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {'operation': 'mul'})


class EltwiseNMin(EltwiseN):
    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {'operation': 'min'})


class EltwiseNMax(EltwiseN):
    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {'operation': 'max'})


class EltwiseNAdd(EltwiseN):
    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {'operation': 'sum'})
