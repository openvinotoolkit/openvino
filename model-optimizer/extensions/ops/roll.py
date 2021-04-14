
from mo.front.common.partial_infer.elemental import copy_shape_infer
from mo.graph.graph import Graph, Node
from mo.graph.perm_inputs import PermuteInputs
from mo.ops.op import Op


class Roll(Op):
    """
    Roll operation that shifts elements of a tensor along specified axes.
    """
    op = 'Roll'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': self.op,
            'op': self.op,
            'version': 'opset7',
            'infer': roll_infer,
            'in_ports_count': 3,
            'out_ports_count': 1
        }, attrs)


def roll_infer(node: Node):
    PermuteInputs().set_input_permutation(node.in_node(2), node, 'input:0', 'axis')
    copy_shape_infer(node)


class TFRoll(Op):
    op = 'TFRoll'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'infer': copy_shape_infer,
            'in_ports_count': 3,
            'out_ports_count': 1
        }, attrs)
