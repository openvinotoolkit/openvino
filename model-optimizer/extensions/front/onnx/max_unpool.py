from extensions.ops.MaxPoolGrad import MaxPoolGrad
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import Graph

class MaxUnpool(FrontReplacementSubgraph):
    enabled = True

    def pattern(self):
        return dict(
            nodes=[
                ('max_pool0', dict(op='MaxPool')),
                ('max_pool1', dict(op='MaxPool')),
                ('slice', dict(op='AttributedSlice')),
                ('sub', dict(op='Sub')),
                ('unpool', dict(op='Unpooling')),
            ],
            edges=[
                ('max_pool1', 'slice'),
                ('max_pool0', 'sub', {'in': 0}),
                ('slice', 'sub', {'in': 1}),
                ('sub', 'unpool', {'in': 1}),
            ])

    @staticmethod
    def replace_sub_graph(graph: Graph, match: dict):
        max_pool = match['max_pool0']
        max_pool_input = max_pool.in_port(0).get_source().node
        unpool = match['unpool']
        unpool_input = unpool.in_port(0).get_source().node

        max_pool.out_port(1).disconnect()

        # Inputs: [max_pool_input, max_pool_output, unpool_input, shape]
        inputs = [max_pool_input, max_pool, unpool_input]

        res = MaxPoolGrad(graph, dict(name=unpool.name + '/fused')).create_node(inputs)
        unpool.out_port(0).get_connection().set_source(res.out_port(0))

        if len(unpool.in_ports()) == 3:
            unpool.in_port(2).get_source().connect(res.in_port(3))
        else:
            max_pool_input.out_port(0).connect(res.in_port(3))
