from mo.graph.graph import Node

pytorch_op_extractors = {}

def pytorch_op_extractor(node: Node, lowered_keys_map: dict) -> (bool, dict):
    result = {}
    supported = False
    op = node['op'].lower()
    if op in lowered_keys_map:
        op = lowered_keys_map[op]
        assert op in pytorch_op_extractors
        attrs = pytorch_op_extractors[op](node)
        if attrs:
            result.update(attrs)
            supported = True
    return supported, result
