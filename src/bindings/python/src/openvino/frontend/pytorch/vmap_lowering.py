# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from contextlib import ExitStack

import torch


def _vmap_name(target):
    if getattr(target, "__module__", None) == "torch._functorch.predispatch":
        return getattr(target, "__name__", "")
    return ""


class _VmapInterpreter(torch.fx.Interpreter):
    def __init__(self, module):
        super().__init__(module)
        self.levels = {}
        self.active_levels = []

    def call_function(self, target, args, kwargs):
        name = _vmap_name(target)
        if name == "_vmap_increment_nesting":
            level = target(*args, **kwargs)
            self.active_levels.append(level)
            return level
        if name == "_add_batch_dim":
            args = list(args)
            args[2] = self.levels.setdefault(args[2], self.active_levels[-1])
        elif name == "_remove_batch_dim":
            args = list(args)
            args[1] = self.levels.get(args[1], args[1])
            if args[3] < 0:
                args[3] += args[0].dim() + 1
            if torch._C._functorch._unwrap_batched(args[0], args[1])[1] is None:
                shape = list(args[0].shape)
                shape.insert(args[3], args[2])
                return args[0].unsqueeze(args[3]).expand(shape)
        result = target(*args, **kwargs)
        if name == "_vmap_decrement_nesting":
            self.active_levels.pop()
        return result

    def run(self, *args):
        self.levels.clear()
        try:
            return super().run(*args)
        finally:
            while self.active_levels:
                torch._C._functorch._vmap_decrement_nesting()
                self.active_levels.pop()


def lower_vmap_regions(module, guard_context):
    from torch.fx.experimental.proxy_tensor import make_fx
    from .vmap_capture import check_vmap_call

    if not any(_vmap_name(node.target) == "_vmap_increment_nesting" for node in module.graph.nodes):
        return module

    graph = torch.fx.Graph()
    graph.output(graph.graph_copy(module.graph, {}))
    result = torch.fx.GraphModule(module, graph)
    ordinal = 0
    while True:
        start = next((node for node in graph.nodes if _vmap_name(node.target) == "_vmap_increment_nesting"), None)
        if start is None:
            break
        region = []
        depth = 0
        for node in list(graph.nodes)[list(graph.nodes).index(start):]:
            name = _vmap_name(node.target)
            if name == "_vmap_increment_nesting":
                depth += 1
                if node.args[1] != "error":
                    raise RuntimeError("vmap supports only randomness='error'.")
            elif name == "_vmap_decrement_nesting":
                depth -= 1
            region.append(node)
            if depth == 0:
                break
        if depth:
            raise RuntimeError("Unbalanced vmap nesting in exported graph.")
        region_set = set(region)
        operands = list(dict.fromkeys(value for node in region for value in node.all_input_nodes
                                      if value not in region_set and value.op != "get_attr"))
        outputs = [node for node in region if any(user not in region_set for user in node.users)]
        body_graph = torch.fx.Graph()
        values = {}
        examples = []
        for operand in operands:
            example = operand.meta.get("val")
            if not isinstance(example, (torch.Tensor, torch.SymInt, torch.SymFloat, torch.SymBool)):
                raise RuntimeError("vmap lowering requires tensor or symbolic FX metadata for its operands.")
            examples.append(example)
            values[operand] = body_graph.placeholder(operand.name)
        for node in region:
            for operand in node.all_input_nodes:
                if operand not in values and operand.op == "get_attr":
                    values[operand] = body_graph.node_copy(operand, values.__getitem__)
            values[node] = body_graph.node_copy(node, values.__getitem__)
        body_graph.output(tuple(values[node] for node in outputs))
        body = torch.fx.GraphModule(result, body_graph)
        for submodule in body.modules():
            if isinstance(submodule, torch.fx.GraphModule):
                for node in submodule.graph.nodes:
                    if node.op == "call_function":
                        check_vmap_call(node.target, node.args, node.kwargs)
        with ExitStack() as stack:
            environments = set()
            for example in examples:
                owner = getattr(example, "fake_mode", None) if isinstance(example, torch.Tensor) else example.node
                environment = getattr(owner, "shape_env", None)
                if environment is not None and environment not in environments:
                    environments.add(environment)
                    stack.enter_context(guard_context(environment))
            lowered = make_fx(_VmapInterpreter(body).run, tracing_mode="fake", _allow_non_fake_inputs=True)(*examples)
        prefix = f"_ov_vmap_{ordinal}"
        ordinal += 1
        result.add_submodule(prefix, lowered)
        replacement = {}
        with graph.inserting_before(start):
            for node, operand in zip((node for node in lowered.graph.nodes if node.op == "placeholder"), operands):
                replacement[node] = operand
            for node in lowered.graph.nodes:
                if node.op == "placeholder":
                    continue
                if node.op == "output":
                    returned = torch.fx.map_arg(node.args[0], replacement.__getitem__)
                    for original, value in zip(outputs, returned):
                        original.replace_all_uses_with(value)
                elif node.op == "get_attr":
                    replacement[node] = graph.get_attr(f"{prefix}.{node.target}")
                    replacement[node].meta = node.meta.copy()
                else:
                    replacement[node] = graph.node_copy(node, replacement.__getitem__)
        for node in reversed(region):
            graph.erase_node(node)
    for node in list(graph.nodes):
        if _vmap_name(node.target) == "lazy_load_decompositions":
            if node.users:
                raise RuntimeError("Unexpected users of vmap decomposition initialization.")
            graph.erase_node(node)
    graph.lint()
    result.recompile()
    return result
