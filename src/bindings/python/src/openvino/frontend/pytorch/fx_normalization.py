# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# mypy: ignore-errors

import operator
from itertools import count

import torch

from openvino.frontend.pytorch.utils import FX_RUNTIME_TYPES, fetch_attr, trace_without_new_guards


def predispatch_name(target):
    if getattr(target, "__module__", None) == "torch._functorch.predispatch":
        return getattr(target, "__name__", "")
    return ""


def _is_flat_apply(node):
    return node.op == "call_function" and str(node.target) == "flat_apply"


def _inline(graph, owner, traced, operands, prefix):
    """Copy traced at the current insertion point of graph and return its output structure."""
    owner.add_submodule(prefix, traced)
    replacement = dict(zip((node for node in traced.graph.nodes if node.op == "placeholder"), operands))
    for node in traced.graph.nodes:
        if node.op == "placeholder":
            continue
        if node.op == "output":
            return torch.fx.map_arg(node.args[0], replacement.__getitem__)
        if node.op == "get_attr":
            replacement[node] = graph.get_attr(f"{prefix}.{node.target}")
            replacement[node].meta = node.meta.copy()
        else:
            replacement[node] = graph.node_copy(node, replacement.__getitem__)


def flat_apply_example(value):
    """Return a logical FakeTensor for metadata recorded inside vmap."""
    if not isinstance(value, FX_RUNTIME_TYPES):
        raise RuntimeError("flat_apply runtime operands require tensor or symbolic FX metadata")
    functorch = torch._C._functorch
    if isinstance(value, torch.Tensor) and functorch.is_batchedtensor(value):
        from torch._subclasses.fake_tensor import FakeTensor
        physical = value
        while functorch.is_batchedtensor(physical):
            physical = functorch.get_unwrapped(physical)
        if not isinstance(physical, FakeTensor):
            raise RuntimeError("flat_apply batched operands require FakeTensor metadata")
        with physical.fake_mode:
            value = torch.empty_strided(value.shape, value.stride(), dtype=value.dtype, device=value.device)
    return value


def _replace_with_structure(node, value):
    if isinstance(value, torch.fx.Node):
        node.replace_all_uses_with(value)
        return
    for user in list(node.users):
        if user.target is operator.getitem and user.args[0] is node and isinstance(user.args[1], int):
            _replace_with_structure(user, value[user.args[1]])
            node.graph.erase_node(user)
        else:
            user.args = torch.fx.map_arg(user.args, lambda arg: value if arg is node else arg)
            user.kwargs = torch.fx.map_arg(user.kwargs, lambda arg: value if arg is node else arg)


def _inline_flat_apply(graph, owner, node, prefix):
    from torch._higher_order_ops.flat_apply import impl

    if len(node.args) < 2 or set(node.kwargs) - {"checked_output"}:
        raise RuntimeError("flat_apply requires callable and input specifications")

    def resolve(arg):
        if isinstance(arg, torch.fx.Node) and arg.op == "get_attr":
            return fetch_attr(owner, arg.target)
        return arg

    function, input_spec = (resolve(arg) for arg in node.args[:2])
    flat_args = [resolve(arg) for arg in node.args[2:]]
    positions = [index for index, arg in enumerate(node.args[2:]) if isinstance(arg, torch.fx.Node) and (
        arg.op != "get_attr" or isinstance(flat_args[index], FX_RUNTIME_TYPES))]
    operands = [node.args[2 + index] for index in positions]
    examples = [flat_apply_example(operand.meta.get("val", flat_args[index]))
                for index, operand in zip(positions, operands)]
    checked_output = node.kwargs.get("checked_output", True)

    def body(*values):
        args = flat_args.copy()
        for index, value in zip(positions, values):
            args[index] = value
        return impl(function, input_spec, tuple(args), checked_output)

    try:
        traced = trace_without_new_guards(body, examples)
    except Exception as error:
        raise RuntimeError(f"Cannot trace flat_apply callable without specializing runtime data: {error}") from error
    with graph.inserting_before(node):
        result = _inline(graph, owner, traced, operands, prefix)
    leaves = []
    torch.fx.node.map_aggregate(result, leaves.append)
    if not all(isinstance(leaf, torch.fx.Node) for leaf in leaves):
        raise RuntimeError("flat_apply supports tensor or symbolic outputs and nested lists/tuples of them")
    _replace_with_structure(node, result)
    metadata = [arg for arg in node.args[:2] if isinstance(arg, torch.fx.Node)]
    graph.erase_node(node)
    for arg in metadata:
        if not arg.users:
            graph.erase_node(arg)


class _VmapInterpreter(torch.fx.Interpreter):
    """Run a vmap region with runtime vmap levels instead of the exported ones."""

    def __init__(self, module):
        super().__init__(module)
        self.levels = {}
        self.active_levels = []

    def call_function(self, target, args, kwargs):
        name = predispatch_name(target)
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


def _check_vmap_region(module):
    for submodule in module.modules():
        if not isinstance(submodule, torch.fx.GraphModule):
            continue
        for node in submodule.graph.nodes:
            if node.op != "call_function":
                continue
            if predispatch_name(node.target) == "_vmap_increment_nesting" and node.args[1] != "error":
                raise RuntimeError("vmap supports only randomness='error'.")
            if isinstance(node.target, torch._ops.OpOverload):
                if node.target._schema.is_mutable:
                    raise RuntimeError("Mutation inside vmap is not supported.")
                if torch.Tag.nondeterministic_seeded in node.target.tags:
                    raise RuntimeError("Random operations inside vmap are not supported.")


def _vmap_region(start):
    region = []
    depth = 0
    node = start
    while node.op != "output":
        name = predispatch_name(node.target)
        if name == "_vmap_increment_nesting":
            depth += 1
        elif name == "_vmap_decrement_nesting":
            depth -= 1
        region.append(node)
        if depth == 0:
            return region
        node = node.next
    raise RuntimeError("Unbalanced vmap nesting in exported graph.")


def _lower_vmap_region(graph, owner, start, prefix):
    """Replace the vmap region beginning at start with its vectorized trace and return the node after it."""
    region = _vmap_region(start)
    region_set = set(region)
    operands = list(dict.fromkeys(value for node in region for value in node.all_input_nodes
                                  if value not in region_set and value.op != "get_attr"))
    outputs = [node for node in region if any(user not in region_set for user in node.users)]
    body_graph = torch.fx.Graph()
    values = {}
    examples = []
    for operand in operands:
        example = operand.meta.get("val")
        if not isinstance(example, FX_RUNTIME_TYPES):
            raise RuntimeError("vmap lowering requires tensor or symbolic FX metadata for its operands.")
        examples.append(example)
        values[operand] = body_graph.placeholder(operand.name)
    for node in region:
        for operand in node.all_input_nodes:
            if operand not in values and operand.op == "get_attr":
                values[operand] = body_graph.node_copy(operand, values.__getitem__)
        values[node] = body_graph.node_copy(node, values.__getitem__)
    body_graph.output(tuple(values[node] for node in outputs))
    body = torch.fx.GraphModule(owner, body_graph)
    _check_vmap_region(body)
    lowered = trace_without_new_guards(_VmapInterpreter(body).run, examples)
    with graph.inserting_before(start):
        returned = _inline(graph, owner, lowered, operands, prefix)
    for original, value in zip(outputs, returned):
        original.replace_all_uses_with(value)
    following = region[-1].next
    for node in reversed(region):
        graph.erase_node(node)
    return following


def normalize_graph_module(module):
    """Inline flat_apply callables and lower vmap regions in module and its GraphModule children.

    The original module is left unchanged; it is returned as is when there is nothing to normalize.
    """
    children = {name: normalize_graph_module(child) for name, child in module.named_children()
                if isinstance(child, torch.fx.GraphModule)}
    children = {name: child for name, child in children.items() if child is not getattr(module, name)}
    if not children and not any(_is_flat_apply(node) or predispatch_name(node.target) for node in module.graph.nodes):
        return module

    graph = torch.fx.Graph()
    graph.output(graph.graph_copy(module.graph, {}))
    result = torch.fx.GraphModule(module, graph)
    for name, child in children.items():
        setattr(result, name, child)
    ordinal = count()
    for node in list(graph.nodes):
        if _is_flat_apply(node):
            _inline_flat_apply(graph, result, node, f"_ov_flat_apply_{next(ordinal)}")
    node = next(iter(graph.nodes))
    while node.op != "output":
        if predispatch_name(node.target) == "_vmap_increment_nesting":
            node = _lower_vmap_region(graph, result, node, f"_ov_vmap_{next(ordinal)}")
        else:
            node = node.next
    for node in list(graph.nodes):
        if predispatch_name(node.target) == "lazy_load_decompositions":
            if node.users:
                raise RuntimeError("Unexpected users of vmap decomposition initialization.")
            graph.erase_node(node)
    graph.lint()
    result.recompile()
    return result
