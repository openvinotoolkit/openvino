# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# mypy: ignore-errors

from contextlib import ExitStack

import torch


def _predispatch_name(node):
    if node.op == "call_function" and getattr(node.target, "__module__", None) == "torch._functorch.predispatch":
        return node.target.__name__
    return ""


def _region(start):
    region = []
    depth = 0
    node = start
    while node.op != "output":
        name = _predispatch_name(node)
        depth += (name == "_vmap_increment_nesting") - (name == "_vmap_decrement_nesting")
        region.append(node)
        if depth == 0:
            return region
        node = node.next
    raise RuntimeError("Unbalanced vmap nesting in exported graph.")


def _check(body):
    for module in body.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue
        for node in module.graph.nodes:
            if _predispatch_name(node) == "_vmap_increment_nesting" and node.args[1] != "error":
                raise RuntimeError("vmap supports only randomness='error'.")
            if isinstance(node.target, torch._ops.OpOverload):
                if node.target._schema.is_mutable:
                    raise RuntimeError("Mutation inside vmap is not supported.")
                if torch.Tag.nondeterministic_seeded in node.target.tags:
                    raise RuntimeError("Random operations inside vmap are not supported.")


def _trace(body, examples):
    """Trace body with vectorized ATen operations, failing instead of specializing symbolic shapes."""
    from torch.fx.experimental.proxy_tensor import make_fx

    _check(body)

    def run(*args):
        level = torch._C._functorch.maybe_current_level()
        try:
            return torch.fx.Interpreter(body).run(*args)
        finally:
            while torch._C._functorch.maybe_current_level() != level:
                torch._C._functorch._vmap_decrement_nesting()

    shape_envs = {getattr(example.fake_mode if isinstance(example, torch.Tensor) else example.node, "shape_env", None)
                  for example in examples} - {None}
    with ExitStack() as stack:
        for shape_env in shape_envs:
            stack.enter_context(shape_env.error_on_new_guards())
        return make_fx(run, tracing_mode="fake", _allow_non_fake_inputs=True)(*examples)


def _preserve_output_aliases(lowered, outputs):
    from torch.utils._pytree import tree_map

    traced_outputs = next(node for node in lowered.graph.nodes if node.op == "output").args[0]
    aliases = [(traced.meta.get("val"), original.meta.get("val"))
               for traced, original in zip(traced_outputs, outputs)]
    aliases = [(traced, original) for traced, original in aliases
               if isinstance(traced, torch.Tensor) and isinstance(original, torch.Tensor)]

    def reconcile(value):
        if isinstance(value, torch.Tensor):
            for traced, original in aliases:
                if torch._C._is_alias_of(value, traced):
                    # Aliases can have different dtypes, so align storage offsets in bytes.
                    offset_bytes = (value.storage_offset() * value.element_size()
                                    - traced.storage_offset() * traced.element_size()
                                    + original.storage_offset() * original.element_size())
                    return value.new_empty((0,)).set_(
                        original.untyped_storage(), offset_bytes // value.element_size(), value.shape, value.stride())
        return value

    for node in lowered.graph.nodes:
        if "val" in node.meta:
            node.meta["val"] = tree_map(reconcile, node.meta["val"])


def lower_vmap_regions(module):
    """Return a copy of module with each torch.vmap region replaced by its vectorized trace.

    The module itself is returned when it has no vmap regions.
    """
    if not any(_predispatch_name(node) == "_vmap_increment_nesting" for node in module.graph.nodes):
        return module

    graph = torch.fx.Graph()
    graph.set_codegen(module.graph._codegen)
    graph.output(graph.graph_copy(module.graph, {}))
    result = torch.fx.GraphModule(module, graph)
    ordinal = 0
    start = next(iter(graph.nodes))
    while start.op != "output":
        if _predispatch_name(start) != "_vmap_increment_nesting":
            start = start.next
            continue
        region = _region(start)
        region_set = set(region)
        operands = list(dict.fromkeys(value for node in region for value in node.all_input_nodes
                                      if value not in region_set and value.op != "get_attr"))
        outputs = [node for node in region if any(user not in region_set for user in node.users)]
        if any("val" not in operand.meta for operand in operands):
            raise RuntimeError("vmap lowering requires FX metadata for the region operands.")
        body_graph = torch.fx.Graph()
        values = {operand: body_graph.placeholder(operand.name) for operand in operands}
        for node in region:
            for operand in node.all_input_nodes:
                if operand not in values:
                    values[operand] = body_graph.node_copy(operand, values.__getitem__)
            values[node] = body_graph.node_copy(node, values.__getitem__)
        body_graph.output(tuple(values[node] for node in outputs))
        lowered = _trace(torch.fx.GraphModule(result, body_graph), [operand.meta["val"] for operand in operands])
        _preserve_output_aliases(lowered, outputs)

        while hasattr(result, f"_ov_vmap_{ordinal}"):
            ordinal += 1
        prefix = f"_ov_vmap_{ordinal}"
        ordinal += 1
        result.add_submodule(prefix, lowered)
        replacement = dict(zip((node for node in lowered.graph.nodes if node.op == "placeholder"), operands))
        with graph.inserting_before(start):
            for node in lowered.graph.nodes:
                if node.op == "output":
                    for original, value in zip(outputs, torch.fx.map_arg(node.args[0], replacement.__getitem__)):
                        original.replace_all_uses_with(value)
                elif node.op == "get_attr":
                    replacement[node] = graph.get_attr(f"{prefix}.{node.target}")
                    replacement[node].meta = node.meta.copy()
                elif node.op != "placeholder":
                    replacement[node] = graph.node_copy(node, replacement.__getitem__)
        start = region[-1].next
        for node in reversed(region):
            graph.erase_node(node)
    for node in list(graph.nodes):
        if _predispatch_name(node) == "lazy_load_decompositions" and not node.users:
            graph.erase_node(node)
    graph.lint()
    result.recompile()
    return result
