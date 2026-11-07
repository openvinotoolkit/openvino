# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Helpers shared by the vLLM FX rewrites (paged attention, Gated DeltaNet, fused MoE)."""

# mypy: ignore-errors

import math
import operator


def is_auto_functionalized(node, get_op) -> bool:
    """Match auto_functionalized_v2(op, ...) with op = get_op(); False if op is not registered."""
    import torch
    if node.op != "call_function" or not node.args:
        return False
    try:
        return (node.target is torch.ops.higher_order.auto_functionalized_v2
                and node.args[0] is get_op())
    except (AttributeError, RuntimeError):
        return False


def replace_mutated_output(gm, node, new_node, base_index) -> bool:
    """Route users of the mutated base `base_index` of `node` to `new_node`.

    auto_functionalized_v2 returns the op result at 0 and each mutated base at 1 + index.
    Erases `node` and its now-unused base allocations once nothing else reads them;
    returns whether `node` was erased.
    """
    out_index = 1 + (base_index or 0)
    for user in list(node.users):
        if (user.op == "call_function" and user.target is operator.getitem
                and len(user.args) == 2 and user.args[1] == out_index):
            user.replace_all_uses_with(new_node)
            gm.graph.erase_node(user)
    if node.users:
        return False
    bases = node.kwargs.get("_all_bases") or []
    gm.graph.erase_node(node)
    for base in bases:
        # Only computed buffers: erasing a placeholder would change the graph signature.
        if getattr(base, "op", None) == "call_function" and not base.users:
            gm.graph.erase_node(base)
    return True


def packing_permutation(pack, shape, dtype):
    """Flat source index of each position produced by `pack`, or None if it is not a pure permutation.

    Packs the base-16 digits of every flat index (exact in any float dtype) and reads them back.
    """
    import torch
    count = math.prod(shape)
    index = torch.arange(count)
    source = torch.zeros(count, dtype=torch.long)
    digit = 0
    while 16 ** digit < count:
        nibble = ((index >> (4 * digit)) & 15).to(dtype).reshape(shape)
        source += pack(nibble).reshape(-1).long() << (4 * digit)
        digit += 1
    return source if torch.equal(torch.sort(source)[0], index) else None
