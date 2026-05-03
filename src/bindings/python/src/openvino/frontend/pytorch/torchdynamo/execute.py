# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# mypy: ignore-errors

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from warnings import warn

import torch
import torch.overrides

from torch.fx import GraphModule
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from openvino.frontend import FrontEndManager
from openvino.frontend.pytorch.fx_decoder import TorchFXPythonDecoder
from openvino.frontend.pytorch.torchdynamo.partition import Partitioner
from openvino.frontend.pytorch.torchdynamo.compile import openvino_compile
from openvino import Core, Type, PartialShape
from openvino.frontend.pytorch.torchdynamo.backend_utils import _get_cache_dir, _get_device, _get_aot_autograd

from typing import Optional, Any

from torch.fx.experimental.proxy_tensor import make_fx, wrapper_and_args_for_make_fx

import logging
logger = logging.getLogger(__name__)


DEFAULT_OPENVINO_PYTHON_CONFIG = MappingProxyType(
    {
        "use_python_fusion_cache": True,
        "allow_single_op_fusion": True,
    },
)

compiled_cache = {}
req_cache = {}
max_openvino_partitions = 0
partitioned_modules = {}
# Cache keyed by structural hash of submodule FX graph + input dtype/rank.
# This lets us reuse a compiled OV model across dynamo retraces where the
# graph is structurally identical (typical during decode loops).
structural_cache = {}


def _structural_key(gm, args):
    """Structural hash of the FX graph that's stable across re-traces.

    Uses normalized node ops + consumer chain instead of gm.code (which has
    arbitrary name suffixes like 'arg99_1' vs 'arg132_1' that differ across
    traces despite identical structure).
    """
    try:
        parts = []
        # Assign index-based ids to placeholders so arg99_1/arg132_1 don't
        # produce different hashes for structurally identical graphs.
        node_id = {}
        ph_i = 0
        for n in gm.graph.nodes:
            if n.op == "placeholder":
                node_id[n] = f"ph{ph_i}"
                ph_i += 1
                parts.append(f"placeholder")
                continue
            # node target (stable)
            t = str(n.target) if hasattr(n, 'target') else str(n.op)
            # input edge descriptor: refer by node_id if known, else by op
            arg_descs = []
            for a in n.args:
                arg_descs.append(node_id.get(a, type(a).__name__))
            parts.append(f"{n.op}:{t}({','.join(arg_descs)})")
            node_id[n] = f"n{len(node_id)}"
    except Exception:
        parts = [str(id(gm))]
    sig = ["|".join(parts)]
    for a in args:
        if isinstance(a, torch.Tensor):
            sig.append(f"T{a.dtype}:{tuple(a.size())}")
        elif isinstance(a, int):
            sig.append(f"I:{a}")
        else:
            sig.append(f"S{type(a).__name__}")
    import hashlib
    return hashlib.sha256("|".join(sig).encode()).hexdigest()


def execute(
    gm: GraphModule,
    *args,
    executor: str = "openvino",
    executor_parameters: Optional[dict] = None,
    options: Optional[Any] = None,
):
    if executor == "openvino":
        return openvino_execute_partitioned(
            gm, *args, executor_parameters=executor_parameters, options=options
        )
    elif executor == "strictly_openvino":
        return openvino_execute(gm, *args, executor_parameters=executor_parameters)

    msg = (
        "Received unexpected value for 'executor': {0}. "
        "Allowed values are: openvino, strictly_openvino."
    ).format(executor)
    raise ValueError(msg)


import numpy as np


def execute_cached(compiled_model, *args):
    ov_inputs = [a.detach().cpu().numpy() for a in args]
    ov_inputs.reverse()
    res = compiled_model(ov_inputs)
    result = [torch.from_numpy(res[out]) for out in compiled_model.outputs]
    return result


def openvino_execute(
    gm: GraphModule,
    *args,
    executor_parameters=None,
    partition_id: int = 0,
    options=None,
):

    executor_parameters = executor_parameters or DEFAULT_OPENVINO_PYTHON_CONFIG

    use_cache = executor_parameters.get(
        "use_python_fusion_cache",
        DEFAULT_OPENVINO_PYTHON_CONFIG["use_python_fusion_cache"],
    )
    global compiled_cache  # noqa: F824

    model_hash_str = executor_parameters.get("model_hash_str", None)
    if model_hash_str is not None:
        fully_supported = False
        if len(model_hash_str) > 3 and model_hash_str[-3:] == "_fs":
            fully_supported = True
        if not fully_supported:
            model_hash_str = model_hash_str + "_p" + str(partition_id)

    # Include input shape in the cache key: OV bakes concrete shapes into the
    # compiled model, so reusing a compiled partition with different input
    # shapes yields zero-sized outputs (observed on vLLM decode where each
    # step has a different seq-length).
    shape_key = tuple(
        tuple(a.size()) if isinstance(a, torch.Tensor) else (type(a).__name__, a)
        for a in args
    )
    cache_key = (partition_id, shape_key)

    if use_cache and (cache_key in compiled_cache):
        compiled = compiled_cache[cache_key]
        req = req_cache[cache_key]
    else:
        struct_key = _structural_key(gm, args)
        if use_cache and struct_key in structural_cache:
            compiled, req = structural_cache[struct_key]
        else:
            compiled = openvino_compile(gm, *args, model_hash_str=model_hash_str, options=options)
            req = compiled.create_infer_request()
            structural_cache[struct_key] = (compiled, req)
        compiled_cache[cache_key] = compiled
        req_cache[cache_key] = req

    flat_args, _ = tree_flatten(args)
    ov_inputs = []
    for arg in flat_args:
        if isinstance(arg, int):
            # Int args are baked into the compiled OV model as Constants
            # (see compile.py). Don't pass them at infer time.
            continue
        t = arg.detach()
        if not t.is_contiguous():
            t = t.contiguous()
        # numpy view shares memory with torch tensor for CPU dense tensors
        ov_inputs.append(t.numpy())

    res = req.infer(ov_inputs, share_inputs=True, share_outputs=False)

    results1 = [torch.from_numpy(res[out]) for out in compiled.outputs]
    if len(results1) == 1:
        return results1[0]
    return results1


class OpenVINOGraphModule(torch.nn.Module):
    def __init__(self, gm, partition_id, use_python_fusion_cache, model_hash_str: str = None, options=None):
        super().__init__()
        self.gm = gm
        self.partition_id = partition_id
        self.executor_parameters = {"use_python_fusion_cache": use_python_fusion_cache,
                                    "model_hash_str": model_hash_str}
        self.perm_fallback = False
        self.options = options

    def __call__(self, *args):
        if self.perm_fallback:
            return self.gm(*args)

        try:
            result = openvino_execute(
                self.gm,
                *args,
                executor_parameters=self.executor_parameters,
                partition_id=self.partition_id,
                options=self.options,
            )
            logger.debug("OpenVINO graph execution successful")
        except Exception as e:
            logger.debug(
                f"OpenVINO execution failed with {e}. Falling back to native PyTorch execution."
            )
            self.perm_fallback = True
            return self.gm(*args)

        return result


def partition_graph(gm: GraphModule, use_python_fusion_cache: bool, model_hash_str: str = None, options=None):
    global max_openvino_partitions
    partition_id = max_openvino_partitions
    for node in gm.graph.nodes:
        # TODO: use a better way to identify fused submodule
        if node.op == "call_module" and "fused_" in node.name:
            openvino_submodule = getattr(gm, node.name)
            gm.delete_submodule(node.target)
            gm.add_submodule(
                node.target,
                OpenVINOGraphModule(openvino_submodule, partition_id, use_python_fusion_cache,
                                    model_hash_str=model_hash_str, options=options),
            )
            partition_id = partition_id + 1

    max_openvino_partitions = partition_id

    return gm


def openvino_execute_partitioned(gm: GraphModule, *args, executor_parameters=None, options=None):
    executor_parameters = executor_parameters or DEFAULT_OPENVINO_PYTHON_CONFIG

    global partitioned_modules  # noqa: F824

    use_python_fusion_cache = executor_parameters.get(
        "use_python_fusion_cache",
        DEFAULT_OPENVINO_PYTHON_CONFIG["use_python_fusion_cache"],
    )
    model_hash_str = executor_parameters.get("model_hash_str", None)

    signature = str(id(gm))
    if (not _get_aot_autograd(options)):
        for idx, input_data in enumerate(args):
            if isinstance(input_data, torch.Tensor):
                # Shape-agnostic: key only on dtype/rank so dynamic OV model is reused
                # across varying seq-lengths during decode instead of recompiling.
                signature = (
                    signature
                    + "_"
                    + str(idx)
                    + ":"
                    + str(input_data.type())[6:]
                    + ":rank"
                    + str(input_data.dim())
                )
            else:
                signature = (
                    signature
                    + "_"
                    + str(idx)
                    + ":"
                    + type(input_data).__name__
                )

    if signature not in partitioned_modules:
        partitioned_modules[signature] = partition_graph(
            gm, use_python_fusion_cache=use_python_fusion_cache, model_hash_str=model_hash_str, options=options
        )
    return partitioned_modules[signature](*args)


def clear_caches():
    global partitioned_modules  # noqa: F824
    global compiled_cache  # noqa: F824

    compiled_cache.clear()
    partitioned_modules.clear()
