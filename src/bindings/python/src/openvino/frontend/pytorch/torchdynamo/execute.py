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


_PA_FIELDS = (
    "key_cache", "value_cache", "past_lens", "subsequence_begins",
    "block_indices", "block_indices_begins", "max_context_len",
)


def _bind_paged_attention_side_channel(compiled):
    """For every compiled-model input named "__pa__<layer>__<field>", look up
    the tensor from vllm.forward_context.get_forward_context() and return a
    mapping {name: numpy_array}.

    Relies on vLLM CPU attention metadata layout (CPUAttentionMetadata).
    """
    try:
        from vllm.forward_context import get_forward_context
    except Exception:
        return {}

    try:
        ctx = get_forward_context()
    except AssertionError:
        # No ForwardContext set (e.g. during CPU warmup paths); fall back to
        # empty tensors so PA at least doesn't segfault.
        ctx = None

    result = {}
    # Group the PA inputs by layer_name first
    layer_to_fields = {}  # layer_name -> {field: Input}
    for inp in compiled.inputs:
        for nm in inp.get_names():
            if not nm.startswith("__pa__"):
                continue
            # "__pa__<layer>__<field>"
            rest = nm[len("__pa__"):]
            # field is one of _PA_FIELDS; find which one matches the tail
            for field in _PA_FIELDS:
                suffix = "__" + field
                if rest.endswith(suffix):
                    layer_name = rest[: -len(suffix)]
                    layer_to_fields.setdefault(layer_name, {})[field] = nm
                    break
            break

    for layer_name, fields in layer_to_fields.items():
        attn_meta = None
        kv_cache = None
        if ctx is not None:
            try:
                am_map = ctx.attn_metadata
                if isinstance(am_map, dict):
                    attn_meta = am_map.get(layer_name)
                # kv cache: vLLM stores it in static_forward_context
                nc_layers = ctx.no_compile_layers
                layer_obj = nc_layers.get(layer_name) if isinstance(nc_layers, dict) else None
                if layer_obj is not None and hasattr(layer_obj, "kv_cache"):
                    kv_cache = layer_obj.kv_cache
                    # kv_cache may be list indexed by virtual_engine
                    if isinstance(kv_cache, list):
                        kv_cache = kv_cache[ctx.virtual_engine]
            except Exception:
                pass

        # Prepare numpy arrays for each field
        def _nz(dtype, shape=(1,)):
            return np.zeros(shape, dtype=dtype)

        key_cache_np = value_cache_np = None
        if kv_cache is not None:
            try:
                # CPU layout: [2, num_blocks, num_kv_heads, block_size, head_size]
                kc, vc = kv_cache.unbind(0)
                key_cache_np = kc.contiguous().numpy()
                value_cache_np = vc.contiguous().numpy()
            except Exception:
                pass
        if key_cache_np is None:
            key_cache_np = _nz(np.float32, (1, 1, 1, 1))
            value_cache_np = _nz(np.float32, (1, 1, 1, 1))

        past_lens_np = _nz(np.int32)
        subseq_begins_np = _nz(np.int32, (2,))
        block_indices_np = _nz(np.int32)
        block_indices_begins_np = _nz(np.int32, (2,))
        max_ctx_len_np = np.array(0, dtype=np.int32)

        if attn_meta is not None:
            try:
                # CPUAttentionMetadata fields:
                #   seq_lens, query_start_loc, block_table, slot_mapping, ...
                seq_lens = getattr(attn_meta, "seq_lens", None)
                qsl = getattr(attn_meta, "query_start_loc", None)
                block_table = getattr(attn_meta, "block_table", None)
                if seq_lens is not None and qsl is not None:
                    # past_lens = seq_lens - (query_start_loc[1:] - query_start_loc[:-1])
                    q_lens = qsl[1:] - qsl[:-1]
                    pl = (seq_lens - q_lens).to(torch.int32).contiguous()
                    past_lens_np = pl.numpy()
                    subseq_begins_np = qsl.to(torch.int32).contiguous().numpy()
                    max_ctx_len_np = np.array(int(seq_lens.max().item()), dtype=np.int32)
                if block_table is not None:
                    bt = block_table.to(torch.int32).contiguous()
                    # Flatten per-row into a single block_indices vector
                    # plus block_indices_begins giving row starts
                    rows = bt.shape[0] if bt.ndim > 0 else 1
                    flat = bt.flatten().numpy()
                    row_len = bt.shape[1] if bt.ndim > 1 else 1
                    begins = np.arange(rows + 1, dtype=np.int32) * row_len
                    block_indices_np = flat
                    block_indices_begins_np = begins
            except Exception:
                pass

        for field, name in fields.items():
            if field == "key_cache":
                result[name] = key_cache_np
            elif field == "value_cache":
                result[name] = value_cache_np
            elif field == "past_lens":
                result[name] = past_lens_np
            elif field == "subsequence_begins":
                result[name] = subseq_begins_np
            elif field == "block_indices":
                result[name] = block_indices_np
            elif field == "block_indices_begins":
                result[name] = block_indices_begins_np
            elif field == "max_context_len":
                result[name] = max_ctx_len_np

    return result


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

    # Bind vLLM PagedAttention side-channel Parameters (KV cache / block
    # tables / past_lens / ...) that the paged_attention C++ translator added
    # as extra Parameters with names like "__pa__<layer_name>__<field>".
    # The tensors come from vllm.forward_context.get_forward_context().
    _pa_inputs_by_pos = {}
    if any(inp.get_names() and any(n.startswith("__pa__") for n in inp.get_names())
           for inp in compiled.inputs):
        _pa_inputs_by_pos = _bind_paged_attention_side_channel(compiled)

    if _pa_inputs_by_pos:
        # Use dict form so we can set by Input object
        _call_kwargs = {}
        _tensor_pos = 0
        for i, inp in enumerate(compiled.inputs):
            _names = inp.get_names()
            pa_tensor = None
            for n in _names:
                if n.startswith("__pa__") and n in _pa_inputs_by_pos:
                    pa_tensor = _pa_inputs_by_pos[n]
                    break
            if pa_tensor is not None:
                _call_kwargs[inp] = pa_tensor
            else:
                _call_kwargs[inp] = ov_inputs[_tensor_pos]
                _tensor_pos += 1
        res = req.infer(_call_kwargs, share_inputs=True, share_outputs=False)
    else:
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
            import os as _os
            if _os.environ.get("OV_TRACE_FALLBACK"):
                print(f"[OV_FALLBACK partition={self.partition_id}] {type(e).__name__}: {str(e)[:800]}", flush=True)
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
