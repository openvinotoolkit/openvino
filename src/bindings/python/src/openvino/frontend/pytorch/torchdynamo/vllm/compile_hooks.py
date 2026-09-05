# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""vLLM-specific compile-time hooks.

Functions called from torchdynamo.compile.openvino_compile to keep the
generic compile path free of vLLM-specific knowledge. Each hook is a
no-op when the input graph does not have the corresponding vLLM marker
(e.g. __pa__ Parameter prefix, vLLM-style Concat patterns).
"""

import logging
import os

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Consolidated entry points called from torchdynamo.compile.openvino_compile.
# Keeps compile.py free of vLLM-hook boilerplate: one try/except at each of
# two call sites, in place of the previous three.
# ---------------------------------------------------------------------------

def apply_post_convert(om, options):
    """Run all vLLM hooks that operate on the freshly-converted OV Model.

    Called right after ``fe.convert(im)`` and before the model is serialized
    or its input shapes are set. Currently: register unregistered ``__pa__``
    Parameters, normalize symint-heavy Concat ranks, and (when
    options["fc_decompress"] is True) rewrite MatMul(X, Const_f16/bf16) into
    the oneDNN BRGEMM decompression form.

    Each sub-hook is a no-op on graphs without the matching pattern.
    """
    register_pa_parameters(om)
    normalize_concat_ranks(om)
    from openvino.frontend.pytorch.torchdynamo.vllm.preset import bool_opt
    if bool_opt(options, "fc_decompress", True):
        rewrite_fc_decompression(om)


def apply_input_shapes(om, args, options, gm=None):
    """vLLM-shaped input handling: resolve Python-int FX inputs and set the
    partial shapes of the remaining tensor Parameters.

    ``gm`` is the FX GraphModule being compiled. It is optional so older
    callers keep working, but without it the int inputs can only be frozen at
    their trace-time values (see bake_symint_constants).

    Returns True if this hook handled the input shaping; False if it should
    fall through to the caller's upstream loop. Falls through when there are
    no int args AND the caller did not opt into the vLLM preset.
    """
    from openvino.frontend.pytorch.torchdynamo.vllm.preset import bool_opt
    if not (bool_opt(options, "vllm", False) or any(isinstance(a, int) for a in args)):
        return False
    bake_symint_constants(
        om, args, dyn_shapes=bool_opt(options, "dynamic_shapes", True), gm=gm)
    return True


def apply_post_config(config, device, options):
    """Run all vLLM hooks that fill in the OV core.compile_model config.

    Called after the caller has built ``config`` from ``_get_config(options)``
    and set CACHE_DIR. No-op on non-CPU devices.
    """
    apply_kv_cache_config_defaults(config, device, options)


def widen_affinity_if_needed(options):
    """Widen process CPU affinity to all cores when the current mask is
    narrower than the requested OV thread count.

    vLLM's ``init_cpu_threads_env`` pins the worker process to a single CPU
    before ``torch.compile`` runs. TBB/OV sample process affinity on their
    first parallel use, so a 1-CPU mask would lock ``INFERENCE_NUM_THREADS=1``
    regardless of the config we pass. Widen the mask before ``core.compile``
    so the OV thread pool inherits a useful mask at creation time.

    No-op on non-Linux systems (``sched_getaffinity`` unavailable) and on
    graphs whose affinity is already at least as wide as the requested
    thread count.
    """
    try:
        cur = os.sched_getaffinity(0)
        from openvino.frontend.pytorch.torchdynamo.backend_utils import _get_config
        cfg = _get_config(options) or {}
        req = int(cfg.get("INFERENCE_NUM_THREADS",
                          os.environ.get("OV_INFERENCE_NUM_THREADS", "0")) or 0)
        if req == 0 or len(cur) < req:
            os.sched_setaffinity(0, set(range(os.cpu_count() or 1)))
    except Exception as _e:
        logger.debug("widen_affinity skipped: %s", _e)


def symint_shape_sources(gm, args):
    """Map int FX inputs to the tensor input dimension that carries them.

    Under ``torch.compile(dynamic=True)`` dynamo hands the backend the
    symbolic sizes as ordinary Python-int graph inputs alongside the tensors
    whose shapes those symbols describe. A vLLM prefill graph, for instance,
    arrives as::

        ph[0] arg164_1  TENSOR int32 [s72]     <- input_ids
        ph[1] arg163_1  SYMINT expr=s72        <- num_tokens
        ph[2] arg167_1  TENSOR int64 [s72]     <- positions
        ph[4] arg166_1  SYMINT expr=s72

    Freezing ph[1]/ph[4] at their trace-time value is what makes the compiled
    model valid for one prefill length only. But the graph states outright
    that both equal ``input_ids.shape[0]``, so they can be rebuilt from a
    ShapeOf of the live tensor instead, and the model becomes valid for every
    length.

    Returns ``{int_arg_index: (tensor_arg_index, dim)}``, covering only the
    symbols that some tensor input's shape actually carries. The mapping is
    read off ``meta['val']`` (a SymInt for size inputs, a FakeTensor with
    SymInt dims for tensors), so it is exact -- no matching of trace-time
    integer values, which would confuse two symbols that happen to coincide
    on this trace.

    Returns ``{}`` when ``gm`` is None, when the placeholders do not line up
    with ``args``, or for a static trace (where the sizes are plain ints with
    no symbol to source).
    """
    if gm is None:
        return {}
    try:
        import torch
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        if len(placeholders) != len(args):
            return {}
        # symbol -> (tensor arg index, dim). First tensor carrying a symbol
        # wins; any of them is equally valid as a source.
        symbol_src = {}
        for idx, node in enumerate(placeholders):
            val = node.meta.get("val", None)
            if not isinstance(val, torch.Tensor):
                continue
            for dim, extent in enumerate(val.shape):
                if isinstance(extent, torch.SymInt):
                    symbol_src.setdefault(str(extent.node.expr), (idx, dim))
        sources = {}
        for idx, node in enumerate(placeholders):
            if not isinstance(args[idx], int):
                continue
            val = node.meta.get("val", None)
            if not isinstance(val, torch.SymInt):
                continue
            src = symbol_src.get(str(val.node.expr))
            if src is not None:
                sources[idx] = src
        return sources
    except Exception as e:
        logger.debug("symint shape sourcing unavailable: %s", e)
        return {}


def bake_symint_constants(om, args, dyn_shapes: bool = True, gm=None):
    """Resolve integer FX inputs and drop their Parameters.

    vLLM's decode FX graphs are symint-heavy: seq_lens, past_lens, and
    block-table sizes appear as Python-int placeholders. If we leave them as
    OV Parameters, ov shape inference uses the unset Parameter upper bound of
    0 and collapses downstream Broadcast/Reshape outputs to size 0. So each
    one is replaced by a value that shape inference can propagate, and its
    Parameter removed. There are two ways to produce that value:

    * ``Gather(ShapeOf(tensor_input), dim)`` when the graph tells us which
      tensor dimension the symbol denotes (see symint_shape_sources). The
      value then tracks the real input at every call, so one compiled model
      serves all shapes.
    * a Constant holding this trace's value otherwise -- correct only for
      as long as dynamo's shape guards force a retrace per distinct value,
      which is exactly why a static trace costs a recompile per prefill
      length.

    Also sets the element-type and partial-shape of the remaining tensor
    Parameters. Tensor inputs get all-dynamic shapes when ``dyn_shapes`` is
    True, or when every int input was sourced from a ShapeOf: in that case
    pinning the tensors to their trace-time shapes would const-fold the
    ShapeOf right back into the frozen value we just avoided.

    Non-vLLM callers can skip this entirely; the caller is responsible for
    setting element_type and partial_shape on the remaining Parameters.
    """
    import torch
    import numpy as np
    from openvino import Type, PartialShape, opset1 as _opset1, opset8 as _opset8

    _dtype_mapping = {
        torch.float32: Type.f32, torch.float64: Type.f64,
        torch.float16: Type.f16, torch.int64: Type.i64,
        torch.int32: Type.i32, torch.uint8: Type.u8,
        torch.int8: Type.i8, torch.bool: Type.boolean,
    }

    sources = symint_shape_sources(gm, args)
    n_int_args = sum(1 for a in args if isinstance(a, int))

    params_to_remove = []
    for idx, input_data in enumerate(args):
        if isinstance(input_data, int):
            param_node = om.inputs[idx].get_node()
            src = sources.get(idx)
            if src is None:
                repl = _opset1.constant(np.array([int(input_data)], dtype=np.int64))
            else:
                # om.inputs is still 1:1 with args here -- Parameters are
                # removed only after this loop -- so the tensor arg index
                # indexes om.inputs directly. i64[1] matches the shape and
                # type of the Constant this replaces, so consumers that were
                # built for the baked form keep working unchanged.
                tensor_arg_idx, dim = src
                shape_of = _opset8.shape_of(om.inputs[tensor_arg_idx], output_type="i64")
                repl = _opset8.gather(
                    shape_of,
                    _opset1.constant(np.array([dim], dtype=np.int64)),
                    _opset1.constant(np.array(0, dtype=np.int64)))
            for consumer in list(param_node.output(0).get_target_inputs()):
                consumer.replace_source_output(repl.output(0))
            params_to_remove.append(param_node)
    for p in params_to_remove:
        om.remove_parameter(p)

    all_symints_sourced = n_int_args > 0 and len(sources) == n_int_args
    if all_symints_sourced and not dyn_shapes:
        logger.debug("symint inputs sourced from ShapeOf; forcing dynamic input shapes")
    dyn = dyn_shapes or all_symints_sourced

    tensor_idx = 0
    for input_data in args:
        if isinstance(input_data, int):
            continue
        om.inputs[tensor_idx].get_node().set_element_type(_dtype_mapping[input_data.dtype])
        if dyn:
            om.inputs[tensor_idx].get_node().set_partial_shape(
                PartialShape([-1] * input_data.ndim))
        else:
            om.inputs[tensor_idx].get_node().set_partial_shape(
                PartialShape(list(input_data.size())))
        tensor_idx += 1

    # NOTE: set_partial_shape above only touches the Parameter -- every
    # downstream node still holds the shape the frontend recorded at conversion,
    # i.e. this trace's concrete sizes. Those have to be re-inferred, or
    # ConstantFolding evaluates the ShapeOf we just built from the *node's*
    # stale output shape and folds it straight back to the frozen size. The
    # caller does it: openvino_compile calls validate_nodes_and_infer_types()
    # immediately after apply_input_shapes() returns, and it is the only caller.


def register_pa_parameters(om):
    """Register dangling ``__pa__``-prefixed Parameters as model inputs.

    The vLLM paged_attention C++ translator emits side-channel Parameters
    for KV cache, block tables, past_lens, etc. Without this registration
    the Model fails validation with ``unregistered_parameters`` errors.

    No-op on graphs without ``__pa__`` Parameters.
    """
    try:
        existing_ids = {id(p) for p in om.get_parameters()}
        to_add = []
        for node in om.get_ordered_ops():
            if node.get_type_name() != "Parameter":
                continue
            if id(node) in existing_ids:
                continue
            if node.get_friendly_name().startswith("__pa__"):
                to_add.append(node)
        if to_add:
            om.add_parameters(to_add)
    except Exception as e:
        logger.debug("PA parameter registration skipped: %s", e)


def normalize_concat_ranks(om):
    """Strip redundant Unsqueeze wrappers feeding Concat.

    Some FX graphs (notably vLLM's symint-heavy ones) emit Unsqueeze
    wrappers that leave rank-mismatched Concat inputs for list-construct
    nodes. Walk the graph until validate_nodes_and_infer_types succeeds,
    bypassing each Unsqueeze whose inner input is already rank>=1.

    No-op on graphs that already pass shape inference.
    """
    def _rank_ge_1(val):
        n = val.get_node()
        ps = val.get_partial_shape()
        if ps.rank.is_static and ps.rank.get_length() >= 1:
            return True
        if n.get_type_name() == "Constant":
            return len(n.get_output_shape(0)) >= 1
        return False

    try:
        for _ in range(64):
            try:
                om.validate_nodes_and_infer_types()
                return
            except Exception:
                pass
            made_change = False
            for node in list(om.get_ordered_ops()):
                if node.get_type_name() != "Concat":
                    continue
                if node.get_input_size() < 2:
                    continue
                for i in range(node.get_input_size()):
                    src = node.input_value(i)
                    src_node = src.get_node()
                    if src_node.get_type_name() != "Unsqueeze":
                        continue
                    inner = src_node.input_value(0)
                    if _rank_ge_1(inner):
                        node.input(i).replace_source_output(inner)
                        made_change = True
            if not made_change:
                return
    except Exception as e:
        logger.debug("concat-rank normalization skipped: %s", e)


def apply_kv_cache_config_defaults(config, device, options=None):
    """Fill vLLM-specific KV-cache and FC-quantization defaults into the OV
    CPU config dict.

    Only applies when device == "CPU". Caller-supplied entries in `config`
    take priority. Merges the vLLM preset config dict when the caller
    opted in with options[\"vllm\"]=True. Also reads env-var fallbacks
    for backward compat with the legacy environment-driven setup.

    No-op on non-CPU devices.
    """
    if device != "CPU":
        return
    import os
    # Merge the vLLM preset dict (KV_CACHE_PRECISION=bf16 etc.) when the
    # caller opted into options["vllm"]=True. Caller-supplied keys win.
    try:
        from openvino.frontend.pytorch.torchdynamo.vllm import preset as _preset
        if _preset.is_vllm_preset(options):
            for k, v in _preset._PRESET_CONFIG.items():
                config.setdefault(k, v)
    except Exception:
        pass
    if "KV_CACHE_PRECISION" not in config:
        # f32 is the verified-correct default for the OV CPU PA op; the
        # vLLM preset overrides this to bf16 when options["vllm"]=True.
        config["KV_CACHE_PRECISION"] = os.environ.get("OV_KV_CACHE_PRECISION", "f32")
    if "DYNAMIC_QUANTIZATION_GROUP_SIZE" not in config:
        # Quantize FC activations to int8 on the fly (vnni int8 GEMM is
        # much faster than f32 GEMM). Matches OV GenAI CPU behavior.
        config["DYNAMIC_QUANTIZATION_GROUP_SIZE"] = int(
            os.environ.get("DYNAMIC_QUANTIZATION_GROUP_SIZE", "32"))
    inf_hint = os.environ.get("OV_INFERENCE_PRECISION_HINT", "f16")
    if "INFERENCE_PRECISION_HINT" not in config and inf_hint:
        # Let the plugin pick its narrow-float GEMM path. PA op is fenced
        # with Convert(f32) in the translator so it stays f32 regardless.
        config["INFERENCE_PRECISION_HINT"] = inf_hint


def rewrite_fc_decompression(om):
    """Rewrite MatMul(X, Const_f16/bf16) into the oneDNN-BRGEMM-friendly form.

    For each MatMul that consumes a constant fp16/bf16 weight (optionally
    transposed via a [1,0] permutation), insert a Convert to f32 marked as
    decompression so the CPU plugin ConvertMatMulToFC pass routes it to
    brgemm_avx512_f32 instead of the slower gemm_mlas_f32 fallback.

    Activation is upcast to f32 and its consumers downcast back to the native
    dtype so downstream ops keep their precision. f32 weights and quantized
    paths are skipped.

    No-op on graphs without matching MatMul patterns. Lives here so the
    generic compile.py stays small; not vLLM-specific by itself but we keep
    all narrow-float / KV-cache / PA-related compile-time edits together.
    """
    from openvino import opset1 as _o1
    from openvino import Type
    try:
        for mm in list(om.get_ordered_ops()):
            if mm.get_type_name() != "MatMul":
                continue
            try:
                tb = mm.get_transpose_b()
            except Exception:
                continue
            if tb:
                continue  # already transpose_b=true
            src = mm.input_value(1).get_node()
            const = None
            new_tb = False
            if src.get_type_name() == "Transpose":
                inner = src.input_value(0).get_node()
                if inner.get_type_name() == "Constant":
                    perm_node = src.input_value(1).get_node()
                    if perm_node.get_type_name() == "Constant":
                        perm = list(perm_node.get_data().flatten())
                        if perm == [1, 0]:
                            const = inner
                            new_tb = True
            elif src.get_type_name() == "Constant":
                const = src
            if const is None:
                continue
            # Plugin\x27s weight-decompression FC path accepts inputType=f32
            # with weightsType in {f16, bf16}. f32 weights need no decompression.
            w_et = const.get_element_type()
            if w_et not in (Type.f16, Type.bf16):
                continue
            conv_w = _o1.convert(const.output(0), "f32")
            try:
                # Mark the Convert as decompression so the plugin pattern
                # matcher accepts it (key == "decompression_0", matching the
                # internal is_decompression() probe).
                conv_w.get_rt_info()["decompression_0"] = True
            except Exception:
                pass
            mm.input(1).replace_source_output(conv_w.output(0))
            # Upcast activation to f32; downcast each consumer of MatMul back.
            act_src = mm.input_value(0)
            act_et = act_src.get_element_type()
            if act_et in (Type.f16, Type.bf16):
                conv_a = _o1.convert(act_src, "f32")
                mm.input(0).replace_source_output(conv_a.output(0))
                out = mm.output(0)
                consumers = list(out.get_target_inputs())
                down = _o1.convert(out, act_et)
                for cin in consumers:
                    cin.replace_source_output(down.output(0))
            try:
                mm.set_transpose_b(new_tb if new_tb else mm.get_transpose_b())
            except Exception:
                pass
        om.validate_nodes_and_infer_types()
    except Exception as e:
        logger.debug("FC_DECOMPRESS rewrite failed: %s", e)
