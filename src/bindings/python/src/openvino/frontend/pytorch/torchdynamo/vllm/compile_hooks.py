# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""vLLM-specific compile-time hooks.

Called from torchdynamo.compile.openvino_compile to keep the generic compile
path free of vLLM knowledge. Each hook no-ops on graphs lacking its marker (a
__pa__ Parameter prefix, a vLLM Concat pattern).
"""

import logging
import os

from openvino.frontend.pytorch.torchdynamo.vllm import preset as _preset

logger = logging.getLogger(__name__)


# The four apply_* entry points below are grouped so compile.py needs one
# try/except per call site rather than one per hook.

def apply_post_convert(om, options):
    """Run vLLM hooks on the freshly-converted Model.

    Runs right after ``fe.convert(im)`` and before serialization or input
    shaping.
    """
    register_pa_parameters(om)
    normalize_concat_ranks(om)
    from openvino.frontend.pytorch.torchdynamo.vllm.preset import bool_opt
    if bool_opt(options, "fc_decompress", True):
        rewrite_fc_decompression(om)


def apply_input_shapes(om, args, options, gm=None):
    """Resolve Python-int FX inputs and shape the remaining tensor Parameters.

    ``gm`` is optional for older callers, but without it int inputs can only be
    frozen at trace-time values (see bake_symint_constants).

    Returns False when the caller should fall through to its own loop: no int
    args and no vLLM preset.
    """
    from openvino.frontend.pytorch.torchdynamo.vllm.preset import bool_opt
    if not (bool_opt(options, "vllm", False) or any(isinstance(a, int) for a in args)):
        return False
    bake_symint_constants(
        om, args, dyn_shapes=bool_opt(options, "dynamic_shapes", True), gm=gm)
    return True


def apply_post_config(config, device, options, om=None):
    """Fill the vLLM defaults into the core.compile_model config.

    Called once the caller has built ``config`` and set CACHE_DIR. ``om`` is
    optional for older callers, but without it float precisions cannot be
    derived from the model and fall back to the preset default.
    """
    apply_kv_cache_config_defaults(config, device, options, om=om)


def widen_affinity_if_needed(options):
    """Widen process CPU affinity to all cores when narrower than needed.

    Needed when the mask is narrower than the requested OV thread count.
    vLLM's ``init_cpu_threads_env`` pins the worker to one CPU before
    ``torch.compile``, and TBB/OV sample affinity on first parallel use -- so a
    1-CPU mask locks ``INFERENCE_NUM_THREADS=1`` whatever config we pass.
    Widening before ``core.compile`` is what lets the pool inherit a useful
    mask at creation. No-op without sched_getaffinity, or if already wide
    enough.
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

    Under dynamic tracing dynamo passes symbolic sizes as plain Python-int
    graph inputs alongside the tensors they describe. A vLLM prefill graph::

        ph[0] arg164_1  TENSOR int32 [s72]     <- input_ids
        ph[1] arg163_1  SYMINT expr=s72        <- num_tokens
        ph[2] arg167_1  TENSOR int64 [s72]     <- positions
        ph[4] arg166_1  SYMINT expr=s72

    Freezing ph[1]/ph[4] is what limits the model to one prefill length; the
    graph states both equal ``input_ids.shape[0]``, so rebuilding them from a
    ShapeOf of the live tensor makes it valid for every length.

    Returns ``{int_arg_index: (tensor_arg_index, dim)}``, read off
    ``meta['val']`` -- matching on trace-time integer values would instead
    conflate two symbols that happen to coincide on this trace. Empty when
    ``gm`` is None, placeholders do not line up with ``args``, or the trace is
    static.
    """
    if gm is None:
        return {}
    try:
        import torch
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        if len(placeholders) != len(args):
            return {}
        # symbol -> (tensor arg index, dim). First carrier wins; any tensor
        # carrying it is an equally valid source.
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

    vLLM decode graphs are symint-heavy: seq_lens, past_lens and block-table
    sizes arrive as Python-int placeholders. Left as OV Parameters, shape
    inference takes their unset upper bound of 0 and collapses downstream
    Broadcast/Reshape outputs to size 0. Each is replaced by a propagatable
    value and its Parameter removed, one of two ways:

    * ``Gather(ShapeOf(tensor_input), dim)`` when the graph says which tensor
      dim the symbol denotes (see symint_shape_sources) -- tracks the real
      input, so one compiled model serves all shapes.
    * otherwise a Constant of this trace's value, correct only while dynamo
      guards force a retrace per distinct value -- which is why a static trace
      costs a recompile per prefill length.

    Also sets element type and partial shape on the remaining tensor
    Parameters, all-dynamic when ``dyn_shapes`` is set or every int input came
    from a ShapeOf: pinning trace-time shapes would const-fold that ShapeOf
    back into the frozen value just avoided.
    """
    import torch
    import numpy as np
    from openvino import Type, PartialShape, opset1 as _opset1, opset8 as _opset8

    _dtype_mapping = {
        torch.float32: Type.f32, torch.float64: Type.f64,
        torch.float16: Type.f16, torch.bfloat16: Type.bf16,
        torch.int64: Type.i64,
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
                # om.inputs is still 1:1 with args (Parameters are removed only
                # after the loop), so the tensor arg index indexes it directly.
                # i64[1] matches the Constant this replaces, so consumers built
                # for the baked form keep working.
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

    # NOTE: set_partial_shape only touches the Parameter; downstream nodes still
    # hold this trace's concrete sizes from conversion and must be re-inferred,
    # or ConstantFolding evaluates the new ShapeOf against a *stale* output
    # shape and folds it back to the frozen size. openvino_compile -- the only
    # caller -- calls validate_nodes_and_infer_types() right after this returns.


def register_pa_parameters(om):
    """Register dangling ``__pa__`` Parameters as model inputs.

    The paged_attention translator emits side-channel Parameters (KV cache,
    block tables, past_lens); unregistered, the Model fails validation with
    ``unregistered_parameters``.
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

    vLLM's symint-heavy graphs emit Unsqueeze wrappers that leave
    rank-mismatched Concat inputs on list-construct nodes. Bypasses each
    Unsqueeze whose inner input is already rank>=1, until shape inference
    succeeds. No-op on graphs that already pass it.
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


def model_float_precision(om):
    """Return the model's float dtype as an OV type name, "bf16"/"f16".

    None when the graph carries no narrow float. The PagedAttention
    key_cache Parameter wins: it is what compute precision
    must agree with, since the CPU plugin picks ``AttentionExecutor<compute_t,
    key_cache_t, value_cache_t>`` off its element type and only some triples
    exist (see preset.precision_config). The frontend creates it at the query
    dtype, so it *is* the model dtype.

    Graphs with no PA op (MLP-only partitions, the sampler) fall back to output
    then input element types, which carry the same dtype. Reading ports rather
    than walking Constants keeps this O(#ports) on graphs with thousands of
    frozen weights.
    """
    if om is None:
        return None

    def narrow(element_type):
        name = element_type.get_type_name()
        return name if name in _preset.SUPPORTED_FLOAT_PRECISIONS else None

    try:
        for port in om.inputs:
            if _is_kv_cache_port(port, field=("key_cache",)):
                if (et := narrow(port.get_element_type())) is not None:
                    return et
        for port in list(om.outputs) + list(om.inputs):
            if (et := narrow(port.get_element_type())) is not None:
                return et
    except Exception as _e:
        logger.debug("model_float_precision failed, falling back to preset: %s", _e)
    return None


def _port_names(port):
    """Every name a port answers to.

    Tensor names plus the node's friendly name (the frontend sets the
    latter, not always the former).
    """
    names = set(port.get_names())
    names.add(port.get_node().get_friendly_name())
    return names


def _is_kv_cache_port(port, field=("key_cache", "value_cache")):
    """True if `port` is a PagedAttention KV-cache Parameter.

    Substring match, not suffix: the frontend appends the layer index,
    giving names like ``__pa__unknown_layer__key_cache_15``.
    """
    return any(n.startswith("__pa__") and any(f in n for f in field)
               for n in _port_names(port))


def retype_kv_cache_parameters(om, et_name):
    """Declare the PagedAttention key_cache/value_cache Parameters as `et_name`.

    The CPU plugin runs ConvertPrecision with ``convert_input_output_precision
    = false``, so a Parameter declared at the model dtype but differing from
    the enforced inference precision keeps its type and gets a Convert spliced
    in after it. On the KV-cache path that Convert is fatal twice over: PA
    writes K and V *in place into its input buffer*, so the writes land in the
    Convert's temporary and never reach the persistent cache, and
    ConvertPagedAttnInputs bails out because its ``as_type_ptr<v0::Parameter>``
    now sees a Convert. Declaring the Parameters at the precision we are about
    to request keeps the cache wired straight into the PA node.

    Returns the number changed. Retyping is legal: the PA op does not constrain
    these element types (``input_check(this, 3, "key_cache", ..., {})``), and
    the plugin picks its executor from the resulting triple.
    """
    from openvino import Type
    target = {"bf16": Type.bf16, "f16": Type.f16, "f32": Type.f32}.get(et_name)
    if om is None or target is None:
        return 0
    changed = 0
    try:
        for port in om.inputs:
            if _is_kv_cache_port(port) and port.get_element_type() != target:
                port.get_node().set_element_type(target)
                changed += 1
        if changed:
            om.validate_nodes_and_infer_types()
            logger.debug("retyped %d KV-cache Parameters to %s", changed, et_name)
    except Exception as _e:
        logger.debug("KV-cache Parameter retype to %s skipped: %s", et_name, _e)
    return changed


def apply_kv_cache_config_defaults(config, device, options=None, om=None):
    """Fill the vLLM KV-cache and FC-quantization defaults into the OV CPU config.

    Caller-supplied entries win. No-op on non-CPU devices.
    """
    if device != "CPU":
        return
    if _preset.is_vllm_preset(options):
        for k, v in _preset._PRESET_CONFIG.items():
            config.setdefault(k, v)

    # Derived together from the model's float dtype, because the CPU PA kernel
    # only exists for matching (compute, cache) pairs. Applies with or without
    # the preset: the old unconditional bf16 pair broke every f16 model, and
    # the old non-preset pair (f32 cache, f16 compute) was mismatched the other
    # way.
    precisions = _preset.precision_config(model_float_precision(om))
    # Env vars remain an escape hatch. Setting only one of the pair is how you
    # get a mismatch, hence the warning below.
    for key, env in (("KV_CACHE_PRECISION", "OV_KV_CACHE_PRECISION"),
                     ("INFERENCE_PRECISION_HINT", "OV_INFERENCE_PRECISION_HINT")):
        if (override := os.environ.get(env)):
            config.setdefault(key, override)
        else:
            config.setdefault(key, precisions[key])
    if config["KV_CACHE_PRECISION"] != config["INFERENCE_PRECISION_HINT"]:
        logger.warning(
            "KV_CACHE_PRECISION=%s and INFERENCE_PRECISION_HINT=%s differ; the OV CPU "
            "PagedAttention kernel is only built for matching pairs and compile_model "
            "may reject this combination.",
            config["KV_CACHE_PRECISION"], config["INFERENCE_PRECISION_HINT"])

    # Make the graph agree with the config just resolved, or the cache
    # Parameters keep the model dtype and the plugin splices in a Convert that
    # silently drops every cache write.
    retype_kv_cache_parameters(om, config["KV_CACHE_PRECISION"])

    if "DYNAMIC_QUANTIZATION_GROUP_SIZE" not in config:
        # On-the-fly int8 FC activations: vnni int8 GEMM beats f32 GEMM by a
        # lot. Matches OV GenAI CPU behavior.
        config["DYNAMIC_QUANTIZATION_GROUP_SIZE"] = int(
            os.environ.get("DYNAMIC_QUANTIZATION_GROUP_SIZE", "32"))


def rewrite_fc_decompression(om):
    """Rewrite MatMul(X, Const_f16/bf16) into the oneDNN-BRGEMM-friendly form.

    For each MatMul on a constant f16/bf16 weight (optionally transposed via a
    [1,0] permutation), inserts a Convert to f32 marked as decompression so the
    plugin's ConvertMatMulToFC routes it to brgemm_avx512_f32 instead of the
    slower gemm_mlas_f32 fallback. The activation is upcast to f32 and its
    consumers downcast back, keeping downstream precision. f32 weights and
    quantized paths are skipped.

    Not vLLM-specific, but kept with the other narrow-float / KV-cache / PA
    compile-time edits so compile.py stays small.
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
            # The plugin's weight-decompression FC path takes inputType=f32 with
            # weightsType in {f16, bf16}; f32 needs no decompression.
            w_et = const.get_element_type()
            if w_et not in (Type.f16, Type.bf16):
                continue
            conv_w = _o1.convert(const.output(0), "f32")
            try:
                # Key name matters: is_decompression() looks for exactly
                # "decompression_0".
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
