# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""vLLM-specific compile-time hooks.

Functions called from torchdynamo.compile.openvino_compile to keep the
generic compile path free of vLLM-specific knowledge. Each hook is a
no-op when the input graph does not have the corresponding vLLM marker
(e.g. __pa__ Parameter prefix, vLLM-style Concat patterns).
"""

import logging

logger = logging.getLogger(__name__)


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
    take priority. Reads env-var fallbacks for backward compat with the
    legacy environment-driven setup; new callers should pass the values
    via options["config"] instead.

    No-op on non-CPU devices.
    """
    if device != "CPU":
        return
    import os
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
