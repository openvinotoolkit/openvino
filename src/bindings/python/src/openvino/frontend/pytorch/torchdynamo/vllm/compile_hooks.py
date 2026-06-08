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
