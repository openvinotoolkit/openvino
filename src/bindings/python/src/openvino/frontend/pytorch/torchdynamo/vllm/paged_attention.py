# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""vLLM PagedAttention integration for the OV torchdynamo backend.

Registers a custom torch op
`openvino::paged_attention(q, k, v, layer_name, scale, kv_sharing_target)`
whose Python impl delegates to vLLM's `unified_attention_with_output`, plus an
FX pre-pass that rewrites `auto_functionalized_v2(unified_attention_with_output)`
call sites into it. That turns attention from an untranslatable HOP into an
op the partitioner can keep inside an OV partition.

The C++ translator emits a PagedAttentionExtension for the op; side_channel.py
binds the KV cache, block tables and lengths from vllm.forward_context.

`scale` and `kv_sharing_target` are carried on the op because neither can be
recovered from the FX graph, and guessing either one silently corrupts output:

* **scale** -- the frontend otherwise falls back to ``1/sqrt(head_dim)``. That
  is wrong for any model that scales differently, e.g. Gemma-4, which uses
  ``scaling = 1.0`` and folds the normalization into its learnable Q/K norm
  weights. Guessing divides every score by ``sqrt(head_dim)`` a second time
  and flattens the softmax toward uniform. 0.0 means "not supplied, derive it".

* **kv_sharing_target** -- layers that reuse an earlier layer's KV cache are
  handed *raw* k/v by vLLM (no k_norm, no v_norm, no RoPE), because its
  attention backends suppress the cache write for them::

      # vllm/v1/attention/backends/cpu_attn.py
      if (self.kv_sharing_target_layer_name is None
              and key is not None and value is not None):
          ops.cpu_attn_reshape_and_cache(...)

  `PagedAttentionExtension` has no read-only mode and always writes, so without
  this the shared layers scribble that placeholder junk over the cache they are
  supposed to be reading. Naming the target lets the translator reuse its
  already-normed k/v, which makes the write idempotent. "" means not shared.
"""

# mypy: ignore-errors

import logging

from .fx_utils import is_auto_functionalized, replace_mutated_output

logger = logging.getLogger(__name__)

_REGISTERED = False


def _register_custom_op():
    """Register torch.ops.openvino.paged_attention once per process."""
    global _REGISTERED
    if _REGISTERED:
        return
    import torch

    @torch.library.custom_op(
        "openvino::paged_attention",
        mutates_args=(),
    )
    def paged_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_name: str,
        scale: float,
        kv_sharing_target: str,
    ) -> torch.Tensor:
        # Only hit on the torch-eager fallback path. vLLM's CPU backend
        # implements just the "_with_output" variant, so pass in an output.
        # scale/kv_sharing_target are consumed by the OV translator; on this
        # path vLLM's own Attention layer already applies both.
        out = torch.empty_like(query).contiguous()
        torch.ops.vllm.unified_attention_with_output(
            query, key, value, out, layer_name
        )
        return out

    @paged_attention.register_fake
    def _paged_attention_fake(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_name: str,
        scale: float,
        kv_sharing_target: str,
    ) -> torch.Tensor:
        return torch.empty_like(query).contiguous()

    _REGISTERED = True
    logger.debug("Registered torch.ops.openvino.paged_attention")


def _attention_layer_meta(layer_name):
    """(scale, kv_sharing_target) for a vLLM attention layer, by name.

    Read from the live forward context, which is the same place
    `unified_attention_with_output` resolves `layer_name` and the same source
    side_channel uses at infer time. The rewrite runs inside the model's first
    forward, so the context exists.

    Returns (0.0, "") when the layer cannot be resolved, which tells the
    translator to fall back to its own derivation rather than trust a guess.
    """
    try:
        from vllm.forward_context import get_forward_context

        layers = get_forward_context().no_compile_layers
        layer = layers.get(layer_name) if isinstance(layers, dict) else None
    except Exception as e:
        logger.debug("no forward context for %s: %s", layer_name, e)
        return 0.0, ""
    if layer is None:
        logger.warning(
            "paged_attention: layer %r not in no_compile_layers; the OV "
            "translator will derive the scale and assume no KV sharing",
            layer_name)
        return 0.0, ""
    # Attention.extra_repr treats impl.scale as the authoritative value.
    scale = getattr(getattr(layer, "impl", None), "scale", None)
    target = getattr(layer, "kv_sharing_target_layer_name", None)
    return (float(scale) if scale is not None else 0.0), str(target or "")


def _is_unified_attention_with_output(node) -> bool:
    """Match auto_functionalized_v2(unified_attention_with_output, ...)."""
    import torch
    return is_auto_functionalized(node, lambda: torch.ops.vllm.unified_attention_with_output.default)


def rewrite_unified_attention_to_paged_attention(gm) -> int:
    """Rewrite every attention HOP node to a call of the OV paged_attention op.

    Matches auto_functionalized_v2(unified_attention_with_output, ...) nodes
    and rewrites them to torch.ops.openvino.paged_attention.default. Returns
    the number of rewrites performed.
    """
    import torch

    _register_custom_op()

    paged_attention_op = torch.ops.openvino.paged_attention.default

    # Collect first; do not mutate while iterating.
    to_rewrite = [n for n in gm.graph.nodes if _is_unified_attention_with_output(n)]
    if not to_rewrite:
        return 0

    rewrites = 0
    for node in to_rewrite:
        kw = dict(node.kwargs)
        query = kw.get("query")
        key = kw.get("key")
        value = kw.get("value")
        layer_name = kw.get("layer_name")
        if query is None or key is None or value is None or layer_name is None:
            logger.warning(
                "Skipping unified_attention rewrite: missing q/k/v/layer_name in kwargs"
            )
            continue

        scale, kv_sharing_target = _attention_layer_meta(layer_name)
        with gm.graph.inserting_after(node):
            new_node = gm.graph.call_function(
                paged_attention_op,
                args=(query, key, value, layer_name, scale,
                      kv_sharing_target),
            )

        # auto_functionalized_v2 writes output to _all_bases[_output_base_index].
        if not replace_mutated_output(gm, node, new_node, kw.get("_output_base_index")):
            # Consumers remain for bases we did not rewrite, so the original
            # node stays and the partitioner splits here. Correct, just slower.
            logger.debug(
                f"Left original auto_functionalized_v2 node for layer "
                f"{layer_name}: it still has {len(node.users)} non-attention "
                f"consumers"
            )

        rewrites += 1

    if rewrites:
        gm.graph.lint()
        gm.recompile()

    return rewrites
