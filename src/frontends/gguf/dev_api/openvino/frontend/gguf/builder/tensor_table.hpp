// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "openvino/frontend/gguf/builder/value.hpp"
#include "openvino/frontend/gguf/visibility.hpp"

namespace ov {
namespace frontend {
namespace gguf {

class GgufGraphContext;

// The weights of one decoder layer, named exactly as in llama.cpp's `llama_layer`.
//
// The field names are the point of this struct. A llama.cpp model file addresses weights as
// `model.layers[il].attn_norm`; the GGUF frontend addresses them by their on-disk string name,
// "blk.<il>.attn_norm.weight". Mirroring llama_layer's spelling turns that difference into a
// mechanical prefix substitution (`model.layers[il].` -> `ctx.layer(il).`) instead of a rewrite,
// which is what keeps a port of an upstream model file small enough to review against its original.
//
// Every field is a GgufValue, EMPTY when the file has no such tensor -- so llama.cpp's null-tensor
// idiom (`build_norm(cur, layer.attn_norm, NULL, ...)`, `layer.bq ? ... : ...`) ports unchanged.
// Looking a weight up also emits it into the graph, once; repeated lookups of the same weight are
// deduplicated, so a port may read a field as many times as the original does.
struct GGUF_FRONTEND_API LayerTensors {
    // attention: norms
    GgufValue attn_norm;
    GgufValue attn_norm_2;
    GgufValue attn_q_norm;
    GgufValue attn_k_norm;
    GgufValue attn_post_norm;

    // attention: projections and their biases
    GgufValue wq;
    GgufValue wk;
    GgufValue wv;
    GgufValue wo;
    GgufValue wqkv;  // fused QKV (phi-3, minicpm)
    GgufValue bq;
    GgufValue bk;
    GgufValue bv;
    GgufValue bo;
    GgufValue bqkv;

    // attention: sinks (gpt-oss) and output gate (muse-glimmer)
    GgufValue attn_sinks;
    GgufValue wqkv_gate;

    // feed-forward: norms
    GgufValue ffn_norm;
    GgufValue ffn_post_norm;

    // feed-forward: dense projections and their biases
    GgufValue ffn_gate;
    GgufValue ffn_up;
    GgufValue ffn_down;
    GgufValue ffn_gate_b;
    GgufValue ffn_up_b;
    GgufValue ffn_down_b;

    // feed-forward: MoE routing and experts
    GgufValue ffn_gate_inp;
    GgufValue ffn_gate_inp_b;
    GgufValue ffn_gate_exps;
    GgufValue ffn_up_exps;
    GgufValue ffn_down_exps;
    GgufValue ffn_exp_probs_b;

    // feed-forward: always-active shared experts
    GgufValue ffn_gate_shexp;
    GgufValue ffn_up_shexp;
    GgufValue ffn_down_shexp;
};

// Lookup of a model's weights by GGUF tensor name, as a llama.cpp model file addresses them.
//
// A lookup EMITS the weight into the graph (as the GGML_OP_NONE leaf the translators expect) and
// returns a handle to it. Emission is idempotent: a weight referenced by several ops, or read
// repeatedly by a ported layer loop, is emitted exactly once.
//
// A tensor the file does not contain yields an EMPTY GgufValue rather than an error, because
// "absent" is how GGUF encodes structure -- no `blk.0.attn_q_norm.weight` means the architecture
// has no QK-norm. A builder that requires a tensor says so itself, via require().
class GGUF_FRONTEND_API GgufTensors {
public:
    explicit GgufTensors(GgufGraphContext& ctx) : m_ctx(&ctx) {}

    // Weight by full GGUF name (e.g. "token_embd.weight", "blk.3.attn_q.bias"). Empty if absent.
    GgufValue operator()(const std::string& gguf_name) const;

    // As operator(), but fails with a diagnostic naming the tensor and the architecture when the
    // file does not have it. Use for a tensor whose absence means the file does not match the
    // architecture being built, so the error names the real problem instead of surfacing later as
    // a shape mismatch.
    GgufValue require(const std::string& gguf_name) const;

    // True when the file carries this tensor, WITHOUT emitting it. For a structure probe
    // ("does layer 0 have a QK-norm?") that must not add a leaf to the graph.
    bool has(const std::string& gguf_name) const;

    // Weight of layer `il` by suffix: layer(3, "attn_norm.weight") -> "blk.3.attn_norm.weight".
    GgufValue layer(int il, const std::string& suffix) const;

    // All of layer `il`'s weights, named as in llama.cpp's llama_layer. Absent tensors are empty
    // values. Emits only the tensors the file actually has.
    LayerTensors layer(int il) const;

private:
    GgufGraphContext* m_ctx;
};

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
