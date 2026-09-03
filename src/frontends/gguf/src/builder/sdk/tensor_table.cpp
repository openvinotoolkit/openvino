// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Weight lookup for the builder SDK, and the small out-of-line pieces of the value handle and the
// model-builder base.

#include "openvino/frontend/gguf/builder/tensor_table.hpp"

#include "builder/sdk/graph_context_impl.hpp"
#include "openvino/core/except.hpp"
#include "openvino/frontend/gguf/builder/graph_context.hpp"
#include "openvino/frontend/gguf/builder/model_builder.hpp"
#include "openvino/frontend/gguf/builder/value.hpp"

namespace ov {
namespace frontend {
namespace gguf {

int64_t GgufValue::ne(size_t i) const {
    if (m_empty || m_shape.rank().is_dynamic()) {
        return -1;
    }
    const size_t rank = m_shape.size();
    // ggml's ne[0] is the fastest-varying dimension, i.e. the LAST entry of the stored shape.
    // Beyond the tensor's rank ggml reports 1, every tensor being nominally 4D with trailing 1s.
    if (i >= rank) {
        return 1;
    }
    const auto& d = m_shape[rank - 1 - i];
    return d.is_static() ? d.get_length() : -1;
}

ModelBuilder::~ModelBuilder() = default;

namespace {

// The graph leaf a weight becomes carries the tensor under its full ".weight" name; a tensor that
// is not a ".weight" (a bias, a norm scale stored without the suffix) goes through the plain path.
bool is_dot_weight(const std::string& name) {
    static const std::string suffix = ".weight";
    return name.size() > suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
}

}  // namespace

bool GgufTensors::has(const std::string& gguf_name) const {
    return m_ctx->impl().emitter.has_weight(gguf_name);
}

GgufValue GgufTensors::operator()(const std::string& gguf_name) const {
    auto& impl = m_ctx->impl();
    auto& e = impl.emitter;
    if (!e.has_weight(gguf_name)) {
        // Absent is a normal, meaningful state: it is how GGUF encodes structure. Report it as an
        // empty value so a ported `if (layer.attn_q_norm)` works.
        return GgufValue();
    }
    // Emission is idempotent -- a weight read repeatedly by a layer loop, or shared between ops,
    // becomes exactly one leaf.
    if (!e.weight_emitted(gguf_name)) {
        if (is_dot_weight(gguf_name)) {
            e.add_weight(gguf_name);
        } else {
            e.add_named_weight(gguf_name);
        }
    }
    return GgufValue(gguf_name, e.shape_of_tensor(gguf_name), e.type_of_tensor(gguf_name));
}

GgufValue GgufTensors::require(const std::string& gguf_name) const {
    auto v = (*this)(gguf_name);
    OPENVINO_ASSERT(v,
                    "[GGUF] model is missing expected weight tensor '",
                    gguf_name,
                    "' for architecture '",
                    m_ctx->arch(),
                    "'");
    return v;
}

GgufValue GgufTensors::layer(int il, const std::string& suffix) const {
    return (*this)("blk." + std::to_string(il) + "." + suffix);
}

LayerTensors GgufTensors::layer(int il) const {
    const auto w = [&](const std::string& suffix) {
        return layer(il, suffix);
    };
    LayerTensors t;

    t.attn_norm = w("attn_norm.weight");
    t.attn_norm_2 = w("attn_norm_2.weight");
    t.attn_q_norm = w("attn_q_norm.weight");
    t.attn_k_norm = w("attn_k_norm.weight");
    t.attn_post_norm = w("post_attention_norm.weight");

    t.wq = w("attn_q.weight");
    t.wk = w("attn_k.weight");
    t.wv = w("attn_v.weight");
    t.wo = w("attn_output.weight");
    t.wqkv = w("attn_qkv.weight");
    t.bq = w("attn_q.bias");
    t.bk = w("attn_k.bias");
    t.bv = w("attn_v.bias");
    t.bo = w("attn_output.bias");
    t.bqkv = w("attn_qkv.bias");

    t.attn_sinks = w("attn_sinks.weight");
    t.wqkv_gate = w("attn_gate.weight");

    t.ffn_norm = w("ffn_norm.weight");
    t.ffn_post_norm = w("post_ffw_norm.weight");

    t.ffn_gate = w("ffn_gate.weight");
    t.ffn_up = w("ffn_up.weight");
    t.ffn_down = w("ffn_down.weight");
    t.ffn_gate_b = w("ffn_gate.bias");
    t.ffn_up_b = w("ffn_up.bias");
    t.ffn_down_b = w("ffn_down.bias");

    t.ffn_gate_inp = w("ffn_gate_inp.weight");
    t.ffn_gate_inp_b = w("ffn_gate_inp.bias");
    t.ffn_gate_exps = w("ffn_gate_exps.weight");
    t.ffn_up_exps = w("ffn_up_exps.weight");
    t.ffn_down_exps = w("ffn_down_exps.weight");
    t.ffn_exp_probs_b = w("exp_probs_b.bias");

    t.ffn_gate_shexp = w("ffn_gate_shexp.weight");
    t.ffn_up_shexp = w("ffn_up_shexp.weight");
    t.ffn_down_shexp = w("ffn_down_shexp.weight");

    return t;
}

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
