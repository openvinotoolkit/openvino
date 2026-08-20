// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "builder/blocks/ffn.hpp"

#include <limits>

#include "builder/blocks/common.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace blocks {

using ov::element::f32;

namespace {

// "<name>.weight" -> "<name>.bias": every weight tensor in this file follows that naming
// convention, so a projection's bias weight name can always be derived from its own.
std::string bias_weight_name(const std::string& weight_name) {
    static const std::string suffix = ".weight";
    return weight_name.substr(0, weight_name.size() - suffix.size()) + ".bias";
}

}  // namespace

// GATE/UP projection -> SwiGLU, shared by dense_ffn's non-fused branch and moe_ffn's shared-expert
// branch (deepseek2-ocr, bailingmoe2, exaone-moe): the same shape in both, differing only in
// weight-name prefix and (dense only) an optional per-projection bias. The DOWN projection that
// follows is left to the caller, since its bias/output-merging conventions differ between the two
// (a plain residual-style output for dense_ffn vs an ADD into the routed experts' sum for MoE).
// Callers supply the exact op names so each keeps its own established naming (ffn_* vs shexp_*); a
// bias ADD op, when emitted, is named "<gate_name|up_name>_b" and reads "<weight_name>.bias".
std::string swiglu_gate_up(GraphEmitter& e,
                           const std::string& gate_w,
                           const std::string& up_w,
                           const std::string& ffn_norm,
                           int64_t T,
                           const std::string& gate_name,
                           const std::string& up_name,
                           const std::string& glu_name,
                           bool has_bias) {
    e.add_weight(gate_w);
    const int n_ff = static_cast<int>(e.weight_tensor(gate_w).get_shape()[0]);
    auto gate = e.add_op("GGML_OP_MUL_MAT", gate_name, {gate_w, ffn_norm}, ps({1, 1, T, n_ff}), f32);
    if (has_bias) {
        gate = add_bias(e, gate, bias_weight_name(gate_w), gate_name + "_b");
    }
    e.add_weight(up_w);
    auto up = e.add_op("GGML_OP_MUL_MAT", up_name, {up_w, ffn_norm}, ps({1, 1, T, n_ff}), f32);
    if (has_bias) {
        up = add_bias(e, up, bias_weight_name(up_w), up_name + "_b");
    }
    return e.add_op("GGML_GLU_OP_SWIGLU", glu_name, {gate, up}, ps({1, 1, T, n_ff}), f32, 0, {{"swapped", false}});
}

std::string geglu_ffn(GraphEmitter& e,
                      const DecoderConfig& cfg,
                      const std::string& p,
                      const std::string& ffn_norm,
                      int64_t T) {
    e.add_weight(p + "ffn_gate.weight");
    e.add_weight(p + "ffn_up.weight");
    e.add_weight(p + "ffn_down.weight");
    const int n_ff = static_cast<int>(e.weight_tensor(p + "ffn_gate.weight").get_shape()[0]);
    auto gate =
        e.add_op("GGML_OP_MUL_MAT", p + "ffn_gate", {p + "ffn_gate.weight", ffn_norm}, ps({1, 1, T, n_ff}), f32);
    auto up = e.add_op("GGML_OP_MUL_MAT", p + "ffn_up", {p + "ffn_up.weight", ffn_norm}, ps({1, 1, T, n_ff}), f32);
    auto glu =
        e.add_op("GGML_GLU_OP_GEGLU", p + "ffn_geglu", {gate, up}, ps({1, 1, T, n_ff}), f32, 0, {{"swapped", false}});
    return e.add_op("GGML_OP_MUL_MAT", p + "ffn_out", {p + "ffn_down.weight", glu}, ps({1, 1, T, cfg.n_embd}), f32);
}

std::string dense_ffn(GraphEmitter& e,
                      const DecoderConfig& cfg,
                      const std::string& p,
                      const std::string& ffn_norm,
                      int64_t T) {
    e.add_weight(p + "ffn_up.weight");
    e.add_weight(p + "ffn_down.weight");
    const bool fused_ffn = !e.has_weight(p + "ffn_gate.weight");  // phi-3: fused gate+up
    const bool has_ffn_bias = e.has_weight(p + "ffn_up.bias");
    std::string glu;
    if (fused_ffn) {
        const int up2 = static_cast<int>(e.weight_tensor(p + "ffn_up.weight").get_shape()[0]);  // 2*n_ff
        auto up = e.add_op("GGML_OP_MUL_MAT", p + "ffn_up", {p + "ffn_up.weight", ffn_norm}, ps({1, 1, T, up2}), f32);
        glu = e.add_op("GGML_GLU_OP_SWIGLU",
                       p + "ffn_swiglu",
                       {up},
                       ps({1, 1, T, up2 / 2}),
                       f32,
                       0,
                       {{"swapped", false}});
    } else {
        glu = swiglu_gate_up(e,
                             p + "ffn_gate.weight",
                             p + "ffn_up.weight",
                             ffn_norm,
                             T,
                             p + "ffn_gate",
                             p + "ffn_up",
                             p + "ffn_swiglu",
                             has_ffn_bias);
    }
    auto out = e.add_op("GGML_OP_MUL_MAT", p + "ffn_out", {p + "ffn_down.weight", glu}, ps({1, 1, T, cfg.n_embd}), f32);
    if (has_ffn_bias) {
        out = add_bias(e, out, p + "ffn_down.bias", p + "ffn_out_b");
    }
    return out;
}

std::string moe_ffn(GraphEmitter& e,
                    const DecoderConfig& cfg,
                    const std::string& p,
                    const std::string& ffn_norm,
                    int64_t T) {
    const int E = cfg.n_expert, K = cfg.n_expert_used;
    const int n_ff = static_cast<int>(e.weight_tensor(p + "ffn_gate_exps.weight").get_shape()[1]);

    // --- router: logits [1,1,T,E] = gate_inp · x ---
    e.add_weight(p + "ffn_gate_inp.weight");
    auto logits =
        e.add_op("GGML_OP_MUL_MAT", p + "moe_logits", {p + "ffn_gate_inp.weight", ffn_norm}, ps({1, 1, T, E}), f32);
    if (cfg.has_moe_gate_bias) {
        logits = add_bias(e, logits, p + "ffn_gate_inp.bias", p + "moe_logits_b");
    }

    // gating: softmax (OLMoE) or softmax-after-topk (gpt-oss "softmax_weight").
    std::string probs = logits;
    if (!cfg.moe_softmax_weight) {
        probs = e.add_op("GGML_OP_SOFT_MAX",
                         p + "moe_probs",
                         {logits},
                         ps({1, 1, T, E}),
                         f32,
                         0,
                         {{"softmax_axis", int64_t(-1)}});
    }

    // top-k expert selection -> indices [1,1,T,K] (i32).
    auto selected = e.add_op("GGML_OP_TOP_K", p + "moe_topk", {probs}, ps({1, 1, T, K}), ov::element::i32);

    // weights = gather probs by selected; op_case 10 returns a per-expert column
    // [1,T,K,1] (robust to dynamic T). gpt-oss softmaxes over the K (expert) axis.
    auto weights = e.add_op("GGML_OP_GET_ROWS", p + "moe_w", {probs, selected}, ps({1, T, K, 1}), f32, 10);
    if (cfg.moe_softmax_weight) {
        weights = e.add_op("GGML_OP_SOFT_MAX",
                           p + "moe_w_sm",
                           {weights},
                           ps({1, T, K, 1}),
                           f32,
                           0,
                           {{"softmax_axis", int64_t(2)}});
    }
    // Renormalize the selected top-K gating weights to sum to 1 (llama.cpp build_moe_ffn's
    // norm_w path, e.g. qwen3moe/glm4moe/bailingmoe2). On the softmax-before-topk path the
    // weights above are a slice of a softmax over all E experts, so the K<E kept values no
    // longer sum to 1 without this; the softmax-after-topk (gpt-oss) path already sums to 1
    // and needs no correction. weights is [1,T,K,1]; transpose K to the last axis, SUM_ROWS,
    // clamp away from 0 (matches llama.cpp's 6.1e-5 floor), then DIV back (broadcasts the
    // [1,T,1,1] sum over K via repeat_input_to_match).
    if (cfg.expert_weights_norm && !cfg.moe_softmax_weight) {
        auto w_tr = e.add_op("GGML_OP_TRANSPOSE", p + "moe_w_tr", {weights}, ps({1, T, 1, K}), f32);
        auto w_sum = e.add_op("GGML_OP_SUM_ROWS", p + "moe_w_sum", {w_tr}, ps({1, T, 1, 1}), f32);
        w_sum = e.add_op("GGML_OP_CLAMP",
                         p + "moe_w_sum_clamped",
                         {w_sum},
                         ps({1, T, 1, 1}),
                         f32,
                         0,
                         {{"clamp_min", 6.1e-5f}, {"clamp_max", std::numeric_limits<float>::max()}});
        weights = e.add_op("GGML_OP_DIV", p + "moe_w_norm", {weights, w_sum}, ps({1, T, K, 1}), f32);
    }

    // gpt-oss expert_weights_scale: optional constant multiplier applied after softmax
    // (mirrors llama.cpp build_moe_ffn w_scale != 0 && w_scale != 1.0 path).
    if (cfg.expert_weights_scale != 0.0f && cfg.expert_weights_scale != 1.0f) {
        weights = scale(e, weights, cfg.expert_weights_scale, p + "moe_w_scaled");
    }

    // expert FFN via MUL_MAT_ID. The routed input x is broadcast to K slots; the
    // translator gathers each token's selected expert matrices. gpt-oss adds per-expert
    // biases (ADD_ID gathers the selected experts' bias rows).
    e.add_weight(p + "ffn_gate_exps.weight");
    e.add_weight(p + "ffn_up_exps.weight");
    e.add_weight(p + "ffn_down_exps.weight");
    const bool eb = cfg.has_moe_expert_bias;
    auto up = e.add_op("GGML_OP_MUL_MAT_ID",
                       p + "moe_up",
                       {p + "ffn_up_exps.weight", ffn_norm, selected},
                       ps({1, T, K, n_ff}),
                       f32);
    if (eb) {
        e.add_named_weight(p + "ffn_up_exps.bias");
        up = e.add_op("GGML_OP_ADD_ID",
                      p + "moe_up_b",
                      {up, p + "ffn_up_exps.bias", selected},
                      ps({1, T, K, n_ff}),
                      f32);
    }
    auto gate = e.add_op("GGML_OP_MUL_MAT_ID",
                         p + "moe_gate",
                         {p + "ffn_gate_exps.weight", ffn_norm, selected},
                         ps({1, T, K, n_ff}),
                         f32);
    if (eb) {
        e.add_named_weight(p + "ffn_gate_exps.bias");
        gate = e.add_op("GGML_OP_ADD_ID",
                        p + "moe_gate_b",
                        {gate, p + "ffn_gate_exps.bias", selected},
                        ps({1, T, K, n_ff}),
                        f32);
    }
    std::string act;
    if (cfg.moe_swiglu_oai) {
        act = e.add_op("GGML_GLU_OP_SWIGLU_OAI",
                       p + "moe_act",
                       {gate, up},
                       ps({1, T, K, n_ff}),
                       f32,
                       0,
                       // Attribute names must match the cgraph decoder's (ggml-decoder.cpp), which
                       // is the external contract the shared translators read: glu_alpha/glu_limit.
                       {{"swapped", false}, {"glu_alpha", 1.702f}, {"glu_limit", 7.0f}});
    } else {
        act = e.add_op("GGML_GLU_OP_SWIGLU",
                       p + "moe_act",
                       {gate, up},
                       ps({1, T, K, n_ff}),
                       f32,
                       0,
                       {{"swapped", false}});
    }
    auto experts = e.add_op("GGML_OP_MUL_MAT_ID",
                            p + "moe_down",
                            {p + "ffn_down_exps.weight", act, selected},
                            ps({1, T, K, cfg.n_embd}),
                            f32);
    if (eb) {
        e.add_named_weight(p + "ffn_down_exps.bias");
        experts = e.add_op("GGML_OP_ADD_ID",
                           p + "moe_down_b",
                           {experts, p + "ffn_down_exps.bias", selected},
                           ps({1, T, K, cfg.n_embd}),
                           f32);
    }

    // Weighted sum over the K selected experts. weights is [1,T,K,1] (per-expert col).
    //   experts [1,T,K,n_embd] * weights -> [1,T,K,n_embd]
    //   TRANSPOSE last two axes -> [1,T,n_embd,K]; SUM_ROWS over K -> [1,T,n_embd,1]
    //   RESHAPE (dynamic) -> [1,1,T,n_embd].
    // The reshape is op_case 5: collapse to [1, 1, -1, n_embd], the token axis staying dynamic.
    // That is the case the cgraph decoder would assign to this same reshape too -- ggml ne
    // [1,n_embd,T,1] -> [n_embd,T,1,1] satisfies both of its case-5 predicates -- so the shared
    // case applies as-is and needs no builder-specific numbering.
    auto weighted = e.add_op("GGML_OP_MUL", p + "moe_weighted", {experts, weights}, ps({1, T, K, cfg.n_embd}), f32);
    auto tr = e.add_op("GGML_OP_TRANSPOSE", p + "moe_tr", {weighted}, ps({1, T, cfg.n_embd, K}), f32);
    auto summed = e.add_op("GGML_OP_SUM_ROWS", p + "moe_sum", {tr}, ps({1, T, cfg.n_embd, 1}), f32);
    auto moe_out = e.add_op("GGML_OP_RESHAPE", p + "moe_out", {summed}, ps({1, 1, T, cfg.n_embd}), f32, 5);

    // Shared experts (deepseek2-ocr, bailingmoe2, exaone-moe): always-active experts whose
    // output is added to the routed experts' weighted sum. Uses plain SwiGLU dense FFN with
    // ffn_{gate,up,down}_shexp.weight (n_ff_shared = shexp rows). Output added to moe_out.
    if (cfg.n_expert_shared > 0 && e.has_weight(p + "ffn_gate_shexp.weight")) {
        auto s_act = swiglu_gate_up(e,
                                    p + "ffn_gate_shexp.weight",
                                    p + "ffn_up_shexp.weight",
                                    ffn_norm,
                                    T,
                                    p + "shexp_gate",
                                    p + "shexp_up",
                                    p + "shexp_act",
                                    /*has_bias=*/false);
        e.add_weight(p + "ffn_down_shexp.weight");
        auto s_down = e.add_op("GGML_OP_MUL_MAT",
                               p + "shexp_out",
                               {p + "ffn_down_shexp.weight", s_act},
                               ps({1, 1, T, cfg.n_embd}),
                               f32);
        moe_out = e.add_op("GGML_OP_ADD", p + "moe_shared_out", {moe_out, s_down}, ps({1, 1, T, cfg.n_embd}), f32);
    }
    return moe_out;
}

}  // namespace blocks
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
