// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "builder/blocks/gated_delta_net.hpp"

#include <vector>

#include "builder/blocks/common.hpp"
#include "openvino/core/except.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace blocks {

std::string gated_delta_net(GraphEmitter& e, const DecoderConfig& cfg, int il, const std::string& attn_norm) {
    using ov::element::f32;
    const std::string p = "blk." + std::to_string(il) + ".";
    auto& graph = *e.graph();

    const int64_t d_conv = cfg.ssm_conv_kernel;
    const int64_t S = cfg.ssm_state_size;     // head_k_dim == head_v_dim
    const int64_t H_k = cfg.ssm_group_count;  // num_k_heads
    const int64_t H_v = cfg.ssm_dt_rank;      // num_v_heads
    OPENVINO_ASSERT(H_v > 0 && cfg.ssm_inner_size % H_v == 0,
                    "[GGUF] Gated-DeltaNet: ssm_inner_size (",
                    cfg.ssm_inner_size,
                    ") is not evenly divisible by ssm_time_step_rank (",
                    H_v,
                    ")");
    const int64_t head_v = cfg.ssm_inner_size / H_v;  // head_v_dim
    const int64_t key_dim = S * H_k;
    const int64_t value_dim = head_v * H_v;
    const int64_t conv_dim = 2 * key_dim + value_dim;

    // ---- input projections ----
    e.add_weight(p + "attn_qkv.weight");
    auto qkv = e.add_op("GGML_OP_MUL_MAT", p + "qkv_mixed", {p + "attn_qkv.weight", attn_norm}, f32);
    e.add_weight(p + "attn_gate.weight");
    auto z = e.add_op("GGML_OP_MUL_MAT", p + "z", {p + "attn_gate.weight", attn_norm}, f32);

    // beta = sigmoid(ssm_beta @ x), one scalar per v-head
    e.add_weight(p + "ssm_beta.weight");
    auto beta = e.add_op("GGML_OP_MUL_MAT", p + "beta", {p + "ssm_beta.weight", attn_norm}, f32);
    beta = e.add_op("GGML_UNARY_OP_SIGMOID", p + "beta_sig", {beta}, f32);
    beta = e.reshape(p + "beta_4d", beta, {0, -1, H_v, 1}, true);

    // g = softplus(ssm_alpha @ x + ssm_dt.bias) * ssm_a   (ggml: -A_log.exp() * softplus)
    e.add_weight(p + "ssm_alpha.weight");
    auto alpha = e.add_op("GGML_OP_MUL_MAT", p + "alpha", {p + "ssm_alpha.weight", attn_norm}, f32);
    e.add_named_weight(p + "ssm_dt.bias");
    alpha = e.add_op("GGML_OP_ADD", p + "alpha_biased", {alpha, p + "ssm_dt.bias"}, f32);
    alpha = e.add_op("GGML_UNARY_OP_SOFTPLUS", p + "alpha_sp", {alpha}, f32);
    e.add_named_weight(p + "ssm_a");
    auto g = e.add_op("GGML_OP_MUL", p + "gate", {alpha, p + "ssm_a"}, f32);
    g = e.reshape(p + "gate_4d", g, {0, -1, H_v, 1}, true);

    // ---- causal depthwise conv over [conv state | this step's tokens] ----
    // conv_state holds the trailing d_conv-1 columns of the previous step's conv input.
    const std::string cs = "conv_state_l" + std::to_string(il);
    if (!e.has_model_input(cs)) {
        e.add_input(cs, f32, ps({1, 1, conv_dim, d_conv - 1}));
    }

    // [1,1,T,conv_dim] -> [1,1,conv_dim,T] so the conv window grows along the last axis.
    auto qkv_t = e.add_op("GGML_OP_TRANSPOSE", p + "qkv_t", {qkv}, f32);
    auto conv_in = e.add_op("GGML_OP_CONCAT", p + "conv_in", {cs, qkv_t}, f32, 0, {{"concat_axis", int{0}}});

    // Next step's state is the trailing d_conv-1 columns of this window.
    const std::vector<int64_t> tail_slice{3, -(d_conv - 1), d_conv - 1};
    auto cs_out = e.add_op("GGML_OP_VIEW", cs + "_out", {conv_in}, f32, 3, {{"view_slice", tail_slice}});
    graph.model_output_names.push_back(cs_out);
    graph.recurrent_states.emplace_back(cs, cs_out);

    e.add_named_weight(p + "ssm_conv1d.weight");
    auto conv = e.add_op("GGML_OP_SSM_CONV", p + "conv_out", {conv_in, p + "ssm_conv1d.weight"}, f32);
    conv = e.add_op("GGML_UNARY_OP_SILU", p + "conv_silu", {conv}, f32);

    // ---- split the conv output into q | k | v and normalize q/k ----
    auto slice_heads = [&](const std::string& name, int64_t off, int64_t width, int64_t heads, int64_t dim) {
        const std::vector<int64_t> sl{3, off, width};
        auto s = e.add_op("GGML_OP_VIEW", p + name + "_s", {conv}, f32, 3, {{"view_slice", sl}});
        return e.reshape(p + name, s, {0, -1, heads, dim}, true);
    };
    auto q = slice_heads("q_conv", 0, key_dim, H_k, S);
    auto k = slice_heads("k_conv", key_dim, key_dim, H_k, S);
    auto v = slice_heads("v_conv", 2 * key_dim, value_dim, H_v, head_v);
    q = e.add_op("GGML_OP_L2_NORM", p + "q_l2", {q}, f32, 0, {{"eps", cfg.rms_eps}});
    k = e.add_op("GGML_OP_L2_NORM", p + "k_l2", {k}, f32, 0, {{"eps", cfg.rms_eps}});

    // ---- recurrent delta rule ----
    // ggml state layout is [B, H_v, value_dim, key_dim]; the translator transposes it for
    // the fused op and transposes the new state back, so the Parameter keeps ggml's layout.
    const std::string ss = "ssm_state_l" + std::to_string(il);
    if (!e.has_model_input(ss)) {
        e.add_input(ss, f32, ps({1, H_v, head_v, S}));
    }

    auto gdn = e.add_op("GGML_OP_GATED_DELTA_NET",
                        p + "gdn",
                        {q, k, v, g, beta, ss},
                        f32,
                        0,
                        {{"gdn_state_slots", int64_t{1}}});

    // Split the packed attention rows and recurrent state; only the token axis is inferred.
    const std::vector<int64_t> attn_view{0, head_v};
    const std::vector<int64_t> state_view{1, head_v};
    auto attn = e.add_op("GGML_OP_VIEW",
                         p + "gdn_attn",
                         {gdn},
                         f32,
                         4,
                         {{"gdn_view", attn_view}, {"view_reshape", std::vector<int64_t>{1, -1, H_v, head_v}}});
    auto new_state = e.add_op("GGML_OP_VIEW",
                              ss + "_out",
                              {gdn},
                              f32,
                              4,
                              {{"gdn_view", state_view}, {"view_reshape", std::vector<int64_t>{1, H_v, head_v, S}}});
    graph.model_output_names.push_back(new_state);
    graph.recurrent_states.emplace_back(ss, new_state);

    // ---- gated output norm + projection ----
    // build_norm_gated: rms_norm(attn, ssm_norm) * silu(z), normalizing the head_v axis.
    auto out = rms_norm(e, attn, p + "ssm_norm.weight", p + "gdn_norm", cfg.rms_eps);
    auto z_4d = e.reshape(p + "z_4d", z, {0, -1, H_v, head_v}, true);
    auto z_silu = e.add_op("GGML_UNARY_OP_SILU", p + "z_silu", {z_4d}, f32);
    out = e.add_op("GGML_OP_MUL", p + "gdn_gated", {out, z_silu}, f32);
    out = e.reshape(p + "gdn_merged", out, {0, 1, -1, value_dim}, true);

    e.add_weight(p + "ssm_out.weight");
    return e.add_op("GGML_OP_MUL_MAT", p + "linear_attn_out", {p + "ssm_out.weight", out}, f32);
}

}  // namespace blocks
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
