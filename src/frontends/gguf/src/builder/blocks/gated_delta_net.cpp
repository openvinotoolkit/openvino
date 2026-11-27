// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "builder/blocks/gated_delta_net.hpp"

#include <algorithm>
#include <array>
#include <numeric>
#include <vector>

#include "builder/blocks/common.hpp"
#include "openvino/core/except.hpp"

namespace ov::frontend::gguf::blocks {

namespace {

std::vector<int64_t> row_permutation(const ov::Shape& shape,
                                     int64_t first,
                                     int64_t rows_per_block,
                                     const std::vector<int64_t>& source) {
    if (shape.empty() || shape[0] == 0 || rows_per_block <= 0 || first < 0 ||
        static_cast<size_t>(first + rows_per_block * source.size()) > shape[0]) {
        return {};
    }
    std::vector<int64_t> indices(shape[0]);
    std::iota(indices.begin(), indices.end(), 0);
    for (size_t block = 0; block < source.size(); ++block) {
        for (int64_t row = 0; row < rows_per_block; ++row) {
            indices[first + block * rows_per_block + row] = first + source[block] * rows_per_block + row;
        }
    }
    return indices;
}

bool concatenable_rows(GraphEmitter& e, const std::string& lhs, const std::string& rhs) {
    const auto& a = e.weight_shape(lhs + ".weight");
    const auto& b = e.weight_shape(rhs + ".weight");
    return !a.empty() && a.size() == b.size() && std::equal(a.begin() + 1, a.end(), b.begin() + 1) &&
           e.weight_qtype(lhs) == e.weight_qtype(rhs);
}

}  // namespace

std::string gated_delta_net(GraphEmitter& e, const DecoderConfig& cfg, int il, const std::string& attn_norm) {
    using ov::element::f32;
    const std::string p = "blk." + std::to_string(il) + ".";
    auto& graph = *e.graph();

    const int64_t d_conv = cfg.ssm_conv_kernel;
    const int64_t S = cfg.ssm_state_size;     // head_k_dim == head_v_dim
    const int64_t H_k = cfg.ssm_group_count;  // num_k_heads
    const int64_t H_v = cfg.ssm_dt_rank;      // num_v_heads
    OPENVINO_ASSERT(H_k > 0, "[GGUF] Gated-DeltaNet requires a positive ssm.group_count, got ", H_k);
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

    // ggml pairs V head j with K head j % H_k (tiled order); the fused OV op pairs it with
    // K head j / (H_v / H_k) (grouped order). Storing the V heads in grouped order in every
    // per-V-head weight lets the op run without a runtime Tile of q and k.
    std::string qkv_base = p + "attn_qkv", gate_base = p + "attn_gate", out_base = p + "ssm_out";
    std::string alpha_base = p + "ssm_alpha", beta_base = p + "ssm_beta";
    std::string conv_base = p + "ssm_conv1d", a_base = p + "ssm_a", dt_base = p + "ssm_dt";
    const bool qwen3next = cfg.arch == "qwen3next";
    if (qwen3next)
        OPENVINO_ASSERT(H_v >= H_k && H_v % H_k == 0, "[GGUF] Gated-DeltaNet requires an integral V/K head ratio");
    if (qwen3next && !e.has_weight(qkv_base + ".weight")) {
        const auto fused = p + "ssm_in.weight";
        const int64_t group_value = value_dim / H_k;
        const int64_t group_width = 2 * S + 2 * group_value;
        OPENVINO_ASSERT(e.weight_rows(fused) == H_k * group_width, "[GGUF] invalid legacy Qwen3-Next QKVZ rows");
        std::vector<int64_t> qkv_order, gate_order;
        const std::array<std::pair<int64_t, int64_t>, 4> ranges{std::make_pair(0, S),
                                                                {S, S},
                                                                {2 * S, group_value},
                                                                {2 * S + group_value, group_value}};
        for (size_t i = 0; i < ranges.size(); ++i) {
            const auto& range = ranges[i];
            auto& order = i == ranges.size() - 1 ? gate_order : qkv_order;
            for (int64_t h = 0; h < H_k; ++h)
                for (int64_t row = 0; row < range.second; ++row)
                    order.push_back(h * group_width + range.first + row);
        }
        e.derive_weight(qkv_base + ".weight",
                        "GGML_OP_VIEW",
                        {fused},
                        105,
                        {{"gather_axis", int64_t{0}}, {"gather_indices", qkv_order}});
        e.derive_weight(gate_base + ".weight",
                        "GGML_OP_VIEW",
                        {fused},
                        105,
                        {{"gather_axis", int64_t{0}}, {"gather_indices", gate_order}});
    }
    if (qwen3next && !e.has_weight(beta_base + ".weight")) {
        const auto fused = p + "ssm_ba.weight";
        OPENVINO_ASSERT(e.has_weight(fused) && e.weight_rows(fused) == 2 * H_v,
                        "[GGUF] invalid Qwen3-Next beta/alpha projection");
        const auto per_group = H_v / H_k;
        for (int64_t part = 0; part < 2; ++part) {
            std::vector<int64_t> order;
            for (int64_t h = 0; h < H_k; ++h)
                for (int64_t row = 0; row < per_group; ++row)
                    order.push_back(h * 2 * per_group + part * per_group + row);
            e.derive_weight((part == 0 ? beta_base : alpha_base) + ".weight",
                            "GGML_OP_VIEW",
                            {fused},
                            105,
                            {{"gather_axis", int64_t{0}}, {"gather_indices", order}});
        }
    }
    bool gqa_grouped = qwen3next;
    const bool grouped_hadamard = e.has_grouped_hadamard_gdn();
    if (!gqa_grouped && (!e.has_hadamard() || grouped_hadamard) && H_v != H_k && H_v % H_k == 0) {
        const int64_t rep = H_v / H_k;
        std::vector<int64_t> src(H_v);
        for (int64_t n = 0; n < H_v; ++n) {
            src[n] = (n % rep) * H_k + n / rep;
        }
        struct Grouped {
            std::string* base;
            std::string suffix;
            int64_t axis;
            std::vector<int64_t> indices;
        };
        std::vector<Grouped> grouped;
        const auto rows = [&](std::string& base, const std::string& suffix, int64_t first, int64_t per_block) {
            const auto& shape = e.weight_shape(base + suffix);
            grouped.push_back({&base, suffix, 0, row_permutation(shape, first, per_block, src)});
        };
        rows(qkv_base, ".weight", 2 * key_dim, head_v);
        rows(gate_base, ".weight", 0, head_v);
        rows(alpha_base, ".weight", 0, 1);
        rows(beta_base, ".weight", 0, 1);
        rows(conv_base, ".weight", 2 * key_dim, head_v);
        rows(a_base, "", 0, 1);
        rows(dt_base, ".bias", 0, 1);
        if (!grouped_hadamard) {
            const auto& shape = e.weight_shape(out_base + ".weight");
            auto indices = shape.size() == 2 && shape[1] == static_cast<size_t>(value_dim)
                               ? row_permutation({shape[1]}, 0, head_v, src)
                               : std::vector<int64_t>{};
            if (!e.column_groups_aligned(out_base + ".weight", indices))
                indices.clear();
            grouped.push_back({&out_base, ".weight", 1, std::move(indices)});
        }
        if (std::all_of(grouped.begin(), grouped.end(), [](const Grouped& weight) {
                return !weight.indices.empty();
            })) {
            for (auto& weight : grouped) {
                const auto source = *weight.base + weight.suffix;
                *weight.base += "_grouped";
                e.derive_weight(*weight.base + weight.suffix,
                                "GGML_OP_VIEW",
                                {source},
                                105,
                                {{"gather_axis", weight.axis}, {"gather_indices", weight.indices}});
            }
            gqa_grouped = true;
        }
    }

    // ---- input projections ----
    e.add_weight(qkv_base + ".weight");
    auto qkv = e.add_op("GGML_OP_MUL_MAT", p + "qkv_mixed", {qkv_base + ".weight", attn_norm});

    // Merge compatible projections without widening z. Standalone FP32 scalar projections
    // remain separate so their bias and activations can fuse into FCs.
    std::string z, beta, alpha;
    const bool dense_gates = e.weight_qtype(beta_base) == GGUF_TYPE_F32 && e.weight_qtype(alpha_base) == GGUF_TYPE_F32;
    const bool ba = (!dense_gates || !e.has_hadamard()) && concatenable_rows(e, beta_base, alpha_base);
    const bool zba = ba && !e.has_hadamard() && (!dense_gates || e.weight_qtype(gate_base) == GGUF_TYPE_F32) &&
                     concatenable_rows(e, gate_base, beta_base);
    if (zba) {
        concat_weight_rows(e,
                           p + "ssm_z_beta_alpha.weight",
                           {gate_base + ".weight", beta_base + ".weight", alpha_base + ".weight"});
        auto zba_out = e.add_op("GGML_OP_MUL_MAT", p + "z_beta_alpha", {p + "ssm_z_beta_alpha.weight", attn_norm});
        z = e.add_op("GGML_OP_VIEW", p + "z", {zba_out}, 3, {{"view_slice", std::vector<int64_t>{3, 0, value_dim}}});
        beta = e.add_op("GGML_OP_VIEW",
                        p + "beta",
                        {zba_out},
                        3,
                        {{"view_slice", std::vector<int64_t>{3, value_dim, H_v}}});
        alpha = e.add_op("GGML_OP_VIEW",
                         p + "alpha",
                         {zba_out},
                         3,
                         {{"view_slice", std::vector<int64_t>{3, value_dim + H_v, H_v}}});
    } else if (ba && !dense_gates) {
        e.add_weight(gate_base + ".weight");
        z = e.add_op("GGML_OP_MUL_MAT", p + "z", {gate_base + ".weight", attn_norm});
        concat_weight_rows(e, p + "ssm_beta_alpha.weight", {beta_base + ".weight", alpha_base + ".weight"});
        auto ba_out = e.add_op("GGML_OP_MUL_MAT", p + "beta_alpha", {p + "ssm_beta_alpha.weight", attn_norm});
        beta = e.add_op("GGML_OP_VIEW", p + "beta", {ba_out}, 3, {{"view_slice", std::vector<int64_t>{3, 0, H_v}}});
        alpha = e.add_op("GGML_OP_VIEW", p + "alpha", {ba_out}, 3, {{"view_slice", std::vector<int64_t>{3, H_v, H_v}}});
    } else {
        const std::map<std::string, ov::Any> gate_attrs{{"activation_rank", dense_gates ? 3 : 4}};
        e.add_weight(gate_base + ".weight");
        z = e.add_op("GGML_OP_MUL_MAT", p + "z", {gate_base + ".weight", attn_norm});
        e.add_weight(beta_base + ".weight");
        beta = e.add_op("GGML_OP_MUL_MAT", p + "beta", {beta_base + ".weight", attn_norm}, 0, gate_attrs);
        e.add_weight(alpha_base + ".weight");
        alpha = e.add_op("GGML_OP_MUL_MAT", p + "alpha", {alpha_base + ".weight", attn_norm}, 0, gate_attrs);
    }

    // beta = sigmoid(ssm_beta @ x), one scalar per v-head
    beta = e.add_op("GGML_UNARY_OP_SIGMOID", p + "beta_sig", {beta});
    beta = e.add_op("GGML_OP_RESHAPE",
                    p + "beta_4d",
                    {beta},
                    6,
                    {{"reshape_target", std::vector<int64_t>{0, -1, H_v, 1}}, {"special_zero", true}});

    // g = softplus(ssm_alpha @ x + ssm_dt.bias) * ssm_a   (ggml: -A_log.exp() * softplus)
    const auto dt_w = dt_base + ".bias";
    e.add_weight(dt_w);
    alpha = e.add_op("GGML_OP_ADD", p + "alpha_biased", {alpha, dt_w});
    alpha = e.add_op("GGML_UNARY_OP_SOFTPLUS", p + "alpha_sp", {alpha});
    e.add_weight(a_base);
    auto g = e.add_op("GGML_OP_MUL", p + "gate", {alpha, a_base});
    g = e.add_op("GGML_OP_RESHAPE",
                 p + "gate_4d",
                 {g},
                 6,
                 {{"reshape_target", std::vector<int64_t>{0, -1, H_v, 1}}, {"special_zero", true}});

    // ---- causal depthwise conv over [conv state | this step's tokens] ----
    // conv_state holds the trailing d_conv-1 columns of the previous step's conv input.
    const std::string cs = "conv_state_l" + std::to_string(il);
    if (!e.has_model_input(cs)) {
        e.add_input(cs, f32, ps({1, 1, conv_dim, d_conv - 1}));
    }

    // [1,1,T,conv_dim] -> [1,1,conv_dim,T] so the conv window grows along the last axis.
    auto qkv_t = e.add_op("GGML_OP_TRANSPOSE", p + "qkv_t", {qkv});
    auto conv_in = e.add_op("GGML_OP_CONCAT", p + "conv_in", {cs, qkv_t}, 0, {{"concat_axis", int{0}}});

    // Next step's state is the trailing d_conv-1 columns of this window.
    const std::vector<int64_t> tail_slice{3, -(d_conv - 1), d_conv - 1};
    auto cs_out = e.add_op("GGML_OP_VIEW", cs + "_out", {conv_in}, 3, {{"view_slice", tail_slice}});
    graph.model_output_names.push_back(cs_out);
    graph.recurrent_states.emplace_back(cs, cs_out);

    const auto conv_w = conv_base + ".weight";
    e.add_weight(conv_w);
    auto conv = e.add_op("GGML_OP_SSM_CONV",
                         p + "conv_silu",
                         {conv_in, conv_w},
                         0,
                         {{"batch_major", true}, {"fuse_silu", true}});

    // ---- split the conv output into q | k | v; the fused op L2-normalizes q/k ----
    auto slice_heads = [&](const std::string& name, int64_t off, int64_t width, int64_t heads, int64_t dim) {
        const std::vector<int64_t> sl{3, off, width};
        auto s = e.add_op("GGML_OP_VIEW", p + name + "_s", {conv}, 3, {{"view_slice", sl}});
        return e.add_op("GGML_OP_RESHAPE",
                        p + name,
                        {s},
                        6,
                        {{"reshape_target", std::vector<int64_t>{0, -1, heads, dim}}, {"special_zero", true}});
    };
    auto q = slice_heads("q_conv", 0, key_dim, H_k, S);
    auto k = slice_heads("k_conv", key_dim, key_dim, H_k, S);
    auto v = slice_heads("v_conv", 2 * key_dim, value_dim, H_v, head_v);

    // ---- recurrent delta rule ----
    // The state keeps the fused op's [B, H_v, key_dim, value_dim] layout, and the op returns the
    // attention and the new state as separate outputs.
    const std::string ss = "ssm_state_l" + std::to_string(il);
    if (!e.has_model_input(ss)) {
        e.add_input(ss, f32, ps({1, H_v, S, head_v}));
    }

    const auto new_state = ss + "_out";
    auto attn = e.add_op("GGML_OP_GATED_DELTA_NET",
                         p + "gdn",
                         {q, k, v, g, beta, ss},
                         0,
                         {{"gdn_state_slots", int64_t{1}},
                          {"fuse_qk_l2norm", true},
                          {"qk_l2_norm_eps", cfg.rms_eps},
                          {"gqa_grouped", gqa_grouped},
                          {"split_outputs", true}},
                         {new_state});
    graph.model_output_names.push_back(new_state);
    graph.recurrent_states.emplace_back(ss, new_state);

    // ---- gated output norm + projection ----
    // build_norm_gated: rms_norm(attn, ssm_norm) * silu(z), normalizing the head_v axis.
    auto out = rms_norm(e, attn, p + "ssm_norm.weight", p + "gdn_norm", cfg.rms_eps);
    auto z_4d = e.add_op("GGML_OP_RESHAPE",
                         p + "z_4d",
                         {z},
                         6,
                         {{"reshape_target", std::vector<int64_t>{0, -1, H_v, head_v}}, {"special_zero", true}});
    auto z_silu = e.add_op("GGML_UNARY_OP_SILU", p + "z_silu", {z_4d});
    out = e.add_op("GGML_OP_MUL", p + "gdn_gated", {out, z_silu});
    out = e.add_op("GGML_OP_RESHAPE",
                   p + "gdn_merged",
                   {out, attn_norm},
                   0,
                   {{"reshape_target", std::vector<int64_t>{1, 1, -1, value_dim}},
                    {"shape_axes", std::vector<int64_t>{0, 1, 2, -1}}});

    e.add_weight(out_base + ".weight");
    return e.add_op("GGML_OP_MUL_MAT",
                    p + "linear_attn_out",
                    {out_base + ".weight", out},
                    0,
                    {{"gqa_grouped", gqa_grouped}});
}

}  // namespace ov::frontend::gguf::blocks
