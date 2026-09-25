// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "builder/blocks/gated_delta_net.hpp"

#include <algorithm>
#include <cstring>
#include <optional>
#include <tuple>
#include <vector>

#include "builder/blocks/common.hpp"
#include "openvino/core/except.hpp"

namespace ov::frontend::gguf::blocks {

namespace {

// Copy of `t` viewed as `outer` rows of bytes, where the chunks [first, first + src.size())
// of `chunk_bytes` in every row are reordered so that new chunk n holds old chunk src[n].
// Returns an empty tensor when the byte layout does not split evenly.
ov::Tensor permute_chunks(const ov::Tensor& t,
                          size_t outer,
                          size_t chunk_bytes,
                          size_t first,
                          const std::vector<int64_t>& src) {
    const size_t bytes = t.get_byte_size();
    if (outer == 0 || bytes % outer != 0 || chunk_bytes == 0 || (first + src.size()) * chunk_bytes > bytes / outer) {
        return {};
    }
    const size_t row_bytes = bytes / outer;
    ov::Tensor out(t.get_element_type(), t.get_shape());
    const auto* in_data = static_cast<const uint8_t*>(t.data());
    auto* out_data = static_cast<uint8_t*>(out.data());
    std::memcpy(out_data, in_data, bytes);
    for (size_t r = 0; r < outer; ++r) {
        for (size_t n = 0; n < src.size(); ++n) {
            std::memcpy(out_data + r * row_bytes + (first + n) * chunk_bytes,
                        in_data + r * row_bytes + (first + static_cast<size_t>(src[n])) * chunk_bytes,
                        chunk_bytes);
        }
    }
    return out;
}

// Reorder `src.size()` blocks of `rows_per_block` rows (dim 0) starting at row `first_row`.
ov::Tensor permute_row_blocks(const ov::Tensor& t,
                              size_t first_row,
                              size_t rows_per_block,
                              const std::vector<int64_t>& src) {
    const auto& shape = t.get_shape();
    if (shape.empty() || t.get_byte_size() % shape[0] != 0) {
        return {};
    }
    const size_t chunk = t.get_byte_size() / shape[0] * rows_per_block;
    if (first_row % rows_per_block != 0) {
        return {};
    }
    return permute_chunks(t, 1, chunk, first_row / rows_per_block, src);
}

// Reorder `src.size()` column blocks of `block_cols` of a [rows, logical_cols] weight part.
// Per-row tensors with a single column (channel-wise scales) are left as they are.
ov::Tensor permute_col_blocks(const ov::Tensor& t,
                              size_t logical_cols,
                              size_t block_cols,
                              const std::vector<int64_t>& src) {
    const auto& shape = t.get_shape();
    if (shape.size() == 2 && shape[1] == 1) {
        return t;
    }
    if (shape.size() != 2 || t.get_byte_size() % shape[0] != 0) {
        return {};
    }
    const size_t row_bytes = t.get_byte_size() / shape[0];
    if ((row_bytes * block_cols) % logical_cols != 0) {
        return {};
    }
    return permute_chunks(t, shape[0], row_bytes * block_cols / logical_cols, 0, src);
}

// Apply `fn` to every present part; fails (returns false) if any present part cannot be transformed.
template <typename Fn>
bool transform_parts(const WeightTensors& in, WeightTensors& out, Fn&& fn) {
    const std::vector<std::pair<const ov::Tensor*, ov::Tensor*>> parts{{&in.weight, &out.weight},
                                                                       {&in.scales, &out.scales},
                                                                       {&in.zero_point, &out.zero_point}};
    for (const auto& [src, dst] : parts) {
        if (*src) {
            *dst = fn(*src);
            if (!*dst) {
                return false;
            }
        }
    }
    return true;
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
    std::string conv_w = p + "ssm_conv1d.weight", a_w = p + "ssm_a", dt_w = p + "ssm_dt.bias";
    bool gqa_grouped = false;
    if (H_v != H_k && H_v % H_k == 0) {
        const int64_t rep = H_v / H_k;
        std::vector<int64_t> src(H_v);
        for (int64_t n = 0; n < H_v; ++n) {
            src[n] = (n % rep) * H_k + n / rep;
        }
        const auto rows = [&](size_t first, size_t per_block) {
            return [&, first, per_block](const ov::Tensor& t) {
                return permute_row_blocks(t, first, per_block, src);
            };
        };
        WeightTensors qkv, gate, alpha, beta, out;
        ov::Tensor conv, a, dt;
        bool ok = transform_parts(weight_parts(e, qkv_base), qkv, rows(2 * key_dim, head_v)) &&
                  transform_parts(weight_parts(e, gate_base), gate, rows(0, head_v)) &&
                  transform_parts(weight_parts(e, alpha_base), alpha, rows(0, 1)) &&
                  transform_parts(weight_parts(e, beta_base), beta, rows(0, 1)) &&
                  transform_parts(weight_parts(e, out_base), out, [&](const ov::Tensor& t) {
                      return permute_col_blocks(t, value_dim, head_v, src);
                  });
        if (ok) {
            conv = permute_row_blocks(e.weight_tensor(conv_w), 2 * key_dim, head_v, src);
            a = permute_row_blocks(e.weight_tensor(a_w), 0, 1, src);
            dt = permute_row_blocks(e.weight_tensor(dt_w), 0, 1, src);
            ok = conv && a && dt;
        }
        if (ok) {
            const std::string g = "_grouped";
            store_parts(e, qkv_base + g, qkv, weight_qtype(e, qkv_base));
            store_parts(e, gate_base + g, gate, weight_qtype(e, gate_base));
            store_parts(e, alpha_base + g, alpha, weight_qtype(e, alpha_base));
            store_parts(e, beta_base + g, beta, weight_qtype(e, beta_base));
            store_parts(e, out_base + g, out, weight_qtype(e, out_base));
            qkv_base += g, gate_base += g, alpha_base += g, beta_base += g, out_base += g;
            conv_w = p + "ssm_conv1d" + g + ".weight", a_w = p + "ssm_a" + g, dt_w = p + "ssm_dt" + g + ".bias";
            e.weights()[conv_w] = conv;
            e.weights()[a_w] = a;
            e.weights()[dt_w] = dt;
            gqa_grouped = true;
        }
    }

    // ---- input projections ----
    e.add_weight(qkv_base + ".weight");
    auto qkv = e.add_op("GGML_OP_MUL_MAT", p + "qkv_mixed", {qkv_base + ".weight", attn_norm});

    // z, ssm_beta and ssm_alpha share the input. beta/alpha produce one scalar per v-head, and
    // such a narrow matmul costs about as much as a wide one, so all three run as one matmul when
    // their quantization layouts can be expressed in a common one.
    std::string z, beta, alpha;
    auto ba = concat_rows(weight_parts(e, beta_base),
                          weight_parts(e, alpha_base),
                          weight_qtype(e, beta_base),
                          weight_qtype(e, alpha_base));
    auto zba = ba ? concat_rows_widened(weight_parts(e, gate_base),
                                        *ba,
                                        weight_qtype(e, gate_base),
                                        weight_qtype(e, beta_base))
                  : std::nullopt;
    if (zba) {
        store_parts(e, p + "ssm_z_beta_alpha", zba->first, zba->second);
        e.add_weight(p + "ssm_z_beta_alpha.weight");
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
    } else if (ba) {
        e.add_weight(gate_base + ".weight");
        z = e.add_op("GGML_OP_MUL_MAT", p + "z", {gate_base + ".weight", attn_norm});
        store_parts(e, p + "ssm_beta_alpha", *ba, weight_qtype(e, beta_base));
        e.add_weight(p + "ssm_beta_alpha.weight");
        auto ba_out = e.add_op("GGML_OP_MUL_MAT", p + "beta_alpha", {p + "ssm_beta_alpha.weight", attn_norm});
        beta = e.add_op("GGML_OP_VIEW", p + "beta", {ba_out}, 3, {{"view_slice", std::vector<int64_t>{3, 0, H_v}}});
        alpha = e.add_op("GGML_OP_VIEW", p + "alpha", {ba_out}, 3, {{"view_slice", std::vector<int64_t>{3, H_v, H_v}}});
    } else {
        e.add_weight(gate_base + ".weight");
        z = e.add_op("GGML_OP_MUL_MAT", p + "z", {gate_base + ".weight", attn_norm});
        e.add_weight(beta_base + ".weight");
        beta = e.add_op("GGML_OP_MUL_MAT", p + "beta", {beta_base + ".weight", attn_norm});
        e.add_weight(alpha_base + ".weight");
        alpha = e.add_op("GGML_OP_MUL_MAT", p + "alpha", {alpha_base + ".weight", attn_norm});
    }

    // beta = sigmoid(ssm_beta @ x), one scalar per v-head
    beta = e.add_op("GGML_UNARY_OP_SIGMOID", p + "beta_sig", {beta});
    beta = e.add_op("GGML_OP_RESHAPE",
                    p + "beta_4d",
                    {beta},
                    6,
                    {{"reshape_target", std::vector<int64_t>{0, -1, H_v, 1}}, {"special_zero", true}});

    // g = softplus(ssm_alpha @ x + ssm_dt.bias) * ssm_a   (ggml: -A_log.exp() * softplus)
    e.add_named_weight(dt_w);
    alpha = e.add_op("GGML_OP_ADD", p + "alpha_biased", {alpha, dt_w});
    alpha = e.add_op("GGML_UNARY_OP_SOFTPLUS", p + "alpha_sp", {alpha});
    e.add_named_weight(a_w);
    auto g = e.add_op("GGML_OP_MUL", p + "gate", {alpha, a_w});
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

    e.add_named_weight(conv_w);
    auto conv = e.add_op("GGML_OP_SSM_CONV", p + "conv_out", {conv_in, conv_w}, 0, {{"batch_major", true}});
    conv = e.add_op("GGML_UNARY_OP_SILU", p + "conv_silu", {conv});

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
    // ggml state layout is [B, H_v, value_dim, key_dim]; the translator transposes it for
    // the fused op and transposes the new state back, so the Parameter keeps ggml's layout.
    const std::string ss = "ssm_state_l" + std::to_string(il);
    if (!e.has_model_input(ss)) {
        e.add_input(ss, f32, ps({1, H_v, head_v, S}));
    }

    auto gdn = e.add_op("GGML_OP_GATED_DELTA_NET",
                        p + "gdn",
                        {q, k, v, g, beta, ss},
                        0,
                        {{"gdn_state_slots", int64_t{1}},
                         {"fuse_qk_l2norm", true},
                         {"qk_l2_norm_eps", cfg.rms_eps},
                         {"gqa_grouped", gqa_grouped}});

    // Split the packed attention rows and recurrent state; only the token axis is inferred.
    const std::vector<int64_t> attn_view{0, head_v};
    const std::vector<int64_t> state_view{1, head_v};
    auto attn = e.add_op("GGML_OP_VIEW", p + "gdn_attn", {gdn}, 4, {{"gdn_view", attn_view}});
    auto new_state = e.add_op("GGML_OP_VIEW", ss + "_out", {gdn}, 4, {{"gdn_view", state_view}});
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
    return e.add_op("GGML_OP_MUL_MAT", p + "linear_attn_out", {out_base + ".weight", out});
}

}  // namespace ov::frontend::gguf::blocks
