// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Configuration: Enable loop-based Q@K computation to avoid materialized K/V broadcast
// Set to 1 to enable grouped computation, 0 to use traditional broadcast
// Disabled by default because NPU compiler optimizations are suboptimal
#define ENABLE_HFA_LOOP_BASED_COMPUTATION 0

#include "host_flash_attention.hpp"

#include "intel_npu/ops/flash_attention_tile.hpp"
#include "logging.hpp"
#include "npuw_transformations/detect_causal_mask.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/openvino.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "util.hpp"

namespace ov {
namespace npuw {
namespace function {

namespace opp = ov::pass::pattern;

// Helper struct: Holds all input parameter nodes for HFA tile model creation
// Contains 7 parameters: past_acc, past_max, past_d, k_tile, v_tile, q, mask_tile
struct HFATileInputs {
    std::shared_ptr<ov::op::v0::Parameter> past_acc;
    std::shared_ptr<ov::op::v0::Parameter> past_max;
    std::shared_ptr<ov::op::v0::Parameter> past_d;
    std::shared_ptr<ov::op::v0::Parameter> k_tile;
    std::shared_ptr<ov::op::v0::Parameter> v_tile;
    std::shared_ptr<ov::op::v0::Parameter> q;
    std::shared_ptr<ov::op::v0::Parameter> mask_tile;
};

// Helper struct: Holds f32-converted nodes from input parameters for computation
// All computations are performed in f32 for numerical stability
struct HFATileF32Nodes {
    std::shared_ptr<ov::Node> past_acc_f32;
    std::shared_ptr<ov::Node> past_max_f32;
    std::shared_ptr<ov::Node> past_d_f32;
    std::shared_ptr<ov::Node> k_tile_f32;
    std::shared_ptr<ov::Node> v_tile_f32;
    std::shared_ptr<ov::Node> q_f32;
    std::shared_ptr<ov::Node> mask_tile_f32;
};

// Helper struct: Flash attention computation results (all in f32 precision)
// Contains: acc (accumulator), maxx (maximum values), d (normalization denominator)
struct FlashAttentionResults {
    ov::Output<ov::Node> acc;
    ov::Output<ov::Node> maxx;
    ov::Output<ov::Node> d;
};

// ============================================================================
// Helper function: Create input parameters for HFA tile model
// ============================================================================
// state_dtype  : element type for past_acc / past_max / past_d (internal HFA state;
//               typically f16 to match KV-block storage dtype).
// kv_tile_dtype: element type for k_tile / v_tile (the KV slice fed into each tile).
//               May differ from state_dtype — e.g. f32 for the final tile that
//               receives the freshly-computed present-KV from the upstream graph,
//               vs f16 for regular tiles that read stored KV blocks.
// ============================================================================
static HFATileInputs create_hfa_tile_inputs(const ov::Shape& q_shape,
                                            const ov::element::Type& state_dtype,
                                            const ov::element::Type& kv_tile_dtype,
                                            const ov::element::Type& q_dtype,
                                            const ov::element::Type& mask_dtype,
                                            int64_t tile_size,
                                            size_t kv_num_heads,
                                            size_t mask_num_heads,
                                            size_t k_seq_dim,
                                            bool v_transposed = true) {
    auto batch = q_shape[0];
    auto num_heads = q_shape[1];
    auto seq_len = q_shape[2];
    auto head_dim = q_shape[3];

    HFATileInputs inputs;

    auto set_param_name = [](std::shared_ptr<ov::op::v0::Parameter>& param, HFATileInputId id) {
        const char* name = hfa_tile_input_id_to_string(id);
        param->set_friendly_name(name);
        param->output(0).get_tensor().set_names({name});
    };

    // State tensors (acc / max / d) use state_dtype so they stay consistent between
    // the regular tile output and the next tile's input without any conversion.
    // past_acc: [batch, num_heads, seq_len, head_dim]
    inputs.past_acc =
        std::make_shared<ov::op::v0::Parameter>(state_dtype, ov::Shape{batch, num_heads, seq_len, head_dim});
    set_param_name(inputs.past_acc, HFATileInputId::PAST_ACC);

    // past_max: [batch, num_heads, seq_len, 1]
    inputs.past_max = std::make_shared<ov::op::v0::Parameter>(state_dtype, ov::Shape{batch, num_heads, seq_len, 1});
    set_param_name(inputs.past_max, HFATileInputId::PAST_MAX);

    // past_d: [batch, num_heads, seq_len, 1]
    inputs.past_d = std::make_shared<ov::op::v0::Parameter>(state_dtype, ov::Shape{batch, num_heads, seq_len, 1});
    set_param_name(inputs.past_d, HFATileInputId::PAST_D);

    // KV tile tensors use kv_tile_dtype (may differ from state_dtype).
    // k_tile follows the model's own K cache layout: [batch, kv_num_heads, tile_size, head_dim]
    // in the usual case, [batch, tile_size, kv_num_heads, head_dim] when the sequence is at
    // dim 1 (the tile graph transposes it back before the QK MatMul).
    if (k_seq_dim == 1) {
        inputs.k_tile = std::make_shared<ov::op::v0::Parameter>(
            kv_tile_dtype,
            ov::Shape{batch, static_cast<size_t>(tile_size), kv_num_heads, head_dim});
    } else {
        inputs.k_tile = std::make_shared<ov::op::v0::Parameter>(
            kv_tile_dtype,
            ov::Shape{batch, kv_num_heads, static_cast<size_t>(tile_size), head_dim});
    }
    set_param_name(inputs.k_tile, HFATileInputId::K_TILE);

    // v_tile: [batch, kv_num_heads, head_dim, tile_size] when V is pre-transposed by OptimizeValueTensors,
    //          [batch, kv_num_heads, tile_size, head_dim] when V is in normal (non-transposed) layout.
    if (v_transposed) {
        inputs.v_tile = std::make_shared<ov::op::v0::Parameter>(
            kv_tile_dtype,
            ov::Shape{batch, kv_num_heads, head_dim, static_cast<size_t>(tile_size)});
    } else {
        inputs.v_tile = std::make_shared<ov::op::v0::Parameter>(
            kv_tile_dtype,
            ov::Shape{batch, kv_num_heads, static_cast<size_t>(tile_size), head_dim});
    }
    set_param_name(inputs.v_tile, HFATileInputId::V_TILE);

    // q: [batch, num_heads, seq_len, head_dim]
    // Q may run at a different precision from KV cache (e.g. f32 vs f16); use q_dtype.
    inputs.q = std::make_shared<ov::op::v0::Parameter>(q_dtype, ov::Shape{batch, num_heads, seq_len, head_dim});
    set_param_name(inputs.q, HFATileInputId::Q);

    // mask_tile: [batch, mask_num_heads, seq_len, tile_size] - use mask's original dtype.
    // mask_num_heads follows the model's own mask layout (1 when broadcast over heads).
    inputs.mask_tile = std::make_shared<ov::op::v0::Parameter>(
        mask_dtype,
        ov::Shape{batch, mask_num_heads, seq_len, static_cast<size_t>(tile_size)});
    set_param_name(inputs.mask_tile, HFATileInputId::MASK_TILE);

    return inputs;
}

// ============================================================================
// Helper function: Convert input parameters to f32
// ============================================================================
static HFATileF32Nodes convert_inputs_to_f32(const HFATileInputs& inputs,
                                             const ov::element::Type& mask_dtype,
                                             const ov::element::Type& compute_dtype,
                                             bool use_mask = true) {
    HFATileF32Nodes f32_nodes;

    f32_nodes.past_acc_f32 = std::make_shared<ov::op::v0::Convert>(inputs.past_acc, compute_dtype);
    f32_nodes.past_acc_f32->set_friendly_name("past_acc_f32");

    f32_nodes.past_max_f32 = std::make_shared<ov::op::v0::Convert>(inputs.past_max, compute_dtype);
    f32_nodes.past_max_f32->set_friendly_name("past_max_f32");

    f32_nodes.past_d_f32 = std::make_shared<ov::op::v0::Convert>(inputs.past_d, compute_dtype);
    f32_nodes.past_d_f32->set_friendly_name("past_d_f32");

    f32_nodes.k_tile_f32 = std::make_shared<ov::op::v0::Convert>(inputs.k_tile, compute_dtype);
    f32_nodes.k_tile_f32->set_friendly_name("k_tile_f32");

    f32_nodes.v_tile_f32 = std::make_shared<ov::op::v0::Convert>(inputs.v_tile, compute_dtype);
    f32_nodes.v_tile_f32->set_friendly_name("v_tile_f32");

    f32_nodes.q_f32 = std::make_shared<ov::op::v0::Convert>(inputs.q, compute_dtype);
    f32_nodes.q_f32->set_friendly_name("q_f32");

    if (use_mask) {
        // Convert mask to f32 if needed
        if (mask_dtype == compute_dtype) {
            f32_nodes.mask_tile_f32 = inputs.mask_tile;
        } else {
            f32_nodes.mask_tile_f32 = std::make_shared<ov::op::v0::Convert>(inputs.mask_tile, compute_dtype);
            f32_nodes.mask_tile_f32->set_friendly_name("mask_tile_f32");
        }
    }

    return f32_nodes;
}

// ============================================================================
// Helper function: Execute flash attention tile implementation using NPU fused op
// ============================================================================
static FlashAttentionResults execute_fused_flash_attention(const HFATileF32Nodes& f32_nodes,
                                                           const std::shared_ptr<ov::Node>& q_input,
                                                           const std::shared_ptr<ov::Node>& k_input,
                                                           const std::shared_ptr<ov::Node>& v_input,
                                                           bool is_last_tile = false,
                                                           bool is_first_tile = false,
                                                           bool v_transposed = true) {
    ov::intel_npu::op::FlashAttentionTile::Config config;
    config.is_head = is_first_tile;
    config.is_tail = is_last_tile;

    auto v_shape = v_input->get_output_partial_shape(0);
    auto rank = v_shape.rank().get_length();
    if (rank != 4)
        OPENVINO_THROW("v_input rank must be 4 for flash attention");

    // When V is pre-transposed by OptimizeValueTensors, the tile parameter is [B,H,head_dim,tile_size];
    // we transpose it back to [B,H,tile_size,head_dim] before feeding FlashAttentionTile (which expects
    // normal [B,H,seq_len,head_dim] layout).  When V was NOT transposed, the tile parameter already
    // holds normal layout [B,H,tile_size,head_dim] and no transpose is needed.
    std::shared_ptr<ov::Node> v_for_attn;
    if (v_transposed) {
        std::vector<int64_t> transpose_v_order({0, 1, 3, 2});
        auto transpose_order = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                      ov::Shape{static_cast<size_t>(rank)},
                                                                      transpose_v_order);
        auto v_transpose = std::make_shared<ov::op::v1::Transpose>(v_input, transpose_order);
        v_transpose->set_friendly_name("v_input_transposed");
        v_for_attn = v_transpose;
    } else {
        // V is already in [B,H,tile_size,head_dim] — pass directly.
        v_for_attn = v_input;
    }

    auto squeeze = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});

    auto past_max_squeezed = std::make_shared<ov::op::v0::Squeeze>(f32_nodes.past_max_f32, squeeze);
    past_max_squeezed->set_friendly_name("past_max_squeezed");

    auto past_sum_squeezed = std::make_shared<ov::op::v0::Squeeze>(f32_nodes.past_d_f32, squeeze);
    past_sum_squeezed->set_friendly_name("past_sum_squeezed");

    const bool use_mask = is_last_tile || static_cast<bool>(f32_nodes.mask_tile_f32);
    OPENVINO_ASSERT(!is_last_tile || f32_nodes.mask_tile_f32,
                    "Final fused HFA tile requires mask input, but mask_tile_f32 is missing");

    std::shared_ptr<ov::intel_npu::op::FlashAttentionTile> flash_attn_tile;
    if (use_mask) {
        flash_attn_tile = std::make_shared<ov::intel_npu::op::FlashAttentionTile>(q_input,
                                                                                  k_input,
                                                                                  v_for_attn,
                                                                                  f32_nodes.past_acc_f32,
                                                                                  past_max_squeezed,
                                                                                  past_sum_squeezed,
                                                                                  f32_nodes.mask_tile_f32,
                                                                                  config);
    } else {
        flash_attn_tile = std::make_shared<ov::intel_npu::op::FlashAttentionTile>(q_input,
                                                                                  k_input,
                                                                                  v_for_attn,
                                                                                  f32_nodes.past_acc_f32,
                                                                                  past_max_squeezed,
                                                                                  past_sum_squeezed,
                                                                                  config);
    }

    flash_attn_tile->set_friendly_name("npu_op_flash_attention_tile");
    FlashAttentionResults results;
    results.acc = flash_attn_tile->output(0);
    results.maxx = flash_attn_tile->output(1);
    results.d = flash_attn_tile->output(2);
    return results;
}

// ============================================================================
// Helper function: Execute flash attention algorithm (unified implementation)
// Supports both traditional broadcast and loop-based grouped computation
// ============================================================================
// Parameters:
//   use_grouped: If true, uses loop-based grouped computation (Q/P reshape)
//                If false, uses traditional broadcast K/V approach
static FlashAttentionResults execute_host_flash_attention(const HFATileF32Nodes& f32_nodes,
                                                          const std::shared_ptr<ov::Node>& q_input,
                                                          const std::shared_ptr<ov::Node>& k_input,
                                                          const std::shared_ptr<ov::Node>& v_input,
                                                          size_t batch,
                                                          size_t num_heads,
                                                          size_t kv_num_heads,
                                                          size_t seq_len,
                                                          size_t tile_size,
                                                          size_t head_dim,
                                                          bool use_grouped = false) {
    FlashAttentionResults results;

    // ========================================================================
    // Step 1: Compute QK (method differs based on use_grouped flag)
    // ========================================================================
    std::shared_ptr<ov::Node> qk;

    if (use_grouped) {
        // Loop-based grouped computation: Q and K are grouped format
        // Q_input:  [batch, kv_num_heads, factor * seq_len, head_dim]
        // K_input:  [batch, kv_num_heads, tile_size, head_dim]
        // QK_grouped: [batch, kv_num_heads, factor * seq_len, tile_size]
        auto qk_grouped = std::make_shared<ov::op::v0::MatMul>(q_input, k_input, false, true);
        qk_grouped->set_friendly_name("qk_grouped");

        // Reshape QK back: [batch, kv_num_heads, factor * seq_len, tile_size] -> [batch, num_heads, seq_len, tile_size]
        auto qk_reshape_pattern =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                   ov::Shape{4},
                                                   std::vector<int64_t>{static_cast<int64_t>(batch),
                                                                        static_cast<int64_t>(num_heads),
                                                                        static_cast<int64_t>(seq_len),
                                                                        static_cast<int64_t>(tile_size)});
        qk = std::make_shared<ov::op::v1::Reshape>(qk_grouped, qk_reshape_pattern, false);
        qk->set_friendly_name("qk");
    } else {
        // Traditional broadcast computation: use broadcast K directly
        // Q_input:  [batch, num_heads, seq_len, head_dim]
        // K_input:  [batch, num_heads, tile_size, head_dim] (already broadcast)
        // QK:       [batch, num_heads, seq_len, tile_size]
        qk = std::make_shared<ov::op::v0::MatMul>(q_input, k_input, false, true);
        qk->set_friendly_name("qk");
    }

    // ========================================================================
    // Step 2: Flash Attention core algorithm (same for both methods)
    // ========================================================================

    // qkm = qk + mask.  Regular tiles may be built without a mask input at all
    // (mask skipping): then the raw QK scores are used directly.
    std::shared_ptr<ov::Node> qkm = qk;
    if (f32_nodes.mask_tile_f32) {
        auto qkm_add = std::make_shared<ov::op::v1::Add>(qk, f32_nodes.mask_tile_f32);
        qkm_add->set_friendly_name("qkm");
        qkm = qkm_add;
    }

    // maxx = max(past_max, reduce_max(qkm, axis=-1, keepdims=True))
    auto axes_const = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
    auto qkm_max = std::make_shared<ov::op::v1::ReduceMax>(qkm, axes_const, true);
    qkm_max->set_friendly_name("qkm_max");

    auto maxx_node = std::make_shared<ov::op::v1::Maximum>(qkm_max, f32_nodes.past_max_f32);
    maxx_node->set_friendly_name("maxx");
    results.maxx = maxx_node->output(0);

    // p = exp(qkm - maxx)
    auto qkm_sub_maxx = std::make_shared<ov::op::v1::Subtract>(qkm, results.maxx);
    auto p = std::make_shared<ov::op::v0::Exp>(qkm_sub_maxx);
    p->set_friendly_name("p");

    // l = reduce_sum(p, axis=-1, keepdims=True)
    auto l = std::make_shared<ov::op::v1::ReduceSum>(p, axes_const, true);
    l->set_friendly_name("l");

    // alpha = exp(past_max - maxx)
    auto past_max_sub_maxx = std::make_shared<ov::op::v1::Subtract>(f32_nodes.past_max_f32, results.maxx);
    auto alpha = std::make_shared<ov::op::v0::Exp>(past_max_sub_maxx);
    alpha->set_friendly_name("alpha");

    // d = past_d * alpha + l
    auto past_d_alpha = std::make_shared<ov::op::v1::Multiply>(f32_nodes.past_d_f32, alpha);
    auto d_node = std::make_shared<ov::op::v1::Add>(past_d_alpha, l);
    d_node->set_friendly_name("d");
    results.d = d_node->output(0);

    // ========================================================================
    // Step 3: Compute PV and final accumulator (method differs based on use_grouped flag)
    // ========================================================================

    auto past_acc_alpha = std::make_shared<ov::op::v1::Multiply>(f32_nodes.past_acc_f32, alpha);
    std::shared_ptr<ov::Node> pv;

    if (use_grouped) {
        // Loop-based grouped computation: reshape P, multiply with V, reshape back
        size_t factor = num_heads / kv_num_heads;

        // Reshape P for grouped V multiplication: [batch, num_heads, seq_len, tile_size]
        //                                      -> [batch, kv_num_heads, factor * seq_len, tile_size]
        auto p_reshape_pattern =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                   ov::Shape{4},
                                                   std::vector<int64_t>{static_cast<int64_t>(batch),
                                                                        static_cast<int64_t>(kv_num_heads),
                                                                        static_cast<int64_t>(factor * seq_len),
                                                                        static_cast<int64_t>(tile_size)});
        auto p_grouped = std::make_shared<ov::op::v1::Reshape>(p, p_reshape_pattern, false);
        p_grouped->set_friendly_name("p_grouped");

        // pv_grouped = matmul(p_grouped, v^T)
        // P_grouped: [batch, kv_num_heads, factor * seq_len, tile_size]
        // V_input:   [batch, kv_num_heads, head_dim, tile_size]
        // PV_grouped: [batch, kv_num_heads, factor * seq_len, head_dim]
        auto pv_grouped = std::make_shared<ov::op::v0::MatMul>(p_grouped, v_input, false, true);
        pv_grouped->set_friendly_name("pv_grouped");

        // Reshape PV back: [batch, kv_num_heads, factor * seq_len, head_dim] -> [batch, num_heads, seq_len, head_dim]
        auto pv_reshape_pattern =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                   ov::Shape{4},
                                                   std::vector<int64_t>{static_cast<int64_t>(batch),
                                                                        static_cast<int64_t>(num_heads),
                                                                        static_cast<int64_t>(seq_len),
                                                                        static_cast<int64_t>(head_dim)});
        pv = std::make_shared<ov::op::v1::Reshape>(pv_grouped, pv_reshape_pattern, false);
        pv->set_friendly_name("pv");
    } else {
        // Traditional broadcast computation: use broadcast V directly
        // P:        [batch, num_heads, seq_len, tile_size]
        // V_input:  [batch, num_heads, head_dim, tile_size] (already broadcast)
        // PV:       [batch, num_heads, seq_len, head_dim]
        pv = std::make_shared<ov::op::v0::MatMul>(p, v_input, false, true);
        pv->set_friendly_name("pv");
    }

    // acc = past_acc * alpha + pv
    auto acc_node = std::make_shared<ov::op::v1::Add>(past_acc_alpha, pv);
    acc_node->set_friendly_name("acc");
    results.acc = acc_node->output(0);

    return results;
}

// ============================================================================
// Helper function: Broadcast KV from kv_num_heads to num_heads
// ============================================================================
static std::pair<std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>> broadcast_kv_tiles(
    const std::shared_ptr<ov::Node>& k_tile_f32,
    const std::shared_ptr<ov::Node>& v_tile_f32,
    size_t batch,
    size_t num_heads,
    size_t kv_num_heads,
    size_t tile_size,
    size_t head_dim) {
    size_t head_expansion = num_heads / kv_num_heads;

    // Broadcast K: [batch, kv_num_heads, tile_size, head_dim] -> [batch, num_heads, tile_size, head_dim]
    auto unsqueeze_axes_k =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{2});
    auto k_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(k_tile_f32, unsqueeze_axes_k);

    auto repeats_k =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                               ov::Shape{5},
                                               std::vector<int64_t>{1, 1, static_cast<int64_t>(head_expansion), 1, 1});
    auto k_tiled = std::make_shared<ov::op::v0::Tile>(k_unsqueezed, repeats_k);

    auto k_reshape_pattern =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                               ov::Shape{4},
                                               std::vector<int64_t>{static_cast<int64_t>(batch),
                                                                    static_cast<int64_t>(num_heads),
                                                                    static_cast<int64_t>(tile_size),
                                                                    static_cast<int64_t>(head_dim)});
    auto k_tile_broadcast = std::make_shared<ov::op::v1::Reshape>(k_tiled, k_reshape_pattern, false);
    k_tile_broadcast->set_friendly_name("k_tile_broadcast");

    // Broadcast V: [batch, kv_num_heads, head_dim, tile_size] -> [batch, num_heads, head_dim, tile_size]
    auto unsqueeze_axes_v =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{2});
    auto v_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(v_tile_f32, unsqueeze_axes_v);

    auto repeats_v =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                               ov::Shape{5},
                                               std::vector<int64_t>{1, 1, static_cast<int64_t>(head_expansion), 1, 1});
    auto v_tiled = std::make_shared<ov::op::v0::Tile>(v_unsqueezed, repeats_v);

    auto v_reshape_pattern =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                               ov::Shape{4},
                                               std::vector<int64_t>{static_cast<int64_t>(batch),
                                                                    static_cast<int64_t>(num_heads),
                                                                    static_cast<int64_t>(head_dim),
                                                                    static_cast<int64_t>(tile_size)});
    auto v_tile_broadcast = std::make_shared<ov::op::v1::Reshape>(v_tiled, v_reshape_pattern, false);
    v_tile_broadcast->set_friendly_name("v_tile_broadcast");

    return {k_tile_broadcast, v_tile_broadcast};
}

#if ENABLE_HFA_LOOP_BASED_COMPUTATION
// ============================================================================
// Helper function: Reshape Q for grouped computation (loop-based approach)
// Avoids materializing broadcasted K/V tensors by reshaping Q to match KV heads
// ============================================================================
// Q: [batch, num_heads, seq_len, head_dim] -> [batch, kv_num_heads, factor * seq_len, head_dim]
// where factor = num_heads / kv_num_heads
static std::shared_ptr<ov::Node> reshape_q_for_groups(const std::shared_ptr<ov::Node>& q_f32,
                                                      size_t batch,
                                                      size_t num_heads,
                                                      size_t kv_num_heads,
                                                      size_t seq_len,
                                                      size_t head_dim) {
    size_t factor = num_heads / kv_num_heads;

    // Reshape Q: [batch, num_heads, seq_len, head_dim] -> [batch, kv_num_heads, factor * seq_len, head_dim]
    auto q_reshape_pattern =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                               ov::Shape{4},
                                               std::vector<int64_t>{static_cast<int64_t>(batch),
                                                                    static_cast<int64_t>(kv_num_heads),
                                                                    static_cast<int64_t>(factor * seq_len),
                                                                    static_cast<int64_t>(head_dim)});
    auto q_grouped = std::make_shared<ov::op::v1::Reshape>(q_f32, q_reshape_pattern, false);
    q_grouped->set_friendly_name("q_grouped");

    return q_grouped;
}
#endif  // ENABLE_HFA_LOOP_BASED_COMPUTATION

// ============================================================================
// Helper function: Clone the model's own attention tail into the final tile model
// ============================================================================
// Everything the original graph does between the P@V MatMul and its Result(s) - the
// head/sequence reshuffling that turns [batch, heads, seq, head_dim] into the block's
// output layout, plus any output Convert - is model-specific and must be reproduced
// rather than assumed.  Returns nullopt when the tail depends on anything the tile
// model does not have (e.g. another Parameter).
static std::optional<ov::ResultVector> clone_attention_tail(const std::shared_ptr<ov::Model>& model,
                                                            const std::shared_ptr<ov::Node>& matmul2,
                                                            const ov::Output<ov::Node>& new_source) {
    std::set<ov::Node*> tail;
    std::vector<std::shared_ptr<ov::Node>> queue{matmul2};
    while (!queue.empty()) {
        auto node = queue.back();
        queue.pop_back();
        for (const auto& out : node->outputs()) {
            for (const auto& consumer : out.get_target_inputs()) {
                auto next = consumer.get_node()->shared_from_this();
                if (tail.insert(next.get()).second) {
                    queue.push_back(next);
                }
            }
        }
    }

    std::map<ov::Node*, std::shared_ptr<ov::Node>> clones;
    ov::ResultVector results;
    for (const auto& node : model->get_ordered_ops()) {
        if (tail.count(node.get()) == 0) {
            continue;
        }
        ov::OutputVector new_inputs;
        new_inputs.reserve(node->get_input_size());
        for (const auto& input : node->inputs()) {
            const auto source = input.get_source_output();
            auto* source_node = source.get_node();
            if (source_node == matmul2.get()) {
                new_inputs.push_back(new_source);
            } else if (auto it = clones.find(source_node); it != clones.end()) {
                new_inputs.push_back(it->second->output(source.get_index()));
            } else if (ov::is_type<ov::op::v0::Constant>(source_node)) {
                new_inputs.push_back(source);
            } else {
                LOG_WARN("HFA: attention tail depends on " << source_node->get_friendly_name()
                                                           << ", which the tile model does not have");
                return std::nullopt;
            }
        }
        auto clone = node->clone_with_new_inputs(new_inputs);
        clone->set_friendly_name(node->get_friendly_name());
        clones[node.get()] = clone;
        if (auto result = ov::as_type_ptr<ov::op::v0::Result>(clone)) {
            results.push_back(result);
        }
    }

    if (results.empty()) {
        LOG_WARN("HFA: no Result reachable from the attention output MatMul");
        return std::nullopt;
    }
    return results;
}

// ============================================================================
// Helper function: Create final tile model outputs (division, transpose, reshape)
// ============================================================================
// ============================================================================
// Helper function: Produce the normalized attention output of the final tile
// ============================================================================
static std::shared_ptr<ov::Node> create_final_tile_output(const FlashAttentionResults& results,
                                                          bool fused_flash_attention) {
    std::shared_ptr<ov::Node> final_result;
    if (fused_flash_attention) {
        // If using FlashAttentionTile node, the output is already normalized, so skip division
        final_result = results.acc.get_node_shared_ptr();
    } else {
        // Division: result = acc / d
        final_result =
            std::make_shared<ov::op::v1::Divide>(results.acc.get_node_shared_ptr(), results.d.get_node_shared_ptr());
    }
    final_result->set_friendly_name("final_result");
    return final_result;
}

// ============================================================================
// Helper function: Create regular tile model outputs (intermediate states: acc, max, d)
// ============================================================================
static ov::ResultVector create_regular_tile_outputs(const FlashAttentionResults& results,
                                                    const ov::element::Type& input_dtype) {
    // Convert outputs back to input_dtype (f16)
    auto acc_output = std::make_shared<ov::op::v0::Convert>(results.acc, input_dtype);
    acc_output->set_friendly_name("acc_output");
    acc_output->output(0).get_tensor().set_names({"acc"});

    auto maxx_output = std::make_shared<ov::op::v0::Convert>(results.maxx, input_dtype);
    maxx_output->set_friendly_name("maxx_output");
    maxx_output->output(0).get_tensor().set_names({"maxx"});

    auto d_output = std::make_shared<ov::op::v0::Convert>(results.d, input_dtype);
    d_output->set_friendly_name("d_output");
    d_output->output(0).get_tensor().set_names({"d"});

    // Create results
    auto out_acc = std::make_shared<ov::op::v0::Result>(acc_output);
    out_acc->set_friendly_name("out_acc");

    auto out_maxx = std::make_shared<ov::op::v0::Result>(maxx_output);
    out_maxx->set_friendly_name("out_maxx");

    auto out_d = std::make_shared<ov::op::v0::Result>(d_output);
    out_d->set_friendly_name("out_d");

    return {out_acc, out_maxx, out_d};
}

// ============================================================================
// Helper function: Create regular tile model outputs for single flash attention node (intermediate states: acc, max, d)
// ============================================================================
static ov::ResultVector create_regular_tile_outputs_fused(const FlashAttentionResults& results,
                                                          const ov::element::Type& input_dtype) {
    auto acc_output = std::make_shared<ov::op::v0::Convert>(results.acc, input_dtype);
    acc_output->set_friendly_name("acc_output");
    acc_output->output(0).get_tensor().set_names({"acc"});

    auto axes = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});

    auto maxx_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(results.maxx, axes);
    maxx_unsqueezed->set_friendly_name("maxx_unsqueezed");
    auto maxx_output = std::make_shared<ov::op::v0::Convert>(maxx_unsqueezed, input_dtype);
    maxx_output->set_friendly_name("maxx_output");
    maxx_output->output(0).get_tensor().set_names({"maxx"});

    auto d_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(results.d, axes);
    d_unsqueezed->set_friendly_name("d_unsqueezed");
    auto d_output = std::make_shared<ov::op::v0::Convert>(d_unsqueezed, input_dtype);
    d_output->set_friendly_name("d_output");
    d_output->output(0).get_tensor().set_names({"d"});

    // Create results
    auto out_acc = std::make_shared<ov::op::v0::Result>(acc_output);
    out_acc->set_friendly_name("out_acc");

    auto out_maxx = std::make_shared<ov::op::v0::Result>(maxx_output);
    out_maxx->set_friendly_name("out_maxx");

    auto out_d = std::make_shared<ov::op::v0::Result>(d_output);
    out_d->set_friendly_name("out_d");

    return {out_acc, out_maxx, out_d};
}

// ============================================================================
// Helper function: Create individual tile model (regular or final)
// ============================================================================
// Parameters:
//   state_dtype    : element type for past_acc / past_max / past_d state tensors
//                    (shared between regular and final tile — typically f16).
//   kv_tile_dtype  : element type for k_tile / v_tile.
//                    Regular tiles: f16 (KV-block storage dtype).
//                    Final tile:    f32 (present-KV output dtype from upstream graph).
//   is_final_tile  : If true, the tile produces the block's real output by cloning the
//                    original model's attention tail (see clone_attention_tail).
//   src_model / src_matmul2 : the original SDPA model and its P@V MatMul, used only
//                    when is_final_tile=true.
static std::shared_ptr<ov::Model> create_hfa_tile_model(const ov::Shape& q_shape,
                                                        const ov::element::Type& state_dtype,
                                                        const ov::element::Type& kv_tile_dtype,
                                                        const ov::element::Type& q_dtype,
                                                        const ov::element::Type& mask_dtype,
                                                        int64_t tile_size,
                                                        size_t kv_num_heads,
                                                        size_t mask_num_heads,
                                                        size_t k_seq_dim,
                                                        float k_dequant_scale = 1.0f,
                                                        float v_dequant_scale = 1.0f,
                                                        bool is_final_tile = false,
                                                        bool fused_flash_attention = false,
                                                        bool enable_mask_skipping = false,
                                                        bool v_transposed = true,
                                                        const std::shared_ptr<ov::Model>& src_model = nullptr,
                                                        const std::shared_ptr<ov::Node>& src_matmul2 = nullptr) {
    LOG_DEBUG("Creating HFA " << (is_final_tile ? "FINAL " : "") << "tile model with tile_size=" << tile_size
                              << ", kv_num_heads=" << kv_num_heads << ", state_dtype=" << state_dtype
                              << ", kv_tile_dtype=" << kv_tile_dtype << ", mask_dtype=" << mask_dtype
                              << ", fused_flash_attention=" << fused_flash_attention);

    // Extract dimensions
    NPUW_ASSERT(q_shape.size() == 4);
    auto batch = q_shape[0];
    auto num_heads = q_shape[1];
    auto seq_len = q_shape[2];
    auto head_dim = q_shape[3];

    NPUW_ASSERT(num_heads % kv_num_heads == 0 && "Q heads must be divisible by KV heads");

    auto compute_dtype = ov::element::f32;
    LOG_DEBUG("Using compute_dtype=f32 for all operations");

    // Create input parameters
    auto inputs = create_hfa_tile_inputs(q_shape,
                                         state_dtype,
                                         kv_tile_dtype,
                                         q_dtype,
                                         mask_dtype,
                                         tile_size,
                                         kv_num_heads,
                                         mask_num_heads,
                                         k_seq_dim,
                                         v_transposed);

    // Convert all inputs to f32.
    // Only the final tile always uses a mask; regular tiles skip it when mask skipping is
    // enabled (see HostFlashAttention::from() for when that is safe).
    const bool use_mask = is_final_tile || !enable_mask_skipping;
    auto f32_nodes = convert_inputs_to_f32(inputs, mask_dtype, compute_dtype, use_mask);

    // Bring the K tile into the [batch, heads, tile, head_dim] layout the rest of the
    // tile graph expects.
    if (k_seq_dim == 1) {
        auto k_transpose_order =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 2, 1, 3});
        auto k_transposed = std::make_shared<ov::op::v1::Transpose>(f32_nodes.k_tile_f32, k_transpose_order);
        k_transposed->set_friendly_name("k_tile_f32_transposed");
        f32_nodes.k_tile_f32 = k_transposed;
    }

    // Replay the KV cache dequantization the original graph applies after the KV Concat.
    auto apply_dequant = [&](std::shared_ptr<ov::Node>& node, float dequant_scale, const char* name) {
        if (dequant_scale == 1.0f) {
            return;
        }
        auto scale_const =
            std::make_shared<ov::op::v0::Constant>(compute_dtype, ov::Shape{1, 1, 1, 1}, std::vector<float>{dequant_scale});
        auto scaled = std::make_shared<ov::op::v1::Multiply>(node, scale_const);
        scaled->set_friendly_name(name);
        node = scaled;
    };
    apply_dequant(f32_nodes.k_tile_f32, k_dequant_scale, "k_tile_dequant");
    apply_dequant(f32_nodes.v_tile_f32, v_dequant_scale, "v_tile_dequant");

    FlashAttentionResults results;

#if ENABLE_HFA_LOOP_BASED_COMPUTATION
    // ========================================================================
    // Loop-based computation: Reshape Q to avoid K/V broadcast materialization
    // ========================================================================
    LOG_DEBUG("Using loop-based grouped computation (ENABLED) - avoids K/V broadcast");

    // Reshape Q for grouped computation
    auto q_grouped = reshape_q_for_groups(f32_nodes.q_f32, batch, num_heads, kv_num_heads, seq_len, head_dim);

    // Execute flash attention with grouped computation (K and V remain 4D, no broadcast)
    results = execute_host_flash_attention(f32_nodes,
                                           q_grouped,             // Q: grouped format
                                           f32_nodes.k_tile_f32,  // K: original 4D
                                           f32_nodes.v_tile_f32,  // V: original 4D
                                           batch,
                                           num_heads,
                                           kv_num_heads,
                                           seq_len,
                                           tile_size,
                                           head_dim,
                                           true);  // use_grouped = true
#else
    // ========================================================================
    // Traditional broadcast-based computation: Materialize K/V broadcast
    // ========================================================================
    LOG_DEBUG("Using traditional broadcast computation (DISABLED loop-based) - materializes K/V broadcast");

    if (fused_flash_attention) {
        // Execute fused flash attention node MHA, GQA
        results = execute_fused_flash_attention(f32_nodes,
                                                f32_nodes.q_f32,
                                                f32_nodes.k_tile_f32,
                                                f32_nodes.v_tile_f32,
                                                is_final_tile,
                                                false,  // is_first_tile
                                                v_transposed);
    } else {
        // Broadcast K and V tiles from kv_num_heads to num_heads
        auto [k_broadcast, v_broadcast] = broadcast_kv_tiles(f32_nodes.k_tile_f32,
                                                             f32_nodes.v_tile_f32,
                                                             batch,
                                                             num_heads,
                                                             kv_num_heads,
                                                             tile_size,
                                                             head_dim);

        // Execute flash attention algorithm with broadcasted K/V
        results = execute_host_flash_attention(f32_nodes,
                                               f32_nodes.q_f32,  // Q: original 4D
                                               k_broadcast,      // K: broadcast to num_heads
                                               v_broadcast,      // V: broadcast to num_heads
                                               batch,
                                               num_heads,
                                               kv_num_heads,
                                               seq_len,
                                               tile_size,
                                               head_dim,
                                               false);  // use_grouped = false
    }
#endif  // ENABLE_HFA_LOOP_BASED_COMPUTATION

    // Create model outputs and name based on tile type
    ov::ResultVector model_results;
    std::string model_name;

    if (is_final_tile) {
        // === FINAL TILE: reproduce the block's own output layout ===
        NPUW_ASSERT(src_model && src_matmul2);
        auto tail = clone_attention_tail(src_model,
                                         src_matmul2,
                                         create_final_tile_output(results, fused_flash_attention)->output(0));
        if (!tail) {
            return nullptr;
        }
        model_results = std::move(*tail);
        model_name = "HFA_Final_Tile";
        LOG_DEBUG("HFA FINAL tile model created: state=" << state_dtype << ", kv_tile=" << kv_tile_dtype
                                                         << ", compute=" << compute_dtype);
    } else {
        // === REGULAR TILE: Output intermediate states (acc, max, d) ===
        // State outputs use state_dtype so they can be directly reused as the next
        // tile's state inputs without any type conversion.
        if (fused_flash_attention) {
            LOG_DEBUG("Using fused flash attention implementation - outputs acc, max, d from separate nodes");
            model_results = create_regular_tile_outputs_fused(results, state_dtype);

        } else {
            LOG_DEBUG("Using host flash attention implementation - outputs acc, max, d from the same node");
            model_results = create_regular_tile_outputs(results, state_dtype);
        }
        model_name = "HFA_Tile";
        LOG_DEBUG("HFA tile model created: state=" << state_dtype << ", kv_tile=" << kv_tile_dtype
                                                   << ", compute=" << compute_dtype << ", state_out=" << state_dtype);
    }

    // Create model parameters
    ov::ParameterVector model_params =
        {inputs.past_acc, inputs.past_max, inputs.past_d, inputs.k_tile, inputs.v_tile, inputs.q};
    if (use_mask) {
        model_params.push_back(inputs.mask_tile);
    }

    // Create and return model
    return std::make_shared<ov::Model>(model_results, model_params, model_name);
}

// ============================================================================
// Helper function: Extract actual Parameter by skipping Convert nodes
// ============================================================================
static std::shared_ptr<ov::Node> skip_convert_nodes(const std::shared_ptr<ov::Node>& node) {    auto current = node;
    while (current && ov::is_type<ov::op::v0::Convert>(current.get())) {
        if (current->get_input_size() > 0) {
            current = current->get_input_node_shared_ptr(0);
        } else {
            break;
        }
    }
    return current;
}

// ============================================================================
// Helper function: Collect the scalar dequantization applied to a KV branch
// ============================================================================
// HFA rebuilds attention from the KV Concat outputs, so anything the original graph
// applies between the Concat and the attention MatMul must be reproduced inside the
// tile model.  Quantized KV caches put a scalar dequantization Multiply there.
// Returns the accumulated scale, or nullopt if the branch contains an operation the
// tile graph cannot reproduce (in which case HFA must not be used for this subgraph).
static std::optional<float> collect_kv_dequant_scale(const std::shared_ptr<ov::Node>& concat,
                                                     const std::shared_ptr<ov::Node>& matmul,
                                                     const char* branch_name) {
    float scale = 1.0f;
    auto node = concat;
    while (node && node != matmul) {
        const auto& consumers = node->output(0).get_target_inputs();
        if (consumers.size() != 1) {
            LOG_WARN("HFA: " << branch_name << " branch fans out at " << node->get_friendly_name());
            return std::nullopt;
        }
        auto next = consumers.begin()->get_node()->shared_from_this();
        if (next == matmul) {
            return scale;
        }
        if (auto mul = ov::as_type_ptr<ov::op::v1::Multiply>(next)) {
            const size_t data_port = (mul->get_input_node_shared_ptr(0) == node) ? 0u : 1u;
            auto factor = ov::as_type_ptr<ov::op::v0::Constant>(
                skip_convert_nodes(mul->get_input_node_shared_ptr(1u - data_port)));
            if (!factor || ov::shape_size(factor->get_shape()) != 1) {
                LOG_WARN("HFA: " << branch_name << " branch has a non-scalar Multiply at "
                                 << mul->get_friendly_name());
                return std::nullopt;
            }
            scale *= factor->cast_vector<float>().front();
        } else if (!ov::is_type<ov::op::v0::Convert>(next) && !ov::is_type<ov::op::v1::Transpose>(next) &&
                   !ov::is_type<ov::op::v1::Reshape>(next) && !ov::is_type<ov::op::v3::Broadcast>(next) &&
                   !ov::is_type<ov::op::v0::Unsqueeze>(next)) {
            LOG_WARN("HFA: unsupported op " << next->get_type_name() << " '" << next->get_friendly_name()
                                            << "' on the " << branch_name << " branch");
            return std::nullopt;
        }
        node = next;
    }
    return scale;
}

// ============================================================================
// Helper function: Build SDPA parameter index mapping
// ============================================================================
static void build_sdpa_param_mapping(HostFlashAttention& hfa,
                                     const std::shared_ptr<ov::Model>& model,
                                     const ov::npuw::util::SDPAPatternNodes& pattern_nodes,
                                     ov::npuw::util::KVOrder kv_order) {
    LOG_INFO("Building SDPA input parameter index mapping...");

    // Helper lambda to safely extract parameter from node (skipping Convert ops)
    auto extract_param = [&](const std::shared_ptr<ov::Node>& node) -> std::shared_ptr<ov::op::v0::Parameter> {
        return ov::as_type_ptr<ov::op::v0::Parameter>(skip_convert_nodes(node));
    };

    // Extract Q (query) parameter - input 0 of MatMul1
    if (auto q_param = extract_param(pattern_nodes.matmul1_node->get_input_node_shared_ptr(0))) {
        hfa._query_param_idx = model->get_parameter_index(q_param);
    }

    // Extract past/present KV parameters from a Concat node.  The present input is the
    // one that does *not* trace back to a past_key_values Parameter; every other input is
    // a past block.  Its position follows kv_order.  Key and value are handled alike.
    auto extract_kv_params = [&](const std::shared_ptr<ov::Node>& concat_node,
                                 std::vector<std::size_t>& block_indices,
                                 std::size_t& present_idx_out,
                                 const char* kv_name) {
        if (!concat_node)
            return;
        const size_t n = concat_node->get_input_size();
        const size_t present_pos = (kv_order == ov::npuw::util::KVOrder::PresentFirst) ? 0u : n - 1u;
        block_indices.clear();
        block_indices.reserve(n - 1);
        for (size_t i = 0; i < n; ++i) {
            auto param = extract_param(concat_node->get_input_node_shared_ptr(i));
            if (!param) {
                LOG_WARN("Could not extract parameter from " << kv_name << " Concat input[" << i << "]");
                continue;
            }
            const std::size_t idx = model->get_parameter_index(param);
            if (i == present_pos) {
                present_idx_out = idx;
                LOG_DEBUG("  Found " << kv_name << "_present at parameter index " << idx);
            } else {
                block_indices.push_back(idx);
                LOG_DEBUG("  Found " << kv_name << " block[" << block_indices.size() - 1 << "] at parameter index "
                                     << idx);
            }
        }
    };

    extract_kv_params(pattern_nodes.past_key_concat_node,
                      hfa._past_key_block_indices,
                      hfa._present_key_param_idx,
                      "past_key");
    extract_kv_params(pattern_nodes.past_value_concat_node,
                      hfa._past_value_block_indices,
                      hfa._present_value_param_idx,
                      "past_value");

    // Extract mask parameter - input 1 of add_node
    if (auto add_param = extract_param(pattern_nodes.add_node->get_input_node_shared_ptr(1))) {
        hfa._attention_mask_param_idx = model->get_parameter_index(add_param);
    }

    LOG_INFO("Built SDPA input mapping: query="
             << hfa._query_param_idx << ", present_key=" << hfa._present_key_param_idx
             << ", present_value=" << hfa._present_value_param_idx << ", mask=" << hfa._attention_mask_param_idx);
    LOG_INFO("  Past key blocks: " << hfa._past_key_block_indices.size());
    LOG_INFO("  Past value blocks: " << hfa._past_value_block_indices.size());

    // Print KV cache blocks
    LOG_DEBUG("Past key blocks (" << hfa._past_key_block_indices.size() << "):");
    for (size_t i = 0; i < hfa._past_key_block_indices.size(); ++i) {
        LOG_DEBUG("  block[" << i << "] -> parameter[" << hfa._past_key_block_indices[i] << "]");
    }

    LOG_DEBUG("Past value blocks (" << hfa._past_value_block_indices.size() << "):");
    for (size_t i = 0; i < hfa._past_value_block_indices.size(); ++i) {
        LOG_DEBUG("  block[" << i << "] -> parameter[" << hfa._past_value_block_indices[i] << "]");
    }

    LOG_DEBUG("=============================================");
}

// ============================================================================
// Helper function: Build tile model parameter index mapping
// ============================================================================
static void build_tile_param_mapping(HostFlashAttention& hfa, const std::shared_ptr<ov::Model>& tile_model) {
    LOG_INFO("Building HFA Tile Model input index mapping...");

    // Parse tile model inputs by their tensor names
    // Expected input order: [past_acc, past_max, past_d, k_tile, v_tile, q, mask_tile]
    const auto& tile_inputs = tile_model->inputs();
    for (std::size_t i = 0; i < tile_inputs.size(); ++i) {
        const auto& tensor_names = tile_inputs[i].get_names();
        if (tensor_names.empty()) {
            LOG_WARN("Tile model input[" << i << "] has no tensor name");
            continue;
        }

        const std::string& name = *tensor_names.begin();

        // Map tensor name to enum ID
        if (name == hfa_tile_input_id_to_string(HFATileInputId::PAST_ACC)) {
            hfa._tile_param_index_map[HFATileInputId::PAST_ACC] = i;
        } else if (name == hfa_tile_input_id_to_string(HFATileInputId::PAST_MAX)) {
            hfa._tile_param_index_map[HFATileInputId::PAST_MAX] = i;
        } else if (name == hfa_tile_input_id_to_string(HFATileInputId::PAST_D)) {
            hfa._tile_param_index_map[HFATileInputId::PAST_D] = i;
        } else if (name == hfa_tile_input_id_to_string(HFATileInputId::K_TILE)) {
            hfa._tile_param_index_map[HFATileInputId::K_TILE] = i;
        } else if (name == hfa_tile_input_id_to_string(HFATileInputId::V_TILE)) {
            hfa._tile_param_index_map[HFATileInputId::V_TILE] = i;
        } else if (name == hfa_tile_input_id_to_string(HFATileInputId::Q)) {
            hfa._tile_param_index_map[HFATileInputId::Q] = i;
        } else if (name == hfa_tile_input_id_to_string(HFATileInputId::MASK_TILE)) {
            hfa._tile_param_index_map[HFATileInputId::MASK_TILE] = i;
        } else {
            LOG_WARN("Unknown tile model input name: " << name);
        }
    }

    // Print the tile input mapping
    LOG_DEBUG("");
    LOG_DEBUG("========== HFA Tile Model Input Mapping ==========");
    LOG_DEBUG("Total entries: " << hfa._tile_param_index_map.size());

    for (const auto& [input_id, input_idx] : hfa._tile_param_index_map) {
        LOG_DEBUG("  " << hfa_tile_input_id_to_string(input_id) << " -> input[" << input_idx << "]");
    }
    LOG_DEBUG("==================================================");
}

// ============================================================================
// Helper function: Build tile model output index mapping
// ============================================================================
static void build_tile_output_mapping(HostFlashAttention& hfa, const std::shared_ptr<ov::Model>& tile_model) {
    LOG_INFO("Building HFA Tile Model output index mapping...");

    // Parse tile model outputs by their tensor names
    // Expected output order: [acc, maxx, d]
    const auto& tile_outputs = tile_model->outputs();
    for (std::size_t i = 0; i < tile_outputs.size(); ++i) {
        const auto& tensor_names = tile_outputs[i].get_names();
        if (tensor_names.empty()) {
            LOG_WARN("Tile model output[" << i << "] has no tensor name");
            continue;
        }

        const std::string& name = *tensor_names.begin();

        // Map tensor name to enum ID
        if (name == "acc") {
            hfa._tile_output_index_map[HFATileOutputId::ACC] = i;
        } else if (name == "maxx") {
            hfa._tile_output_index_map[HFATileOutputId::MAXX] = i;
        } else if (name == "d") {
            hfa._tile_output_index_map[HFATileOutputId::D] = i;
        } else {
            LOG_WARN("Unknown tile model output name: " << name);
        }
    }

    // Print the tile output mapping
    LOG_DEBUG("");
    LOG_DEBUG("========== HFA Tile Model Output Mapping ==========");
    LOG_DEBUG("Total entries: " << hfa._tile_output_index_map.size());

    for (const auto& [output_id, output_idx] : hfa._tile_output_index_map) {
        LOG_DEBUG("  " << hfa_tile_output_id_to_string(output_id) << " -> output[" << output_idx << "]");
    }
    LOG_DEBUG("==================================================");
}

// ============================================================================
// Helper function: Extract sequence dimension from Concat node
// ============================================================================
static std::optional<std::size_t> extract_sequence_dim_from_concat(const std::shared_ptr<ov::Node>& concat_node,
                                                                   const std::string& tensor_name) {
    if (!concat_node) {
        LOG_WARN("Failed to extract " << tensor_name << " concat node");
        return std::nullopt;
    }

    auto concat_op = std::dynamic_pointer_cast<ov::op::v0::Concat>(concat_node);
    if (!concat_op) {
        LOG_WARN("Failed to cast " << tensor_name << "_concat to Concat op");
        return std::nullopt;
    }

    const auto& concat_out_shape = concat_op->get_output_partial_shape(0);
    return ov::util::try_normalize_axis(concat_op->get_axis(), concat_out_shape.rank(), *concat_op);
}

std::optional<HostFlashAttention> HostFlashAttention::from(const std::shared_ptr<ov::Model>& model,
                                                           bool fused_flash_attention,
                                                           bool enable_mask_skipping) {
    LOG_INFO("Attempting to create HostFlashAttention"
             << (fused_flash_attention ? " with fused flash attention node" : ""));
    LOG_BLOCK();

    // ========================================================================
    // Step 1: Validate SDPA pattern and extract key nodes
    // ========================================================================
    auto pattern_nodes = ov::npuw::util::find_sdpa_pattern_nodes(model);
    if (!pattern_nodes.is_valid()) {
        LOG_WARN("Failed to re-find SDPA pattern nodes");
        return std::nullopt;
    }

    auto q_input = pattern_nodes.matmul1_node->get_input_node_shared_ptr(0);
    auto k_concat = pattern_nodes.past_key_concat_node;

    // Skip Convert nodes to get to the actual Parameter/input
    q_input = skip_convert_nodes(q_input);

    if (!q_input || !k_concat) {
        LOG_WARN("Failed to extract Q input or K concat from pattern");
        return std::nullopt;
    }

    // ========================================================================
    // Step 2: Extract shape and data type information
    // ========================================================================
    auto q_shape = q_input->get_output_partial_shape(0);
    if (q_shape.is_dynamic()) {
        LOG_WARN("Dynamic shapes not yet supported for HFA");
        return std::nullopt;
    }

    auto q_shape_static = q_shape.to_shape();

    // The mask and the KV tiles must be walked in the same order, so K and V have to
    // agree on where the present KV sits.
    using ov::npuw::util::KVOrder;
    const auto detected_order = ov::npuw::util::kv_concat_order(pattern_nodes.past_key_concat_node,
                                                                pattern_nodes.past_value_concat_node);
    if (!detected_order) {
        LOG_WARN("Could not determine a consistent K/V concat order");
        return std::nullopt;
    }
    const KVOrder kv_order = *detected_order;

    const bool present_first = (kv_order == KVOrder::PresentFirst);
    const std::size_t present_k_concat_input = present_first ? 0u : k_concat->get_input_size() - 1u;
    const std::size_t first_past_k_concat_input = present_first ? 1u : 0u;

    // KV cache and Q may have different element types (e.g. f16 KV vs f32 Q).
    // block_kv_dtype: skip any Convert(f16→f32) that sits between the block Parameter
    // and the Concat; the Concat output may be upcast to f32 (Gemma-4) but the block
    // manager allocates tensors at the underlying storage dtype (f16).
    auto first_kv_node = skip_convert_nodes(k_concat->get_input_node_shared_ptr(first_past_k_concat_input));
    const ov::element::Type block_kv_dtype = first_kv_node->get_output_element_type(0);

    // present_kv_dtype: dtype of the freshly-computed present-KV tensors that the
    // upstream NPU subgraph passes at runtime (typically f32).
    auto present_kv_node = skip_convert_nodes(k_concat->get_input_node_shared_ptr(present_k_concat_input));
    const ov::element::Type present_kv_dtype = present_kv_node->get_output_element_type(0);

    const ov::element::Type q_dtype = q_input->get_output_element_type(0);
    LOG_DEBUG("HFA dtypes: block_kv=" << block_kv_dtype << ", present_kv=" << present_kv_dtype << ", q=" << q_dtype);

    // Validate Q shape and extract query_size (seq_len dimension)
    if (q_shape_static.size() != 4) {
        LOG_WARN("Q shape must be 4D, got " << q_shape_static.size() << "D shape");
        return std::nullopt;
    }
    std::size_t query_size = q_shape_static[2];  // seq_len at index 2
    LOG_DEBUG("Extracted query_size (seq_len) from Q shape: " << query_size);

    auto mask_param = ov::npuw::util::find_mask_parameter(pattern_nodes.add_node);
    if (!mask_param) {
        LOG_WARN("Could not find mask parameter in model");
        return std::nullopt;
    }
    auto mask_dtype = mask_param->get_output_element_type(0);
    const auto mask_shape = mask_param->get_shape();
    if (mask_shape.size() != 4) {
        LOG_WARN("Attention mask must be 4D, got " << mask_shape.size() << "D shape");
        return std::nullopt;
    }
    const std::size_t mask_num_heads = mask_shape[1];

    auto output_dtype = ov::element::f16;  // Default fallback
    if (model->outputs().size() > 0) {
        output_dtype = model->output(0).get_element_type();
        LOG_DEBUG("Original SDPA output data type: " << output_dtype);
    } else {
        LOG_WARN("No outputs found in model, using default output dtype: " << output_dtype);
    }

    // ========================================================================
    // Step 3: Extract K/V sequence dimensions from Concat nodes
    // ========================================================================
    auto k_seq_dim_opt = extract_sequence_dim_from_concat(pattern_nodes.past_key_concat_node, "K");
    if (!k_seq_dim_opt) {
        return std::nullopt;
    }
    std::size_t k_seq_dim = k_seq_dim_opt.value();

    auto v_seq_dim_opt = extract_sequence_dim_from_concat(pattern_nodes.past_value_concat_node, "V");
    if (!v_seq_dim_opt) {
        return std::nullopt;
    }
    std::size_t v_seq_dim = v_seq_dim_opt.value();

    // ========================================================================
    // Step 4: Extract KV heads configuration and context size
    // ========================================================================
    size_t kv_num_heads = 0;
    size_t context_size = 0;
    if (!k_concat->get_output_partial_shape(0).is_static()) {
        return std::nullopt;
    }

    auto k_full_shape = k_concat->get_output_partial_shape(0).to_shape();
    // K shape after concat: [batch, kv_num_heads, kv_cache_size, head_dim], or
    // [batch, kv_cache_size, kv_num_heads, head_dim] when the sequence sits at dim 1.
    if (k_full_shape.size() != 4) {
        return std::nullopt;
    }
    if (k_seq_dim != 1 && k_seq_dim != 2) {
        LOG_WARN("Unsupported K sequence dimension for HFA: " << k_seq_dim);
        return std::nullopt;
    }
    if (v_seq_dim != 2 && v_seq_dim != 3) {
        LOG_WARN("Unsupported V sequence dimension for HFA: " << v_seq_dim);
        return std::nullopt;
    }

    kv_num_heads = k_full_shape[k_seq_dim == 1 ? 2 : 1];  // the non-sequence of dims 1/2
    context_size = k_full_shape[k_seq_dim];               // Extract context size from sequence dimension

    if (kv_num_heads == 0) {
        LOG_WARN("Failed to determine KV num_heads");
        return std::nullopt;
    }

    if (context_size == 0) {
        LOG_WARN("Failed to determine context_size");
        return std::nullopt;
    }

    // ========================================================================
    // Step 5: Create tile models
    // ========================================================================
    // The tile size is the extent of the present KV, not of Q: the final tile must consume
    // the whole present KV in a single inference, while Q is never tiled.  The two coincide
    // for the usual LLM layout, but not when Q is GQA-grouped ahead of the SDPA (then
    // Q seq_len = group * present_kv_len).
    const auto& present_k_shape = k_concat->get_input_partial_shape(present_k_concat_input);
    if (present_k_shape.is_dynamic() || present_k_shape.rank().get_length() != 4) {
        LOG_WARN("Present K input of the KV Concat must be static and 4D");
        return std::nullopt;
    }
    const std::size_t tile_size = present_k_shape.to_shape()[k_seq_dim];
    if (tile_size == 0 || context_size % tile_size != 0) {
        LOG_WARN("HFA context size " << context_size << " is not a multiple of the tile size " << tile_size);
        return std::nullopt;
    }

    // The tile graph is rebuilt from the KV Concat outputs, so any post-Concat KV
    // processing (notably the scalar dequantization of a quantized cache) has to be
    // replayed inside it - otherwise it would be silently dropped.
    const auto k_dequant_scale_opt =
        collect_kv_dequant_scale(pattern_nodes.past_key_concat_node, pattern_nodes.matmul1_node, "K");
    const auto v_dequant_scale_opt =
        collect_kv_dequant_scale(pattern_nodes.past_value_concat_node, pattern_nodes.matmul2_node, "V");
    if (!k_dequant_scale_opt || !v_dequant_scale_opt) {
        return std::nullopt;
    }
    const float k_dequant_scale = *k_dequant_scale_opt;
    const float v_dequant_scale = *v_dequant_scale_opt;

    // V tensors are pre-transposed (stored as [B,H,head_dim,seq]) only when OptimizeValueTensors
    // succeeded, which is reflected by the V-concat axis being 3 instead of the default 2.
    const bool v_transposed = (v_seq_dim == 3);
    // Regular tile: state and KV-tile both use block_kv_dtype (f16).
    //   past_acc/max/d: f16   k_tile/v_tile: f16  (from KV blocks)
    // Final tile: state still uses block_kv_dtype (f16) for zero-copy with regular
    //   tile outputs; KV-tile uses present_kv_dtype (f32) matching the upstream graph.
    //   past_acc/max/d: f16   k_tile/v_tile: f32  (present-KV from upstream)
    // The flash-attention state (acc / max / d) is an accumulator, so it needs a real
    // type.  Reuse the KV storage type when it is one - that keeps the regular tile's
    // outputs zero-copy compatible with the next tile's inputs - and fall back to f16
    // for a quantized KV cache.
    const ov::element::Type state_dtype = block_kv_dtype.is_real() ? block_kv_dtype : ov::element::f16;

    LOG_INFO("Creating HFA tile models: tile_size=" << tile_size << ", query_size=" << query_size
                                                    << ", present_first=" << present_first
                                                    << ", v_transposed=" << v_transposed
                                                    << ", k_seq_dim=" << k_seq_dim << ", kv_num_heads=" << kv_num_heads
                                                    << ", k_scale=" << k_dequant_scale
                                                    << ", v_scale=" << v_dequant_scale
                                                    << ", block_kv=" << block_kv_dtype
                                                    << ", present_kv=" << present_kv_dtype << ", q=" << q_dtype
                                                    << ", state=" << state_dtype);

    // Per-SDPA mask-skipping decision
    // DetectAttentionMask (run earlier on the original SDPA node) may have annotated
    // this subgraph's Add(QK, mask) node with its mask kind, carried here via
    // copy_runtime_info() during SDPA decomposition. Per NPUW_SDPA_MASK_RT_KEY's
    // encoding, the mask-skipping decision is:
    //
    //   no annotation (Unknown) : DISABLED -- unknown mask shape, skipping it could
    //                             silently change results.
    //   value <  0 (Causal)     : ENABLED unconditionally for non-final tiles (a
    //                             causal mask never excludes anything a non-final
    //                             regular tile would otherwise include).
    //   value >= 0 (SlidingWindow, value = window_size)
    //                           : ENABLED only if window_size >= context_size (then it
    //                             behaves exactly like Causal); otherwise the window
    //                             would truncate positions a regular tile must still
    //                             respect, so it stays DISABLED.
    //
    // This replaces the enable_mask_skipping flag passed in from the caller: that flag
    // is now just a master kill switch (NPUW_ATTN_HFA_MASK_SKIPPING=NO disables the
    // optimization outright, regardless of mask kind).
    bool local_enable_mask_skipping = false;
    if (enable_mask_skipping && pattern_nodes.add_node) {
        const auto& rt_info = pattern_nodes.add_node->get_rt_info();
        const auto it = rt_info.find(ov::npuw::NPUW_SDPA_MASK_RT_KEY);
        if (it != rt_info.end()) {
            const auto encoded = it->second.as<int64_t>();
            if (encoded < 0) {
                local_enable_mask_skipping = true;
                LOG_DEBUG("Per-SDPA mask annotation: Causal → mask skipping ENABLED for this ATTN subgraph");
            } else if (encoded >= static_cast<int64_t>(context_size)) {
                local_enable_mask_skipping = true;
                LOG_DEBUG("Per-SDPA mask annotation: SlidingWindow(window_size="
                          << encoded << ") covers the full context (" << context_size
                          << ") → mask skipping ENABLED for this ATTN subgraph");
            } else {
                LOG_DEBUG("Per-SDPA mask annotation: SlidingWindow(window_size="
                          << encoded << ") is narrower than the context (" << context_size
                          << ") → mask skipping DISABLED for this ATTN subgraph");
            }
        } else {
            LOG_DEBUG("No per-SDPA mask annotation (Unknown) → mask skipping DISABLED for this ATTN subgraph");
        }
    }
    auto tile_model = create_hfa_tile_model(q_shape_static,
                                            state_dtype,
                                            block_kv_dtype,  // kv_tile_dtype (past blocks)
                                            q_dtype,
                                            mask_dtype,
                                            tile_size,
                                            kv_num_heads,
                                            mask_num_heads,
                                            k_seq_dim,
                                            k_dequant_scale,
                                            v_dequant_scale,
                                            false,
                                            fused_flash_attention,
                                            local_enable_mask_skipping,
                                            v_transposed);
    if (!tile_model) {
        LOG_WARN("Failed to create HFA tile model");
        return std::nullopt;
    }

    auto final_tile_model = create_hfa_tile_model(q_shape_static,
                                                  state_dtype,       // consistent with the regular tile
                                                  present_kv_dtype,  // kv_tile_dtype (present-KV, f32)
                                                  q_dtype,
                                                  mask_dtype,
                                                  tile_size,
                                                  kv_num_heads,
                                                  mask_num_heads,
                                                  k_seq_dim,
                                                  k_dequant_scale,
                                                  v_dequant_scale,
                                                  true,
                                                  fused_flash_attention,
                                                  local_enable_mask_skipping,
                                                  v_transposed,
                                                  model,
                                                  pattern_nodes.matmul2_node);
    if (!final_tile_model) {
        LOG_WARN("Failed to create HFA final tile model");
        return std::nullopt;
    }

    // ========================================================================
    // Step 6: Create HostFlashAttention structure and set configuration
    // ========================================================================
    HostFlashAttention hfa;
    hfa._tile_model = tile_model;
    hfa._final_tile_model = final_tile_model;
    hfa._query_size = query_size;
    hfa._context_size = context_size;
    hfa._tile_size = static_cast<int64_t>(tile_size);
    hfa._k_seq_dim = k_seq_dim;
    hfa._v_seq_dim = v_seq_dim;
    hfa._kv_order = kv_order;

    // ========================================================================
    // Step 7: Build SDPA parameter index mapping
    // ========================================================================
    build_sdpa_param_mapping(hfa, model, pattern_nodes, kv_order);

    // ========================================================================
    // Step 8: Build tile model parameter index mapping
    // The first 6 input indices are identical in both models regular and final
    // final_tile_model has mask_tile (index 6)
    // ========================================================================
    build_tile_param_mapping(hfa, final_tile_model);

    // ========================================================================
    // Step 9: Build tile model output index mapping
    // ========================================================================
    build_tile_output_mapping(hfa, tile_model);

    LOG_INFO("Successfully created HostFlashAttention with query_size="
             << query_size << ", context_size=" << context_size << ", tile_size=" << tile_size
             << ", present_first=" << present_first);

    return hfa;
}

}  // namespace function

namespace compiled {

// Constructor implementation - extracts metadata
HostFlashAttention::HostFlashAttention(const function::HostFlashAttention& func_hfa) {
    LOG_INFO("Constructing compiled::HostFlashAttention");
    LOG_BLOCK();

    // Extract tile configuration from function HFA
    _tile_size = func_hfa._tile_size;

    // Store the tile models for later compilation
    _tile_model_to_compile = func_hfa._tile_model;
    _final_tile_model_to_compile = func_hfa._final_tile_model;

    // Copy query size, context size, and K/V sequence dimensions from function HFA
    _sdpa_attention_info._query_size = func_hfa._query_size;
    _sdpa_attention_info._context_size = func_hfa._context_size;
    _sdpa_attention_info._k_seq_dim = func_hfa._k_seq_dim;
    _sdpa_attention_info._v_seq_dim = func_hfa._v_seq_dim;
    _sdpa_attention_info._kv_order = func_hfa._kv_order;

    // Pre-cache all indices from function HFA maps
    LOG_INFO("Pre-caching SDPA and tile indices...");

    // Pre-cache SDPA parameter indices (direct field access — no map lookup)
    _sdpa_attention_info._sdpa_indices.query = func_hfa._query_param_idx;

    // Copy all KV cache block indices
    _sdpa_attention_info._sdpa_indices.past_key_blocks = func_hfa._past_key_block_indices;
    _sdpa_attention_info._sdpa_indices.past_value_blocks = func_hfa._past_value_block_indices;

    _sdpa_attention_info._sdpa_indices.present_key = func_hfa._present_key_param_idx;
    _sdpa_attention_info._sdpa_indices.present_value = func_hfa._present_value_param_idx;
    _sdpa_attention_info._sdpa_indices.attention_mask = func_hfa._attention_mask_param_idx;

    // Pre-cache tile input indices
    auto get_tile_input_idx = [&](HFATileInputId input_id) -> std::size_t {
        auto it = func_hfa._tile_param_index_map.find(input_id);
        if (it == func_hfa._tile_param_index_map.end()) {
            OPENVINO_THROW("HFA: Tile input mapping not found for input ID: ", static_cast<uint8_t>(input_id));
        }
        return it->second;
    };

    auto get_tile_output_idx = [&](HFATileOutputId output_id) -> std::size_t {
        auto it = func_hfa._tile_output_index_map.find(output_id);
        if (it == func_hfa._tile_output_index_map.end()) {
            OPENVINO_THROW("HFA: Tile output mapping not found for output ID: ", static_cast<uint8_t>(output_id));
        }
        return it->second;
    };

    // Cache all tile input indices
    _sdpa_attention_info._tile_input_indices.q = get_tile_input_idx(HFATileInputId::Q);
    _sdpa_attention_info._tile_input_indices.k = get_tile_input_idx(HFATileInputId::K_TILE);
    _sdpa_attention_info._tile_input_indices.v = get_tile_input_idx(HFATileInputId::V_TILE);
    _sdpa_attention_info._tile_input_indices.mask = get_tile_input_idx(HFATileInputId::MASK_TILE);
    _sdpa_attention_info._tile_input_indices.acc = get_tile_input_idx(HFATileInputId::PAST_ACC);
    _sdpa_attention_info._tile_input_indices.max = get_tile_input_idx(HFATileInputId::PAST_MAX);
    _sdpa_attention_info._tile_input_indices.d = get_tile_input_idx(HFATileInputId::PAST_D);

    // Cache all tile output indices
    _sdpa_attention_info._tile_output_indices.acc = get_tile_output_idx(HFATileOutputId::ACC);
    _sdpa_attention_info._tile_output_indices.max = get_tile_output_idx(HFATileOutputId::MAXX);
    _sdpa_attention_info._tile_output_indices.d = get_tile_output_idx(HFATileOutputId::D);

    LOG_INFO("Pre-cached SDPA indices: [query="
             << _sdpa_attention_info._sdpa_indices.query
             << ", present_key=" << _sdpa_attention_info._sdpa_indices.present_key
             << ", present_value=" << _sdpa_attention_info._sdpa_indices.present_value
             << ", attention_mask=" << _sdpa_attention_info._sdpa_indices.attention_mask << "]");
    LOG_INFO("  Past key blocks: " << _sdpa_attention_info._sdpa_indices.past_key_blocks.size());
    LOG_INFO("  Past value blocks: " << _sdpa_attention_info._sdpa_indices.past_value_blocks.size());
    LOG_INFO("Attention configuration: query_size="
             << _sdpa_attention_info._query_size << ", context_size=" << _sdpa_attention_info._context_size
             << ", k_seq_dim=" << _sdpa_attention_info._k_seq_dim << ", v_seq_dim=" << _sdpa_attention_info._v_seq_dim);
    LOG_INFO("Pre-cached tile indices: inputs[q=" << _sdpa_attention_info._tile_input_indices.q
                                                  << ", k=" << _sdpa_attention_info._tile_input_indices.k
                                                  << ", v=" << _sdpa_attention_info._tile_input_indices.v
                                                  << ", mask=" << _sdpa_attention_info._tile_input_indices.mask
                                                  << ", acc=" << _sdpa_attention_info._tile_input_indices.acc
                                                  << ", max=" << _sdpa_attention_info._tile_input_indices.max
                                                  << ", d=" << _sdpa_attention_info._tile_input_indices.d
                                                  << "], outputs[acc=" << _sdpa_attention_info._tile_output_indices.acc
                                                  << ", max=" << _sdpa_attention_info._tile_output_indices.max
                                                  << ", d=" << _sdpa_attention_info._tile_output_indices.d << "]");

    // Note: _compiled_tile_model and _compiled_final_tile_model will be set later by
    // compile_host_flash_attention_model()
}
}  // namespace compiled

namespace runtime {
namespace host_flash_attention {

// PositionIDs constructor
PositionIDs::PositionIDs(std::size_t param_idx,
                         std::size_t tile_size,
                         ov::npuw::util::KVOrder kv_order,
                         const ov::ISyncInferRequest& rq)
    : _position_ids_idx(param_idx),
      _tile_size(tile_size),
      _kv_order(kv_order),
      _rq(rq) {
    // FIXME: speculative decode is indistinguishable at this point!
    _case = _tile_size == 1 ? Case::GENERATE : Case::PREFILL;
}

Selector::Ptr PositionIDs::find(std::size_t tile_size,
                                ov::npuw::util::KVOrder kv_order,
                                const ov::ISyncInferRequest& rq) {
    auto is_position_ids = [](const ov::Output<const ov::Node>& p) {
        const auto& shape = p.get_shape();
        // FIXME: 2D/3D position IDs are not supported here YET
        return p.get_node()->get_friendly_name() == "position_ids" &&
               (shape.size() == 1 || (shape.size() == 2 && shape[0] == 1));
    };

    const auto& inputs = rq.get_inputs();
    auto pos_ids_iter = std::find_if(inputs.begin(), inputs.end(), is_position_ids);
    if (pos_ids_iter != inputs.end()) {
        const auto param_idx = std::distance(inputs.begin(), pos_ids_iter);
        return Selector::Ptr{new PositionIDs(param_idx, tile_size, kv_order, rq)};
    }
    return Selector::Ptr{};
}

void PositionIDs::prepare(int64_t past_len) {
    const auto& iport = _rq.get().get_compiled_model()->inputs()[_position_ids_idx];
    const auto in_tensor = _rq.get().get_tensor(iport);
    const auto in_dims = in_tensor->get_shape();
    const int64_t step = static_cast<int64_t>(_tile_size);

    const auto elem_type = in_tensor->get_element_type();
    OPENVINO_ASSERT(elem_type == ov::element::i64 || elem_type == ov::element::i32,
                    "Unsupported element type for position IDs: ",
                    elem_type.get_type_name());
    auto get_pos = [&](size_t i) -> int64_t {
        return elem_type == ov::element::i64 ? in_tensor->data<int64_t>()[i]
                                             : static_cast<int64_t>(in_tensor->data<int32_t>()[i]);
    };

    // Same logic as regular attention PositionIDs
    for (int64_t idx = static_cast<int64_t>(in_dims.back()) - 1; idx >= 0; idx--) {
        const auto pos = get_pos(static_cast<size_t>(idx));
        if (pos > 0) {
            // Initialize fields
            _current_length = pos;
            switch (_case) {
            case Case::GENERATE:
                // decode case, we have pos_id-1 past elements to take from kvcache
                _past_length = _current_length;
                break;
            case Case::PREFILL:
                // chunked prefill case. calculate the past_length in full chunks
                // FIXME: We know too much about chunking here
                // A present-first graph is driven externally (not by LLMInferRequest), so
                // the supplied history length is unreliable; derive the past from the
                // position ids instead - every chunk but the current one is full.
                _past_length = _kv_order == ov::npuw::util::KVOrder::PresentFirst
                                   ? (_current_length / step) * step
                                   : ((past_len + step - 1) / step) * step;
                break;
            default:
                NPUW_ASSERT(false && "Reached the unreachable code");
            }
            return;
        }
    }
    LOG_WARN("Dynamic selector - no data found in the feature?");
    _current_length = -1;
}

int64_t PositionIDs::context_length() const {
    return static_cast<int64_t>(_tile_size) + _past_length;
}

// ============================================================================
// HFARuntimeContext Implementation
// ============================================================================

void HFARuntimeContext::reset() {
    m_mask_tile_cache.clear();
    m_mask_tile_buffers.clear();
    m_state_buffers.reset();
    m_current_buffer_idx = 0;
}

ov::SoPtr<ov::ITensor> HFARuntimeContext::find_cached_mask_tile(const ov::SoPtr<ov::ITensor>& mask_tensor,
                                                                int64_t mask_offset,
                                                                int64_t tile_length) const {
    HFATileMaskKey cache_key{mask_tensor, mask_offset, tile_length};
    auto it = m_mask_tile_cache.find(cache_key);
    if (it != m_mask_tile_cache.end()) {
        return it->second;
    }
    return {};
}

ov::SoPtr<ov::ITensor> HFARuntimeContext::get_mask_tile_buffer(size_t index) const {
    if (index >= m_mask_tile_buffers.size()) {
        throw std::out_of_range("HFA: mask tile buffer index " + std::to_string(index) + " out of range [0, " +
                                std::to_string(m_mask_tile_buffers.size()) + ")");
    }
    return m_mask_tile_buffers[index];
}

void HFARuntimeContext::cache_mask_tile(const ov::SoPtr<ov::ITensor>& mask_tensor,
                                        int64_t mask_offset,
                                        int64_t tile_length,
                                        const ov::SoPtr<ov::ITensor>& cached_tile) {
    HFATileMaskKey cache_key{mask_tensor, mask_offset, tile_length};
    m_mask_tile_cache[cache_key] = cached_tile;
}

void HFARuntimeContext::clear_mask_cache() {
    m_mask_tile_cache.clear();
}

void HFARuntimeContext::initialize_state_tensors(ov::SoPtr<ov::ITensor>& acc,
                                                 ov::SoPtr<ov::ITensor>& max,
                                                 ov::SoPtr<ov::ITensor>& sum) {
    const auto type = acc->get_element_type();
    if (type == ov::element::f16) {
        std::memset(acc->data<ov::float16>(), 0, acc->get_byte_size());
        std::fill_n(max->data<ov::float16>(), max->get_size(), std::numeric_limits<ov::float16>::lowest());
        std::memset(sum->data<ov::float16>(), 0, sum->get_byte_size());
    } else if (type == ov::element::f32) {
        std::memset(acc->data<float>(), 0, acc->get_byte_size());
        std::fill_n(max->data<float>(), max->get_size(), std::numeric_limits<float>::lowest());
        std::memset(sum->data<float>(), 0, sum->get_byte_size());
    } else {
        throw std::runtime_error("HFA: Unsupported state tensor type");
    }
}

void HFARuntimeContext::prepare_next_state_buffers() {
    if (!m_state_buffers.has_value()) {
        return;
    }
    size_t next_idx = 1 - m_current_buffer_idx;
    auto& next_buffer = (*m_state_buffers)[next_idx];
    initialize_state_tensors(next_buffer.acc, next_buffer.max, next_buffer.sum);
}

void HFARuntimeContext::switch_buffers() {
    if (m_state_buffers.has_value()) {
        m_current_buffer_idx = 1 - m_current_buffer_idx;
    }
}

}  // namespace host_flash_attention
}  // namespace runtime

}  // namespace npuw
}  // namespace ov
