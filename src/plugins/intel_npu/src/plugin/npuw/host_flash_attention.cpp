// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Configuration: Enable loop-based Q@K computation to avoid materialized K/V broadcast
// Set to 1 to enable grouped computation, 0 to use traditional broadcast
// Disabled by default because NPU compiler optimizations are suboptimal
#define ENABLE_HFA_LOOP_BASED_COMPUTATION 0

#include "host_flash_attention.hpp"

#include "logging.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/openvino.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "pyramid_attention.hpp"
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
    std::shared_ptr<ov::Node> acc;
    std::shared_ptr<ov::Node> maxx;
    std::shared_ptr<ov::Node> d;
};

// ============================================================================
// Helper function: Create input parameters for HFA tile model
// ============================================================================
static HFATileInputs create_hfa_tile_inputs(const ov::Shape& q_shape,
                                            const ov::element::Type& input_dtype,
                                            const ov::element::Type& mask_dtype,
                                            int64_t tile_size,
                                            size_t kv_num_heads) {
    auto batch = q_shape[0];
    auto num_heads = q_shape[1];
    auto seq_len = q_shape[2];
    auto head_dim = q_shape[3];

    HFATileInputs inputs;

    // past_acc: [batch, num_heads, seq_len, head_dim]
    inputs.past_acc =
        std::make_shared<ov::op::v0::Parameter>(input_dtype, ov::Shape{batch, num_heads, seq_len, head_dim});
    inputs.past_acc->set_friendly_name("past_acc");
    inputs.past_acc->output(0).get_tensor().set_names({"past_acc"});

    // past_max: [batch, num_heads, seq_len, 1]
    inputs.past_max = std::make_shared<ov::op::v0::Parameter>(input_dtype, ov::Shape{batch, num_heads, seq_len, 1});
    inputs.past_max->set_friendly_name("past_max");
    inputs.past_max->output(0).get_tensor().set_names({"past_max"});

    // past_d: [batch, num_heads, seq_len, 1]
    inputs.past_d = std::make_shared<ov::op::v0::Parameter>(input_dtype, ov::Shape{batch, num_heads, seq_len, 1});
    inputs.past_d->set_friendly_name("past_d");
    inputs.past_d->output(0).get_tensor().set_names({"past_d"});

    // k_tile: [batch, kv_num_heads, tile_size, head_dim]
    inputs.k_tile = std::make_shared<ov::op::v0::Parameter>(
        input_dtype,
        ov::Shape{batch, kv_num_heads, static_cast<size_t>(tile_size), head_dim});
    inputs.k_tile->set_friendly_name("k_tile");
    inputs.k_tile->output(0).get_tensor().set_names({"k_tile"});

    // v_tile: [batch, kv_num_heads, head_dim, tile_size]
    inputs.v_tile = std::make_shared<ov::op::v0::Parameter>(
        input_dtype,
        ov::Shape{batch, kv_num_heads, head_dim, static_cast<size_t>(tile_size)});
    inputs.v_tile->set_friendly_name("v_tile");
    inputs.v_tile->output(0).get_tensor().set_names({"v_tile"});

    // q: [batch, num_heads, seq_len, head_dim]
    inputs.q = std::make_shared<ov::op::v0::Parameter>(input_dtype, ov::Shape{batch, num_heads, seq_len, head_dim});
    inputs.q->set_friendly_name("q");
    inputs.q->output(0).get_tensor().set_names({"q"});

    // mask_tile: [batch, 1, seq_len, tile_size] - use mask's original dtype
    inputs.mask_tile =
        std::make_shared<ov::op::v0::Parameter>(mask_dtype,
                                                ov::Shape{batch, 1, seq_len, static_cast<size_t>(tile_size)});
    inputs.mask_tile->set_friendly_name("mask_tile");
    inputs.mask_tile->output(0).get_tensor().set_names({"mask_tile"});

    return inputs;
}

// ============================================================================
// Helper function: Convert input parameters to f32
// ============================================================================
static HFATileF32Nodes convert_inputs_to_f32(const HFATileInputs& inputs,
                                             const ov::element::Type& mask_dtype,
                                             const ov::element::Type& compute_dtype) {
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

    // Convert mask to f32 if needed
    if (mask_dtype == compute_dtype) {
        f32_nodes.mask_tile_f32 = inputs.mask_tile;
    } else {
        f32_nodes.mask_tile_f32 = std::make_shared<ov::op::v0::Convert>(inputs.mask_tile, compute_dtype);
        f32_nodes.mask_tile_f32->set_friendly_name("mask_tile_f32");
    }

    return f32_nodes;
}

// ============================================================================
// Helper function: Execute flash attention algorithm (unified implementation)
// Supports both traditional broadcast and loop-based grouped computation
// ============================================================================
// Parameters:
//   use_grouped: If true, uses loop-based grouped computation (Q/P reshape)
//                If false, uses traditional broadcast K/V approach
static FlashAttentionResults execute_flash_attention(const HFATileF32Nodes& f32_nodes,
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

    // qkm = qk + mask
    auto qkm = std::make_shared<ov::op::v1::Add>(qk, f32_nodes.mask_tile_f32);
    qkm->set_friendly_name("qkm");

    // maxx = max(past_max, reduce_max(qkm, axis=-1, keepdims=True))
    auto axes_const = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
    auto qkm_max = std::make_shared<ov::op::v1::ReduceMax>(qkm, axes_const, true);
    qkm_max->set_friendly_name("qkm_max");

    results.maxx = std::make_shared<ov::op::v1::Maximum>(qkm_max, f32_nodes.past_max_f32);
    results.maxx->set_friendly_name("maxx");

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
    results.d = std::make_shared<ov::op::v1::Add>(past_d_alpha, l);
    results.d->set_friendly_name("d");

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
    results.acc = std::make_shared<ov::op::v1::Add>(past_acc_alpha, pv);
    results.acc->set_friendly_name("acc");

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
// Helper function: Create final tile model outputs (division, transpose, reshape)
// ============================================================================
static ov::ResultVector create_final_tile_outputs(const FlashAttentionResults& results,
                                                  const ov::element::Type& output_dtype,
                                                  size_t batch,
                                                  size_t seq_len,
                                                  size_t num_heads,
                                                  size_t head_dim) {
    // Division: result = acc / d
    auto final_result = std::make_shared<ov::op::v1::Divide>(results.acc, results.d);
    final_result->set_friendly_name("final_result");

    // Transpose (0,2,1,3): [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, num_heads, head_dim]
    auto transpose_order =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 2, 1, 3});
    auto transposed_result = std::make_shared<ov::op::v1::Transpose>(final_result, transpose_order);
    transposed_result->set_friendly_name("transposed_result");

    // Reshape: [batch, seq_len, num_heads, head_dim] -> [batch, seq_len, num_heads*head_dim]
    auto reshape_pattern =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                               ov::Shape{3},
                                               std::vector<int64_t>{static_cast<int64_t>(batch),
                                                                    static_cast<int64_t>(seq_len),
                                                                    static_cast<int64_t>(num_heads * head_dim)});
    auto reshaped_result = std::make_shared<ov::op::v1::Reshape>(transposed_result, reshape_pattern, false);
    reshaped_result->set_friendly_name("reshaped_result");

    // Convert final output to original SDPA output dtype
    auto final_output = std::make_shared<ov::op::v0::Convert>(reshaped_result, output_dtype);
    final_output->set_friendly_name("final_output");
    final_output->output(0).get_tensor().set_names({"output"});

    // Create result - only ONE output
    auto out_result = std::make_shared<ov::op::v0::Result>(final_output);
    out_result->set_friendly_name("out_result");

    return {out_result};
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
// Helper function: Create individual tile model (regular or final)
// ============================================================================
// Parameters:
//   is_final_tile: If true, creates final tile with division/transpose/reshape
//   output_dtype: Output data type (only used when is_final_tile=true)
static std::shared_ptr<ov::Model> create_hfa_tile_model(const ov::Shape& q_shape,
                                                        const ov::element::Type& input_dtype,
                                                        const ov::element::Type& mask_dtype,
                                                        int64_t tile_size,
                                                        size_t kv_num_heads,
                                                        bool is_final_tile = false,
                                                        const ov::element::Type& output_dtype = ov::element::f16) {
    LOG_DEBUG("Creating HFA " << (is_final_tile ? "FINAL " : "") << "tile model with tile_size=" << tile_size
                              << ", kv_num_heads=" << kv_num_heads << ", mask_dtype=" << mask_dtype
                              << (is_final_tile ? ", output_dtype=" + output_dtype.get_type_name() : ""));

    // Extract dimensions
    NPUW_ASSERT(q_shape.size() == 4);
    auto batch = q_shape[0];
    auto num_heads = q_shape[1];
    auto seq_len = q_shape[2];
    auto head_dim = q_shape[3];

    NPUW_ASSERT(num_heads % kv_num_heads == 0 && "Q heads must be divisible by KV heads");

    auto compute_dtype = ov::element::f32;
    LOG_DEBUG("Using compute_dtype=f32 for all operations to match mask type");

    // Create input parameters
    auto inputs = create_hfa_tile_inputs(q_shape, input_dtype, mask_dtype, tile_size, kv_num_heads);

    // Convert all inputs to f32
    auto f32_nodes = convert_inputs_to_f32(inputs, mask_dtype, compute_dtype);

    FlashAttentionResults results;

#if ENABLE_HFA_LOOP_BASED_COMPUTATION
    // ========================================================================
    // Loop-based computation: Reshape Q to avoid K/V broadcast materialization
    // ========================================================================
    LOG_DEBUG("Using loop-based grouped computation (ENABLED) - avoids K/V broadcast");

    // Reshape Q for grouped computation
    auto q_grouped = reshape_q_for_groups(f32_nodes.q_f32, batch, num_heads, kv_num_heads, seq_len, head_dim);

    // Execute flash attention with grouped computation (K and V remain 4D, no broadcast)
    results = execute_flash_attention(f32_nodes,
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

    // Broadcast K and V tiles from kv_num_heads to num_heads
    auto [k_broadcast, v_broadcast] = broadcast_kv_tiles(f32_nodes.k_tile_f32,
                                                         f32_nodes.v_tile_f32,
                                                         batch,
                                                         num_heads,
                                                         kv_num_heads,
                                                         tile_size,
                                                         head_dim);

    // Execute flash attention algorithm with broadcasted K/V
    results = execute_flash_attention(f32_nodes,
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
#endif  // ENABLE_HFA_LOOP_BASED_COMPUTATION

    // Create model outputs and name based on tile type
    ov::ResultVector model_results;
    std::string model_name;

    if (is_final_tile) {
        // === FINAL TILE: Add division, transpose and reshape for final output ===
        model_results = create_final_tile_outputs(results, output_dtype, batch, seq_len, num_heads, head_dim);
        model_name = "HFA_Final_Tile";
        LOG_DEBUG("HFA FINAL tile model created: inputs=" << input_dtype << ", compute=" << compute_dtype
                                                          << ", output=" << output_dtype);
    } else {
        // === REGULAR TILE: Output intermediate states (acc, max, d) ===
        model_results = create_regular_tile_outputs(results, input_dtype);
        model_name = "HFA_Tile";
        LOG_DEBUG("HFA tile model created: inputs=" << input_dtype << ", compute=" << compute_dtype
                                                    << ", outputs=" << input_dtype);
    }

    // Create model parameters
    ov::ParameterVector model_params =
        {inputs.past_acc, inputs.past_max, inputs.past_d, inputs.k_tile, inputs.v_tile, inputs.q, inputs.mask_tile};

    // Create and return model
    return std::make_shared<ov::Model>(model_results, model_params, model_name);
}

// ============================================================================
// Helper function: Extract actual Parameter by skipping Convert nodes
// ============================================================================
static std::shared_ptr<ov::Node> skip_convert_nodes(const std::shared_ptr<ov::Node>& node) {
    auto current = node;
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
// Helper function: Build SDPA parameter index mapping
// ============================================================================
static void build_sdpa_param_mapping(HostFlashAttention& hfa,
                                     const std::shared_ptr<ov::Model>& model,
                                     const SDPAPatternNodes& pattern_nodes) {
    LOG_INFO("Building SDPA input parameter index mapping...");

    // Helper lambda to safely extract parameter from node (skipping Convert ops)
    auto extract_param = [&](const std::shared_ptr<ov::Node>& node) -> std::shared_ptr<ov::op::v0::Parameter> {
        return ov::as_type_ptr<ov::op::v0::Parameter>(skip_convert_nodes(node));
    };

    // Extract Q (query) parameter - input 0 of MatMul1
    if (auto q_param = extract_param(pattern_nodes.matmul1_node->get_input_node_shared_ptr(0))) {
        std::size_t q_idx = model->get_parameter_index(q_param);
        hfa._sdpa_param_index_map[SDPAInputId::QUERY] = q_idx;
    }

    // Extract past_key parameter - input 0 of past_key_concat
    if (pattern_nodes.past_key_concat_node) {
        if (auto past_k_param = extract_param(pattern_nodes.past_key_concat_node->get_input_node_shared_ptr(0))) {
            std::size_t past_k_idx = model->get_parameter_index(past_k_param);
            hfa._sdpa_param_index_map[SDPAInputId::PAST_KEY] = past_k_idx;
        }

        // Extract present_key parameter - input 1 of past_key_concat
        if (auto present_k_param = extract_param(pattern_nodes.past_key_concat_node->get_input_node_shared_ptr(1))) {
            std::size_t present_k_idx = model->get_parameter_index(present_k_param);
            hfa._sdpa_param_index_map[SDPAInputId::PRESENT_KEY] = present_k_idx;
        }
    }

    // Extract past_value parameter - input 0 of past_value_concat
    if (pattern_nodes.past_value_concat_node) {
        if (auto past_v_param = extract_param(pattern_nodes.past_value_concat_node->get_input_node_shared_ptr(0))) {
            std::size_t past_v_idx = model->get_parameter_index(past_v_param);
            hfa._sdpa_param_index_map[SDPAInputId::PAST_VALUE] = past_v_idx;
        }

        // Extract present_value parameter - input 1 of past_value_concat
        if (auto present_v_param = extract_param(pattern_nodes.past_value_concat_node->get_input_node_shared_ptr(1))) {
            std::size_t present_v_idx = model->get_parameter_index(present_v_param);
            hfa._sdpa_param_index_map[SDPAInputId::PRESENT_VALUE] = present_v_idx;
        }
    }

    // Extract mask parameter - input 1 of add_node
    if (auto add_param = extract_param(pattern_nodes.add_node->get_input_node_shared_ptr(1))) {
        std::size_t mask_idx = model->get_parameter_index(add_param);
        hfa._sdpa_param_index_map[SDPAInputId::ATTENTION_MASK] = mask_idx;
    }

    LOG_INFO("Built SDPA input mapping with " << hfa._sdpa_param_index_map.size() << " entries");

    // Print the complete mapping table
    LOG_DEBUG("");
    LOG_DEBUG("========== SDPA Input Index Mapping ==========");
    LOG_DEBUG("Total entries: " << hfa._sdpa_param_index_map.size());

    for (const auto& [input_id, param_idx] : hfa._sdpa_param_index_map) {
        LOG_DEBUG("  " << sdpa_input_id_to_string(input_id) << " -> parameter[" << param_idx << "]");
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
        if (name == "past_acc") {
            hfa._tile_param_index_map[HFATileInputId::PAST_ACC] = i;
        } else if (name == "past_max") {
            hfa._tile_param_index_map[HFATileInputId::PAST_MAX] = i;
        } else if (name == "past_d") {
            hfa._tile_param_index_map[HFATileInputId::PAST_D] = i;
        } else if (name == "k_tile") {
            hfa._tile_param_index_map[HFATileInputId::K_TILE] = i;
        } else if (name == "v_tile") {
            hfa._tile_param_index_map[HFATileInputId::V_TILE] = i;
        } else if (name == "q") {
            hfa._tile_param_index_map[HFATileInputId::Q] = i;
        } else if (name == "mask_tile") {
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

std::optional<HostFlashAttention> HostFlashAttention::from(const std::shared_ptr<ov::Model>& model) {
    LOG_INFO("Attempting to create HostFlashAttention from model");
    LOG_BLOCK();

    // ========================================================================
    // Step 1: Validate SDPA pattern and extract key nodes
    // ========================================================================
    auto pattern_nodes = find_sdpa_pattern_nodes(model);
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
    auto dtype = q_input->get_output_element_type(0);

    // Validate Q shape and extract query_size (seq_len dimension)
    if (q_shape_static.size() != 4) {
        LOG_WARN("Q shape must be 4D, got " << q_shape_static.size() << "D shape");
        return std::nullopt;
    }
    std::size_t query_size = q_shape_static[2];  // seq_len at index 2
    LOG_DEBUG("Extracted query_size (seq_len) from Q shape: " << query_size);

    auto mask_param = find_mask_parameter(pattern_nodes.add_node);
    if (!mask_param) {
        LOG_WARN("Could not find mask parameter in model");
        return std::nullopt;
    }
    auto mask_dtype = mask_param->get_output_element_type(0);

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
    // K shape after concat: [batch, kv_num_heads, kv_cache_size, head_dim]
    if (k_full_shape.size() != 4) {
        return std::nullopt;
    }

    kv_num_heads = k_full_shape[1];          // Extract kv_num_heads from K shape
    context_size = k_full_shape[k_seq_dim];  // Extract context size from sequence dimension

    if (kv_num_heads == 0) {
        LOG_WARN("Failed to determine KV num_heads");
        return std::nullopt;
    }

    if (context_size == 0) {
        LOG_WARN("Failed to determine context_size");
        return std::nullopt;
    }

    // ========================================================================
    // Step 5: Create tile models using query_size as tile_size
    // ========================================================================
    LOG_INFO("Creating HFA tile models with tile_size=" << query_size);
    auto tile_model = create_hfa_tile_model(q_shape_static, dtype, mask_dtype, query_size, kv_num_heads, false);
    if (!tile_model) {
        LOG_WARN("Failed to create HFA tile model");
        return std::nullopt;
    }

    auto final_tile_model =
        create_hfa_tile_model(q_shape_static, dtype, mask_dtype, query_size, kv_num_heads, true, output_dtype);
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
    hfa._tile_size = query_size;
    hfa._k_seq_dim = k_seq_dim;
    hfa._v_seq_dim = v_seq_dim;

    // ========================================================================
    // Step 7: Build SDPA parameter index mapping
    // ========================================================================
    build_sdpa_param_mapping(hfa, model, pattern_nodes);

    // ========================================================================
    // Step 8: Build tile model parameter index mapping
    // ========================================================================
    build_tile_param_mapping(hfa, tile_model);

    // ========================================================================
    // Step 9: Build tile model output index mapping
    // ========================================================================
    build_tile_output_mapping(hfa, tile_model);

    LOG_INFO("Successfully created HostFlashAttention with query_size="
             << query_size << ", context_size=" << context_size << ", tile_size=" << query_size);

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

    // Pre-cache all indices from function HFA maps
    LOG_INFO("Pre-caching SDPA and tile indices...");

    // Pre-cache SDPA parameter indices
    auto get_sdpa_param_idx = [&](SDPAInputId input_id) -> std::size_t {
        auto it = func_hfa._sdpa_param_index_map.find(input_id);
        if (it == func_hfa._sdpa_param_index_map.end()) {
            OPENVINO_THROW("HFA: SDPA parameter mapping not found for input ID: ", static_cast<uint8_t>(input_id));
        }
        return it->second;
    };

    _sdpa_attention_info._sdpa_indices.query = get_sdpa_param_idx(SDPAInputId::QUERY);
    _sdpa_attention_info._sdpa_indices.past_key = get_sdpa_param_idx(SDPAInputId::PAST_KEY);
    _sdpa_attention_info._sdpa_indices.past_value = get_sdpa_param_idx(SDPAInputId::PAST_VALUE);
    _sdpa_attention_info._sdpa_indices.present_key = get_sdpa_param_idx(SDPAInputId::PRESENT_KEY);
    _sdpa_attention_info._sdpa_indices.present_value = get_sdpa_param_idx(SDPAInputId::PRESENT_VALUE);
    _sdpa_attention_info._sdpa_indices.attention_mask = get_sdpa_param_idx(SDPAInputId::ATTENTION_MASK);

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
             << _sdpa_attention_info._sdpa_indices.query << ", past_key=" << _sdpa_attention_info._sdpa_indices.past_key
             << ", past_value=" << _sdpa_attention_info._sdpa_indices.past_value
             << ", present_key=" << _sdpa_attention_info._sdpa_indices.present_key
             << ", present_value=" << _sdpa_attention_info._sdpa_indices.present_value
             << ", attention_mask=" << _sdpa_attention_info._sdpa_indices.attention_mask << "]");
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
PositionIDs::PositionIDs(std::size_t param_idx, std::size_t query_size, const ov::ISyncInferRequest& rq)
    : _position_ids_idx(param_idx),
      _query_size(query_size),
      _rq(rq) {
    // FIXME: speculative decode is indistinguishable at this point!
    _case = _query_size == 1 ? Case::GENERATE : Case::PREFILL;
}

Selector::Ptr PositionIDs::find(std::size_t query_size, const ov::ISyncInferRequest& rq) {
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
        return Selector::Ptr{new PositionIDs(param_idx, query_size, rq)};
    }
    return Selector::Ptr{};
}

void PositionIDs::prepare(int64_t past_len) {
    const auto& iport = _rq.get().get_compiled_model()->inputs()[_position_ids_idx];
    const auto in_tensor = _rq.get().get_tensor(iport);
    const auto in_dims = in_tensor->get_shape();

    // Same logic as regular attention PositionIDs
    auto* pos_data_ptr = in_tensor->data<int64_t>();
    for (auto idx = in_dims.back() - 1; idx >= 0; idx--) {
        if (pos_data_ptr[idx] > 0) {
            // Initialize fields
            _current_length = pos_data_ptr[idx];
            switch (_case) {
            case Case::GENERATE:
                // decode case, we have pos_id-1 past elements to take from kvcache
                _past_length = _current_length;
                break;
            case Case::PREFILL:
                // chunked prefill case. calculate the past_length in full chunks
                // FIXME: We know too much about chunking here
                _past_length = ((past_len + _query_size - 1) / _query_size) * _query_size;
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
    return _query_size + _past_length;
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
