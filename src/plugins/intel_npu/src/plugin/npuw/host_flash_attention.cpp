// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "host_flash_attention.hpp"

#include "logging.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/openvino.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "pyramid_attention.hpp"
#include "util.hpp"

namespace ov {
namespace npuw {
namespace function {

namespace opp = ov::pass::pattern;

// Helper function to create a single HFA tile computation
// Implements the flash attention tile algorithm from hfa.py::ov_hfa_tile
static std::shared_ptr<ov::Model> create_hfa_tile_model(const ov::Shape& q_shape,
                                                        const ov::element::Type& input_dtype,
                                                        int64_t tile_size,
                                                        size_t kv_num_heads) {
    LOG_DEBUG("Creating HFA tile model with tile_size=" << tile_size << ", kv_heads=" << kv_num_heads);

    // Extract dimensions from Q shape [batch, num_heads, seq_len, head_dim]
    NPUW_ASSERT(q_shape.size() == 4);
    auto batch = q_shape[0];
    auto num_heads = q_shape[1];
    auto seq_len = q_shape[2];
    auto head_dim = q_shape[3];

    // Calculate head expansion factor
    NPUW_ASSERT(num_heads % kv_num_heads == 0 && "Q heads must be divisible by KV heads");
    size_t head_expansion = num_heads / kv_num_heads;

    // Use input_dtype for ALL operations - no type conversion (same as hfa.py)
    auto compute_dtype = input_dtype;

    // Input parameters for HFA tile (using input_dtype, typically f16)
    // past_acc: [batch, num_heads, seq_len, head_dim]
    auto in_past_acc =
        std::make_shared<ov::op::v0::Parameter>(input_dtype, ov::Shape{batch, num_heads, seq_len, head_dim});
    in_past_acc->set_friendly_name("past_acc");
    in_past_acc->output(0).get_tensor().set_names({"past_acc"});

    // past_max: [batch, num_heads, seq_len, 1]
    auto in_past_max = std::make_shared<ov::op::v0::Parameter>(input_dtype, ov::Shape{batch, num_heads, seq_len, 1});
    in_past_max->set_friendly_name("past_max");
    in_past_max->output(0).get_tensor().set_names({"past_max"});

    // past_d: [batch, num_heads, seq_len, 1]
    auto in_past_d = std::make_shared<ov::op::v0::Parameter>(input_dtype, ov::Shape{batch, num_heads, seq_len, 1});
    in_past_d->set_friendly_name("past_d");
    in_past_d->output(0).get_tensor().set_names({"past_d"});

    // k_tile: [batch, kv_num_heads, tile_size, head_dim] - CHANGED: use kv_num_heads instead of num_heads
    auto in_k_tile = std::make_shared<ov::op::v0::Parameter>(
        input_dtype,
        ov::Shape{batch, kv_num_heads, static_cast<size_t>(tile_size), head_dim});
    in_k_tile->set_friendly_name("k_tile");
    in_k_tile->output(0).get_tensor().set_names({"k_tile"});

    // v_tile: [batch, kv_num_heads, head_dim, tile_size] - CHANGED: use kv_num_heads instead of num_heads
    auto in_v_tile = std::make_shared<ov::op::v0::Parameter>(
        input_dtype,
        ov::Shape{batch, kv_num_heads, head_dim, static_cast<size_t>(tile_size)});
    in_v_tile->set_friendly_name("v_tile");
    in_v_tile->output(0).get_tensor().set_names({"v_tile"});

    // q: [batch, num_heads, seq_len, head_dim]
    auto in_q = std::make_shared<ov::op::v0::Parameter>(input_dtype, ov::Shape{batch, num_heads, seq_len, head_dim});
    in_q->set_friendly_name("q");
    in_q->output(0).get_tensor().set_names({"q"});

    // mask_tile: [batch, 1, seq_len, tile_size]
    auto in_mask_tile =
        std::make_shared<ov::op::v0::Parameter>(input_dtype,
                                                ov::Shape{batch, 1, seq_len, static_cast<size_t>(tile_size)});
    in_mask_tile->set_friendly_name("mask_tile");
    in_mask_tile->output(0).get_tensor().set_names({"mask_tile"});

    // === NEW: Broadcast K and V tiles from kv_num_heads to num_heads ===
    // For GQA: kv_num_heads=8, num_heads=32, head_expansion=4
    // Each KV head needs to be repeated head_expansion times consecutively
    // Input:  [batch, 8, tile_size, head_dim]
    // Desired: [batch, 32, tile_size, head_dim] where each of 8 heads is repeated 4 times
    // Pattern: [h0,h0,h0,h0, h1,h1,h1,h1, ..., h7,h7,h7,h7]

    // Strategy: Unsqueeze -> Tile -> Reshape
    // Step 1: [batch, 8, tile_size, head_dim] -> [batch, 8, 1, tile_size, head_dim]
    // Step 2: [batch, 8, 1, tile_size, head_dim] -> [batch, 8, 4, tile_size, head_dim]
    // Step 3: [batch, 8, 4, tile_size, head_dim] -> [batch, 32, tile_size, head_dim]

    // Broadcast K: [batch, kv_num_heads, tile_size, head_dim] -> [batch, num_heads, tile_size, head_dim]
    // Step 1: Unsqueeze at axis 2
    auto unsqueeze_axes_k =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{2});
    auto k_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(in_k_tile, unsqueeze_axes_k);
    // Now shape: [batch, kv_num_heads, 1, tile_size, head_dim]

    // Step 2: Tile along the new dimension (axis 2) - repeat head_expansion times
    auto repeats_k =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                               ov::Shape{5},
                                               std::vector<int64_t>{1, 1, static_cast<int64_t>(head_expansion), 1, 1});
    auto k_tiled = std::make_shared<ov::op::v0::Tile>(k_unsqueezed, repeats_k);
    // Now shape: [batch, kv_num_heads, head_expansion, tile_size, head_dim]

    // Step 3: Reshape to merge kv_num_heads and head_expansion dimensions
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
    // Step 1: Unsqueeze at axis 2
    auto unsqueeze_axes_v =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{2});
    auto v_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(in_v_tile, unsqueeze_axes_v);
    // Now shape: [batch, kv_num_heads, 1, head_dim, tile_size]

    // Step 2: Tile along the new dimension (axis 2)
    auto repeats_v =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                               ov::Shape{5},
                                               std::vector<int64_t>{1, 1, static_cast<int64_t>(head_expansion), 1, 1});
    auto v_tiled = std::make_shared<ov::op::v0::Tile>(v_unsqueezed, repeats_v);
    // Now shape: [batch, kv_num_heads, head_expansion, head_dim, tile_size]

    // Step 3: Reshape to merge kv_num_heads and head_expansion dimensions
    auto v_reshape_pattern =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                               ov::Shape{4},
                                               std::vector<int64_t>{static_cast<int64_t>(batch),
                                                                    static_cast<int64_t>(num_heads),
                                                                    static_cast<int64_t>(head_dim),
                                                                    static_cast<int64_t>(tile_size)});
    auto v_tile_broadcast = std::make_shared<ov::op::v1::Reshape>(v_tiled, v_reshape_pattern, false);
    v_tile_broadcast->set_friendly_name("v_tile_broadcast");

    // Flash Attention Tile Algorithm (from hfa.py::ov_hfa_tile):
    // NO TYPE CONVERSION - all operations use input_dtype directly

    // qk = matmul(q, k^T) - now using broadcasted k_tile
    auto qk = std::make_shared<ov::op::v0::MatMul>(in_q, k_tile_broadcast, false, true);
    qk->set_friendly_name("qk");

    // qkm = qk + mask
    auto qkm = std::make_shared<ov::op::v1::Add>(qk, in_mask_tile);
    qkm->set_friendly_name("qkm");

    // maxx = max(past_max, reduce_max(qkm, axis=-1, keepdims=True))
    auto axes_const = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
    auto qkm_max = std::make_shared<ov::op::v1::ReduceMax>(qkm, axes_const, true);
    qkm_max->set_friendly_name("qkm_max");

    auto maxx = std::make_shared<ov::op::v1::Maximum>(in_past_max, qkm_max);
    maxx->set_friendly_name("maxx");

    // p = exp(qkm - maxx)
    auto qkm_sub_maxx = std::make_shared<ov::op::v1::Subtract>(qkm, maxx);
    auto p = std::make_shared<ov::op::v0::Exp>(qkm_sub_maxx);
    p->set_friendly_name("p");

    // l = reduce_sum(p, axis=-1, keepdims=True)
    auto l = std::make_shared<ov::op::v1::ReduceSum>(p, axes_const, true);
    l->set_friendly_name("l");

    // alpha = exp(past_max - maxx)
    auto past_max_sub_maxx = std::make_shared<ov::op::v1::Subtract>(in_past_max, maxx);
    auto alpha = std::make_shared<ov::op::v0::Exp>(past_max_sub_maxx);
    alpha->set_friendly_name("alpha");

    // d = past_d * alpha + l
    auto past_d_alpha = std::make_shared<ov::op::v1::Multiply>(in_past_d, alpha);
    auto d = std::make_shared<ov::op::v1::Add>(past_d_alpha, l);
    d->set_friendly_name("d");

    // acc = past_acc * alpha + matmul(p, v) - now using broadcasted v_tile
    auto past_acc_alpha = std::make_shared<ov::op::v1::Multiply>(in_past_acc, alpha);
    auto pv = std::make_shared<ov::op::v0::MatMul>(p, v_tile_broadcast, false, true);
    pv->set_friendly_name("pv");
    auto acc = std::make_shared<ov::op::v1::Add>(past_acc_alpha, pv);
    acc->set_friendly_name("acc");

    // Set output tensor names
    acc->output(0).get_tensor().set_names({"acc"});
    maxx->output(0).get_tensor().set_names({"maxx"});
    d->output(0).get_tensor().set_names({"d"});

    // Create results - NO TYPE CONVERSION
    auto out_acc = std::make_shared<ov::op::v0::Result>(acc);
    out_acc->set_friendly_name("out_acc");

    auto out_maxx = std::make_shared<ov::op::v0::Result>(maxx);
    out_maxx->set_friendly_name("out_maxx");

    auto out_d = std::make_shared<ov::op::v0::Result>(d);
    out_d->set_friendly_name("out_d");

    // Create model
    auto tile_model = std::make_shared<ov::Model>(
        ov::ResultVector{out_acc, out_maxx, out_d},
        ov::ParameterVector{in_past_acc, in_past_max, in_past_d, in_k_tile, in_v_tile, in_q, in_mask_tile},
        "HFA_Tile");

    LOG_DEBUG("HFA tile model created successfully with uniform dtype=" << input_dtype << " (NO type conversion)");
    return tile_model;
}

// Helper function to create the FINAL HFA tile computation (with division and transpose)
// This fuses the final tile computation with acc/d division and transpose (0,2,1,3)
static std::shared_ptr<ov::Model> create_hfa_final_tile_model(const ov::Shape& q_shape,
                                                              const ov::element::Type& input_dtype,
                                                              int64_t tile_size,
                                                              size_t kv_num_heads) {
    LOG_DEBUG("Creating HFA FINAL tile model with tile_size=" << tile_size << ", kv_num_heads=" << kv_num_heads
                                                              << " (with division, transpose and reshape)");

    // Extract dimensions from Q shape [batch, num_heads, seq_len, head_dim]
    NPUW_ASSERT(q_shape.size() == 4);
    auto batch = q_shape[0];
    auto num_heads = q_shape[1];
    auto seq_len = q_shape[2];
    auto head_dim = q_shape[3];

    // Calculate head expansion factor for GQA (e.g., 32 / 8 = 4)
    size_t head_expansion = num_heads / kv_num_heads;
    LOG_DEBUG("Head expansion factor: " << head_expansion);

    // Input parameters (same as regular tile model)
    auto in_past_acc =
        std::make_shared<ov::op::v0::Parameter>(input_dtype, ov::Shape{batch, num_heads, seq_len, head_dim});
    in_past_acc->set_friendly_name("past_acc");
    in_past_acc->output(0).get_tensor().set_names({"past_acc"});

    auto in_past_max = std::make_shared<ov::op::v0::Parameter>(input_dtype, ov::Shape{batch, num_heads, seq_len, 1});
    in_past_max->set_friendly_name("past_max");
    in_past_max->output(0).get_tensor().set_names({"past_max"});

    auto in_past_d = std::make_shared<ov::op::v0::Parameter>(input_dtype, ov::Shape{batch, num_heads, seq_len, 1});
    in_past_d->set_friendly_name("past_d");
    in_past_d->output(0).get_tensor().set_names({"past_d"});

    // K and V tiles now use kv_num_heads (e.g., 8 instead of 32)
    auto in_k_tile = std::make_shared<ov::op::v0::Parameter>(
        input_dtype,
        ov::Shape{batch, kv_num_heads, static_cast<size_t>(tile_size), head_dim});
    in_k_tile->set_friendly_name("k_tile");
    in_k_tile->output(0).get_tensor().set_names({"k_tile"});

    auto in_v_tile = std::make_shared<ov::op::v0::Parameter>(
        input_dtype,
        ov::Shape{batch, kv_num_heads, head_dim, static_cast<size_t>(tile_size)});
    in_v_tile->set_friendly_name("v_tile");
    in_v_tile->output(0).get_tensor().set_names({"v_tile"});

    auto in_q = std::make_shared<ov::op::v0::Parameter>(input_dtype, ov::Shape{batch, num_heads, seq_len, head_dim});
    in_q->set_friendly_name("q");
    in_q->output(0).get_tensor().set_names({"q"});

    auto in_mask_tile =
        std::make_shared<ov::op::v0::Parameter>(input_dtype,
                                                ov::Shape{batch, 1, seq_len, static_cast<size_t>(tile_size)});
    in_mask_tile->set_friendly_name("mask_tile");
    in_mask_tile->output(0).get_tensor().set_names({"mask_tile"});

    // Broadcast K and V from kv_num_heads to num_heads using Unsqueeze->Tile->Reshape
    // This ensures each KV head is repeated consecutively (not interleaved)

    // Broadcast K: [batch, kv_num_heads, tile_size, head_dim] -> [batch, num_heads, tile_size, head_dim]
    auto unsqueeze_axes_k =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{2});
    auto k_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(in_k_tile, unsqueeze_axes_k);
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
    auto v_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(in_v_tile, unsqueeze_axes_v);
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

    // Flash Attention Tile Algorithm (now using broadcasted K and V)
    auto qk = std::make_shared<ov::op::v0::MatMul>(in_q, k_tile_broadcast, false, true);
    qk->set_friendly_name("qk");

    auto qkm = std::make_shared<ov::op::v1::Add>(qk, in_mask_tile);
    qkm->set_friendly_name("qkm");

    auto axes_const = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
    auto qkm_max = std::make_shared<ov::op::v1::ReduceMax>(qkm, axes_const, true);
    qkm_max->set_friendly_name("qkm_max");

    auto maxx = std::make_shared<ov::op::v1::Maximum>(in_past_max, qkm_max);
    maxx->set_friendly_name("maxx");

    auto qkm_sub_maxx = std::make_shared<ov::op::v1::Subtract>(qkm, maxx);
    auto p = std::make_shared<ov::op::v0::Exp>(qkm_sub_maxx);
    p->set_friendly_name("p");

    auto l = std::make_shared<ov::op::v1::ReduceSum>(p, axes_const, true);
    l->set_friendly_name("l");

    auto past_max_sub_maxx = std::make_shared<ov::op::v1::Subtract>(in_past_max, maxx);
    auto alpha = std::make_shared<ov::op::v0::Exp>(past_max_sub_maxx);
    alpha->set_friendly_name("alpha");

    auto past_d_alpha = std::make_shared<ov::op::v1::Multiply>(in_past_d, alpha);
    auto d = std::make_shared<ov::op::v1::Add>(past_d_alpha, l);
    d->set_friendly_name("d");

    // acc = past_acc * alpha + matmul(p, v) - now using broadcasted v_tile
    auto past_acc_alpha = std::make_shared<ov::op::v1::Multiply>(in_past_acc, alpha);
    auto pv = std::make_shared<ov::op::v0::MatMul>(p, v_tile_broadcast, false, true);
    pv->set_friendly_name("pv");
    auto acc = std::make_shared<ov::op::v1::Add>(past_acc_alpha, pv);
    acc->set_friendly_name("acc");

    // === NEW: Add division and transpose for final output ===

    // Division: result = acc / d
    // acc shape: [batch, num_heads, seq_len, head_dim]
    // d shape:   [batch, num_heads, seq_len, 1]
    // Result after division: [batch, num_heads, seq_len, head_dim]
    auto final_result = std::make_shared<ov::op::v1::Divide>(acc, d);
    final_result->set_friendly_name("final_result");

    // Transpose (0,2,1,3): [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, num_heads, head_dim]
    auto transpose_order =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 2, 1, 3});
    auto transposed_result = std::make_shared<ov::op::v1::Transpose>(final_result, transpose_order);
    transposed_result->set_friendly_name("transposed_result");

    // Reshape: [batch, seq_len, num_heads, head_dim] -> [batch, seq_len, num_heads*head_dim]
    // This matches the expected output shape of the original SDPA
    auto reshape_pattern =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                               ov::Shape{3},
                                               std::vector<int64_t>{static_cast<int64_t>(batch),
                                                                    static_cast<int64_t>(seq_len),
                                                                    static_cast<int64_t>(num_heads * head_dim)});
    auto reshaped_result = std::make_shared<ov::op::v1::Reshape>(transposed_result, reshape_pattern, false);
    reshaped_result->set_friendly_name("reshaped_result");

    // Set output tensor name
    reshaped_result->output(0).get_tensor().set_names({"output"});

    // Create result - only ONE output (the final reshaped result)
    auto out_result = std::make_shared<ov::op::v0::Result>(reshaped_result);
    out_result->set_friendly_name("out_result");

    // Create model
    auto final_tile_model = std::make_shared<ov::Model>(
        ov::ResultVector{out_result},
        ov::ParameterVector{in_past_acc, in_past_max, in_past_d, in_k_tile, in_v_tile, in_q, in_mask_tile},
        "HFA_Final_Tile");

    LOG_DEBUG("HFA FINAL tile model created successfully (with division, transpose and reshape fused)");
    return final_tile_model;
}

std::optional<HostFlashAttention> HostFlashAttention::from(const std::shared_ptr<ov::Model>& model) {
    LOG_INFO("Attempting to create HostFlashAttention from model");
    LOG_BLOCK();

    // Validate and setup using shared function from pyramid attention
    auto validation_result = validate_and_setup_pyramid_attention(model);
    if (!validation_result) {
        LOG_WARN("Failed to validate SDPA pattern for HFA");
        std::cout << "HostFlashAttention::from - pattern validation failed" << std::endl;
        return std::nullopt;
    }

    LOG_INFO("Successfully validated decomposed SDPA pattern");

    // Extract pre-computed dimensions
    const auto& past_key_sequence_dims = validation_result->past_key_sequence_dims;
    const auto& past_value_sequence_dims = validation_result->past_value_sequence_dims;

    // Create Attention instance from model using shared function
    auto attention_opt = create_attention_from_model(model, past_key_sequence_dims, past_value_sequence_dims);
    if (!attention_opt) {
        LOG_WARN("Failed to create attention from model");
        return std::nullopt;
    }

    LOG_INFO("Successfully created attention metadata from model");

    // Re-find pattern nodes to extract Q input and K concat for tile model creation
    auto pattern_nodes = find_sdpa_pattern_nodes(model);
    if (!pattern_nodes.is_valid()) {
        LOG_WARN("Failed to re-find SDPA pattern nodes");
        return std::nullopt;
    }

    auto q_input = pattern_nodes.matmul1_node->get_input_node_shared_ptr(0);
    auto k_concat = pattern_nodes.past_key_concat_node;

    // Skip Convert nodes to get to the actual Parameter/input
    while (q_input && std::dynamic_pointer_cast<ov::op::v0::Convert>(q_input)) {
        if (q_input->get_input_size() > 0) {
            q_input = q_input->get_input_node_shared_ptr(0);
            LOG_DEBUG("Skipped Convert node, now at: " << q_input->get_friendly_name());
        } else {
            break;
        }
    }

    if (!q_input || !k_concat) {
        LOG_WARN("Failed to extract Q input or K concat from pattern");
        return std::nullopt;
    }

    LOG_INFO("Successfully extracted Q input and K concat nodes");

    // Extract shape information
    auto q_shape = q_input->get_output_partial_shape(0);
    if (q_shape.is_dynamic()) {
        LOG_WARN("Dynamic shapes not yet supported for HFA");
        return std::nullopt;
    }

    auto q_shape_static = q_shape.to_shape();
    auto dtype = q_input->get_output_element_type(0);

    LOG_DEBUG("Q shape: " << q_shape_static);
    LOG_DEBUG("Data type: " << dtype);

    // Determine KV cache size from Concat node
    int64_t kv_cache_size = 0;
    size_t kv_num_heads = 0;
    if (k_concat->get_output_partial_shape(0).is_static()) {
        auto k_full_shape = k_concat->get_output_partial_shape(0).to_shape();
        // K shape after concat: [batch, kv_num_heads, kv_cache_size, head_dim]
        if (k_full_shape.size() >= 3) {
            kv_num_heads = k_full_shape[1];  // Extract kv_num_heads (e.g., 8)
            kv_cache_size = k_full_shape[2];
            LOG_DEBUG("Detected KV num_heads: " << kv_num_heads);
            LOG_DEBUG("Detected KV cache size: " << kv_cache_size);
        }
    }

    if (kv_cache_size == 0 || kv_num_heads == 0) {
        LOG_WARN("Failed to determine KV cache size or num_heads");
        return std::nullopt;
    }

    // Create HFA tile model with kv_num_heads parameter
    constexpr int64_t DEFAULT_TILE_SIZE = 1024;
    auto tile_model = create_hfa_tile_model(q_shape_static, dtype, DEFAULT_TILE_SIZE, kv_num_heads);

    if (!tile_model) {
        LOG_WARN("Failed to create HFA tile model");
        return std::nullopt;
    }

    // Create HFA FINAL tile model (with division and transpose) - also with kv_num_heads
    auto final_tile_model = create_hfa_final_tile_model(q_shape_static, dtype, DEFAULT_TILE_SIZE, kv_num_heads);

    if (!final_tile_model) {
        LOG_WARN("Failed to create HFA final tile model");
        return std::nullopt;
    }

    // Save original model to file
    try {
        std::string original_model_path = "hfa_original_model.xml";
        ov::serialize(model, original_model_path);
        LOG_INFO("Saved original model to: " << original_model_path);
        std::cout << "Saved original decomposed SDPA model to: " << original_model_path << std::endl;
    } catch (const std::exception& e) {
        LOG_WARN("Failed to save original model: " << e.what());
    }

    // Save generated flash attention tile model to file
    try {
        std::string tile_model_path = "hfa_tile_model.xml";
        ov::serialize(tile_model, tile_model_path);
        LOG_INFO("Saved HFA tile model to: " << tile_model_path);
        std::cout << "Saved flash attention tile model to: " << tile_model_path << std::endl;
    } catch (const std::exception& e) {
        LOG_WARN("Failed to save tile model: " << e.what());
    }

    // Save generated flash attention FINAL tile model to file
    try {
        std::string final_tile_model_path = "hfa_final_tile_model.xml";
        ov::serialize(final_tile_model, final_tile_model_path);
        LOG_INFO("Saved HFA final tile model to: " << final_tile_model_path);
        std::cout << "Saved flash attention final tile model to: " << final_tile_model_path << std::endl;
    } catch (const std::exception& e) {
        LOG_WARN("Failed to save final tile model: " << e.what());
    }

    // Create HostFlashAttention structure
    HostFlashAttention hfa;
    hfa._original_model = model;  // Store original SDPA model for parameter extraction
    hfa._tile_model = tile_model;
    hfa._final_tile_model = final_tile_model;
    hfa._tile_size = DEFAULT_TILE_SIZE;
    hfa._kv_cache_size = kv_cache_size;
    hfa._sdpa_attention = std::move(attention_opt.value());  // Store SDPA attention metadata

    LOG_INFO("Successfully created HostFlashAttention");
    std::cout << "HostFlashAttention created with tile_size=" << hfa._tile_size
              << ", kv_cache_size=" << hfa._kv_cache_size << std::endl;

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
    _kv_cache_size = func_hfa._kv_cache_size;

    // Store the tile models for later compilation
    _tile_model_to_compile = func_hfa._tile_model;
    _final_tile_model_to_compile = func_hfa._final_tile_model;

    // Extract attention parameter info from original SDPA model (not from tile models)
    const auto& sdpa_attn = func_hfa._sdpa_attention;
    const auto& original_model = func_hfa._original_model;

    _sdpa_attention_info.params.reserve(sdpa_attn._inputs.size());
    for (const auto& input : sdpa_attn._inputs) {
        std::size_t p_idx = original_model->get_parameter_index(input.param);
        _sdpa_attention_info.params.push_back({p_idx, input.dim});
    }
    _sdpa_attention_info.mask_idx = original_model->get_parameter_index(sdpa_attn._mask);
    _sdpa_attention_info.query_size = sdpa_attn.query_len();

    LOG_INFO("Extracted HFA config: tile_size=" << _tile_size << ", kv_cache_size=" << _kv_cache_size);
    LOG_INFO("Extracted " << _sdpa_attention_info.params.size() << " past KV parameters from original SDPA model");

    // Note: _compiled_tile_model and _compiled_final_tile_model will be set later by
    // compile_host_flash_attention_model()
}

}  // namespace compiled

namespace runtime {
namespace host_flash_attention {

// PositionIDs constructor
PositionIDs::PositionIDs(std::size_t param_idx, std::size_t query_size, const ov::ISyncInferRequest& rq)
    : m_position_ids_idx(param_idx),
      m_query_size(query_size),
      m_rq(rq) {
    // FIXME: speculative decode is indistinguishable at this point!
    m_case = m_query_size == 1 ? Case::GENERATE : Case::PREFILL;
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
    const auto& iport = m_rq.get_compiled_model()->inputs()[m_position_ids_idx];
    const auto in_tensor = m_rq.get_tensor(iport);
    const auto in_dims = in_tensor->get_shape();

    // Same logic as regular attention PositionIDs
    auto* pos_data_ptr = in_tensor->data<int64_t>();
    for (auto idx = in_dims.back() - 1; idx >= 0; idx--) {
        if (pos_data_ptr[idx] > 0) {
            // Initialize fields
            m_current_length = pos_data_ptr[idx];
            switch (m_case) {
            case Case::GENERATE:
                // decode case, we have pos_id-1 past elements to take from kvcache
                m_past_length = m_current_length;
                break;
            case Case::PREFILL:
                // chunked prefill case. calculate the past_length in full chunks
                // FIXME: We know too much about chunking here
                m_past_length = ((past_len + m_query_size - 1) / m_query_size) * m_query_size;
                break;
            default:
                NPUW_ASSERT(false && "Reached the unreachable code");
            }
            return;
        }
    }
    LOG_WARN("Dynamic selector - no data found in the feature?");
    m_current_length = -1;
}

int64_t PositionIDs::length() const {
    return m_query_size;
}

int64_t PositionIDs::past_length() const {
    return m_past_length;
}

}  // namespace host_flash_attention
}  // namespace runtime

}  // namespace npuw
}  // namespace ov
