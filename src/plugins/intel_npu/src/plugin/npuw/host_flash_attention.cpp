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

// Helper: Create input parameters for HFA tile model
struct HFATileInputs {
    std::shared_ptr<ov::op::v0::Parameter> past_acc;
    std::shared_ptr<ov::op::v0::Parameter> past_max;
    std::shared_ptr<ov::op::v0::Parameter> past_d;
    std::shared_ptr<ov::op::v0::Parameter> k_tile;
    std::shared_ptr<ov::op::v0::Parameter> v_tile;
    std::shared_ptr<ov::op::v0::Parameter> q;
    std::shared_ptr<ov::op::v0::Parameter> mask_tile;
};

// Helper: Create converted f32 nodes from input parameters
struct HFATileF32Nodes {
    std::shared_ptr<ov::Node> past_acc_f32;
    std::shared_ptr<ov::Node> past_max_f32;
    std::shared_ptr<ov::Node> past_d_f32;
    std::shared_ptr<ov::Node> k_tile_f32;
    std::shared_ptr<ov::Node> v_tile_f32;
    std::shared_ptr<ov::Node> q_f32;
    std::shared_ptr<ov::Node> mask_tile_f32;
};

// Helper: Flash attention computation results (all in f32)
struct FlashAttentionResults {
    std::shared_ptr<ov::Node> acc;
    std::shared_ptr<ov::Node> maxx;
    std::shared_ptr<ov::Node> d;
};

// Helper function: Create input parameters for HFA tile model
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

// Helper function: Convert input parameters to f32
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

// Helper function: Broadcast KV from kv_num_heads to num_heads
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

// Helper function: Execute flash attention algorithm (all in f32)
static FlashAttentionResults execute_flash_attention(const HFATileF32Nodes& f32_nodes,
                                                     const std::shared_ptr<ov::Node>& k_broadcast,
                                                     const std::shared_ptr<ov::Node>& v_broadcast) {
    FlashAttentionResults results;

    // qk = matmul(q, k^T)
    auto qk = std::make_shared<ov::op::v0::MatMul>(f32_nodes.q_f32, k_broadcast, false, true);
    qk->set_friendly_name("qk");

    // qkm = qk + mask
    auto qkm = std::make_shared<ov::op::v1::Add>(qk, f32_nodes.mask_tile_f32);
    qkm->set_friendly_name("qkm");

    // maxx = max(past_max, reduce_max(qkm, axis=-1, keepdims=True))
    auto axes_const = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
    auto qkm_max = std::make_shared<ov::op::v1::ReduceMax>(qkm, axes_const, true);
    qkm_max->set_friendly_name("qkm_max");

    results.maxx = std::make_shared<ov::op::v1::Maximum>(f32_nodes.past_max_f32, qkm_max);
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

    // acc = past_acc * alpha + matmul(p, v)
    auto past_acc_alpha = std::make_shared<ov::op::v1::Multiply>(f32_nodes.past_acc_f32, alpha);
    auto pv = std::make_shared<ov::op::v0::MatMul>(p, v_broadcast, false, true);
    pv->set_friendly_name("pv");
    results.acc = std::make_shared<ov::op::v1::Add>(past_acc_alpha, pv);
    results.acc->set_friendly_name("acc");

    return results;
}

// Helper function to create a single HFA tile computation
// Implements the flash attention tile algorithm
static std::shared_ptr<ov::Model> create_hfa_tile_model(const ov::Shape& q_shape,
                                                        const ov::element::Type& input_dtype,
                                                        const ov::element::Type& mask_dtype,
                                                        int64_t tile_size,
                                                        size_t kv_num_heads) {
    LOG_DEBUG("Creating HFA tile model with tile_size=" << tile_size << ", kv_heads=" << kv_num_heads
                                                        << ", mask_dtype=" << mask_dtype);

    // Extract dimensions
    NPUW_ASSERT(q_shape.size() == 4);
    auto batch = q_shape[0];
    auto num_heads = q_shape[1];
    auto seq_len = q_shape[2];
    auto head_dim = q_shape[3];

    NPUW_ASSERT(num_heads % kv_num_heads == 0 && "Q heads must be divisible by KV heads");
    size_t head_expansion = num_heads / kv_num_heads;

    auto compute_dtype = ov::element::f32;
    LOG_DEBUG("Using compute_dtype=f32 for all operations to match mask type");

    // Create input parameters
    auto inputs = create_hfa_tile_inputs(q_shape, input_dtype, mask_dtype, tile_size, kv_num_heads);

    // Convert all inputs to f32
    auto f32_nodes = convert_inputs_to_f32(inputs, mask_dtype, compute_dtype);

    // Broadcast K and V tiles
    auto [k_broadcast, v_broadcast] = broadcast_kv_tiles(f32_nodes.k_tile_f32,
                                                         f32_nodes.v_tile_f32,
                                                         batch,
                                                         num_heads,
                                                         kv_num_heads,
                                                         tile_size,
                                                         head_dim);

    // Execute flash attention algorithm
    auto results = execute_flash_attention(f32_nodes, k_broadcast, v_broadcast);

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

    // Create model
    auto tile_model = std::make_shared<ov::Model>(ov::ResultVector{out_acc, out_maxx, out_d},
                                                  ov::ParameterVector{inputs.past_acc,
                                                                      inputs.past_max,
                                                                      inputs.past_d,
                                                                      inputs.k_tile,
                                                                      inputs.v_tile,
                                                                      inputs.q,
                                                                      inputs.mask_tile},
                                                  "HFA_Tile");

    LOG_DEBUG("HFA tile model created successfully: inputs=" << input_dtype << ", compute=" << compute_dtype
                                                             << ", outputs=" << input_dtype);
    return tile_model;
}

// Helper function to create the FINAL HFA tile computation (with division and transpose)
// This fuses the final tile computation with acc/d division and transpose (0,2,1,3)
static std::shared_ptr<ov::Model> create_hfa_final_tile_model(const ov::Shape& q_shape,
                                                              const ov::element::Type& input_dtype,
                                                              const ov::element::Type& mask_dtype,
                                                              const ov::element::Type& output_dtype,
                                                              int64_t tile_size,
                                                              size_t kv_num_heads) {
    LOG_DEBUG("Creating HFA FINAL tile model with tile_size="
              << tile_size << ", kv_num_heads=" << kv_num_heads << ", mask_dtype=" << mask_dtype
              << ", output_dtype=" << output_dtype << " (with division, transpose and reshape)");

    // Extract dimensions
    NPUW_ASSERT(q_shape.size() == 4);
    auto batch = q_shape[0];
    auto num_heads = q_shape[1];
    auto seq_len = q_shape[2];
    auto head_dim = q_shape[3];

    NPUW_ASSERT(num_heads % kv_num_heads == 0 && "Q heads must be divisible by KV heads");
    size_t head_expansion = num_heads / kv_num_heads;

    auto compute_dtype = ov::element::f32;
    LOG_DEBUG("Using compute_dtype=f32 for all operations to match mask type");

    // Create input parameters (reuse helper)
    auto inputs = create_hfa_tile_inputs(q_shape, input_dtype, mask_dtype, tile_size, kv_num_heads);

    // Convert all inputs to f32 (reuse helper)
    auto f32_nodes = convert_inputs_to_f32(inputs, mask_dtype, compute_dtype);

    // Broadcast K and V tiles (reuse helper)
    auto [k_broadcast, v_broadcast] = broadcast_kv_tiles(f32_nodes.k_tile_f32,
                                                         f32_nodes.v_tile_f32,
                                                         batch,
                                                         num_heads,
                                                         kv_num_heads,
                                                         tile_size,
                                                         head_dim);

    // Execute flash attention algorithm (reuse helper)
    auto results = execute_flash_attention(f32_nodes, k_broadcast, v_broadcast);

    // === NEW: Add division, transpose and reshape for final output ===

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

    // Create model
    auto final_tile_model = std::make_shared<ov::Model>(ov::ResultVector{out_result},
                                                        ov::ParameterVector{inputs.past_acc,
                                                                            inputs.past_max,
                                                                            inputs.past_d,
                                                                            inputs.k_tile,
                                                                            inputs.v_tile,
                                                                            inputs.q,
                                                                            inputs.mask_tile},
                                                        "HFA_Final_Tile");

    LOG_DEBUG("HFA FINAL tile model created successfully: inputs=" << input_dtype << ", compute=" << compute_dtype
                                                                   << ", output=" << output_dtype);
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

    // Extract mask type from attention metadata
    auto mask_param = attention_opt->_mask;
    auto mask_dtype = mask_param->get_output_element_type(0);
    LOG_DEBUG("Mask data type: " << mask_dtype);

    // Extract original SDPA output type from model
    auto output_dtype = ov::element::f16;  // Default fallback
    if (model->outputs().size() > 0) {
        output_dtype = model->output(0).get_element_type();
        LOG_DEBUG("Original SDPA output data type: " << output_dtype);
    } else {
        LOG_WARN("No outputs found in model, using default output dtype: " << output_dtype);
    }

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
    auto tile_model = create_hfa_tile_model(q_shape_static, dtype, mask_dtype, DEFAULT_TILE_SIZE, kv_num_heads);

    if (!tile_model) {
        LOG_WARN("Failed to create HFA tile model");
        return std::nullopt;
    }

    // Create HFA FINAL tile model (with division and transpose) - with output_dtype
    auto final_tile_model =
        create_hfa_final_tile_model(q_shape_static, dtype, mask_dtype, output_dtype, DEFAULT_TILE_SIZE, kv_num_heads);

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

    // Build SDPA input parameter index mapping from pattern nodes
    // This mapping will be transferred to compiled::HostFlashAttentionInfo
    LOG_INFO("Building SDPA input parameter index mapping...");

    // Helper lambda to safely extract parameter from node (skipping Convert ops)
    auto extract_param = [&](const std::shared_ptr<ov::Node>& node) -> std::shared_ptr<ov::op::v0::Parameter> {
        auto current = node;
        // Skip Convert nodes to get to actual Parameter
        while (current && ov::is_type<ov::op::v0::Convert>(current.get())) {
            if (current->get_input_size() > 0) {
                current = current->get_input_node_shared_ptr(0);
            } else {
                break;
            }
        }
        return ov::as_type_ptr<ov::op::v0::Parameter>(current);
    };

    // Extract Q (query) parameter - input 0 of MatMul1
    if (auto q_param = extract_param(pattern_nodes.matmul1_node->get_input_node_shared_ptr(0))) {
        std::size_t q_idx = model->get_parameter_index(q_param);
        hfa._sdpa_param_index_map[SDPAInputId::QUERY] = q_idx;
        LOG_DEBUG("Mapped QUERY to parameter index " << q_idx);
    }

    // Extract past_key parameter - input 0 of past_key_concat
    if (pattern_nodes.past_key_concat_node) {
        if (auto past_k_param = extract_param(pattern_nodes.past_key_concat_node->get_input_node_shared_ptr(0))) {
            std::size_t past_k_idx = model->get_parameter_index(past_k_param);
            hfa._sdpa_param_index_map[SDPAInputId::PAST_KEY] = past_k_idx;
            LOG_DEBUG("Mapped PAST_KEY to parameter index " << past_k_idx);
        }

        // Extract present_key parameter - input 1 of past_key_concat
        if (auto present_k_param = extract_param(pattern_nodes.past_key_concat_node->get_input_node_shared_ptr(1))) {
            std::size_t present_k_idx = model->get_parameter_index(present_k_param);
            hfa._sdpa_param_index_map[SDPAInputId::PRESENT_KEY] = present_k_idx;
            LOG_DEBUG("Mapped PRESENT_KEY to parameter index " << present_k_idx);
        }
    }

    // Extract past_value parameter - input 0 of past_value_concat
    if (pattern_nodes.past_value_concat_node) {
        if (auto past_v_param = extract_param(pattern_nodes.past_value_concat_node->get_input_node_shared_ptr(0))) {
            std::size_t past_v_idx = model->get_parameter_index(past_v_param);
            hfa._sdpa_param_index_map[SDPAInputId::PAST_VALUE] = past_v_idx;
            LOG_DEBUG("Mapped PAST_VALUE to parameter index " << past_v_idx);
        }

        // Extract present_value parameter - input 1 of past_value_concat
        if (auto present_v_param = extract_param(pattern_nodes.past_value_concat_node->get_input_node_shared_ptr(1))) {
            std::size_t present_v_idx = model->get_parameter_index(present_v_param);
            hfa._sdpa_param_index_map[SDPAInputId::PRESENT_VALUE] = present_v_idx;
            LOG_DEBUG("Mapped PRESENT_VALUE to parameter index " << present_v_idx);
        }
    }

    // Extract mask parameter - from SDPA attention metadata
    std::size_t mask_idx = model->get_parameter_index(hfa._sdpa_attention._mask);
    hfa._sdpa_param_index_map[SDPAInputId::ATTENTION_MASK] = mask_idx;
    LOG_DEBUG("Mapped ATTENTION_MASK to parameter index " << mask_idx);

    LOG_INFO("Built SDPA input mapping with " << hfa._sdpa_param_index_map.size() << " entries");

    // Print the complete mapping table
    std::cout << "\n========== SDPA Input Index Mapping ==========\n";
    std::cout << "Total entries: " << hfa._sdpa_param_index_map.size() << "\n";

    // Helper to convert enum to string for printing
    auto sdpa_input_id_to_string = [](SDPAInputId id) -> const char* {
        switch (id) {
        case SDPAInputId::PAST_KEY:
            return "PAST_KEY";
        case SDPAInputId::PAST_VALUE:
            return "PAST_VALUE";
        case SDPAInputId::QUERY:
            return "QUERY";
        case SDPAInputId::PRESENT_KEY:
            return "PRESENT_KEY";
        case SDPAInputId::ATTENTION_MASK:
            return "ATTENTION_MASK";
        case SDPAInputId::PRESENT_VALUE:
            return "PRESENT_VALUE";
        default:
            return "UNKNOWN";
        }
    };

    for (const auto& [input_id, param_idx] : hfa._sdpa_param_index_map) {
        std::cout << "  " << sdpa_input_id_to_string(input_id) << " -> parameter[" << param_idx << "]" << std::endl;
    }
    std::cout << "=============================================\n" << std::endl;

    // Build HFA Tile Model input index mapping
    // This mapping allows accessing tile model inputs by semantic name
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
            LOG_DEBUG("Mapped PAST_ACC to tile input[" << i << "]");
        } else if (name == "past_max") {
            hfa._tile_param_index_map[HFATileInputId::PAST_MAX] = i;
            LOG_DEBUG("Mapped PAST_MAX to tile input[" << i << "]");
        } else if (name == "past_d") {
            hfa._tile_param_index_map[HFATileInputId::PAST_D] = i;
            LOG_DEBUG("Mapped PAST_D to tile input[" << i << "]");
        } else if (name == "k_tile") {
            hfa._tile_param_index_map[HFATileInputId::K_TILE] = i;
            LOG_DEBUG("Mapped K_TILE to tile input[" << i << "]");
        } else if (name == "v_tile") {
            hfa._tile_param_index_map[HFATileInputId::V_TILE] = i;
            LOG_DEBUG("Mapped V_TILE to tile input[" << i << "]");
        } else if (name == "q") {
            hfa._tile_param_index_map[HFATileInputId::Q] = i;
            LOG_DEBUG("Mapped Q to tile input[" << i << "]");
        } else if (name == "mask_tile") {
            hfa._tile_param_index_map[HFATileInputId::MASK_TILE] = i;
            LOG_DEBUG("Mapped MASK_TILE to tile input[" << i << "]");
        } else {
            LOG_WARN("Unknown tile model input name: " << name);
        }
    }

    // Print the tile input mapping
    std::cout << "\n========== HFA Tile Model Input Mapping ==========\n";
    std::cout << "Total entries: " << hfa._tile_param_index_map.size() << "\n";

    auto tile_input_id_to_string = [](HFATileInputId id) -> const char* {
        switch (id) {
        case HFATileInputId::PAST_ACC:
            return "PAST_ACC";
        case HFATileInputId::PAST_MAX:
            return "PAST_MAX";
        case HFATileInputId::PAST_D:
            return "PAST_D";
        case HFATileInputId::K_TILE:
            return "K_TILE";
        case HFATileInputId::V_TILE:
            return "V_TILE";
        case HFATileInputId::Q:
            return "Q";
        case HFATileInputId::MASK_TILE:
            return "MASK_TILE";
        default:
            return "UNKNOWN";
        }
    };

    for (const auto& [input_id, input_idx] : hfa._tile_param_index_map) {
        std::cout << "  " << tile_input_id_to_string(input_id) << " -> input[" << input_idx << "]" << std::endl;
    }
    std::cout << "==================================================\n" << std::endl;

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

    // Build parameter info for past key/value tensors
    _sdpa_attention_info.params.reserve(sdpa_attn._inputs.size());
    for (const auto& input : sdpa_attn._inputs) {
        std::size_t p_idx = original_model->get_parameter_index(input.param);
        _sdpa_attention_info.params.push_back({p_idx, input.dim});
    }
    _sdpa_attention_info.mask_idx = original_model->get_parameter_index(sdpa_attn._mask);
    _sdpa_attention_info.query_size = sdpa_attn.query_len();

    // Copy SDPA input index mapping from function HFA (already built in from() method)
    _sdpa_attention_info.sdpa_param_index_map = func_hfa._sdpa_param_index_map;

    // Copy HFA Tile Model input index mapping from function HFA
    _sdpa_attention_info.tile_param_index_map = func_hfa._tile_param_index_map;

    LOG_INFO("Extracted HFA config: tile_size=" << _tile_size << ", kv_cache_size=" << _kv_cache_size);
    LOG_INFO("Extracted " << _sdpa_attention_info.params.size() << " past KV parameters from original SDPA model");
    LOG_INFO("Copied SDPA input mapping with " << _sdpa_attention_info.sdpa_param_index_map.size() << " entries");
    LOG_INFO("Copied Tile input mapping with " << _sdpa_attention_info.tile_param_index_map.size() << " entries");

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
