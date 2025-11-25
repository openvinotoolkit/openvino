// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "host_flash_attention.hpp"

#include "logging.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/openvino.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "util.hpp"

namespace ov {
namespace npuw {
namespace function {

namespace opp = ov::pass::pattern;

// Helper function to create a single HFA tile computation
// Implements the flash attention tile algorithm from hfa.py::ov_hfa_tile
static std::shared_ptr<ov::Model> create_hfa_tile_model(const ov::Shape& q_shape,
                                                        const ov::element::Type& dtype,
                                                        int64_t tile_size) {
    LOG_DEBUG("Creating HFA tile model with tile_size=" << tile_size);

    // Extract dimensions from Q shape [batch, num_heads, seq_len, head_dim]
    NPUW_ASSERT(q_shape.size() == 4);
    auto batch = q_shape[0];
    auto num_heads = q_shape[1];
    auto seq_len = q_shape[2];
    auto head_dim = q_shape[3];

    // Input parameters for HFA tile
    // past_acc: [batch, num_heads, seq_len, head_dim]
    auto in_past_acc = std::make_shared<ov::op::v0::Parameter>(dtype, ov::Shape{batch, num_heads, seq_len, head_dim});
    in_past_acc->set_friendly_name("past_acc");

    // past_max: [batch, num_heads, seq_len, 1]
    auto in_past_max = std::make_shared<ov::op::v0::Parameter>(dtype, ov::Shape{batch, num_heads, seq_len, 1});
    in_past_max->set_friendly_name("past_max");

    // past_d: [batch, num_heads, seq_len, 1]
    auto in_past_d = std::make_shared<ov::op::v0::Parameter>(dtype, ov::Shape{batch, num_heads, seq_len, 1});
    in_past_d->set_friendly_name("past_d");

    // k_tile: [batch, num_heads, tile_size, head_dim]
    auto in_k_tile =
        std::make_shared<ov::op::v0::Parameter>(dtype,
                                                ov::Shape{batch, num_heads, static_cast<size_t>(tile_size), head_dim});
    in_k_tile->set_friendly_name("k_tile");

    // v_tile: [batch, num_heads, head_dim, tile_size]
    auto in_v_tile =
        std::make_shared<ov::op::v0::Parameter>(dtype,
                                                ov::Shape{batch, num_heads, head_dim, static_cast<size_t>(tile_size)});
    in_v_tile->set_friendly_name("v_tile");

    // q: [batch, num_heads, seq_len, head_dim]
    auto in_q = std::make_shared<ov::op::v0::Parameter>(dtype, ov::Shape{batch, num_heads, seq_len, head_dim});
    in_q->set_friendly_name("q");

    // mask_tile: [batch, 1, seq_len, tile_size]
    auto in_mask_tile =
        std::make_shared<ov::op::v0::Parameter>(dtype, ov::Shape{batch, 1, seq_len, static_cast<size_t>(tile_size)});
    in_mask_tile->set_friendly_name("mask_tile");

    // Flash Attention Tile Algorithm (from hfa.py::ov_hfa_tile):
    // qk = matmul(q, k^T)
    auto qk = std::make_shared<ov::op::v0::MatMul>(in_q, in_k_tile, false, true);
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

    // acc = past_acc * alpha + matmul(p, v)
    auto past_acc_alpha = std::make_shared<ov::op::v1::Multiply>(in_past_acc, alpha);
    auto pv = std::make_shared<ov::op::v0::MatMul>(p, in_v_tile, false, true);
    pv->set_friendly_name("pv");
    auto acc = std::make_shared<ov::op::v1::Add>(past_acc_alpha, pv);
    acc->set_friendly_name("acc");

    // Create results
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

    LOG_DEBUG("HFA tile model created successfully");
    return tile_model;
}

std::optional<HostFlashAttention> HostFlashAttention::from(const std::shared_ptr<ov::Model>& model) {
    LOG_INFO("Attempting to create HostFlashAttention from model");
    LOG_BLOCK();

    // Pattern matching to find decomposed SDPA components
    // Look for the key MatMul operations in the SDPA pattern

    std::shared_ptr<ov::Node> q_input = nullptr;
    std::shared_ptr<ov::Node> k_concat = nullptr;
    std::shared_ptr<ov::Node> v_concat = nullptr;
    std::shared_ptr<ov::Node> softmax_node = nullptr;

    // Find the Softmax node (key indicator of attention)
    for (const auto& node : model->get_ordered_ops()) {
        if (auto sm = std::dynamic_pointer_cast<ov::op::v8::Softmax>(node)) {
            softmax_node = sm;
            LOG_DEBUG("Found Softmax node: " << sm->get_friendly_name());

            // Trace back to find Q, K inputs
            // Pattern: MatMul(Q, K) -> Add(mask) -> Softmax
            if (sm->get_input_size() > 0) {
                auto add_node = sm->get_input_node_shared_ptr(0);
                if (auto add_op = std::dynamic_pointer_cast<ov::op::v1::Add>(add_node)) {
                    LOG_DEBUG("Found Add node before Softmax");

                    // Get MatMul(Q, K)
                    if (add_op->get_input_size() > 0) {
                        auto matmul_qk = add_op->get_input_node_shared_ptr(0);
                        if (auto mm = std::dynamic_pointer_cast<ov::op::v0::MatMul>(matmul_qk)) {
                            LOG_DEBUG("Found MatMul(Q,K) node");
                            q_input = mm->get_input_node_shared_ptr(0);

                            // K should come from Reshape <- Broadcast <- Unsqueeze <- Concat
                            auto k_path = mm->get_input_node_shared_ptr(1);
                            while (k_path && !std::dynamic_pointer_cast<ov::op::v0::Concat>(k_path)) {
                                if (k_path->get_input_size() > 0) {
                                    k_path = k_path->get_input_node_shared_ptr(0);
                                } else {
                                    break;
                                }
                            }
                            if (auto concat = std::dynamic_pointer_cast<ov::op::v0::Concat>(k_path)) {
                                k_concat = concat;
                                LOG_DEBUG("Found K Concat node");
                            }
                        }
                    }
                }
            }

            // Trace forward to find V
            // Pattern: Softmax -> MatMul(S, V)
            for (const auto& output : sm->outputs()) {
                for (const auto& target : output.get_target_inputs()) {
                    auto consumer = target.get_node()->shared_from_this();
                    if (auto mm_sv = std::dynamic_pointer_cast<ov::op::v0::MatMul>(consumer)) {
                        LOG_DEBUG("Found MatMul(S,V) node");

                        // V should be the second input
                        auto v_path = mm_sv->get_input_node_shared_ptr(1);
                        while (v_path && !std::dynamic_pointer_cast<ov::op::v0::Concat>(v_path)) {
                            if (v_path->get_input_size() > 0) {
                                v_path = v_path->get_input_node_shared_ptr(0);
                            } else {
                                break;
                            }
                        }
                        if (auto concat = std::dynamic_pointer_cast<ov::op::v0::Concat>(v_path)) {
                            v_concat = concat;
                            LOG_DEBUG("Found V Concat node");
                        }
                    }
                }
            }
            break;
        }
    }

    if (!q_input || !k_concat || !v_concat || !softmax_node) {
        LOG_WARN("Failed to identify decomposed SDPA pattern");
        std::cout << "HostFlashAttention::from - pattern not found" << std::endl;
        return std::nullopt;
    }

    LOG_INFO("Successfully identified decomposed SDPA pattern");

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
    if (k_concat->get_output_partial_shape(0).is_static()) {
        auto k_full_shape = k_concat->get_output_partial_shape(0).to_shape();
        // K shape after concat: [batch, num_heads, kv_cache_size, head_dim]
        if (k_full_shape.size() >= 3) {
            kv_cache_size = k_full_shape[2];
            LOG_DEBUG("Detected KV cache size: " << kv_cache_size);
        }
    }

    if (kv_cache_size == 0) {
        LOG_WARN("Failed to determine KV cache size");
        return std::nullopt;
    }

    // Create HFA tile model
    constexpr int64_t DEFAULT_TILE_SIZE = 1024;
    auto tile_model = create_hfa_tile_model(q_shape_static, dtype, DEFAULT_TILE_SIZE);

    if (!tile_model) {
        LOG_WARN("Failed to create HFA tile model");
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

    // Create HostFlashAttention structure
    HostFlashAttention hfa;
    hfa._tile_model = tile_model;
    hfa._tile_size = DEFAULT_TILE_SIZE;
    hfa._kv_cache_size = kv_cache_size;

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

    // TODO: Extract metadata from function::HostFlashAttention
    LOG_WARN("compiled::HostFlashAttention constructor is not yet implemented");
}

}  // namespace compiled

}  // namespace npuw
}  // namespace ov
