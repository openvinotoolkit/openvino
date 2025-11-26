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
                                                        const ov::element::Type& input_dtype,
                                                        int64_t tile_size) {
    LOG_DEBUG("Creating HFA tile model with tile_size=" << tile_size);

    // Extract dimensions from Q shape [batch, num_heads, seq_len, head_dim]
    NPUW_ASSERT(q_shape.size() == 4);
    auto batch = q_shape[0];
    auto num_heads = q_shape[1];
    auto seq_len = q_shape[2];
    auto head_dim = q_shape[3];

    // Use f32 for internal computation precision, but keep I/O in original dtype (f16)
    auto compute_dtype = ov::element::f32;

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

    // k_tile: [batch, num_heads, tile_size, head_dim]
    auto in_k_tile =
        std::make_shared<ov::op::v0::Parameter>(input_dtype,
                                                ov::Shape{batch, num_heads, static_cast<size_t>(tile_size), head_dim});
    in_k_tile->set_friendly_name("k_tile");
    in_k_tile->output(0).get_tensor().set_names({"k_tile"});

    // v_tile: [batch, num_heads, head_dim, tile_size]
    auto in_v_tile =
        std::make_shared<ov::op::v0::Parameter>(input_dtype,
                                                ov::Shape{batch, num_heads, head_dim, static_cast<size_t>(tile_size)});
    in_v_tile->set_friendly_name("v_tile");
    in_v_tile->output(0).get_tensor().set_names({"v_tile"});

    // q: [batch, num_heads, seq_len, head_dim]
    auto in_q = std::make_shared<ov::op::v0::Parameter>(input_dtype, ov::Shape{batch, num_heads, seq_len, head_dim});
    in_q->set_friendly_name("q");
    in_q->output(0).get_tensor().set_names({"q"});

    // mask_tile: [batch, 1, seq_len, tile_size] - f32 in Decomposed SDPA
    auto in_mask_tile =
        std::make_shared<ov::op::v0::Parameter>(compute_dtype,
                                                ov::Shape{batch, 1, seq_len, static_cast<size_t>(tile_size)});
    in_mask_tile->set_friendly_name("mask_tile");
    in_mask_tile->output(0).get_tensor().set_names({"mask_tile"});

    // Convert inputs to f32 for computation (mask_tile already f32)
    auto past_acc_f32 = std::make_shared<ov::op::v0::Convert>(in_past_acc, compute_dtype);
    auto past_max_f32 = std::make_shared<ov::op::v0::Convert>(in_past_max, compute_dtype);
    auto past_d_f32 = std::make_shared<ov::op::v0::Convert>(in_past_d, compute_dtype);
    auto k_tile_f32 = std::make_shared<ov::op::v0::Convert>(in_k_tile, compute_dtype);
    auto v_tile_f32 = std::make_shared<ov::op::v0::Convert>(in_v_tile, compute_dtype);
    auto q_f32 = std::make_shared<ov::op::v0::Convert>(in_q, compute_dtype);

    // Flash Attention Tile Algorithm (from hfa.py::ov_hfa_tile):
    // qk = matmul(q, k^T)
    auto qk = std::make_shared<ov::op::v0::MatMul>(q_f32, k_tile_f32, false, true);
    qk->set_friendly_name("qk");

    // qkm = qk + mask (mask_tile is already f32)
    auto qkm = std::make_shared<ov::op::v1::Add>(qk, in_mask_tile);
    qkm->set_friendly_name("qkm");

    // maxx = max(past_max, reduce_max(qkm, axis=-1, keepdims=True))
    auto axes_const = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
    auto qkm_max = std::make_shared<ov::op::v1::ReduceMax>(qkm, axes_const, true);
    qkm_max->set_friendly_name("qkm_max");

    auto maxx_f32 = std::make_shared<ov::op::v1::Maximum>(past_max_f32, qkm_max);
    maxx_f32->set_friendly_name("maxx_f32");

    // p = exp(qkm - maxx)
    auto qkm_sub_maxx = std::make_shared<ov::op::v1::Subtract>(qkm, maxx_f32);
    auto p = std::make_shared<ov::op::v0::Exp>(qkm_sub_maxx);
    p->set_friendly_name("p");

    // l = reduce_sum(p, axis=-1, keepdims=True)
    auto l = std::make_shared<ov::op::v1::ReduceSum>(p, axes_const, true);
    l->set_friendly_name("l");

    // alpha = exp(past_max - maxx)
    auto past_max_sub_maxx = std::make_shared<ov::op::v1::Subtract>(past_max_f32, maxx_f32);
    auto alpha = std::make_shared<ov::op::v0::Exp>(past_max_sub_maxx);
    alpha->set_friendly_name("alpha");

    // d = past_d * alpha + l
    auto past_d_alpha = std::make_shared<ov::op::v1::Multiply>(past_d_f32, alpha);
    auto d_f32 = std::make_shared<ov::op::v1::Add>(past_d_alpha, l);
    d_f32->set_friendly_name("d_f32");

    // acc = past_acc * alpha + matmul(p, v)
    auto past_acc_alpha = std::make_shared<ov::op::v1::Multiply>(past_acc_f32, alpha);
    auto pv = std::make_shared<ov::op::v0::MatMul>(p, v_tile_f32, false, true);
    pv->set_friendly_name("pv");
    auto acc_f32 = std::make_shared<ov::op::v1::Add>(past_acc_alpha, pv);
    acc_f32->set_friendly_name("acc_f32");

    // Convert outputs back to input_dtype (f16) for consistency
    auto acc = std::make_shared<ov::op::v0::Convert>(acc_f32, input_dtype);
    acc->set_friendly_name("acc");
    acc->output(0).get_tensor().set_names({"acc"});

    auto maxx = std::make_shared<ov::op::v0::Convert>(maxx_f32, input_dtype);
    maxx->set_friendly_name("maxx");
    maxx->output(0).get_tensor().set_names({"maxx"});

    auto d = std::make_shared<ov::op::v0::Convert>(d_f32, input_dtype);
    d->set_friendly_name("d");
    d->output(0).get_tensor().set_names({"d"});

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

    LOG_DEBUG("HFA tile model created successfully with I/O dtype=" << input_dtype
                                                                    << ", compute dtype=" << compute_dtype);
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

                            // Skip Convert nodes to get to the actual Parameter/input
                            while (q_input && std::dynamic_pointer_cast<ov::op::v0::Convert>(q_input)) {
                                if (q_input->get_input_size() > 0) {
                                    q_input = q_input->get_input_node_shared_ptr(0);
                                    LOG_DEBUG("Skipped Convert node, now at: " << q_input->get_friendly_name());
                                } else {
                                    break;
                                }
                            }

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

    // Extract tile configuration from function HFA
    _tile_size = func_hfa._tile_size;
    _kv_cache_size = func_hfa._kv_cache_size;

    // Store the tile model for later compilation
    _tile_model_to_compile = func_hfa._tile_model;

    LOG_INFO("Extracted HFA config: tile_size=" << _tile_size << ", kv_cache_size=" << _kv_cache_size);

    // Note: _compiled_tile_model will be set later by compile_host_flash_attention_model()
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
