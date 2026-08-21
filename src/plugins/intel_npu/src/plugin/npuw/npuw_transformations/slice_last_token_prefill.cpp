// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "slice_last_token_prefill.hpp"

#include "../logging.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/transpose.hpp"

namespace ov::npuw {

SliceLastTokenPrefill::SliceLastTokenPrefill(uint32_t batch_dim, uint32_t num_last_tokens)
    : m_batch_dim(batch_dim),
      m_num_last_tokens(num_last_tokens < 1u ? 1u : num_last_tokens) {}

bool SliceLastTokenPrefill::run_on_model(const std::shared_ptr<ov::Model>& model) {
    // ================================================================== //
    // PHASE 1: Discovery — find and validate all required nodes.          //
    // No graph modifications are made here.  If anything is unexpected,   //
    // we return false without touching the model.                         //
    // ================================================================== //

    // 1a. Find the last ScaledDotProductAttention in topological order
    //     (last in traversal = last transformer layer).
    std::shared_ptr<ov::op::v13::ScaledDotProductAttention> last_sdpa;
    for (auto& node : model->get_ordered_ops()) {
        if (auto sdpa = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(node)) {
            last_sdpa = sdpa;
        }
    }
    if (!last_sdpa) {
        LOG_DEBUG("SliceLastTokenPrefill: no ScaledDotProductAttention found, skipping");
        return false;
    }

    // 1b. Validate Q shape.  Q is SDPA input port 0: [B, H, N, D_head].
    const auto q_val = last_sdpa->input_value(0);
    const auto q_shape = q_val.get_partial_shape();
    if (q_shape.is_dynamic() || q_shape.rank().get_length() != 4) {
        LOG_WARN("SliceLastTokenPrefill: Q is dynamic or not 4D, skipping");
        return false;
    }
    constexpr int64_t Q_SEQ_DIM = 2;  // [B, H, N, D]
    const int64_t seq_len = q_shape[Q_SEQ_DIM].get_length();
    const int64_t num_last = static_cast<int64_t>(m_num_last_tokens);
    if (seq_len <= num_last) {
        LOG_DEBUG("SliceLastTokenPrefill: seq_len <= num_last_tokens, nothing to slice");
        return false;
    }

    // 1c. Check mask (optional; we proceed even if absent/wrong rank).
    int64_t mask_seq_len = -1;
    if (last_sdpa->get_input_size() > 3) {
        const auto mask_shape = last_sdpa->input_value(3).get_partial_shape();
        if (!mask_shape.is_dynamic() && mask_shape.rank().get_length() == 4) {
            mask_seq_len = mask_shape[2].get_length();
        }
    }

    // 1d. Downstream path: SDPA → Transpose → Reshape.
    auto find_single_consumer = [](const ov::Output<ov::Node>& out, auto type_pred) -> std::shared_ptr<ov::Node> {
        for (auto& ti : out.get_target_inputs()) {
            auto n = ti.get_node()->shared_from_this();
            if (type_pred(n))
                return n;
        }
        return nullptr;
    };

    auto transpose_node = find_single_consumer(last_sdpa->output(0), [](const std::shared_ptr<ov::Node>& n) {
        return ov::is_type<ov::op::v1::Transpose>(n);
    });
    if (!transpose_node) {
        LOG_WARN("SliceLastTokenPrefill: Transpose not found after SDPA");
        return false;
    }

    auto reshape_node = find_single_consumer(transpose_node->output(0), [](const std::shared_ptr<ov::Node>& n) {
        return ov::is_type<ov::op::v1::Reshape>(n);
    });
    if (!reshape_node) {
        LOG_WARN("SliceLastTokenPrefill: Reshape not found after Transpose");
        return false;
    }

    // 1e. Validate Reshape shape constant.
    auto shape_const = ov::as_type_ptr<ov::op::v0::Constant>(reshape_node->input_value(1).get_node_shared_ptr());
    if (!shape_const) {
        LOG_WARN("SliceLastTokenPrefill: Reshape shape input is not a Constant");
        return false;
    }
    const size_t reshape_seq_idx = 1u - m_batch_dim;
    {
        const auto& cshape = shape_const->get_shape();
        if (cshape.size() != 1 || cshape[0] < reshape_seq_idx + 1) {
            LOG_WARN("SliceLastTokenPrefill: Reshape constant shape unexpected");
            return false;
        }
    }

    // 1f. Walk forward: Reshape → o_proj (MatMul) → residual Add.
    constexpr int MAX_HOPS = 8;
    auto find_downstream =
        [](const std::shared_ptr<ov::Node>& start, auto type_pred, int max_hops) -> std::shared_ptr<ov::Node> {
        auto cur = start;
        for (int hop = 0; hop < max_hops; ++hop) {
            std::shared_ptr<ov::Node> next;
            for (auto& ti : cur->output(0).get_target_inputs()) {
                next = ti.get_node()->shared_from_this();
                break;
            }
            if (!next)
                break;
            if (type_pred(next))
                return next;
            cur = next;
        }
        return nullptr;
    };

    auto o_proj = find_downstream(
        reshape_node,
        [](const std::shared_ptr<ov::Node>& n) {
            return ov::is_type<ov::op::v0::MatMul>(n);
        },
        MAX_HOPS);
    if (!o_proj) {
        LOG_WARN("SliceLastTokenPrefill: o_proj MatMul not found after Reshape");
        return false;
    }

    auto residual_add = find_downstream(
        o_proj,
        [](const std::shared_ptr<ov::Node>& n) {
            return ov::is_type<ov::op::v1::Add>(n);
        },
        MAX_HOPS);
    if (!residual_add) {
        LOG_WARN("SliceLastTokenPrefill: residual Add not found after o_proj");
        return false;
    }

    // 1g. Identify the shortcut input of the residual Add.
    int shortcut_port = -1;
    for (int i = 0; i < static_cast<int>(residual_add->get_input_size()); ++i) {
        auto cur = residual_add->input_value(i).get_node_shared_ptr();
        bool from_o_proj = false;
        for (int h = 0; h < MAX_HOPS; ++h) {
            if (cur.get() == o_proj.get()) {
                from_o_proj = true;
                break;
            }
            if (cur->get_input_size() == 0)
                break;
            cur = cur->input_value(0).get_node_shared_ptr();
        }
        if (!from_o_proj) {
            shortcut_port = i;
            break;
        }
    }
    if (shortcut_port < 0) {
        LOG_WARN("SliceLastTokenPrefill: could not identify shortcut input of residual Add");
        return false;
    }

    // 1h. Validate shortcut shape.
    const auto sc_val = residual_add->input_value(shortcut_port);
    const auto sc_shape = sc_val.get_partial_shape();
    const int64_t sc_seq_dim = 1 - static_cast<int64_t>(m_batch_dim);
    int64_t sc_seq_len = -1;
    if (!sc_shape.is_dynamic() && sc_shape.rank().get_length() == 3) {
        sc_seq_len = sc_shape[sc_seq_dim].get_length();
    }
    if (sc_seq_len <= 0) {
        LOG_WARN("SliceLastTokenPrefill: shortcut shape unexpected, skipping");
        return false;
    }

    // ================================================================== //
    // PHASE 2: Apply — all nodes validated; now modify the graph.        //
    // ================================================================== //

    auto make_last_tokens_slice = [&num_last](const ov::Output<ov::Node>& src, int64_t dim, int64_t total) {
        auto start = ov::op::v0::Constant::create(ov::element::i64, {1}, {total - num_last});
        auto stop = ov::op::v0::Constant::create(ov::element::i64, {1}, {total});
        auto step = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {dim});
        return std::make_shared<ov::op::v8::Slice>(src, start, stop, step, axes);
    };

    // 2a. Slice Q: [B, H, N, D] → [B, H, K, D].
    auto q_slice = make_last_tokens_slice(q_val, Q_SEQ_DIM, seq_len);
    last_sdpa->input(0).replace_source_output(q_slice->output(0));

    // 2b. Slice mask (if present): [B, 1, N, S] → [B, 1, K, S].
    if (mask_seq_len > 0) {
        const auto mask_val = last_sdpa->input_value(3);
        auto mask_slice = make_last_tokens_slice(mask_val, 2, mask_seq_len);
        last_sdpa->input(3).replace_source_output(mask_slice->output(0));
    }

    // 2c. Update Reshape shape constant: N → K at seq dimension.
    if (shape_const->get_element_type() == ov::element::i32) {
        auto vals = shape_const->cast_vector<int32_t>();
        vals[reshape_seq_idx] = static_cast<int32_t>(num_last);
        auto new_const = ov::op::v0::Constant::create(ov::element::i32, shape_const->get_shape(), vals);
        reshape_node->input(1).replace_source_output(new_const->output(0));
    } else {
        auto vals = shape_const->cast_vector<int64_t>();
        vals[reshape_seq_idx] = num_last;
        auto new_const = ov::op::v0::Constant::create(ov::element::i64, shape_const->get_shape(), vals);
        reshape_node->input(1).replace_source_output(new_const->output(0));
    }

    // 2d. Slice shortcut: [B, N, hidden] → [B, K, hidden].
    auto sc_slice = make_last_tokens_slice(sc_val, sc_seq_dim, sc_seq_len);
    residual_add->input(shortcut_port).replace_source_output(sc_slice->output(0));

    // 2e. Re-infer shapes; downstream nodes (LayerNorm, FFN, final norm)
    //     propagate the new [B, K, ...] shapes automatically.
    model->validate_nodes_and_infer_types();

    LOG_DEBUG("SliceLastTokenPrefill: applied successfully");
    return true;
}

}  // namespace ov::npuw
