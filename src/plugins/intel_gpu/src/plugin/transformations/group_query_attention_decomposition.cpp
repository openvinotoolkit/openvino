// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "group_query_attention_decomposition.hpp"

#include "intel_gpu/op/sdpa.hpp"

namespace ov::intel_gpu {

std::shared_ptr<ov::Node> GroupQueryAttentionDecomposition::make_sdpa(const ov::Output<ov::Node>& query,
                                                                      const ov::Output<ov::Node>& key,
                                                                      const ov::Output<ov::Node>& value,
                                                                      const ov::Output<ov::Node>& mask,
                                                                      const ov::Output<ov::Node>& scale,
                                                                      const ov::Output<ov::Node>& sink,
                                                                      bool is_causal) {
    ov::OutputVector inputs{query, key, value};
    if (mask.get_node()) {
        inputs.push_back(mask);
    }
    if (scale.get_node()) {
        inputs.push_back(scale);
    }
    if (sink.get_node()) {
        inputs.push_back(sink);
    }

    const auto order = op::SDPA::default_order(query.get_partial_shape().rank().get_length());
    return register_new_node<op::SDPA>(inputs,
                                       is_causal,
                                       order,
                                       order,
                                       order,
                                       order,
                                       ov::element::dynamic,
                                       is_causal ? op::SDPA::CausalMaskAlignment::LOWER_RIGHT : op::SDPA::CausalMaskAlignment::UPPER_LEFT);
}

std::shared_ptr<ov::Node> GroupQueryAttentionDecomposition::make_attention_mask(const ov::Output<ov::Node>& curr_seqlen_scalar,
                                                                                const ov::Output<ov::Node>& kv_len_scalar,
                                                                                const ov::Output<ov::Node>& kv_len_1d,
                                                                                const ov::Output<ov::Node>& past_seqlen,
                                                                                const ov::element::Type& compute_type,
                                                                                bool causal,
                                                                                int64_t local_window_size,
                                                                                const ov::Output<ov::Node>& external_bias,
                                                                                const ov::Output<ov::Node>& bias_col_offset,
                                                                                bool sliding_window_cache,
                                                                                float scale,
                                                                                bool has_sink) {
    if (causal && local_window_size == -1 && !sliding_window_cache && !external_bias.get_node() && scale == 0.0f && !has_sink) {
        return nullptr;
    }

    return ov::pass::GroupQueryAttentionDecomposition::make_attention_mask(curr_seqlen_scalar,
                                                                           kv_len_scalar,
                                                                           kv_len_1d,
                                                                           past_seqlen,
                                                                           compute_type,
                                                                           causal,
                                                                           local_window_size,
                                                                           external_bias,
                                                                           bias_col_offset,
                                                                           sliding_window_cache,
                                                                           scale,
                                                                           has_sink);
}

}  // namespace ov::intel_gpu