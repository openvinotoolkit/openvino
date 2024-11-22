// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "spatial.hpp"

#include "util.hpp"

ov::npuw::runtime::spatial::AttentionMask::AttentionMask(std::size_t param_idx, const ov::ISyncInferRequest& rq)
    : m_attn_mask_param_idx(param_idx),
      m_rq(rq) {}

ov::npuw::runtime::spatial::Selector::Ptr ov::npuw::runtime::spatial::AttentionMask::find(
    const ov::ISyncInferRequest& rq) {
    auto is_attn_mask = [](const ov::Output<const ov::Node>& p) {
        const auto shape = p.get_shape();
        return p.get_node()->get_friendly_name() == "attention_mask" &&
               (shape.size() == 1 || (shape.size() == 2 && shape[0] == 1));
    };

    const auto& inputs = rq.get_inputs();
    auto attn_mask_iter = std::find_if(inputs.begin(), inputs.end(), is_attn_mask);
    if (attn_mask_iter != inputs.end()) {
        const auto param_idx = std::distance(inputs.begin(), attn_mask_iter);
        return Selector::Ptr{new AttentionMask(param_idx, rq)};
    }
    return Selector::Ptr{};
}

void ov::npuw::runtime::spatial::AttentionMask::prepare() {
    // Find the current valid range for this attention mask
    // Here we have the following (very strong) assumption:
    // The attention mask is dense (that is, has zero or one continuous interest region)
    const auto& iport = m_rq.get_compiled_model()->inputs()[m_attn_mask_param_idx];
    std::tie(m_valid_range_begin, m_valid_range_end) = ov::npuw::util::validMaskRange(m_rq.get_tensor(iport));
}

bool ov::npuw::runtime::spatial::AttentionMask::need_submit(std::size_t offset, std::size_t len) const {
    // We don't submit this request if
    // - it is completely below the valid range
    // - it is completely above the valid range
    // in all other cases, we do
    return !(offset + len < m_valid_range_begin || offset >= m_valid_range_end);
}
