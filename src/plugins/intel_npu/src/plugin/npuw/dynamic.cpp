// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "dynamic.hpp"

#include "util.hpp"

ov::npuw::runtime::dynamic::PositionIDs::PositionIDs(std::size_t param_idx, const ov::ISyncInferRequest& rq)
    : m_position_ids_idx(param_idx),
      m_rq(rq) {}

ov::npuw::runtime::dynamic::Selector::Ptr ov::npuw::runtime::dynamic::PositionIDs::find(
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
        return Selector::Ptr{new PositionIDs(param_idx, rq)};
    }
    return Selector::Ptr{};
}

void ov::npuw::runtime::dynamic::PositionIDs::prepare() {
    const auto& iport = m_rq.get_compiled_model()->inputs()[m_position_ids_idx];
    const auto in_tensor = m_rq.get_tensor(iport);
    const auto in_dims = in_tensor->get_shape();

    // There's several cases possible:
    // a. Prefill input_ids, including chunk
    // b. Generate input_ids, 1
    // c. Generate input_ids, N (speculative)
    // Prefill (even chunked) is left-padded, so for (a) it's enough to take the last element.
    // Same works for b (there's no choise).
    // c may require traversing the tensor backwards as Generate with N>1 is right_padded (?)

    auto* pos_data_ptr = in_tensor->data<int64_t>();
    for (auto idx = in_dims.back() - 1; idx >= 0; idx--) {
        if (pos_data_ptr[idx] > 0) {
            m_current_length = pos_data_ptr[idx];
            return;
        }
    }
    LOG_WARN("Dynamic selector - no data found in the feature?");
    m_current_length = -1;
}

int64_t ov::npuw::runtime::dynamic::PositionIDs::length() {
    return m_current_length;
}
