// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "attention.hpp"

#include "util.hpp"

ov::npuw::runtime::attention::PositionIDs::PositionIDs(std::size_t param_idx, const ov::npuw::compiled::Attention &d, const ov::ISyncInferRequest& rq)
    : m_position_ids_idx(param_idx),
      m_d(d),
      m_rq(rq) {
    // FIXME: speculative decode is indistinguishable at this point!
    m_case = m_d.query_size == 1 ? Case::GENERATE : Case::PREFILL;
}

ov::npuw::runtime::attention::Selector::Ptr ov::npuw::runtime::attention::PositionIDs::find(
    const ov::npuw::compiled::Attention &d,
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
        return Selector::Ptr{new PositionIDs(param_idx, d, rq)};
    }
    return Selector::Ptr{};
}

void ov::npuw::runtime::attention::PositionIDs::prepare() {
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
                m_past_length = (m_current_length / m_d.query_size) * m_d.query_size;
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

int64_t ov::npuw::runtime::attention::PositionIDs::length() const {
    return m_current_length;
}

int64_t ov::npuw::runtime::attention::PositionIDs::past_length() const {
    return m_past_length;
}
