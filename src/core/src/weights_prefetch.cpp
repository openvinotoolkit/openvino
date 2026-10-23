// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/weights_prefetch.hpp"

#include <utility>

#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/weight_sharing_util.hpp"
#include "openvino/op/constant.hpp"

namespace ov::util {

OrderedPrefetchScheduler::OrderedPrefetchScheduler(const std::vector<std::shared_ptr<ov::Node>>& ops,
                                                   size_t lookahead,
                                                   size_t min_bytes)
    : m_lookahead(lookahead),
      m_min_bytes(min_bytes) {
    m_constants.reserve(ops.size());
    for (const auto& op : ops) {
        if (auto constant = ov::as_type_ptr<ov::op::v0::Constant>(op);
            constant && m_indices.emplace(constant.get(), m_constants.size()).second) {
            m_constants.push_back(std::move(constant));
        }
    }
    m_states.assign(m_constants.size(), State::pending);
    m_futures.resize(m_constants.size());
    prefetch_ahead();
}

void OrderedPrefetchScheduler::acquire(const ov::Node& node) noexcept {
    const auto found = m_indices.find(&node);
    if (found == m_indices.end()) {
        return;
    }
    const auto index = found->second;
    if (m_states[index] == State::pending) {
        // Consumed ahead of the planned order: still load it in the background, on several threads.
        m_futures[index] = ov::weight_sharing::Extension::hint_prefetch(*m_constants[index]);
        m_states[index] = State::prefetched;
        ++m_in_flight;
        m_in_flight_bytes += m_constants[index]->get_byte_size();
    }
}

void OrderedPrefetchScheduler::release(const ov::Node& node) noexcept {
    const auto found = m_indices.find(&node);
    if (found == m_indices.end() || m_states[found->second] == State::released) {
        return;
    }
    const auto index = found->second;
    if (m_states[index] == State::prefetched) {
        --m_in_flight;
        m_in_flight_bytes -= m_constants[index]->get_byte_size();
    }
    m_states[index] = State::released;
    m_futures[index] = {};
    ov::weight_sharing::Extension::hint_evict(*m_constants[index]);
    prefetch_ahead();
}

void OrderedPrefetchScheduler::prefetch_ahead() noexcept {
    for (; (m_in_flight < m_lookahead || m_in_flight_bytes < m_min_bytes) && m_next < m_constants.size(); ++m_next) {
        if (m_states[m_next] == State::pending) {
            m_futures[m_next] = ov::weight_sharing::Extension::hint_prefetch(*m_constants[m_next]);
            m_states[m_next] = State::prefetched;
            ++m_in_flight;
            m_in_flight_bytes += m_constants[m_next]->get_byte_size();
        }
    }
}

}  // namespace ov::util
