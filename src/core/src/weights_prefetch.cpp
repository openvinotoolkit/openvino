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
      m_min_bytes(min_bytes),
      m_next(0),
      m_in_flight(0),
      m_in_flight_bytes(0) {
    m_constants.reserve(ops.size());
    for (const auto& op : ops) {
        if (auto constant = ov::as_type_ptr<ov::op::v0::Constant>(op);
            constant && m_indices.emplace(constant.get(), m_constants.size()).second) {
            m_sizes.push_back(constant->get_byte_size());
            m_constants.push_back(std::move(constant));
        }
    }
    m_states.assign(m_constants.size(), State::pending);
    m_futures.resize(m_constants.size());
    prefetch_ahead();
}

std::optional<size_t> OrderedPrefetchScheduler::index_of(const ov::Node& node) const noexcept {
    // The lock check guards against a dead constant whose address was reused by another node.
    if (const auto found = m_indices.find(&node);
        found != m_indices.end() && m_constants[found->second].lock().get() == &node) {
        return found->second;
    }
    return std::nullopt;
}

void OrderedPrefetchScheduler::acquire(const ov::Node& node) noexcept {
    if (const auto index = index_of(node); index && m_states[*index] == State::pending) {
        // Consumed ahead of the planned order: still load it in the background, on several threads.
        m_futures[*index] =
            ov::weight_sharing::Extension::hint_prefetch(static_cast<const ov::op::v0::Constant&>(node));
        if (m_futures[*index].valid()) {
            m_states[*index] = State::prefetch_requested;
            ++m_in_flight;
            m_in_flight_bytes += m_sizes[*index];
        }
    }
}

void OrderedPrefetchScheduler::release(const ov::Node& node) noexcept {
    if (const auto index = index_of(node)) {
        if (m_states[*index] == State::prefetch_requested) {
            if (m_futures[*index].valid()) {
                m_futures[*index].wait();
            }
            --m_in_flight;
            m_in_flight_bytes -= m_sizes[*index];
        }
        m_states[*index] = State::released;
        m_futures[*index] = {};
        // Alive: the caller holds node and index_of() verified it.
        ov::weight_sharing::Extension::hint_evict(*m_constants[*index].lock());
        prefetch_ahead();
    }
}

void OrderedPrefetchScheduler::prefetch_ahead() noexcept {
    for (; (m_in_flight < m_lookahead || m_in_flight_bytes < m_min_bytes) && m_next < m_constants.size(); ++m_next) {
        if (m_states[m_next] == State::pending) {
            if (const auto constant = m_constants[m_next].lock()) {
                m_futures[m_next] = ov::weight_sharing::Extension::hint_prefetch(*constant);
                if (m_futures[m_next].valid()) {
                    m_states[m_next] = State::prefetch_requested;
                    ++m_in_flight;
                    m_in_flight_bytes += m_sizes[m_next];
                }
            } else {
                m_states[m_next] = State::released;
            }
        }
    }
}

}  // namespace ov::util
