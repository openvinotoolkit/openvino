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
    m_entries.reserve(ops.size());
    for (const auto& op : ops) {
        if (auto constant = ov::as_type_ptr<ov::op::v0::Constant>(op);
            constant && m_indices.emplace(constant.get(), m_entries.size()).second) {
            m_entries.push_back({constant, constant->get_byte_size()});
        }
    }
    prefetch_ahead();
}

std::optional<size_t> OrderedPrefetchScheduler::index_of(const ov::Node& node) const noexcept {
    // The lock check guards against a dead constant whose address was reused by another node.
    if (const auto found = m_indices.find(&node);
        found != m_indices.end() && m_entries[found->second].constant.lock().get() == &node) {
        return found->second;
    }
    return std::nullopt;
}

void OrderedPrefetchScheduler::prefetch(Entry& entry, const ov::op::v0::Constant& constant) noexcept {
    entry.future = ov::weight_sharing::Extension::hint_prefetch(constant);
    if (entry.future.valid()) {
        entry.state = State::prefetch_requested;
        ++m_in_flight;
        m_in_flight_bytes += entry.size;
    }
}

void OrderedPrefetchScheduler::acquire(const ov::Node& node) noexcept {
    if (const auto index = index_of(node); index && m_entries[*index].state == State::pending) {
        // Consumed ahead of the planned order: still load it in the background, on several threads.
        prefetch(m_entries[*index], static_cast<const ov::op::v0::Constant&>(node));
    }
}

void OrderedPrefetchScheduler::release(const ov::Node& node) noexcept {
    if (const auto index = index_of(node)) {
        auto& entry = m_entries[*index];
        if (entry.state == State::prefetch_requested) {
            entry.future.wait();
            --m_in_flight;
            m_in_flight_bytes -= entry.size;
        }
        entry.state = State::released;
        entry.future = {};
        // Alive: the caller holds node and index_of() verified it.
        ov::weight_sharing::Extension::hint_evict(*entry.constant.lock());
        prefetch_ahead();
    }
}

void OrderedPrefetchScheduler::prefetch_ahead() noexcept {
    for (; (m_in_flight < m_lookahead || m_in_flight_bytes < m_min_bytes) && m_next < m_entries.size(); ++m_next) {
        if (auto& entry = m_entries[m_next]; entry.state == State::pending) {
            if (const auto constant = entry.constant.lock()) {
                prefetch(entry, *constant);
            } else {
                entry.state = State::released;
            }
        }
    }
}

}  // namespace ov::util
