// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/weights_prefetch.hpp"

#include <charconv>
#include <cstdlib>
#include <limits>
#include <string_view>
#include <system_error>
#include <utility>

#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/weight_sharing_util.hpp"
#include "openvino/op/constant.hpp"

namespace ov::weight_sharing {

PrefetchScheduler::PrefetchScheduler(const std::vector<std::shared_ptr<ov::Node>>& ops, size_t lookahead)
    : m_lookahead(lookahead) {
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

void PrefetchScheduler::acquire(const ov::op::v0::Constant& constant) noexcept {
    const auto found = m_indices.find(&constant);
    if (found == m_indices.end()) {
        return;
    }
    const auto index = found->second;
    if (m_states[index] == State::pending) {
        // Consumed ahead of the planned order: still load it in the background, on several threads.
        m_futures[index] = Extension::hint_prefetch(*m_constants[index]);
        m_states[index] = State::prefetched;
        ++m_in_flight;
    }
    if (m_futures[index].valid()) {
        m_futures[index].wait();
    }
}

void PrefetchScheduler::release(const ov::op::v0::Constant& constant) noexcept {
    const auto found = m_indices.find(&constant);
    if (found == m_indices.end() || m_states[found->second] == State::released) {
        return;
    }
    const auto index = found->second;
    if (m_states[index] == State::prefetched) {
        --m_in_flight;
    }
    m_states[index] = State::released;
    m_futures[index] = {};
    Extension::hint_evict(*m_constants[index]);
    prefetch_ahead();
}

void PrefetchScheduler::prefetch_ahead() noexcept {
    for (; m_in_flight < m_lookahead && m_next < m_constants.size(); ++m_next) {
        if (m_states[m_next] == State::pending) {
            m_futures[m_next] = Extension::hint_prefetch(*m_constants[m_next]);
            m_states[m_next] = State::prefetched;
            ++m_in_flight;
        }
    }
}

size_t PrefetchScheduler::get_lookahead() {
    const char* value = std::getenv("OV_WEIGHTS_PREFETCH_LOOKAHEAD");
    if (!value) {
        return 0;
    }
    const std::string_view text(value);
    if (text == "all") {
        return std::numeric_limits<size_t>::max();
    }
    size_t lookahead = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), lookahead);
    return error == std::errc{} && end == text.data() + text.size() ? lookahead : 0;
}

}  // namespace ov::weight_sharing
