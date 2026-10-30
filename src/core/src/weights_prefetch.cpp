// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/weights_prefetch.hpp"

#include <algorithm>
#include <cstring>
#include <tuple>
#include <utility>

#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/weight_sharing_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/runtime/make_tensor.hpp"

namespace ov::util {

SequentialPrefetchScheduler::SequentialPrefetchScheduler(ov::Tensor tensor, size_t window, size_t step)
    : m_tensor(std::move(tensor)),
      m_size(m_tensor.get_byte_size()),
      m_window(window),
      m_step(std::max<size_t>(step, 1)) {
    OPENVINO_ASSERT(m_tensor.get_element_type() == ov::element::u8 && m_tensor.get_shape().size() == 1,
                    "Prefetch tensor must be 1D u8");
    advance(0, 0);
}

std::shared_ptr<IMemoryHints> SequentialPrefetchScheduler::chunk_hints(size_t begin, size_t end) const {
    return ov::get_memory_hints(ov::Tensor{m_tensor, ov::Coordinate{begin}, ov::Coordinate{end}});
}

void SequentialPrefetchScheduler::advance(size_t begin, size_t end) noexcept {
    try {
        begin = std::min(begin, m_size);
        end = std::min(std::max(begin, end), m_size);
        if (m_evict) {
            // Whole steps only, the read position may be in the middle of a page.
            const auto evict_end = begin == m_size ? m_size : begin / m_step * m_step;
            if (evict_end > m_evicted) {
                if (const auto hints = chunk_hints(m_evicted, evict_end)) {
                    hints->hint_evict();
                }
                m_evicted = evict_end;
            }
        }
        m_prefetched = std::max(m_prefetched, begin);
        const auto target = std::min(end + m_window, m_size);
        while (m_prefetched < target) {
            const auto chunk_end = std::min(m_prefetched + m_step, m_size);
            if (const auto hints = chunk_hints(m_prefetched, chunk_end)) {
                std::ignore = hints->hint_prefetch();
            }
            m_prefetched = chunk_end;
        }
    } catch (...) {
        // Hints are advisory, the data is read correctly without them.
    }
}

PrefetchStreamBuf::PrefetchStreamBuf(const ov::Tensor& tensor, bool prefetch) {
    const auto begin = static_cast<char*>(const_cast<void*>(tensor.data()));
    setg(begin, begin, begin + tensor.get_byte_size());
    if (prefetch && ov::get_memory_hints(tensor)) {
        m_scheduler = std::make_unique<SequentialPrefetchScheduler>(tensor);
    }
}

std::streamsize PrefetchStreamBuf::xsgetn(char_type* dst, std::streamsize count) {
    const auto n = std::min<std::streamsize>(count, egptr() - gptr());
    if (n <= 0) {
        return 0;
    }
    const auto pos = static_cast<size_t>(gptr() - eback());
    if (m_scheduler) {
        m_scheduler->advance(pos, pos + static_cast<size_t>(n));
    }
    std::memcpy(dst, gptr(), static_cast<size_t>(n));
    // gbump() takes int, reads may exceed 2 GB.
    setg(eback(), gptr() + n, egptr());
    return n;
}

PrefetchStreamBuf::pos_type PrefetchStreamBuf::seekoff(off_type off,
                                                       std::ios_base::seekdir dir,
                                                       std::ios_base::openmode which) {
    if (!(which & std::ios_base::in)) {
        return pos_type(off_type(-1));
    }
    const off_type base = dir == std::ios_base::beg   ? 0
                          : dir == std::ios_base::cur ? gptr() - eback()
                                                      : egptr() - eback();
    const off_type pos = base + off;
    if (pos < 0 || pos > egptr() - eback()) {
        return pos_type(off_type(-1));
    }
    setg(eback(), eback() + pos, egptr());
    return pos_type(pos);
}

PrefetchStreamBuf::pos_type PrefetchStreamBuf::seekpos(pos_type pos, std::ios_base::openmode which) {
    return seekoff(off_type(pos), std::ios_base::beg, which);
}

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
                m_states[m_next] = State::prefetch_requested;
                ++m_in_flight;
                m_in_flight_bytes += m_sizes[m_next];
            } else {
                m_states[m_next] = State::released;
            }
        }
    }
}

}  // namespace ov::util
