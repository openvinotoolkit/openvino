// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "openvino/core/core_visibility.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/util/memory.hpp"

namespace ov::util {

/** @brief Loads the weights before they are read and evicts them after use. */
class OPENVINO_API IPrefetchScheduler {
public:
    virtual ~IPrefetchScheduler() = default;

    /** @brief Requests asynchronous prefetch of the node's weights if not already requested. Call it right before the
     * node is read. */
    virtual void acquire(const ov::Node& node) noexcept = 0;

    /** @brief Evicts the node weights and prefetches the next ones. Call it once the node is not read. */
    virtual void release(const ov::Node& node) noexcept = 0;
};

/**
 * @brief Loads the constants ahead of their consumer and evicts them after use.
 *
 * Fits a plugin that reads each constant once, in the order of ops. With lookahead = 2 and constants A B C D:
 * prefetch A, B; acquire A, release A (evict A, prefetch C); acquire B, release B (evict B, prefetch D); ...
 *
 * Non-constant nodes and constants not backed by a file mapping are ignored. Not thread safe.
 *
 * Entry states:
 * @verbatim
   pending ---(1)---> prefetch_requested ---(2)---> released
      |                                                ^
      +----------------------(3)-----------------------+
   @endverbatim
 * (1) prefetch_ahead() or acquire(), only if hint_prefetch() started loading (constant is file-backed).
 * (2) release(): waits for the load to finish, then evicts.
 * (3) release() of a constant that was never prefetched, or prefetch_ahead() finds the constant destroyed.
 *
 * released is final: the entry is never prefetched again.
 */
class OPENVINO_API OrderedPrefetchScheduler : public IPrefetchScheduler {
public:
    /**
     * @param ops Nodes in the order they are consumed; only the constants among them are tracked,
     *            duplicates are ignored.
     * @param lookahead Minimum number of constants prefetched and not released yet.
     * @param min_bytes More constants are prefetched while fewer bytes than this are prefetched and not released yet.
     *
     * Default lookahead and min_bytes showed the best compile time in manual testing.
     */
    explicit OrderedPrefetchScheduler(const std::vector<std::shared_ptr<ov::Node>>& ops,
                                      size_t lookahead = 2,
                                      size_t min_bytes = 32 * one_mib);

    void acquire(const ov::Node& node) noexcept override;
    void release(const ov::Node& node) noexcept override;

private:
    enum class State { pending, prefetch_requested, released };

    struct Entry {
        std::weak_ptr<ov::op::v0::Constant> constant;  // Does not keep the constant alive.
        size_t size;                                   // Byte size, cached since weak_ptr may expire.
        State state = State::pending;
        PrefetchToken future;  // Valid while state is prefetch_requested.
    };

    void prefetch(Entry& entry, const ov::op::v0::Constant& constant) noexcept;
    void prefetch_ahead() noexcept;
    std::optional<size_t> index_of(const ov::Node& node) const noexcept;

    std::vector<Entry> m_entries;                           // In the order of consumption.
    std::unordered_map<const ov::Node*, size_t> m_indices;  // Constant address -> index into m_entries.
    size_t m_lookahead;        // Minimum number of constants kept prefetched and not released.
    size_t m_min_bytes;        // Minimum number of prefetched-and-not-released bytes.
    size_t m_next;             // Index of the next constant to prefetch.
    size_t m_in_flight;        // Number of constants currently prefetched and not released.
    size_t m_in_flight_bytes;  // Bytes of the m_in_flight constants.
};
}  // namespace ov::util
