// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <future>
#include <memory>
#include <unordered_map>
#include <vector>

#include "openvino/core/core_visibility.hpp"

namespace ov {
class Node;
namespace op::v0 {
class Constant;
}  // namespace op::v0
}  // namespace ov

namespace ov::weight_sharing {

/**
 * @brief Loads the constants a fixed number of constants ahead of their consumer and evicts them after use.
 *
 * The constants are expected to be consumed in the given order. With lookahead = 2 and constants A B C D:
 * prefetch A, B; acquire A, release A (evict A, prefetch C); acquire B, release B (evict B, prefetch D); ...
 *
 * Constants not backed by a file mapping are ignored by the underlying hints. Not thread safe.
 */
class OPENVINO_API PrefetchScheduler {
public:
    /**
     * @param ops Nodes in the order they are consumed; only the constants among them are tracked,
     *            duplicates are ignored.
     * @param lookahead Maximum number of constants prefetched and not released yet.
     */
    PrefetchScheduler(const std::vector<std::shared_ptr<ov::Node>>& ops, size_t lookahead);

    /** @brief Waits for the constant data to be loaded. Call it right before the constant is read. */
    void acquire(const ov::op::v0::Constant& constant) noexcept;

    /** @brief Evicts the constant data and prefetches the next constants. Call it once the constant is not read. */
    void release(const ov::op::v0::Constant& constant) noexcept;

    /**
     * @brief Lookahead set by the OV_WEIGHTS_PREFETCH_LOOKAHEAD environment variable.
     *
     * @return 0 (prefetch disabled) if unset or invalid, the maximum size_t value for "all".
     */
    static size_t get_lookahead();

private:
    enum class State { pending, prefetched, released };

    void prefetch_ahead() noexcept;

    std::vector<std::shared_ptr<ov::op::v0::Constant>> m_constants;
    std::vector<State> m_states;
    std::vector<std::shared_future<void>> m_futures;
    std::unordered_map<const ov::op::v0::Constant*, size_t> m_indices;
    size_t m_lookahead;
    size_t m_next = 0;
    size_t m_in_flight = 0;
};

}  // namespace ov::weight_sharing
