// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "intel_npu/common/igraph.hpp"
#include "intel_npu/utils/zero/zero_wrappers.hpp"
#include "openvino/core/except.hpp"

namespace intel_npu {

/**
 * @brief Ordering state shared by every pipeline built on the same graph.
 *
 * Needed only when inferences must run in the order they were first submitted. Two mechanisms live
 * here:
 *  - the event last signalled by each command list, so the next submission can wait on it. This is
 *    a fallback for drivers below command-queue extension 1.1; newer drivers order submissions in
 *    the queue itself.
 *  - a ticket counter, so a pipeline can check it is being pushed in its turn.
 *
 * Lives entirely in the backend, where Event is an ordinary type; the graph it belongs to knows
 * nothing about it. Every accessor is safe to call concurrently, since sibling infer requests share
 * one instance. Obtain an instance from SubmissionOrderPool rather than constructing one directly.
 */
class SubmissionOrder final {
public:
    SubmissionOrder() = default;
    SubmissionOrder(const SubmissionOrder&) = delete;
    SubmissionOrder(SubmissionOrder&&) = delete;
    SubmissionOrder& operator=(const SubmissionOrder&) = delete;
    SubmissionOrder& operator=(SubmissionOrder&&) = delete;

    /**
     * @brief Sizes the per-command-list event slots.
     * @details Shrinking drops the events of the removed slots, which is what a pipeline with a
     * smaller batch expects: there is nothing for it to wait on.
     */
    void resize(size_t numberOfCommandLists) {
        std::lock_guard<std::mutex> lock(_mutex);
        _lastSubmittedEvent.resize(numberOfCommandLists);
    }

    /**
     * @brief The event last signalled by the given command list, or null if there is none yet.
     * @details Returned by value: a reference could dangle as soon as another thread overwrote the
     * slot.
     */
    std::shared_ptr<Event> last_event(size_t indexOfCommandList) const {
        std::lock_guard<std::mutex> lock(_mutex);
        if (indexOfCommandList >= _lastSubmittedEvent.size()) {
            return nullptr;
        }
        return _lastSubmittedEvent[indexOfCommandList];
    }

    void set_last_event(const std::shared_ptr<Event>& event, size_t indexOfCommandList) {
        std::lock_guard<std::mutex> lock(_mutex);
        OPENVINO_ASSERT(indexOfCommandList < _lastSubmittedEvent.size(),
                        "Submission order was not sized for command list ",
                        indexOfCommandList);
        _lastSubmittedEvent[indexOfCommandList] = event;
    }

    /// Hands out the next ticket. Each pipeline takes one when it is built.
    uint32_t next_id() {
        return _nextId.fetch_add(1, std::memory_order_relaxed);
    }

    void set_last_submitted_id(uint32_t id) {
        std::lock_guard<std::mutex> lock(_mutex);
        _lastSubmittedId = id;
    }

    uint32_t last_submitted_id() const {
        std::lock_guard<std::mutex> lock(_mutex);
        return _lastSubmittedId;
    }

private:
    mutable std::mutex _mutex;
    std::vector<std::shared_ptr<Event>> _lastSubmittedEvent;
    std::atomic<uint32_t> _nextId{0};
    uint32_t _lastSubmittedId = 0;
};

/**
 * @brief Hands out the SubmissionOrder shared by all pipelines built on the same graph.
 *
 * A graph is the scope across which inferences are ordered, so it is used here purely as an
 * identity - it holds nothing and knows nothing about the ordering. Keeping the mapping on this
 * side means the submission machinery, and with it Level Zero's Event, stays out of IGraph.
 *
 * Only pipelines that actually order their submissions ask for an instance, which is a regular
 * graph on a driver below command-queue extension 1.1. On any newer driver the pool is never
 * touched.
 */
class SubmissionOrderPool final {
public:
    SubmissionOrderPool(const SubmissionOrderPool&) = delete;
    SubmissionOrderPool(SubmissionOrderPool&&) = delete;
    SubmissionOrderPool& operator=(const SubmissionOrderPool&) = delete;
    SubmissionOrderPool& operator=(SubmissionOrderPool&&) = delete;

    static SubmissionOrderPool& getInstance();

    /**
     * @brief Returns the instance belonging to @p graph, creating it on first use.
     * @details The pool keeps only a weak reference, so the state lives exactly as long as the
     * pipelines using it. Once the last one goes away the ordering constraint is vacuous - there is
     * nothing left to be ordered against - and the next pipeline starts a fresh sequence.
     */
    std::shared_ptr<SubmissionOrder> get(const IGraph& graph);

private:
    SubmissionOrderPool() = default;

    // Keyed by address, which is safe precisely because the entries are weak: a pipeline holds a
    // shared_ptr to its graph, so a graph cannot be destroyed while any instance of its
    // SubmissionOrder is still alive. An address that gets recycled therefore always finds an
    // expired entry and starts over, never another graph's state.
    std::unordered_map<const IGraph*, std::weak_ptr<SubmissionOrder>> _pool;
    std::mutex _mutex;
};

}  // namespace intel_npu
