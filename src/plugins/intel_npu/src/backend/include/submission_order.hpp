// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

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
 * Owned by the backend and reached through IGraph, which is the scope across which inferences are
 * ordered. Every accessor is safe to call concurrently, since sibling infer requests share one
 * instance.
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

}  // namespace intel_npu
