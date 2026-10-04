// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "submission_order.hpp"

#include <iterator>

namespace intel_npu {

SubmissionOrderPool& SubmissionOrderPool::getInstance() {
    // A plain function-local static is enough here, unlike ZeroCmdQueuePool: the entries are weak
    // and nothing calls back into the pool on destruction, so there is no Level Zero object left to
    // clean up and no static-destruction order to get wrong.
    static SubmissionOrderPool instance;
    return instance;
}

std::shared_ptr<SubmissionOrder> SubmissionOrderPool::get(const IGraph& graph) {
    std::lock_guard<std::mutex> lock(_mutex);

    // Drop the entries whose last pipeline is gone. An expired entry can never be revived, so this
    // both bounds the map and is what lets a recycled graph address start a fresh sequence.
    for (auto it = _pool.begin(); it != _pool.end();) {
        it = it->second.expired() ? _pool.erase(it) : std::next(it);
    }

    auto& slot = _pool[&graph];
    if (auto existing = slot.lock()) {
        return existing;
    }

    auto created = std::make_shared<SubmissionOrder>();
    slot = created;
    return created;
}

}  // namespace intel_npu
