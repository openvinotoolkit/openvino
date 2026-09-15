// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <mutex>
#include <set>
#include <vector>

namespace ov::op::v0 {
class Constant;
}  // namespace ov::op::v0

namespace ov::intel_cpu {

/**
 * @brief Populates the pages backing the model weights, once, on the first inference.
 *
 * A single instance is shared by every Graph of a CompiledModel - all the streams and all the inner
 * subgraph bodies - so the weights of a model are prefetched exactly once.
 */
class WeightsPrefetch {
public:
    using Ptr = std::shared_ptr<WeightsPrefetch>;

    /**
     * @brief Appends \p constants. Constants already registered by another stream replicating the
     *        very same model are ignored.
     */
    void registerConstants(const std::vector<std::shared_ptr<const ov::op::v0::Constant>>& constants);

    /**
     * @brief Starts populating the registered weights in the background and releases the collected list.
     *        Returns immediately, subsequent calls are no-ops.
     */
    void prefetchOnce();

private:
    using ConstantRef = std::weak_ptr<const ov::op::v0::Constant>;

    std::once_flag m_once;
    std::mutex m_mutex;
    // Observers only: this class must never keep the weights mapping alive.
    std::set<ConstantRef, std::owner_less<ConstantRef>> m_registered;
    std::vector<ConstantRef> m_constants;
};

}  // namespace ov::intel_cpu
