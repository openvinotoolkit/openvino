// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <mutex>
#include <set>
#include <vector>
#include <cstdlib>

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

    static bool is_disabled() {
        static const bool disable_weights_prefetch = std::getenv("OV_CPU_DISABLE_WEIGHTS_PREFETCH") != nullptr;
        return disable_weights_prefetch;
    }

    // OV_CPU_WEIGHTS_PREFETCH_AT_INFER=1 delays the prefetch start from compile_model to the first infer.
    static bool is_deferred_to_infer() {
        static const bool deferred = std::getenv("OV_CPU_WEIGHTS_PREFETCH_AT_INFER") != nullptr;
        return deferred;
    }

private:
    using ConstantRef = std::weak_ptr<const ov::op::v0::Constant>;

    std::once_flag m_once;
    std::mutex m_mutex;
    // Observers only: this class must never keep the weights mapping alive.
    std::set<ConstantRef, std::owner_less<ConstantRef>> m_registered;
    std::vector<ConstantRef> m_constants;
};

}  // namespace ov::intel_cpu
