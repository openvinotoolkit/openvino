// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <chrono>
#include <map>
#include <optional>
#include <vector>

#include "common.hpp"
#include "compile_service.hpp"

namespace ov {
namespace auto_plugin {

// What the eviction policy knows about one inactive instance that is ready.
struct InstanceUsage {
    ModelInstanceKey instance;
    // a cache location is set, the device supports caching and the location is a directory
    bool reloadable = false;
    std::optional<CompileStats> stats;
    Time last_used{};
};

struct EvictionDecision {
    std::vector<ModelInstanceKey> release;
    // never released for the rest of the compiled model's life
    std::vector<ModelInstanceKey> keep_resident;
};

/**
 * @brief Decides which inactive instances of the dynamic device selection are released: the ones that can be reloaded
 * from the cache, beyond the most recently used ones.
 *
 * Called on the dispatch thread right after an inference was started on the active instance, with every other
 * instance that is ready and not kept alive yet.
 *
 * An instance that cannot be reloaded, or that was compiled again without a cache hit, stays resident for good:
 * releasing it would turn every switch back to it into a full compilation.
 */
class ReloadablePolicy {
public:
    // max_inactive: how many reloadable inactive instances may stay; min_idle: how long one stays after its last use
    explicit ReloadablePolicy(size_t max_inactive = 0, std::chrono::milliseconds min_idle = {});
    EvictionDecision decide(const std::vector<InstanceUsage>& inactive, Time now) const;

private:
    size_t m_max_inactive;
    std::chrono::milliseconds m_min_idle;
};

/**
 * @brief Keeps a device that failed to compile out of the selection for a while, longer after each failure, and for
 * good after too many of them.
 *
 * Not thread safe, the dynamic dispatch thread is its only user.
 */
class FailurePolicy {
public:
    explicit FailurePolicy(std::chrono::milliseconds initial_backoff = std::chrono::seconds(1),
                           size_t max_failures = 3);
    // returns true if the device must not be selected anymore
    bool record_failure(const DeviceName& device, Time now);
    void record_success(const DeviceName& device);
    bool is_available(const DeviceName& device, Time now) const;

private:
    struct Failures {
        size_t count = 0;
        Time retry_after{};
    };
    std::chrono::milliseconds m_initial_backoff;
    size_t m_max_failures;
    std::map<DeviceName, Failures> m_failures;
};

}  // namespace auto_plugin
}  // namespace ov
