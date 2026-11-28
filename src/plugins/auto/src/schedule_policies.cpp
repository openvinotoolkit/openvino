// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "schedule_policies.hpp"

#include <algorithm>

namespace ov {
namespace auto_plugin {

ReloadablePolicy::ReloadablePolicy(size_t max_inactive, std::chrono::milliseconds min_idle)
    : m_max_inactive(max_inactive),
      m_min_idle(min_idle) {}

EvictionDecision ReloadablePolicy::decide(const std::vector<InstanceUsage>& inactive, Time now) const {
    EvictionDecision decision;
    std::vector<const InstanceUsage*> evictable;
    for (const auto& usage : inactive) {
        const bool recompiled_without_cache = usage.stats && usage.stats->compile_count > 1 &&
                                              usage.stats->last_loaded_from_cache.value_or(true) == false;
        if (!usage.reloadable || recompiled_without_cache) {
            decision.keep_resident.push_back(usage.instance);
        } else {
            evictable.push_back(&usage);
        }
    }
    std::sort(evictable.begin(), evictable.end(), [](const InstanceUsage* lhs, const InstanceUsage* rhs) {
        return lhs->last_used > rhs->last_used;
    });
    for (size_t i = m_max_inactive; i < evictable.size(); i++) {
        if (now - evictable[i]->last_used >= m_min_idle) {
            decision.release.push_back(evictable[i]->instance);
        }
    }
    return decision;
}

FailurePolicy::FailurePolicy(std::chrono::milliseconds initial_backoff, size_t max_failures)
    : m_initial_backoff(initial_backoff),
      m_max_failures(max_failures) {}

bool FailurePolicy::record_failure(const DeviceName& device, Time now) {
    auto& failures = m_failures[device];
    failures.count++;
    // 1x, 2x, 4x ... of the initial backoff
    failures.retry_after = now + m_initial_backoff * (1u << std::min<size_t>(failures.count - 1, 16));
    return failures.count >= m_max_failures;
}

void FailurePolicy::record_success(const DeviceName& device) {
    m_failures.erase(device);
}

bool FailurePolicy::is_available(const DeviceName& device, Time now) const {
    const auto iter = m_failures.find(device);
    return iter == m_failures.end() || now >= iter->second.retry_after;
}

}  // namespace auto_plugin
}  // namespace ov
