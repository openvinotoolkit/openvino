// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "refresh_policy.hpp"

#include <algorithm>

namespace ov {
namespace auto_plugin {

namespace {
template <class Duration>
Time::duration to_time(Duration duration) {
    return std::chrono::duration_cast<Time::duration>(duration);
}
}  // namespace

RefreshDecision TelemetryRefreshPolicy::decide(const RefreshInput& input, RefreshState& state, Time now) const {
    const auto& config = m_config;
    RefreshDecision decision;

    std::optional<std::chrono::microseconds> fastest;
    for (const auto& [instance, latency] : input.stats.instances) {
        if (latency.samples > 0 && latency.baseline.count() > 0 && (!fastest || latency.baseline < *fastest)) {
            fastest = latency.baseline;
        }
    }
    decision.mode = RefreshDecision::Mode::Fresh;
    if (fastest) {
        const double ratio = static_cast<double>(input.query_cost.count()) / static_cast<double>(fastest->count());
        if (ratio > config.fresh_ratio) {
            decision.mode = RefreshDecision::Mode::Economy;
        }
    }

    bool slow = false;
    const auto active = input.stats.instances.find(input.stats.active);
    if (active != input.stats.instances.end() && active->second.samples >= config.min_samples) {
        slow = static_cast<double>(active->second.recent_mean.count()) >
               config.slow_factor * static_cast<double>(active->second.baseline.count());
    }
    const bool switched = !state.active.device.empty() && input.stats.active != state.active;
    state.active = input.stats.active;
    const bool near_threshold = input.headroom && *input.headroom <= config.headroom;

    const unsigned max_backoff =
        input.heartbeat == Heartbeat::Locked ? config.max_backoff : config.max_backoff_unreliable;
    if (slow || switched || near_threshold || decision.mode == RefreshDecision::Mode::Fresh) {
        state.backoff = 1;
        state.quiet = 0;
    } else if (++state.quiet >= config.stable_needed) {
        state.backoff = std::min(state.backoff * 2, max_backoff);
        state.quiet = 0;
    }
    state.backoff = std::min(state.backoff, max_backoff);
    decision.backoff = state.backoff;

    switch (input.heartbeat) {
    case Heartbeat::Probing:
        decision.interval = config.probe_interval;
        decision.next_query = now + to_time(config.probe_interval);
        break;
    case Heartbeat::Lost:
        decision.interval = config.default_period * state.backoff;
        decision.next_query = now + to_time(decision.interval);
        break;
    case Heartbeat::Locked: {
        const auto period = input.period.value_or(config.default_period);
        decision.interval = period * state.backoff;
        const auto guard = std::max(
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::duration<double, std::micro>(
                std::chrono::duration_cast<std::chrono::microseconds>(period).count() * config.guard_fraction)),
            input.query_cost * 2);
        const auto expected = input.last_change + to_time(decision.interval) - to_time(guard);
        decision.next_query = std::max(expected, now + to_time(config.probe_interval));
        break;
    }
    }
    return decision;
}

}  // namespace auto_plugin
}  // namespace ov
