// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <chrono>
#include <optional>

#include "instance_stats.hpp"

namespace ov {
namespace auto_plugin {

struct RefreshConfig {
    // assumed refresh period of the backend until it is measured
    std::chrono::milliseconds default_period{1000};
    // a query this cheap relative to the fastest inference keeps the data fresh, a dearer one is backed off
    double fresh_ratio = 0.02;
    unsigned max_backoff = 4;
    unsigned max_backoff_unreliable = 2;
    // quiet decisions in a row before the backoff doubles
    unsigned stable_needed = 3;
    // the device is slow when its recent inferences take this many times its baseline
    double slow_factor = 1.3;
    size_t min_samples = 16;
    // a utilization this close (in percentage points) to a threshold is queried again at once
    float headroom = 10.f;
    // interval between queries while the refresh period is looked for, and around the expected refresh
    std::chrono::milliseconds probe_interval{10};
    // the first query of a refresh is this fraction of the period before the expected refresh
    double guard_fraction = 0.2;
};

enum class Heartbeat {
    Probing,  // the refresh period is being measured
    Locked,   // the backend refreshes regularly, the period is known
    Lost      // no regular refresh was found
};

struct RefreshInput {
    Heartbeat heartbeat = Heartbeat::Probing;
    std::optional<std::chrono::milliseconds> period;
    Time last_change{};
    std::chrono::microseconds query_cost{};
    InstanceStatsSnapshot stats;
    // smallest distance of a device utilization to its threshold
    std::optional<float> headroom;
};

struct RefreshState {
    unsigned backoff = 1;
    unsigned quiet = 0;
    ModelInstanceKey active;
};

struct RefreshDecision {
    enum class Mode { Fresh, Economy };
    Time next_query{};
    Mode mode = Mode::Fresh;
    unsigned backoff = 1;
    // how often the data is refreshed by this decision, for judging when it is too old
    std::chrono::milliseconds interval{};
};

/**
 * @brief Decides when the telemetry backend is queried next, from what a query costs against what an inference
 * costs, and from whether the current device is behaving.
 *
 * A cheap query keeps the data fresh, one refresh period at a time. A query that costs about as much as an
 * inference is backed off, up to a few periods, while nothing changes. A slow current device, a switch or a
 * utilization near its threshold bring the queries back at once.
 */
class TelemetryRefreshPolicy {
public:
    explicit TelemetryRefreshPolicy(RefreshConfig config = {}) : m_config(config) {}
    RefreshDecision decide(const RefreshInput& input, RefreshState& state, Time now) const;
    const RefreshConfig& config() const {
        return m_config;
    }

private:
    RefreshConfig m_config;
};

}  // namespace auto_plugin
}  // namespace ov
