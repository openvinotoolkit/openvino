// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <thread>
#include <vector>

#include "instance_stats.hpp"
#include "observation.hpp"
#include "period_estimator.hpp"
#include "refresh_policy.hpp"

namespace ov {
namespace auto_plugin {

/**
 * @brief Queries the telemetry backend in the background, so that no inference waits for it, and hands out the last
 * observation.
 *
 * The refresh period of the backend is measured at run time; how often it is queried after that is up to the
 * refresh policy. The compiled models that use it subscribe with their devices and the inference stats of their
 * instances, which the policy reads. step() does one scheduling step and takes the time, so it can be driven
 * without a thread.
 */
class SignalHub : public std::enable_shared_from_this<SignalHub> {
public:
    using StatsFn = std::function<InstanceStatsSnapshot()>;
    using HeadroomFn = std::function<std::optional<float>(const Utilizations&)>;
    using ClockFn = std::function<Time()>;

    struct Options {
        RefreshConfig refresh;
        PeriodEstimator::Config estimator;
        // no read for this long pauses the queries until the next one
        std::chrono::milliseconds idle_limit{4000};
        // how long the refresh period is looked for before the backend is taken as not refreshing regularly
        std::chrono::milliseconds probe_budget{3000};
        // how long to wait before looking for the refresh period again after it was not found
        std::chrono::milliseconds reprobe_interval{30000};
        bool start_thread = true;
        ClockFn clock = [] {
            return std::chrono::steady_clock::now();
        };
    };

    class Subscription {
    public:
        ~Subscription();

    private:
        friend class SignalHub;
        struct State {
            DeviceQuery devices;
            StatsFn stats;
            HeadroomFn headroom;
            // held while the callbacks run, so that destroying the subscription waits for them
            std::mutex call_mutex;
            bool active = true;
        };
        std::weak_ptr<SignalHub> m_hub;
        std::shared_ptr<State> m_state;
    };

    explicit SignalHub(std::shared_ptr<ISignalSource> source) : SignalHub(std::move(source), Options{}) {}
    SignalHub(std::shared_ptr<ISignalSource> source, Options options);
    ~SignalHub();
    SignalHub(const SignalHub&) = delete;
    SignalHub& operator=(const SignalHub&) = delete;

    std::shared_ptr<Subscription> subscribe(DeviceQuery devices, StatsFn stats, HeadroomFn headroom = {});

    /**
     * @brief The last observation, if it is recent enough and covers every device.
     * @note Empty means the caller has to query the backend itself.
     */
    std::optional<Observation> read(const std::vector<std::string>& device_names);
    std::optional<Observation> read(const std::vector<std::string>& device_names, Time now);

    /**
     * @brief Query the backend if it is due.
     * @return when the next step is due, never() to wait for a read or a subscription
     */
    Time step(Time now);

    static Time never() {
        return Time::max();
    }

    struct Summary {
        size_t queries = 0;
        uint64_t refreshes = 0;
        std::optional<std::chrono::milliseconds> period;
    };
    // what the backend was asked and what it answered so far
    Summary summary();

private:
    void unsubscribe(const std::shared_ptr<Subscription::State>& state);
    void rebuild_devices();
    void run();

    const std::shared_ptr<ISignalSource> m_source;
    const Options m_options;
    const TelemetryRefreshPolicy m_policy;

    std::mutex m_mutex;
    std::condition_variable m_cv;
    std::vector<std::shared_ptr<Subscription::State>> m_subscriptions;
    DeviceQuery m_query;
    std::set<std::string> m_covered;
    std::shared_ptr<const Observation> m_observation;
    PeriodEstimator m_estimator;
    RefreshState m_state;
    Time m_next_query{};
    Time m_last_query{};
    Time m_last_read{};
    Time m_probe_start{};
    Time m_lost_since{};
    std::chrono::microseconds m_cost{};
    std::chrono::milliseconds m_interval{};
    std::chrono::milliseconds m_stale_limit{300};
    size_t m_queries = 0;
    bool m_probe_started = false;
    bool m_paused = false;
    bool m_wake = false;
    bool m_stop = false;
    std::thread m_thread;
};

}  // namespace auto_plugin
}  // namespace ov
