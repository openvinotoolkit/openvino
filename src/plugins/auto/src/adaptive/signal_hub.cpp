// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "signal_hub.hpp"

#include <algorithm>
#include <map>

namespace ov {
namespace auto_plugin {

SignalHub::Subscription::~Subscription() {
    if (const auto hub = m_hub.lock()) {
        hub->unsubscribe(m_state);
    }
}

SignalHub::SignalHub(std::shared_ptr<ISignalSource> source, Options options)
    : m_source(std::move(source)),
      m_options(std::move(options)),
      m_policy(m_options.refresh),
      m_estimator(m_options.estimator) {
    if (m_options.start_thread) {
        m_thread = std::thread([this] {
            run();
        });
    }
}

SignalHub::~SignalHub() {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_stop = true;
    }
    m_cv.notify_all();
    if (m_thread.joinable()) {
        m_thread.join();
    }
}

std::shared_ptr<SignalHub::Subscription> SignalHub::subscribe(DeviceQuery devices, StatsFn stats, HeadroomFn headroom) {
    auto state = std::make_shared<Subscription::State>();
    state->devices = std::move(devices);
    state->stats = std::move(stats);
    state->headroom = std::move(headroom);
    auto subscription = std::shared_ptr<Subscription>(new Subscription());
    subscription->m_hub = weak_from_this();
    subscription->m_state = state;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_subscriptions.push_back(state);
        rebuild_devices();
        m_last_read = m_options.clock();
        m_paused = false;
        m_wake = true;
        m_next_query = Time{};
    }
    m_cv.notify_all();
    return subscription;
}

void SignalHub::unsubscribe(const std::shared_ptr<Subscription::State>& state) {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_subscriptions.erase(std::remove(m_subscriptions.begin(), m_subscriptions.end(), state),
                              m_subscriptions.end());
        rebuild_devices();
    }
    std::lock_guard<std::mutex> call(state->call_mutex);
    state->active = false;
}

void SignalHub::rebuild_devices() {
    m_query.clear();
    m_covered.clear();
    for (const auto& subscription : m_subscriptions) {
        for (const auto& device : subscription->devices) {
            if (m_covered.insert(device.first).second) {
                m_query.push_back(device);
            }
        }
    }
}

std::optional<Observation> SignalHub::read(const std::vector<std::string>& device_names) {
    return read(device_names, m_options.clock());
}

std::optional<Observation> SignalHub::read(const std::vector<std::string>& device_names, Time now) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_last_read = now;
    if (m_paused) {
        m_paused = false;
        m_wake = true;
        m_next_query = Time{};
        m_cv.notify_all();
    }
    if (!m_observation) {
        return std::nullopt;
    }
    for (const auto& name : device_names) {
        if (m_covered.find(name) == m_covered.end()) {
            return std::nullopt;
        }
    }
    if (now - m_observation->observed_at > m_stale_limit) {
        return std::nullopt;
    }
    return *m_observation;
}

Time SignalHub::step(Time now) {
    std::vector<std::shared_ptr<Subscription::State>> subscriptions;
    DeviceQuery query;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_subscriptions.empty()) {
            return never();
        }
        if (now - m_last_read > m_options.idle_limit) {
            m_paused = true;
            return never();
        }
        if (now < m_next_query) {
            return m_next_query;
        }
        subscriptions = m_subscriptions;
        query = m_query;
    }

    const auto begin = m_options.clock();
    auto result = m_source->query(query);
    const auto cost = std::chrono::duration_cast<std::chrono::microseconds>(m_options.clock() - begin);

    InstanceStatsSnapshot stats;
    std::optional<float> headroom;
    for (const auto& subscription : subscriptions) {
        std::lock_guard<std::mutex> call(subscription->call_mutex);
        if (!subscription->active) {
            continue;
        }
        if (subscription->stats) {
            auto snapshot = subscription->stats();
            for (const auto& [instance, latency] : snapshot.instances) {
                stats.instances.insert_or_assign(instance, latency);
            }
            if (stats.active.device.empty()) {
                stats.active = snapshot.active;
            }
        }
        if (subscription->headroom && !result.empty()) {
            if (const auto distance = subscription->headroom(result)) {
                headroom = headroom ? std::min(*headroom, *distance) : *distance;
            }
        }
    }

    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_probe_started) {
        m_probe_started = true;
        m_probe_start = now;
    }
    m_cost = m_cost.count() == 0 ? cost : (m_cost * 3 + cost) / 4;
    if (!result.empty()) {
        const bool had = m_observation != nullptr;
        const bool changed = !had || m_observation->utilizations != result;
        if (changed) {
            // the query before this one was close, so the refresh happened right before it
            const bool precise =
                had && m_last_query != Time{} && now - m_last_query <= m_options.refresh.probe_interval * 3;
            m_estimator.on_change(now, precise, m_state.backoff);
        }
        auto observation = std::make_shared<Observation>();
        observation->utilizations = std::move(result);
        observation->version = changed ? (had ? m_observation->version + 1 : 1) : m_observation->version;
        observation->observed_at = now;
        observation->last_change = changed ? now : m_observation->last_change;
        observation->query_cost = m_cost;
        observation->period = m_estimator.period();
        m_observation = std::move(observation);
    }
    m_last_query = now;

    RefreshInput input;
    input.query_cost = m_cost;
    input.stats = std::move(stats);
    input.headroom = headroom;
    input.period = m_estimator.period();
    if (m_observation) {
        input.last_change = m_observation->last_change;
    }
    if (m_estimator.reliable()) {
        input.heartbeat = Heartbeat::Locked;
        // no refresh for far longer than the period: the heartbeat stopped
        if (m_observation && now - m_observation->last_change > m_interval * 5 / 2 + std::chrono::milliseconds(100)) {
            m_estimator.on_lost();
            input.heartbeat = Heartbeat::Lost;
            input.period.reset();
        }
    } else {
        input.heartbeat =
            m_estimator.lost() || now - m_probe_start > m_options.probe_budget ? Heartbeat::Lost : Heartbeat::Probing;
    }
    if (input.heartbeat == Heartbeat::Lost) {
        if (m_lost_since == Time{}) {
            m_lost_since = now;
        }
        if (now - m_lost_since > m_options.reprobe_interval) {
            m_estimator.reset();
            m_probe_start = now;
            m_lost_since = Time{};
            input.heartbeat = Heartbeat::Probing;
        }
    } else {
        m_lost_since = Time{};
    }

    const auto decision = m_policy.decide(input, m_state, now);
    m_next_query = decision.next_query;
    m_interval = std::max(decision.interval, std::chrono::milliseconds(1));
    // the data is as old as the interval it is refreshed at, and a read in between may be a bit later
    m_stale_limit = std::max<std::chrono::milliseconds>(m_interval * 3 + std::chrono::milliseconds(200),
                                                        std::chrono::milliseconds(300));
    return m_next_query;
}

void SignalHub::run() {
    for (;;) {
        const auto next = step(m_options.clock());
        std::unique_lock<std::mutex> lock(m_mutex);
        if (m_stop) {
            return;
        }
        if (m_wake) {
            m_wake = false;
            continue;
        }
        if (next == never()) {
            m_cv.wait(lock, [this] {
                return m_stop || m_wake;
            });
        } else {
            m_cv.wait_until(lock, next, [this] {
                return m_stop || m_wake;
            });
        }
        if (m_stop) {
            return;
        }
        m_wake = false;
    }
}

}  // namespace auto_plugin
}  // namespace ov
