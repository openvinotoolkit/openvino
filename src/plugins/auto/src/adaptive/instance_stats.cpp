// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "instance_stats.hpp"

#include <algorithm>

namespace ov {
namespace auto_plugin {

namespace {
constexpr size_t baseline_percentile = 10;
}

InstanceStats::InstanceStats(size_t window) : m_window(std::max<size_t>(window, 1)) {}

void InstanceStats::record_inference(const ModelInstanceKey& instance, std::chrono::microseconds elapsed) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto& window = m_windows[instance];
    if (window.samples.size() < m_window) {
        window.samples.push_back(elapsed.count());
    } else {
        window.samples[window.next] = elapsed.count();
    }
    window.next = (window.next + 1) % m_window;
    window.total++;
}

void InstanceStats::set_active(const ModelInstanceKey& instance) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_active = instance;
}

namespace {
long long newest(const std::vector<long long>& samples, size_t next) {
    return samples[(next + samples.size() - 1) % samples.size()];
}
}  // namespace

std::optional<std::chrono::microseconds> InstanceStats::last_inference(const ModelInstanceKey& instance) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    const auto it = m_windows.find(instance);
    if (it == m_windows.end() || it->second.samples.empty()) {
        return std::nullopt;
    }
    return std::chrono::microseconds(newest(it->second.samples, it->second.next));
}

InstanceStatsSnapshot InstanceStats::snapshot() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    InstanceStatsSnapshot result;
    result.active = m_active;
    for (const auto& [instance, window] : m_windows) {
        if (window.samples.empty()) {
            continue;
        }
        auto sorted = window.samples;
        const auto index = (sorted.size() - 1) * baseline_percentile / 100;
        std::nth_element(sorted.begin(), sorted.begin() + index, sorted.end());
        InstanceLatency latency;
        latency.baseline = std::chrono::microseconds(sorted[index]);
        latency.recent = std::chrono::microseconds(newest(window.samples, window.next));
        latency.samples = window.samples.size();
        latency.total = window.total;
        result.instances.emplace(instance, latency);
    }
    return result;
}

}  // namespace auto_plugin
}  // namespace ov
