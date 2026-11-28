// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <chrono>
#include <map>
#include <mutex>
#include <optional>
#include <vector>

#include "../common.hpp"

namespace ov {
namespace auto_plugin {

struct InstanceLatency {
    // low percentile of the recent window, what the instance does when nothing disturbs it
    std::chrono::microseconds baseline{};
    std::chrono::microseconds recent{};
    // mean of the last few samples, steadier than recent
    std::chrono::microseconds recent_mean{};
    // samples in the window
    size_t samples = 0;
    // every sample since the compiled model was created
    size_t total = 0;
};

struct InstanceStatsSnapshot {
    std::map<ModelInstanceKey, InstanceLatency> instances;
    ModelInstanceKey active;
};

/**
 * @brief Inference latency of every instance of one compiled model, over a sliding window.
 *
 * Written by the device completion callback, read by the dispatch thread and the destructor. One lock: it is only
 * written by the dynamic device selection, which runs one inference at a time.
 */
class InstanceStats {
public:
    explicit InstanceStats(size_t window = 64);
    void record_inference(const ModelInstanceKey& instance, std::chrono::microseconds elapsed);
    void set_active(const ModelInstanceKey& instance);
    std::optional<std::chrono::microseconds> last_inference(const ModelInstanceKey& instance) const;
    InstanceStatsSnapshot snapshot() const;

private:
    struct Window {
        std::vector<long long> samples;
        size_t next = 0;
        size_t total = 0;
    };
    const size_t m_window;
    mutable std::mutex m_mutex;
    std::map<ModelInstanceKey, Window> m_windows;
    ModelInstanceKey m_active;
};

}  // namespace auto_plugin
}  // namespace ov
