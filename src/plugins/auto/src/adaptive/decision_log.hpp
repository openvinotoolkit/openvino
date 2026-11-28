// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <string>

#include "instance_stats.hpp"

namespace ov {
namespace auto_plugin {

/**
 * @brief What the dynamic device selection decided, counted for the summary logged when the compiled model is
 * destroyed. Counting is cheap, so the summary is also available when the per-decision logs are off.
 */
class DecisionLog {
public:
    void count_inference() {
        m_inferences++;
    }
    void count_switch() {
        m_switches++;
    }
    // a switch the switch policy kept from happening
    void count_suppressed() {
        m_suppressed++;
    }
    size_t inferences() const {
        return m_inferences.load();
    }
    size_t switches() const {
        return m_switches.load();
    }
    size_t suppressed() const {
        return m_suppressed.load();
    }
    // one line per instance after the totals
    std::string summary(const InstanceStatsSnapshot& stats) const;

private:
    std::atomic<size_t> m_inferences = {0};
    std::atomic<size_t> m_switches = {0};
    std::atomic<size_t> m_suppressed = {0};
};

}  // namespace auto_plugin
}  // namespace ov
