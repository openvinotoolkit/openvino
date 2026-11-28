// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <chrono>
#include <string>

#include "../common.hpp"

namespace ov {
namespace auto_plugin {

struct SwitchConfig {
    // a device keeps serving at least this long, unless it has to be left
    std::chrono::milliseconds min_dwell{1000};
};

struct SwitchInput {
    DeviceName current;
    DeviceName proposed;
    Time now{};
    Time current_since{};
    // the current device can still be selected: it is a candidate and did not fail
    bool current_available = true;
    // the current device is at or over its utilization threshold
    bool current_over_threshold = false;
};

struct SwitchDecision {
    bool stay = false;
    const char* reason = "";
};

/**
 * @brief Keeps the device that serves unless it has to be left, so that a change of the telemetry data, which is
 * now refreshed less often, does not make the selection bounce between devices.
 */
class SwitchPolicy {
public:
    explicit SwitchPolicy(SwitchConfig config = {}) : m_config(config) {}
    SwitchDecision decide(const SwitchInput& input) const;

private:
    SwitchConfig m_config;
};

}  // namespace auto_plugin
}  // namespace ov
