// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../common.hpp"

namespace ov {
namespace auto_plugin {

// (device name, device type) of the devices whose utilization is wanted
using DeviceQuery = std::vector<std::pair<std::string, std::string>>;
using Utilizations = std::unordered_map<std::string, float>;

// What the last query of the telemetry backend returned.
struct Observation {
    Utilizations utilizations;
    // grows only when the backend returned data that differs from the previous query
    uint64_t version = 0;
    Time observed_at{};
    Time last_change{};
    std::chrono::microseconds query_cost{};
    // how often the backend refreshes its data, measured at run time; empty until it is known to be regular
    std::optional<std::chrono::milliseconds> period;
};

class ISignalSource {
public:
    virtual ~ISignalSource() = default;
    // empty when the backend could not answer
    virtual Utilizations query(const DeviceQuery& devices) = 0;
};

}  // namespace auto_plugin
}  // namespace ov
