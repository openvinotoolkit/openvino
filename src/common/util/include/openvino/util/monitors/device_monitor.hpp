// Copyright (C) 2019-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <deque>
#include <memory>
#include <vector>

#include "openvino/util/monitors/performance_counter.hpp"
namespace ov {
namespace util {
namespace monitor {
class DeviceMonitor {
public:
    DeviceMonitor();
    virtual ~DeviceMonitor() = default;
    std::map<std::string, double> get_utilization(const std::string& luid);

private:
    std::shared_ptr<ov::util::monitor::PerformanceCounter> performance_counter;
};
}  // namespace monitor
}  // namespace util
}  // namespace ov