// Copyright (C) 2019-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/monitors/device_monitor.hpp"

#include <algorithm>
#include <iostream>

#include "openvino/util/monitors/cpu_performance_counter.hpp"
#include "openvino/util/monitors/xpu_performance_counter.hpp"

namespace ov {
namespace util {
namespace monitor {

DeviceMonitor::DeviceMonitor() {}

std::map<std::string, double> DeviceMonitor::get_utilization(const std::string& luid) {
    if (luid.empty() && !performance_counter)
        performance_counter = std::make_shared<ov::util::monitor::CpuPerformanceCounter>();
    else
        performance_counter = std::make_shared<ov::util::monitor::XpuPerformanceCounter>(luid);
    return performance_counter->get_utilization();
}
}  // namespace monitor
}  // namespace util
}  // namespace ov
