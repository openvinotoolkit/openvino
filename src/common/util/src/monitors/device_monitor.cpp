// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/device_monitor.hpp"

#include <algorithm>
#include <iostream>

#include "openvino/util/cpu_device_monitor.hpp"
#include "openvino/util/xpu_device_monitor.hpp"

namespace ov {
namespace util {

std::map<std::string, float> get_device_utilization(const std::string& device_id) {
    std::shared_ptr<ov::util::IDeviceMonitor> m_device_monitor;
    if (device_id.empty())
        m_device_monitor = std::make_shared<ov::util::CPUDeviceMonitor>();
    else
        m_device_monitor = std::make_shared<ov::util::XPUDeviceMonitor>(device_id);
    try {
        return m_device_monitor->get_utilization();
    } catch (...) {
        // Handle exceptions and return an empty map
        return {};
    }
}
}  // namespace util
}  // namespace ov
