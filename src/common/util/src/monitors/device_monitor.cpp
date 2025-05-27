// Copyright (C) 2019-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/monitors/device_monitor.hpp"

#include <algorithm>
#include <iostream>

#include "openvino/util/monitors/cpu_device.hpp"
#include "openvino/util/monitors/xpu_device.hpp"

namespace ov {
namespace util {

DeviceMonitor::DeviceMonitor() {}

DeviceMonitor::~DeviceMonitor() = default;

std::map<std::string, float> DeviceMonitor::get_utilization(const std::string& device_id) {
    if (device_id.empty() && !m_device_performance)
        m_device_performance = std::make_shared<ov::util::CPUDevice>();
    else
        m_device_performance = std::make_shared<ov::util::XPUDevice>(device_id);
    return m_device_performance->get_utilization();
}

std::map<std::string, float> get_device_utilization(const std::string& device_id) {
    DeviceMonitor devices_monitor;
    return devices_monitor.get_utilization(device_id);
}
}  // namespace util
}  // namespace ov
