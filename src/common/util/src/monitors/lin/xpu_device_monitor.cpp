// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/xpu_device_monitor.hpp"

#include "xpu_device_monitor_impl.hpp"

namespace ov {
namespace util {

XPUDeviceMonitor::XPUDeviceMonitor(const std::string& device_luid) : IDeviceMonitor("XPU"), m_device_luid(device_luid) {
    m_impl = std::make_shared<XPUDeviceMonitorImpl>(m_device_luid);
}
std::map<std::string, float> XPUDeviceMonitor::get_utilization() {
    return m_impl->get_utilization();
}
}  // namespace util
}  // namespace ov