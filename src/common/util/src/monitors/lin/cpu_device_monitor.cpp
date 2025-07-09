// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/cpu_device_monitor.hpp"

#include <map>

#include "cpu_device_monitor_impl.hpp"

namespace ov {
namespace util {

CPUDeviceMonitor::CPUDeviceMonitor() : IDeviceMonitor("CPU") {
    m_impl = std::make_shared<CPUDeviceMonitorImpl>();
}
std::map<std::string, float> CPUDeviceMonitor::get_utilization() {
    return m_impl->get_utilization();
}
}  // namespace util
}  // namespace ov
