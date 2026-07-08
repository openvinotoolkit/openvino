// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/xpu_device_monitor.hpp"

namespace ov {
namespace util {

XPUDeviceMonitor::XPUDeviceMonitor(const std::string& device_luid, const std::string& device_type)
    : IDeviceMonitor("XPU"),
      m_device_luid(device_luid),
      m_device_type(device_type) {}

std::map<std::string, float> XPUDeviceMonitor::get_utilization() {
    return {};
}

}  // namespace util
}  // namespace ov
