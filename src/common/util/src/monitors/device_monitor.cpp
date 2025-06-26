// Copyright (C) 2019-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/monitors/device_monitor.hpp"

#include <algorithm>
#include <iostream>

#include "openvino/util/monitors/cpu_device_monitor.hpp"
#include "openvino/util/monitors/xpu_device_monitor.hpp"

namespace ov {
namespace util {

std::map<std::string, float> get_device_utilization(const std::string& device_id) {
    std::shared_ptr<ov::util::IDeviceMonitor> m_device_performance;
    if (device_id.empty())
        m_device_performance = std::make_shared<ov::util::CPUDeviceMonitor>();
    else
        m_device_performance = std::make_shared<ov::util::XPUDeviceMonitor>(device_id);
    return m_device_performance->get_utilization();
}
}  // namespace util
}  // namespace ov
