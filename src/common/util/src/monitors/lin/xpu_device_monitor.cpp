// Copyright (C) 2019-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/monitors/xpu_device_monitor.hpp"

#include "openvino/util/monitors/idevice_monitor.hpp"
namespace ov {
namespace util {
class XPUDeviceMonitor::PerformanceImpl {
public:
    PerformanceImpl(const std::string& device_luid) {}

    std::map<std::string, float> get_utilization() {
        return {};
    }
};

XPUDeviceMonitor::XPUDeviceMonitor(const std::string& device_luid)
    : IDeviceMonitor("XPU"),
      m_device_luid(device_luid) {}
std::map<std::string, float> XPUDeviceMonitor::get_utilization() {
    if (!m_perf_impl) {
        m_perf_impl = std::make_shared<PerformanceImpl>(m_device_luid);
    }
    return m_perf_impl->get_utilization();
}
}  // namespace util
}  // namespace ov