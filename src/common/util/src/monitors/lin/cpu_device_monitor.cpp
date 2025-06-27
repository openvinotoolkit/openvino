// Copyright (C) 2019-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/cpu_device_monitor.hpp"

#include <map>

namespace ov {
namespace util {
class CPUDeviceMonitor::PerformanceImpl {
public:
    PerformanceImpl() {}

    std::map<std::string, float> get_utilization() {
        return {{"Total", -1.0f}};
    }
};

CPUDeviceMonitor::CPUDeviceMonitor() : IDeviceMonitor("CPU") {}
std::map<std::string, float> CPUDeviceMonitor::get_utilization() {
    if (!m_perf_impl)
        m_perf_impl = std::make_shared<PerformanceImpl>();
    return m_perf_impl->get_utilization();
}
}  // namespace util
}  // namespace ov