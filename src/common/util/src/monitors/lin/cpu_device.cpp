// Copyright (C) 2019-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/monitors/cpu_device.hpp"

#include <map>

namespace ov {
namespace util {
class CPUDevice::PerformanceImpl {
public:
    PerformanceImpl() {}

    std::map<std::string, float> get_utilization() {
        return {{"Total", -1.0f}};
    }
};

CPUDevice::CPUDevice() : IDevice("CPU") {}
std::map<std::string, float> CPUDevice::get_utilization() {
    if (!m_perf_impl)
        m_perf_impl = std::make_shared<PerformanceImpl>();
    return m_perf_impl->get_utilization();
}
}  // namespace util
}  // namespace ov