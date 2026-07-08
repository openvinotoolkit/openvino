// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/cpu_device_monitor.hpp"

namespace ov {
namespace util {

CPUDeviceMonitor::CPUDeviceMonitor() : IDeviceMonitor("CPU") {}

std::map<std::string, float> CPUDeviceMonitor::get_utilization() {
    return {{"Total", -1.0f}};
}

}  // namespace util
}  // namespace ov
