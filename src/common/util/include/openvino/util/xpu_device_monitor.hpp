// Copyright (C) 2019-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>

#include "openvino/util/idevice_monitor.hpp"

namespace ov::util {
class XPUDeviceMonitor : public IDeviceMonitor {
    // This class is used to monitor GPU/NPU performance data.
    // It uses the PerformanceImpl class to get the actual performance data.
    // The user only needs to call the get_utilization() method to get the performance data.
public:
    XPUDeviceMonitor(const std::string& device_luid);
    virtual ~XPUDeviceMonitor() = default;
    std::map<std::string, float> get_utilization() override;

private:
    std::string m_device_luid;
    class PerformanceImpl;
    std::shared_ptr<PerformanceImpl> m_perf_impl = nullptr;
};
}  // namespace ov::util