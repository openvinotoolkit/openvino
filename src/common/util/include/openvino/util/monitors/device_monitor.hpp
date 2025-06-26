// Copyright (C) 2019-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <deque>
#include <memory>
#include <vector>

#include "openvino/util/monitors/idevice_monitor.hpp"
namespace ov {
namespace util {
class DeviceMonitor {
public:
    DeviceMonitor();
    virtual ~DeviceMonitor();
    std::map<std::string, float> get_utilization(const std::string& device_id);

private:
    std::shared_ptr<ov::util::IDeviceMonitor> m_device_performance;
};

std::map<std::string, float> get_device_utilization(const std::string& device_id);
}  // namespace util
}  // namespace ov