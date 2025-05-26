// Copyright (C) 2019-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <deque>
#include <memory>
#include <vector>

#include "openvino/util/monitors/idevice.hpp"
namespace ov {
namespace util {
class DeviceMonitor {
public:
    DeviceMonitor();
    virtual ~DeviceMonitor();
    std::map<std::string, double> get_utilization(const std::string& device_id);

private:
    std::shared_ptr<ov::util::IDevice> m_device_performance;
};

std::map<std::string, double> get_utilization(const std::string& device_id);
}  // namespace util
}  // namespace ov