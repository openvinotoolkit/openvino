// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace ov::util {
class IDeviceMonitor {
public:
    IDeviceMonitor(std::string device_name) : m_device_name(std::move(device_name)) {}
    virtual std::map<std::string, float> get_utilization() = 0;
    const std::string& name() const {
        return m_device_name;
    }
    virtual ~IDeviceMonitor() = default;

private:
    std::string m_device_name;
};

class IDeviceMonitorImpl {
public:
    virtual ~IDeviceMonitorImpl() = default;
    virtual std::map<std::string, float> get_utilization() = 0;
};
}  // namespace ov::util
