// Copyright (C) 2019-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace ov {
namespace util {
class IDevice {
public:
    IDevice(std::string device_name) : m_device_name(device_name) {}
    virtual std::map<std::string, float> get_utilization() = 0;
    const std::string& name() const {
        return m_device_name;
    }
    virtual ~IDevice() = default;

private:
    std::string m_device_name;
};
}  // namespace util
}  // namespace ov