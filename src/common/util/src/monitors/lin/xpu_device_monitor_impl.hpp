// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <map>

#include "openvino/util/idevice_monitor.hpp"

namespace ov::util {
class XPUDeviceMonitorImpl : public IDeviceMonitorImpl {
public:
    XPUDeviceMonitorImpl(const std::string& device_luid) : m_device_luid(device_luid) {}
    std::map<std::string, float> get_utilization() override {
        return {};
    }

private:
    std::string m_device_luid;
};

}  // namespace ov::util
