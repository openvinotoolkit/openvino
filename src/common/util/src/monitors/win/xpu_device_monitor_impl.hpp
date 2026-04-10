// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <map>
#include <memory>
#include <string>

#include "IpfClient.h"
#include "openvino/util/idevice_monitor.hpp"

namespace ov::util {
class XPUDeviceMonitorImpl : public IDeviceMonitorImpl {
public:
    XPUDeviceMonitorImpl(const std::string& device_luid) : m_device_luid(device_luid) {
        try {
            m_ipf = std::make_unique<Ipf::ClientApi>();
        } catch (...) {
            m_ipf = nullptr;
        }
    }

    std::map<std::string, float> get_utilization() override {
        std::map<std::string, float> result;
        if (m_device_luid.empty() || !m_ipf) {
            return result;
        }
        try {
            // Try device-specific path first (e.g., "Platform.GPU.<luid>.Utilization")
            // then fall back to generic path.
            // The actual namespace structure depends on the IPF GPU provider implementation.
            // TODO: Update path once IPF GPU utilization provider is available.
            std::string device_path = "Platform.GPU." + m_device_luid + ".Utilization";
            auto value_str = m_ipf->GetValue(device_path);
            if (!value_str.empty() && value_str != "null") {
                result[m_device_luid] = std::stof(value_str);
                return result;
            }

            // Fallback: try generic GPU utilization path
            value_str = m_ipf->GetValue("Platform.GPU.Utilization");
            if (!value_str.empty() && value_str != "null") {
                result[m_device_luid] = std::stof(value_str);
            } else {
                result[m_device_luid] = 0.0f;
            }
        } catch (...) {
            result[m_device_luid] = 0.0f;
        }
        return result;
    }

private:
    std::string m_device_luid;
    std::unique_ptr<Ipf::ClientApi> m_ipf;
};

}  // namespace ov::util
