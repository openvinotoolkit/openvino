// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <iostream>
#include <map>
#include <memory>
#include <string>

#include "IpfClient.h"
#include "nlohmann/json.hpp"
#include "openvino/util/idevice_monitor.hpp"

namespace ov::util {
class XPUDeviceMonitorImpl : public IDeviceMonitorImpl {
public:
    XPUDeviceMonitorImpl(const std::string& device_luid, const std::string& device_type = "GPU")
        : m_device_luid(device_luid),
          m_device_type(device_type) {
        std::cerr << "[IPF_DEBUG] XPUDeviceMonitorImpl: creating with LUID=\"" << device_luid
                  << "\", type=\"" << device_type << "\"" << std::endl;
        try {
            m_ipf = std::make_unique<Ipf::ClientApi>();
            std::cerr << "[IPF_DEBUG] XPUDeviceMonitorImpl: IPF ClientApi created successfully" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[IPF_DEBUG] XPUDeviceMonitorImpl: IPF ClientApi creation failed: " << e.what() << std::endl;
            m_ipf = nullptr;
        } catch (...) {
            std::cerr << "[IPF_DEBUG] XPUDeviceMonitorImpl: IPF ClientApi creation failed (unknown exception)"
                      << std::endl;
            m_ipf = nullptr;
        }
    }

    std::map<std::string, float> get_utilization() override {
        std::map<std::string, float> result;
        if (!m_ipf) {
            std::cerr << "[IPF_DEBUG] XPUDeviceMonitorImpl::get_utilization: m_ipf is null" << std::endl;
            return result;
        }
        try {
            auto json_str = m_ipf->GetNode("Platform.Features.AISelector");
            std::cerr << "[IPF_DEBUG] XPUDeviceMonitorImpl::get_utilization: AISelector = " << json_str << std::endl;
            auto j = nlohmann::json::parse(json_str);
            if (j.contains("Performance")) {
                std::string key = m_device_type + "Utilization";
                if (j["Performance"].contains(key)) {
                    result[m_device_luid] = j["Performance"][key].get<float>();
                } else {
                    std::cerr << "[IPF_DEBUG] XPUDeviceMonitorImpl::get_utilization: key \"" << key
                              << "\" not found in Performance" << std::endl;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "[IPF_DEBUG] XPUDeviceMonitorImpl::get_utilization: exception: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "[IPF_DEBUG] XPUDeviceMonitorImpl::get_utilization: unknown exception" << std::endl;
        }
        return result;
    }

private:
    std::string m_device_luid;
    std::string m_device_type;
    std::unique_ptr<Ipf::ClientApi> m_ipf;
};

}  // namespace ov::util
