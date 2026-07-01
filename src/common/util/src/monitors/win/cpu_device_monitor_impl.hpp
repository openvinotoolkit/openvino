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
class CPUDeviceMonitorImpl : public IDeviceMonitorImpl {
public:
    CPUDeviceMonitorImpl() {
        try {
            m_ipf = std::make_unique<Ipf::ClientApi>();
            std::cerr << "[IPF_DEBUG] CPUDeviceMonitorImpl: IPF ClientApi created successfully" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[IPF_DEBUG] CPUDeviceMonitorImpl: IPF ClientApi creation failed: " << e.what() << std::endl;
            m_ipf = nullptr;
        } catch (...) {
            std::cerr << "[IPF_DEBUG] CPUDeviceMonitorImpl: IPF ClientApi creation failed (unknown exception)" << std::endl;
            m_ipf = nullptr;
        }
    }

    std::map<std::string, float> get_utilization() override {
        std::map<std::string, float> result;
        if (!m_ipf) {
            std::cerr << "[IPF_DEBUG] CPUDeviceMonitorImpl::get_utilization: m_ipf is null" << std::endl;
            result["Total"] = 0.0f;
            return result;
        }
        try {
            auto json_str = m_ipf->GetNode("Platform.Features.AISelector");
            std::cerr << "[IPF_DEBUG] CPUDeviceMonitorImpl::get_utilization: AISelector = " << json_str << std::endl;
            auto j = nlohmann::json::parse(json_str);
            if (j.contains("Performance") && j["Performance"].contains("CPUUtilization")) {
                result["Total"] = j["Performance"]["CPUUtilization"].get<float>();
            } else {
                result["Total"] = 0.0f;
            }
        } catch (const std::exception& e) {
            std::cerr << "[IPF_DEBUG] CPUDeviceMonitorImpl::get_utilization: exception: " << e.what() << std::endl;
            result["Total"] = 0.0f;
        } catch (...) {
            result["Total"] = 0.0f;
        }
        return result;
    }

private:
    std::unique_ptr<Ipf::ClientApi> m_ipf;
};

}  // namespace ov::util
