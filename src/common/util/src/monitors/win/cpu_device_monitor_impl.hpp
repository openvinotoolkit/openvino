// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <iostream>
#include <map>
#include <memory>
#include <string>

#include "IpfClient.h"
#include "openvino/util/idevice_monitor.hpp"

namespace ov::util {
class CPUDeviceMonitorImpl : public IDeviceMonitorImpl {
public:
    CPUDeviceMonitorImpl() {
        // Step 1: Create IPF ClientApi (tests SDK -> ClientApiProxy.dll -> ipfsvc connection)
        try {
            m_ipf = std::make_unique<Ipf::ClientApi>();
            std::cerr << "[IPF_DEBUG] CPUDeviceMonitorImpl: IPF ClientApi created successfully" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[IPF_DEBUG] CPUDeviceMonitorImpl: IPF ClientApi creation failed: " << e.what() << std::endl;
            m_ipf = nullptr;
            return;
        } catch (...) {
            std::cerr << "[IPF_DEBUG] CPUDeviceMonitorImpl: IPF ClientApi creation failed (unknown exception)" << std::endl;
            m_ipf = nullptr;
            return;
        }
        // Step 2: Discover available IPF namespace nodes (diagnostic only, does not affect m_ipf)
        try {
            auto nodes = m_ipf->QueryNode("Platform");
            std::cerr << "[IPF_DEBUG] CPUDeviceMonitorImpl: QueryNode(\"Platform\") = " << nodes << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[IPF_DEBUG] CPUDeviceMonitorImpl: QueryNode(\"Platform\") failed: " << e.what()
                      << " (no namespace providers registered yet)" << std::endl;
        } catch (...) {
            std::cerr << "[IPF_DEBUG] CPUDeviceMonitorImpl: QueryNode(\"Platform\") failed (unknown exception)" << std::endl;
        }
    }

    std::map<std::string, float> get_utilization() override {
        std::map<std::string, float> result;
        if (!m_ipf) {
            std::cerr << "[IPF_DEBUG] CPUDeviceMonitorImpl::get_utilization: m_ipf is null, returning 0.0" << std::endl;
            result["Total"] = 0.0f;
            return result;
        }
        try {
            auto value_str = m_ipf->GetValue("Platform.SOC.CPU.Utilization");
            std::cerr << "[IPF_DEBUG] CPUDeviceMonitorImpl::get_utilization: GetValue(\"Platform.SOC.CPU.Utilization\") = \""
                      << value_str << "\"" << std::endl;
            if (!value_str.empty() && value_str != "null") {
                float val = std::stof(value_str);
                std::cerr << "[IPF_DEBUG] CPUDeviceMonitorImpl::get_utilization: parsed value = " << val << std::endl;
                result["Total"] = val;
            } else {
                std::cerr << "[IPF_DEBUG] CPUDeviceMonitorImpl::get_utilization: empty/null response, fallback to 0.0" << std::endl;
                result["Total"] = 0.0f;
            }
        } catch (const std::exception& e) {
            std::cerr << "[IPF_DEBUG] CPUDeviceMonitorImpl::get_utilization: exception: " << e.what() << std::endl;
            result["Total"] = 0.0f;
        } catch (...) {
            std::cerr << "[IPF_DEBUG] CPUDeviceMonitorImpl::get_utilization: unknown exception" << std::endl;
            result["Total"] = 0.0f;
        }
        return result;
    }

private:
    std::unique_ptr<Ipf::ClientApi> m_ipf;
};

}  // namespace ov::util
