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
        // Step 1: Create IPF ClientApi
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
        // Step 2: Verify IPF connection by reading CPU brand string
        try {
            auto brand = m_ipf->GetValue("Platform.SOC.Info.BrandString");
            std::cerr << "[IPF_DEBUG] CPUDeviceMonitorImpl: SOC.Info.BrandString = " << brand << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[IPF_DEBUG] CPUDeviceMonitorImpl: SOC.Info.BrandString query failed: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "[IPF_DEBUG] CPUDeviceMonitorImpl: SOC.Info.BrandString query failed (unknown)" << std::endl;
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
            std::cerr << "[IPF_DEBUG] CPUDeviceMonitorImpl::get_utilization: GetValue = \"" << value_str << "\"" << std::endl;
            if (!value_str.empty() && value_str != "null") {
                float val = std::stof(value_str);
                result["Total"] = val;
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
