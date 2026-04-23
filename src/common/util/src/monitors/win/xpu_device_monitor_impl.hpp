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
class XPUDeviceMonitorImpl : public IDeviceMonitorImpl {
public:
    XPUDeviceMonitorImpl(const std::string& device_luid) : m_device_luid(device_luid) {
        std::cerr << "[IPF_DEBUG] XPUDeviceMonitorImpl: creating with LUID=\"" << device_luid << "\"" << std::endl;
        // Step 1: Create IPF ClientApi
        try {
            m_ipf = std::make_unique<Ipf::ClientApi>();
            std::cerr << "[IPF_DEBUG] XPUDeviceMonitorImpl: IPF ClientApi created successfully" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[IPF_DEBUG] XPUDeviceMonitorImpl: IPF ClientApi creation failed: " << e.what() << std::endl;
            m_ipf = nullptr;
            return;
        } catch (...) {
            std::cerr << "[IPF_DEBUG] XPUDeviceMonitorImpl: IPF ClientApi creation failed (unknown exception)" << std::endl;
            m_ipf = nullptr;
            return;
        }
        // Step 2: Verify IPF connection by reading CPU brand string
        try {
            auto brand = m_ipf->GetValue("Platform.SOC.Info.BrandString");
            std::cerr << "[IPF_DEBUG] XPUDeviceMonitorImpl: SOC.Info.BrandString = " << brand << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[IPF_DEBUG] XPUDeviceMonitorImpl: SOC.Info.BrandString query failed: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "[IPF_DEBUG] XPUDeviceMonitorImpl: SOC.Info.BrandString query failed (unknown)" << std::endl;
        }
    }

    std::map<std::string, float> get_utilization() override {
        std::map<std::string, float> result;
        if (m_device_luid.empty() || !m_ipf) {
            std::cerr << "[IPF_DEBUG] XPUDeviceMonitorImpl::get_utilization: "
                      << (m_device_luid.empty() ? "LUID is empty" : "m_ipf is null")
                      << ", returning empty map" << std::endl;
            return result;
        }
        try {
            std::string device_path = "Platform.GPU." + m_device_luid + ".Utilization";
            std::cerr << "[IPF_DEBUG] XPUDeviceMonitorImpl::get_utilization: trying \"" << device_path << "\"" << std::endl;
            auto value_str = m_ipf->GetValue(device_path);
            if (!value_str.empty() && value_str != "null") {
                result[m_device_luid] = std::stof(value_str);
                return result;
            }
            // Fallback: generic GPU utilization
            value_str = m_ipf->GetValue("Platform.GPU.Utilization");
            if (!value_str.empty() && value_str != "null") {
                result[m_device_luid] = std::stof(value_str);
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
    std::unique_ptr<Ipf::ClientApi> m_ipf;
};

}  // namespace ov::util
