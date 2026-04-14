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
        try {
            m_ipf = std::make_unique<Ipf::ClientApi>();
            std::cerr << "[IPF_DEBUG] XPUDeviceMonitorImpl: IPF ClientApi created successfully" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[IPF_DEBUG] XPUDeviceMonitorImpl: IPF ClientApi creation failed: " << e.what() << std::endl;
            m_ipf = nullptr;
        } catch (...) {
            std::cerr << "[IPF_DEBUG] XPUDeviceMonitorImpl: IPF ClientApi creation failed (unknown exception)" << std::endl;
            m_ipf = nullptr;
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
            // Try device-specific path first
            std::string device_path = "Platform.GPU." + m_device_luid + ".Utilization";
            std::cerr << "[IPF_DEBUG] XPUDeviceMonitorImpl::get_utilization: trying device path: \"" << device_path << "\"" << std::endl;
            auto value_str = m_ipf->GetValue(device_path);
            std::cerr << "[IPF_DEBUG] XPUDeviceMonitorImpl::get_utilization: GetValue(\"" << device_path << "\") = \"" << value_str << "\"" << std::endl;

            if (!value_str.empty() && value_str != "null") {
                float val = std::stof(value_str);
                std::cerr << "[IPF_DEBUG] XPUDeviceMonitorImpl::get_utilization: device-specific value = " << val << std::endl;
                result[m_device_luid] = val;
                return result;
            }

            // Fallback: try generic GPU utilization path
            std::cerr << "[IPF_DEBUG] XPUDeviceMonitorImpl::get_utilization: device path empty/null, trying fallback \"Platform.GPU.Utilization\"" << std::endl;
            value_str = m_ipf->GetValue("Platform.GPU.Utilization");
            std::cerr << "[IPF_DEBUG] XPUDeviceMonitorImpl::get_utilization: GetValue(\"Platform.GPU.Utilization\") = \"" << value_str << "\"" << std::endl;

            if (!value_str.empty() && value_str != "null") {
                float val = std::stof(value_str);
                std::cerr << "[IPF_DEBUG] XPUDeviceMonitorImpl::get_utilization: fallback value = " << val << std::endl;
                result[m_device_luid] = val;
            } else {
                std::cerr << "[IPF_DEBUG] XPUDeviceMonitorImpl::get_utilization: all paths returned empty/null, returning empty map" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "[IPF_DEBUG] XPUDeviceMonitorImpl::get_utilization: exception: " << e.what() << ", returning empty map" << std::endl;
        } catch (...) {
            std::cerr << "[IPF_DEBUG] XPUDeviceMonitorImpl::get_utilization: unknown exception, returning empty map" << std::endl;
        }
        return result;
    }

private:
    std::string m_device_luid;
    std::unique_ptr<Ipf::ClientApi> m_ipf;
};

}  // namespace ov::util
