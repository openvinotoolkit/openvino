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
class CPUDeviceMonitorImpl : public IDeviceMonitorImpl {
public:
    CPUDeviceMonitorImpl() {
        try {
            m_ipf = std::make_unique<Ipf::ClientApi>();
        } catch (...) {
            m_ipf = nullptr;
        }
    }

    std::map<std::string, float> get_utilization() override {
        std::map<std::string, float> result;
        if (!m_ipf) {
            result["Total"] = 0.0f;
            return result;
        }
        try {
            // Query CPU utilization from IPF namespace
            // The actual path may vary depending on IPF provider availability
            auto value_str = m_ipf->GetValue("Platform.SOC.CPU.Utilization");
            if (!value_str.empty() && value_str != "null") {
                result["Total"] = std::stof(value_str);
            } else {
                result["Total"] = 0.0f;
            }
        } catch (...) {
            result["Total"] = 0.0f;
        }
        return result;
    }

private:
    std::unique_ptr<Ipf::ClientApi> m_ipf;
};

}  // namespace ov::util
