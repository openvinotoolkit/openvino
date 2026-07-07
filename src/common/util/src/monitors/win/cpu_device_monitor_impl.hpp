// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
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
        } catch (...) {
            m_ipf = nullptr;
        }
    }

    std::map<std::string, float> get_utilization() override {
        std::map<std::string, float> result;
        if (!m_ipf) {
            result["Total"] = -1.0f;
            return result;
        }
        try {
            auto json_str = m_ipf->GetNode("Platform.Features.AISelector");
            auto j = nlohmann::json::parse(json_str);
            if (j.contains("Performance") && j["Performance"].contains("CPUUtilization")) {
                result["Total"] = j["Performance"]["CPUUtilization"].get<float>();
            } else {
                result["Total"] = -1.0f;
            }
        } catch (...) {
            result["Total"] = -1.0f;
        }
        return result;
    }

private:
    std::unique_ptr<Ipf::ClientApi> m_ipf;
};

}  // namespace ov::util
