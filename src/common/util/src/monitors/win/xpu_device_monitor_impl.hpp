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
class XPUDeviceMonitorImpl : public IDeviceMonitorImpl {
public:
    XPUDeviceMonitorImpl(const std::string& device_luid, const std::string& device_type = "GPU")
        : m_device_luid(device_luid),
          m_device_type(device_type) {
        try {
            m_ipf = std::make_unique<Ipf::ClientApi>();
        } catch (...) {
            m_ipf = nullptr;
        }
    }

    std::map<std::string, float> get_utilization() override {
        std::map<std::string, float> result;
        if (!m_ipf) {
            return result;
        }
        try {
            auto json_str = m_ipf->GetNode("Platform.Features.AISelector");
            auto j = nlohmann::json::parse(json_str);
            if (j.contains("Performance")) {
                std::string key = m_device_type + "Utilization";
                if (j["Performance"].contains(key)) {
                    result[m_device_luid] = j["Performance"][key].get<float>();
                }
            }
        } catch (...) {
            result.clear();
        }
        return result;
    }

private:
    std::string m_device_luid;
    std::string m_device_type;
    std::unique_ptr<Ipf::ClientApi> m_ipf;
};

}  // namespace ov::util
