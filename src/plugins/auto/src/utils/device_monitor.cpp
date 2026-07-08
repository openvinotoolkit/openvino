// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "device_monitor.hpp"

#ifdef OV_AUTO_ENABLE_IPF

#    include <memory>

#    include "ClientApi.h"
#    include "nlohmann/json.hpp"

namespace ov {
namespace auto_plugin {
namespace device_monitor {

namespace {

// Owns a single platform telemetry client shared across queries. Construction may
// fail if the telemetry backend is not present; in that case queries return nullopt.
class TelemetryClient {
public:
    TelemetryClient() {
        try {
            m_client = std::make_unique<Ipf::ClientApi>();
        } catch (...) {
            m_client = nullptr;
        }
    }

    std::optional<float> utilization(const std::string& device_name) {
        if (!m_client) {
            return std::nullopt;
        }
        const std::string metric_key = device_name_to_metric_key(device_name);
        if (metric_key.empty()) {
            return std::nullopt;
        }
        try {
            const auto json_str = m_client->GetNode("Platform.Features.AISelector");
            const auto parsed = nlohmann::json::parse(json_str);
            if (!parsed.contains("Performance") || !parsed["Performance"].contains(metric_key)) {
                return std::nullopt;
            }
            const float value = parsed["Performance"][metric_key].get<float>();
            if (value < 0.0f || value > 100.0f) {
                return std::nullopt;
            }
            return value;
        } catch (...) {
            return std::nullopt;
        }
    }

private:
    std::unique_ptr<Ipf::ClientApi> m_client;
};

}  // namespace

std::optional<float> query_device_utilization(const std::string& device_name, const std::string& device_luid) {
    static_cast<void>(device_luid);
    static TelemetryClient client;
    return client.utilization(device_name);
}

}  // namespace device_monitor
}  // namespace auto_plugin
}  // namespace ov

#else  // OV_AUTO_ENABLE_IPF

namespace ov {
namespace auto_plugin {
namespace device_monitor {

std::optional<float> query_device_utilization(const std::string& device_name, const std::string& device_luid) {
    static_cast<void>(device_name);
    static_cast<void>(device_luid);
    return std::nullopt;
}

}  // namespace device_monitor
}  // namespace auto_plugin
}  // namespace ov

#endif  // OV_AUTO_ENABLE_IPF
