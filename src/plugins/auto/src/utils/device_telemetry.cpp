// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "device_telemetry.hpp"

#ifdef OV_AUTO_ENABLE_IPF

#    include <memory>

#    include "ClientApi.h"
#    include "nlohmann/json.hpp"
#    include "log_util.hpp"

namespace ov {
namespace auto_plugin {
namespace device_monitor {

// Define get_log_tag() to enable LOG_DEBUG_TAG macro for all functions in this namespace
inline std::string get_log_tag() {
    return "[IPF]";
}

namespace {

// Owns a single platform telemetry client shared across queries. Construction may
// fail if the telemetry backend is not present; in that case queries return nullopt.
class TelemetryClient {
public:
    TelemetryClient() {
        try {
            m_client = std::make_unique<Ipf::ClientApi>();
            LOG_DEBUG_TAG("TelemetryClient: IPF ClientApi initialized successfully");
        } catch (const std::exception& e) {
            m_client = nullptr;
            LOG_DEBUG_TAG("TelemetryClient: IPF ClientApi initialization failed: %s", e.what());
        } catch (...) {
            m_client = nullptr;
            LOG_DEBUG_TAG("TelemetryClient: IPF ClientApi initialization failed (unknown exception)");
        }
    }

    std::optional<float> utilization(const std::string& device_name) {
        if (!m_client) {
            LOG_DEBUG_TAG("TelemetryClient::utilization(%s): client not initialized", device_name.c_str());
            return std::nullopt;
        }
        const std::string metric_key = device_name_to_metric_key(device_name);
        if (metric_key.empty()) {
            LOG_DEBUG_TAG("TelemetryClient::utilization(%s): unknown device type, metric_key empty", device_name.c_str());
            return std::nullopt;
        }
        try {
            LOG_DEBUG_TAG("TelemetryClient::utilization(%s): querying IPF for metric_key=%s", device_name.c_str(), metric_key.c_str());
            const auto json_str = m_client->GetNode("Platform.Features.AISelector");
            LOG_DEBUG_TAG("TelemetryClient: raw IPF response: %s", json_str.c_str());
            const auto parsed = nlohmann::json::parse(json_str);
            if (!parsed.contains("Performance")) {
                LOG_DEBUG_TAG("TelemetryClient: JSON missing 'Performance' section");
                return std::nullopt;
            }
            if (!parsed["Performance"].contains(metric_key)) {
                LOG_DEBUG_TAG("TelemetryClient: Performance section missing key: %s", metric_key.c_str());
                return std::nullopt;
            }
            const float value = parsed["Performance"][metric_key].get<float>();
            LOG_DEBUG_TAG("TelemetryClient: parsed utilization=%s for device=%s", std::to_string(value), device_name.c_str());
            const float utilization_percent = value * 100.0f;
            LOG_DEBUG_TAG("TelemetryClient: converted utilization=%s for device=%s", std::to_string(utilization_percent), device_name.c_str());
            return utilization_percent;
        } catch (...) {
            LOG_DEBUG_TAG("TelemetryClient: unknown exception during query for device=%s", device_name.c_str());
            return std::nullopt;
        }
    }

private:
    std::unique_ptr<Ipf::ClientApi> m_client;
};

}  // namespace

std::optional<float> query_device_utilization(const std::string& device_name, const std::string& device_luid) {
    LOG_DEBUG_TAG("query_device_utilization called: device_name=%s", device_name.c_str());
    static_cast<void>(device_luid);
    static TelemetryClient* const client = new TelemetryClient();
    return client->utilization(device_name);
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
