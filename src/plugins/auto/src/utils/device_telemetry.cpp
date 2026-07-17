// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "device_telemetry.hpp"

#ifdef OV_AUTO_ENABLE_IPF

#    include <cmath>
#    include <vector>

#    include "ClientApiC.h"
#    include "log_util.hpp"
#    include "nlohmann/json.hpp"

namespace ov {
namespace auto_plugin {
namespace device_monitor {

// Define get_log_tag() to enable LOG_DEBUG_TAG macro for all functions in this namespace
inline std::string get_log_tag() {
    return "[IPF]";
}

// Calls into IPF's ClientApi DLL exclusively through its plain-C ABI (ClientApiC.h).
// The C ABI uses only plain types (char*, size_t, enums) across the DLL boundary, so it
// stays safe to call regardless of this plugin's own MSVC runtime library (/MD or /MDd),
// even though ClientApi.dll itself is always built with /MD (see thirdparty/CMakeLists.txt).
class TelemetryClient::Impl {
public:
    Impl() {
        const ipf_err_t status = IpfCreate(nullptr, &m_handle);
        if (status != IpfError::IPF_ERR_OK) {
            m_handle = nullptr;
            LOG_DEBUG_TAG("TelemetryClient: IPF ClientApi initialization failed: %s", ipf_ef_error_str(status));
        } else {
            LOG_DEBUG_TAG("TelemetryClient: IPF ClientApi initialized successfully");
        }
    }

    ~Impl() {
        if (m_handle != nullptr) {
            IpfDestroy(m_handle);
            m_handle = nullptr;
        }
    }

    std::optional<float> utilization(const std::string& device_name, const std::string& device_luid) {
        // Keep device_luid for API compatibility; current IPF metric is per device type.
        static_cast<void>(device_luid);
        if (m_handle == nullptr) {
            LOG_DEBUG_TAG("TelemetryClient::utilization(%s): client not initialized", device_name.c_str());
            return std::nullopt;
        }
        const std::string metric_key = device_name_to_metric_key(device_name);
        if (metric_key.empty()) {
            LOG_DEBUG_TAG("TelemetryClient::utilization(%s): unknown device type, metric_key empty", device_name.c_str());
            return std::nullopt;
        }
        LOG_DEBUG_TAG("TelemetryClient::utilization(%s): querying IPF for metric_key=%s", device_name.c_str(), metric_key.c_str());
        const std::string json_str = get_node("Platform.Features.AISelector");
        if (json_str.empty()) {
            return std::nullopt;
        }
        try {
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
            const std::string value_as_string = std::to_string(value);
            LOG_DEBUG_TAG("TelemetryClient: parsed utilization=%s for device=%s",
                          value_as_string.c_str(),
                          device_name.c_str());
            if (!std::isfinite(value) || value < 0.0f || value > 100.0f) {
                LOG_DEBUG_TAG("TelemetryClient: utilization value out of supported range [0,100], value=%s for device=%s",
                              value_as_string.c_str(),
                              device_name.c_str());
                return std::nullopt;
            }
            const float utilization_percent = (value <= 1.0f) ? value * 100.0f : value;
            const std::string utilization_percent_as_string = std::to_string(utilization_percent);
            LOG_DEBUG_TAG("TelemetryClient: converted utilization=%s for device=%s",
                          utilization_percent_as_string.c_str(),
                          device_name.c_str());
            return utilization_percent;
        } catch (...) {
            LOG_DEBUG_TAG("TelemetryClient: unknown exception during query for device=%s", device_name.c_str());
            return std::nullopt;
        }
    }

private:
    // Queries a node from IPF using the buf/len two-call protocol exposed by the C ABI:
    // the first call with an empty buffer reports the required size via IPF_ERR_BUFFERTOOSMALL,
    // then a second call with a sized buffer retrieves the null-terminated JSON payload.
    // Returns an empty string when the query fails.
    std::string get_node(const char* path) {
        size_t len = 0;
        ipf_err_t status = IpfGetNode(m_handle, path, nullptr, &len);
        if (status != IpfError::IPF_ERR_BUFFERTOOSMALL || len == 0) {
            LOG_DEBUG_TAG("TelemetryClient: IpfGetNode(%s) size query failed: %s", path, ipf_ef_error_str(status));
            return {};
        }
        std::vector<char> buf(len);
        status = IpfGetNode(m_handle, path, buf.data(), &len);
        if (status != IpfError::IPF_ERR_OK) {
            LOG_DEBUG_TAG("TelemetryClient: IpfGetNode(%s) failed: %s", path, ipf_ef_error_str(status));
            return {};
        }
        return std::string(buf.data());
    }

    void* m_handle = nullptr;
};

TelemetryClient::TelemetryClient() : m_impl(std::make_unique<Impl>()) {}

TelemetryClient::~TelemetryClient() = default;

std::optional<float> TelemetryClient::utilization(const std::string& device_name, const std::string& device_luid) {
    return m_impl->utilization(device_name, device_luid);
}

}  // namespace device_monitor
}  // namespace auto_plugin
}  // namespace ov

#else  // OV_AUTO_ENABLE_IPF

namespace ov {
namespace auto_plugin {
namespace device_monitor {

class TelemetryClient::Impl {};

TelemetryClient::TelemetryClient() : m_impl(nullptr) {}

TelemetryClient::~TelemetryClient() = default;

std::optional<float> TelemetryClient::utilization(const std::string& device_name, const std::string& device_luid) {
    static_cast<void>(device_name);
    static_cast<void>(device_luid);
    return std::nullopt;
}

}  // namespace device_monitor
}  // namespace auto_plugin
}  // namespace ov

#endif  // OV_AUTO_ENABLE_IPF
