// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "device_telemetry.hpp"

#ifdef OV_AUTO_ENABLE_IPF

#    include <atomic>
#    include <cmath>
#    include <memory>
#    include <mutex>

#    include "log_util.hpp"
#    include "nlohmann/json.hpp"

namespace ov {
namespace auto_plugin {
namespace device_monitor {

inline std::string get_log_tag() {
    return "[IPF]";
}

// IPF namespace/paths confirmed against the DttSampleEfApp sample (Demo.cpp / Constants.h) and
// DTT team feedback. Note the registered event path ends in "OnEpoGearChanged", not
// "OnGearChanged"; the event's own eventPath argument is delivered as the parent node
// ("...Policy.EPO"), with the real event name/value inside the JSON payload instead.
constexpr const char* k_dtt_version_path = "Platform.Features.DTT.Software.Version";
constexpr const char* k_dtt_current_gear_path = "Platform.Features.DTT.Policy.EPO.CurrentGear";
constexpr const char* k_dtt_gear_changed_path = "Platform.Features.DTT.Policy.EPO.OnEpoGearChanged";

namespace {

std::optional<float> parse_utilization_from_aiselector_json_impl(const std::string& json_str,
                                                                 const std::string& metric_key,
                                                                 std::string_view metric_key_view,
                                                                 const std::string& device_name) {
    try {
        LOG_DEBUG_TAG("TelemetryClient: raw IPF response: %s", json_str.c_str());
        const auto parsed = nlohmann::json::parse(json_str);
        if (!parsed.contains("Performance")) {
            LOG_WARNING_TAG("TelemetryClient: JSON missing 'Performance' section");
            return std::nullopt;
        }
        const auto& performance = parsed["Performance"];
        auto metric_it = performance.find(metric_key);
        // IGPU may be reported under either IGPUUtilization or GPUUtilization; fall back to the latter.
        const bool igpu_fallback_attempted = metric_it == performance.end() && metric_key_view == k_igpu_utilization_metric;
        if (igpu_fallback_attempted) {
            static const std::string igpu_fallback_key{k_igpu_utilization_fallback_metric};
            metric_it = performance.find(igpu_fallback_key);
        }
        if (metric_it == performance.end()) {
            if (igpu_fallback_attempted) {
                LOG_WARNING_TAG("TelemetryClient: Performance section missing keys: %s and fallback %.*s",
                                metric_key.c_str(),
                                static_cast<int>(k_igpu_utilization_fallback_metric.size()),
                                k_igpu_utilization_fallback_metric.data());
            } else {
                LOG_WARNING_TAG("TelemetryClient: Performance section missing key: %s", metric_key.c_str());
            }
            return std::nullopt;
        }
        if (!metric_it->is_number()) {
            const auto& resolved_metric_key = metric_it.key();
            LOG_WARNING_TAG("TelemetryClient: Performance value for key %s is not a number", resolved_metric_key.c_str());
            return std::nullopt;
        }
        float value = metric_it->get<float>();
        const std::string value_as_string = std::to_string(value);
        LOG_DEBUG_TAG("TelemetryClient: parsed utilization=%s for device=%s", value_as_string.c_str(), device_name.c_str());
        if (!std::isfinite(value) || value < 0.0f || value > 100.0f) {
            LOG_WARNING_TAG("TelemetryClient: utilization value out of supported range [0,100], value=%s for device=%s",
                            value_as_string.c_str(),
                            device_name.c_str());
            return std::nullopt;
        }

        return value;
    } catch (const nlohmann::json::exception& e) {
        LOG_DEBUG_TAG("TelemetryClient: JSON parsing exception: %s", e.what());
        return std::nullopt;
    }
}

// Stores a parsed gear value; captured by value (shared_ptr) in the event callback closure so it
// remains safe to invoke even if IpfClientApiAdapter ever has to leak a failed-unregister callback.
void store_gear(const std::shared_ptr<std::atomic<int>>& shared_gear, const std::string& gear_str) {
    try {
        shared_gear->store(std::stoi(gear_str));
    } catch (const std::exception&) {
        LOG_WARNING_TAG("TelemetryClient: EPO gear value is not an integer: %s", gear_str.c_str());
    }
}

// Free function (not a TelemetryClient::Impl member) so the event callback closure never needs to
// capture `this`; see store_gear() above for the leak-safety rationale.
void handle_gear_changed_event(const std::shared_ptr<std::atomic<int>>& shared_gear, const std::string& event_json) {
    try {
        const auto data = nlohmann::json::parse(event_json);
        if (!data.is_object() || data.empty()) {
            LOG_WARNING_TAG("TelemetryClient: gear-changed payload must be a non-empty JSON object");
            return;
        }
        const auto event_name = data.begin().key();
        const auto& event_value = data.begin().value();
        const std::string gear_str = event_value.is_string() ? event_value.get<std::string>() : event_value.dump();
        LOG_DEBUG_TAG("TelemetryClient: event name=%s, EPO gear=%s", event_name.c_str(), gear_str.c_str());
        store_gear(shared_gear, gear_str);
    } catch (const nlohmann::json::exception& e) {
        LOG_WARNING_TAG("TelemetryClient: failed to parse gear-changed event data: %s", e.what());
    }
}

}  // namespace

// Business logic; all direct IPF ClientApi calls live in ipf_client.cpp.
class TelemetryClient::Impl {
public:
    // Falls back to a real adapter when client is null.
    explicit Impl(std::unique_ptr<IIpfClient> client)
        : m_client(client ? std::move(client) : std::make_unique<IpfClientApiAdapter>()) {}

    Impl() : Impl(std::make_unique<IpfClientApiAdapter>()) {}

    ~Impl() {
        if (m_gear_event_registered) {
            m_client->unregister_event(k_dtt_gear_changed_path);
        }
    }

    std::optional<float> utilization(const std::string& device_name, const std::string& device_type) {
        if (!m_client->is_valid()) {
            LOG_DEBUG_TAG("TelemetryClient::utilization(%s): client not initialized", device_name.c_str());
            return std::nullopt;
        }
        const auto metric_key_view = device_to_metric_key(device_name, device_type);
        if (metric_key_view.empty()) {
            LOG_WARNING_TAG("TelemetryClient::utilization(%s): unknown device type, metric_key empty", device_name.c_str());
            return std::nullopt;
        }
        const std::string metric_key{metric_key_view};
        LOG_DEBUG_TAG("TelemetryClient::utilization(%s): querying IPF for metric_key=%s", device_name.c_str(), metric_key.c_str());
        const std::string json_str = m_client->get_node("Platform.Features.AISelector");
        if (json_str.empty()) {
            return std::nullopt;
        }
        return parse_utilization_from_aiselector_json_impl(json_str, metric_key, metric_key_view, device_name);
    }

    std::optional<bool> is_low_power_mode() {
        if (!m_client->is_valid()) {
            return std::nullopt;
        }
        std::call_once(m_low_power_init_once, [this]() {
            m_shared_gear = std::make_shared<std::atomic<int>>(-1);
            log_dtt_version();
            log_current_gear();
            m_gear_event_registered =
                m_client->register_event(k_dtt_gear_changed_path,
                                          [shared_gear = m_shared_gear](const std::string& event_json) {
                                              handle_gear_changed_event(shared_gear, event_json);
                                          });
            if (m_gear_event_registered) {
                LOG_INFO_TAG("TelemetryClient: registered for %s", k_dtt_gear_changed_path);
            }
        });
        // After first init, this is an atomic read; the call_once body does not re-run.
        const int gear = m_shared_gear->load();
        if (gear < 0) {
            return std::nullopt;
        }
        return is_low_power_gear(gear);
    }

private:
    // Best-effort one-shot reads; failures are already logged by the underlying IIpfClient.
    void log_dtt_version() {
        const std::string json_str = m_client->get_value(k_dtt_version_path);
        if (json_str.empty()) {
            return;
        }
        try {
            const auto parsed = nlohmann::json::parse(json_str);
            const std::string version = parsed.is_string() ? parsed.get<std::string>() : parsed.dump();
            LOG_INFO_TAG("TelemetryClient: DTT version = %s", version.c_str());
        } catch (const nlohmann::json::exception& e) {
            LOG_WARNING_TAG("TelemetryClient: failed to parse DTT version: %s", e.what());
        }
    }

    void log_current_gear() {
        const std::string json_str = m_client->get_value(k_dtt_current_gear_path);
        if (json_str.empty()) {
            return;
        }
        try {
            const auto parsed = nlohmann::json::parse(json_str);
            const std::string gear_str = parsed.is_string() ? parsed.get<std::string>() : parsed.dump();
            LOG_INFO_TAG("TelemetryClient: current EPO gear = %s", gear_str.c_str());
            store_gear(m_shared_gear, gear_str);
        } catch (const nlohmann::json::exception& e) {
            LOG_WARNING_TAG("TelemetryClient: failed to parse current EPO gear: %s", e.what());
        }
    }

    std::unique_ptr<IIpfClient> m_client;
    bool m_gear_event_registered = false;
    std::once_flag m_low_power_init_once;
    std::shared_ptr<std::atomic<int>> m_shared_gear;
};

TelemetryClient::TelemetryClient() : m_impl(std::make_unique<Impl>()) {}

#if defined(MULTIUNITTEST)
TelemetryClient::TelemetryClient(std::unique_ptr<IIpfClient> client_for_test)
    : m_impl(std::make_unique<Impl>(std::move(client_for_test))) {}
#endif

TelemetryClient::~TelemetryClient() = default;

std::optional<float> TelemetryClient::utilization(const std::string& device_name, const std::string& device_type) {
    return m_impl->utilization(device_name, device_type);
}

std::optional<bool> TelemetryClient::is_low_power_mode() {
    return m_impl->is_low_power_mode();
}

#ifdef MULTIUNITTEST
std::optional<float> parse_utilization_from_aiselector_json_for_test(const std::string& json_str,
                                                                     const std::string& device_name,
                                                                     const std::string& device_type) {
    const auto metric_key_view = device_to_metric_key(device_name, device_type);
    if (metric_key_view.empty()) {
        return std::nullopt;
    }
    return parse_utilization_from_aiselector_json_impl(json_str,
                                                       std::string{metric_key_view},
                                                       metric_key_view,
                                                       device_name);
}
#endif

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

std::optional<float> TelemetryClient::utilization(const std::string&, const std::string&) {
    return std::nullopt;
}

std::optional<bool> TelemetryClient::is_low_power_mode() {
    return std::nullopt;
}

}  // namespace device_monitor
}  // namespace auto_plugin
}  // namespace ov

#endif  // OV_AUTO_ENABLE_IPF
