// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "device_telemetry.hpp"

#ifdef OV_AUTO_ENABLE_IPF

#    include <algorithm>
#    include <algorithm>
#    include <atomic>
#    include <cmath>
#    include <memory>
#    include <mutex>
#    include <optional>

#    include "log_util.hpp"
#    include "nlohmann/json.hpp"

namespace ov {
namespace auto_plugin {
namespace device_monitor {

inline std::string get_log_tag() {
    return "[IPF]";
}

namespace {

// Parses the raw IPF response once; returns the "Performance" section, or nullopt if
// malformed/missing so callers can look up multiple devices' metrics without reparsing.
std::optional<nlohmann::json> parse_performance_section(const std::string& json_str) {
    try {
        LOG_DEBUG_TAG("TelemetryClient: raw IPF response: %s", json_str.c_str());
        const auto parsed = nlohmann::json::parse(json_str);
        if (!parsed.contains("Performance")) {
            LOG_WARNING_TAG("TelemetryClient: JSON missing 'Performance' section");
            return std::nullopt;
        }
        return parsed["Performance"];
    } catch (const nlohmann::json::exception& e) {
        LOG_DEBUG_TAG("TelemetryClient: JSON parsing exception: %s", e.what());
        return std::nullopt;
    }
}

std::optional<float> extract_utilization(const nlohmann::json& performance,
                                         const std::string& metric_key,
                                         std::string_view metric_key_view,
                                         const std::string& device_name) {
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
}

}  // namespace

// Tracks DTT's EPO gear via lazy IPF queries plus an OnEpoGearChanged subscription, exposing
// only the low-power-mode question that TelemetryClient needs.
class DttGearTracker {
public:
    explicit DttGearTracker(IIpfClient& client) : m_client(client) {}

    ~DttGearTracker() {
        if (m_gear_event_registered) {
            m_client.unregister_event(k_dtt_gear_changed_path);
        }
    }

    std::optional<bool> is_low_power_mode() {
        if (!m_client.is_valid()) {
            return std::nullopt;
        }
        ensure_gear_tracking_registered();
        if (!m_shared_gear) {
            return std::nullopt;
        }
        const int gear = m_shared_gear->load();
        if (gear < 0) {
            LOG_DEBUG_TAG("TelemetryClient: EPO gear unknown, low power mode unavailable");
            return std::nullopt;
        }
        const bool low_power = is_low_power_gear(gear);
        LOG_DEBUG_TAG("TelemetryClient: EPO gear=%d, low_power_mode=%s", gear, low_power ? "true" : "false");
        return low_power;
    }

private:
    // The registered event path ends in "OnEpoGearChanged", not "OnGearChanged"; the event's
    // own eventPath argument is delivered as the parent node ("...Policy.EPO"), with the real
    // event name/value inside the JSON payload instead.
    static constexpr const char* k_dtt_root_path = "Platform.Features.DTT";
    static constexpr const char* k_dtt_status_path = "Platform.Features.DTT.Software.Status";
    static constexpr const char* k_dtt_version_path = "Platform.Features.DTT.Software.Version";
    static constexpr const char* k_dtt_epo_status_path = "Platform.Features.DTT.Policy.EPO.Status";
    static constexpr const char* k_dtt_current_gear_path = "Platform.Features.DTT.Policy.EPO.CurrentGear";
    static constexpr const char* k_dtt_gear_changed_path = "Platform.Features.DTT.Policy.EPO.OnEpoGearChanged";

    // DTT is probed once per client.
    void ensure_gear_tracking_registered() {
        std::call_once(m_low_power_init_once, [this]() { initialize_gear_tracking(); });
    }

    // Must only be called from ensure_gear_tracking_registered() (call_once-guarded).
    void initialize_gear_tracking() {
        m_shared_gear = std::make_shared<std::atomic<int>>(-1);
        refresh_dtt_nodes();
        if (!is_dtt_available()) {
            LOG_WARNING_TAG("TelemetryClient: DTT unavailable, EPO gear tracking is disabled");
            return;
        }
        log_dtt_version();
        // Seed the gear only while EPO is enabled; otherwise CurrentGear may be stale.
        if (is_epo_enabled()) {
            log_current_gear();
        }
        m_gear_event_registered =
            m_client.register_event(k_dtt_gear_changed_path,
                                     [shared_gear = m_shared_gear](const std::string& event_json) {
                                         handle_gear_changed_event(shared_gear, event_json);
                                     });
        if (m_gear_event_registered) {
            LOG_INFO_TAG("TelemetryClient: registered for %s", k_dtt_gear_changed_path);
        }
    }

    // DTT requires reading its root node once to refresh the subtree before individual value queries.
    void refresh_dtt_nodes() {
        const std::string json_str = m_client.get_node(k_dtt_root_path);
        LOG_DEBUG_TAG("TelemetryClient: DTT root node refresh %s", json_str.empty() ? "failed" : "succeeded");
    }

    // DTT publishes its own health here; an unreadable status means the driver is missing or stopped.
    bool is_dtt_available() {
        const auto status = get_value_as_string(k_dtt_status_path);
        if (!status.has_value()) {
            LOG_WARNING_TAG("TelemetryClient: DTT status unavailable, DTT driver may not be installed or running");
            return false;
        }
        LOG_INFO_TAG("TelemetryClient: DTT status = %s", status->c_str());
        return true;
    }

    // CurrentGear only reflects the live platform state while EPO is enabled.
    bool is_epo_enabled() {
        const auto status = get_value_as_string(k_dtt_epo_status_path);
        if (!status.has_value()) {
            LOG_WARNING_TAG("TelemetryClient: EPO status unavailable, treating low power mode as unknown");
            return false;
        }
        LOG_INFO_TAG("TelemetryClient: EPO status = %s", status->c_str());
        if (*status != "Enabled") {
            LOG_WARNING_TAG("TelemetryClient: EPO is not enabled, ignoring current EPO gear");
            return false;
        }
        return true;
    }

    // Best-effort one-shot read; failures are already logged by get_value_as_string.
    void log_dtt_version() {
        const auto version = get_value_as_string(k_dtt_version_path);
        if (version.has_value()) {
            LOG_INFO_TAG("TelemetryClient: DTT version = %s", version->c_str());
        }
    }

    // DTT values arrive as JSON; non-string nodes are dumped verbatim so they stay loggable.
    std::optional<std::string> get_value_as_string(const char* path) {
        const std::string json_str = m_client.get_value(path);
        if (json_str.empty()) {
            return std::nullopt;
        }
        LOG_DEBUG_TAG("TelemetryClient: raw IPF value at %s: %s", path, json_str.c_str());
        try {
            const auto parsed = nlohmann::json::parse(json_str);
            if (parsed.is_null()) {
                LOG_WARNING_TAG("TelemetryClient: IPF value at %s is null", path);
                return std::nullopt;
            }
            return parsed.is_string() ? parsed.get<std::string>() : parsed.dump();
        } catch (const nlohmann::json::exception& e) {
            LOG_WARNING_TAG("TelemetryClient: failed to parse value at %s: %s", path, e.what());
            return std::nullopt;
        }
    }

    void log_current_gear() {
        const auto gear_str = get_value_as_string(k_dtt_current_gear_path);
        if (!gear_str.has_value()) {
            LOG_WARNING_TAG("TelemetryClient: current EPO gear unavailable");
            return;
        }
        LOG_INFO_TAG("TelemetryClient: current EPO gear = %s", gear_str->c_str());
        store_gear(m_shared_gear, *gear_str);
    }

    // Rejects anything outside the EPO-defined range so untrusted telemetry cannot force a mode.
    static std::optional<int> parse_gear(const std::string& gear_str) {
        int gear = 0;
        std::size_t parsed_chars = 0;
        try {
            gear = std::stoi(gear_str, &parsed_chars);
        } catch (const std::exception&) {
            LOG_WARNING_TAG("TelemetryClient: EPO gear value is not an integer: %s", gear_str.c_str());
            return std::nullopt;
        }
        // std::stoi accepts a numeric prefix (e.g. "4garbage"); require the whole string to be consumed.
        if (parsed_chars != gear_str.size()) {
            LOG_WARNING_TAG("TelemetryClient: EPO gear value is not an integer: %s", gear_str.c_str());
            return std::nullopt;
        }
        if (!is_valid_gear(gear)) {
            LOG_WARNING_TAG("TelemetryClient: EPO gear %d is out of the supported range [%d, %d]",
                            gear,
                            k_min_gear,
                            k_max_gear);
            return std::nullopt;
        }
        return gear;
    }

    // Static so the event callback closure captures only shared_gear (by value) and never `this`,
    // staying safe to invoke even if IpfClientApiAdapter ever has to leak a failed-unregister callback.
    static void store_gear(const std::shared_ptr<std::atomic<int>>& shared_gear, const std::string& gear_str) {
        if (const auto gear = parse_gear(gear_str)) {
            shared_gear->store(*gear);
        }
    }

    // Static for the same leak-safety reason as store_gear() above.
    static void handle_gear_changed_event(const std::shared_ptr<std::atomic<int>>& shared_gear,
                                          const std::string& event_json) {
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

    IIpfClient& m_client;
    bool m_gear_event_registered = false;
    std::once_flag m_low_power_init_once;
    std::shared_ptr<std::atomic<int>> m_shared_gear;
};

// Business logic; all direct IPF ClientApi calls live in ipf_client.cpp.
class TelemetryClient::Impl {
public:
    // Falls back to a real adapter when client is null.
    explicit Impl(std::unique_ptr<IIpfClient> client)
        : m_client(client ? std::move(client) : std::make_unique<IpfClientApiAdapter>()),
          m_gear_tracker(*m_client) {}

    Impl() : Impl(std::make_unique<IpfClientApiAdapter>()) {}

    bool is_available() const {
        return m_client->is_valid();
    }

    // Single IPF round trip; the returned JSON snapshot covers every device's utilization at
    // once, so callers should fetch it once per decision and reuse it for all candidates.
    std::string fetch_utilization_snapshot() {
        if (!m_client->is_valid()) {
            LOG_DEBUG_TAG("TelemetryClient::fetch_utilization_snapshot: client not initialized");
            return {};
        }
        LOG_DEBUG_TAG("TelemetryClient::fetch_utilization_snapshot: querying IPF for AISelector snapshot");
        return m_client->get_node("Platform.Features.AISelector");
    }

    // Fetches one snapshot and resolves utilization for every requested device from it.
    std::unordered_map<std::string, float> utilizations(
        const std::vector<std::pair<std::string, std::string>>& devices) {
        const bool any_supported = std::any_of(devices.begin(), devices.end(), [](const auto& device) {
            return !device_to_metric_key(device.first, device.second).empty();
        });
        if (!any_supported) {
            return {};
        }
        const std::string snapshot = fetch_utilization_snapshot();
        if (snapshot.empty()) {
            return {};
        }
        return utilization_from_snapshot(snapshot, devices);
    }

    std::optional<bool> is_low_power_mode() {
        return m_gear_tracker.is_low_power_mode();
    }

private:
    std::unique_ptr<IIpfClient> m_client;
    DttGearTracker m_gear_tracker;
};

TelemetryClient::TelemetryClient() : m_impl(std::make_unique<Impl>()) {}

#if defined(MULTIUNITTEST)
TelemetryClient::TelemetryClient(std::unique_ptr<IIpfClient> client_for_test)
    : m_impl(std::make_unique<Impl>(std::move(client_for_test))) {}
#endif

TelemetryClient::~TelemetryClient() = default;

bool TelemetryClient::is_available() const {
    return m_impl->is_available();
}

std::unordered_map<std::string, float> TelemetryClient::utilizations(
    const std::vector<std::pair<std::string, std::string>>& devices) {
    return m_impl->utilizations(devices);
}

std::optional<bool> TelemetryClient::is_low_power_mode() {
    return m_impl->is_low_power_mode();
}

std::unordered_map<std::string, float> utilization_from_snapshot(
    const std::string& snapshot,
    const std::vector<std::pair<std::string, std::string>>& devices) {
    std::unordered_map<std::string, float> result;
    if (snapshot.empty() || devices.empty()) {
        return result;
    }
    const auto performance = parse_performance_section(snapshot);
    if (!performance.has_value()) {
        return result;
    }
    for (const auto& [device_name, device_type] : devices) {
        const auto metric_key_view = device_to_metric_key(device_name, device_type);
        if (metric_key_view.empty()) {
            continue;
        }
        const auto utilization =
            extract_utilization(*performance, std::string{metric_key_view}, metric_key_view, device_name);
        if (utilization.has_value()) {
            result.emplace(device_name, *utilization);
        }
    }
    return result;
}

std::optional<float> utilization_from_snapshot(const std::string& snapshot,
                                                const std::string& device_name,
                                                const std::string& device_type) {
    const auto result = utilization_from_snapshot(snapshot, std::vector<std::pair<std::string, std::string>>{{device_name, device_type}});
    const auto it = result.find(device_name);
    return it != result.end() ? std::optional<float>(it->second) : std::nullopt;
}

#ifdef MULTIUNITTEST
std::optional<float> parse_utilization_from_aiselector_json_for_test(const std::string& json_str,
                                                                     const std::string& device_name,
                                                                     const std::string& device_type) {
    return utilization_from_snapshot(json_str, device_name, device_type);
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

bool TelemetryClient::is_available() const {
    return false;
}

std::unordered_map<std::string, float> TelemetryClient::utilizations(const std::vector<std::pair<std::string, std::string>>&) {
    return {};
}

std::optional<bool> TelemetryClient::is_low_power_mode() {
    return std::nullopt;
}

std::unordered_map<std::string, float> utilization_from_snapshot(
    const std::string&,
    const std::vector<std::pair<std::string, std::string>>&) {
    return {};
}

std::optional<float> utilization_from_snapshot(const std::string&, const std::string&, const std::string&) {
    return std::nullopt;
}

}  // namespace device_monitor
}  // namespace auto_plugin
}  // namespace ov

#endif  // OV_AUTO_ENABLE_IPF
