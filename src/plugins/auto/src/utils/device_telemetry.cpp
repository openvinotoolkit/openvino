// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "device_telemetry.hpp"

#ifdef OV_AUTO_ENABLE_IPF

#    include <atomic>
#    include <cmath>
#    include <memory>
#    include <mutex>
#    include <optional>
#    include <vector>

#    include "ClientApiC.h"
#    include "log_util.hpp"
#    include "nlohmann/json.hpp"

namespace ov {
namespace auto_plugin {
namespace device_monitor {

inline std::string get_log_tag() {
    return "[IPF]";
}

// The registered event path ends in "OnEpoGearChanged", not "OnGearChanged"; the event's
// own eventPath argument is delivered as the parent node ("...Policy.EPO"), with the real
// event name/value inside the JSON payload instead.
constexpr const char* k_dtt_root_path = "Platform.Features.DTT";
constexpr const char* k_dtt_status_path = "Platform.Features.DTT.Software.Status";
constexpr const char* k_dtt_version_path = "Platform.Features.DTT.Software.Version";
constexpr const char* k_dtt_epo_status_path = "Platform.Features.DTT.Policy.EPO.Status";
constexpr const char* k_dtt_current_gear_path = "Platform.Features.DTT.Policy.EPO.CurrentGear";
constexpr const char* k_dtt_gear_changed_path = "Platform.Features.DTT.Policy.EPO.OnEpoGearChanged";

void gear_changed_callback(const char* path, const char* event, void* context);

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

}  // namespace

// Calls into the statically linked IPF ClientApi through the plain-C ABI (ClientApiC.h).
class TelemetryClient::Impl {
public:
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

    static void handle_gear_changed_event(void* context, const std::string& gear_str) {
        auto* sp = static_cast<std::shared_ptr<std::atomic<int>>*>(context);
        if (sp == nullptr || !*sp) {
            return;
        }
        const auto gear = parse_gear(gear_str);
        if (!gear.has_value()) {
            return;
        }
        const int previous_gear = (*sp)->exchange(*gear);
        // Suppress repeated same-gear notifications so a spurious event storm can't flood the log.
        if (*gear != previous_gear) {
            LOG_INFO_TAG("TelemetryClient: EPO gear changed to %d", *gear);
        }
    }

    Impl() {
        const ipf_err_t status = IpfCreate(nullptr, &m_handle);
        if (status != IpfError::IPF_ERR_OK) {
            m_handle = nullptr;
            LOG_WARNING_TAG("TelemetryClient: IPF ClientApi initialization failed: %s: %s",
                            ipf_ef_error_str(status),
                            IpfGetLastErrorMessage());
            return;
        }
        LOG_INFO_TAG("TelemetryClient: IPF ClientApi initialized successfully");
    }

    ~Impl() {
        if (m_handle != nullptr) {
            if (m_gear_event_registered) {
                const ipf_err_t unreg_status =
                    IpfUnregisterEvent(m_handle, k_dtt_gear_changed_path, gear_changed_callback);
                if (unreg_status == IpfError::IPF_ERR_OK) {
                    m_callback_context.reset();
                } else {
                    // Unregister failed: intentionally leak context to prevent use-after-free.
                    LOG_WARNING_TAG("TelemetryClient: IpfUnregisterEvent failed: %s: %s, leaking callback context",
                                    ipf_ef_error_str(unreg_status),
                                    IpfGetLastErrorMessage());
                    (void)m_callback_context.release();
                }
            }
            IpfDestroy(m_handle);
            m_handle = nullptr;
        }
    }

    std::optional<float> utilization(const std::string& device_name, const std::string& device_type) {
        if (m_handle == nullptr) {
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
        const std::string json_str = get_node("Platform.Features.AISelector");
        if (json_str.empty()) {
            return std::nullopt;
        }
        return parse_utilization_from_aiselector_json_impl(json_str, metric_key, metric_key_view, device_name);
    }

    std::optional<bool> is_low_power_mode() {
        if (m_handle == nullptr) {
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

    void on_gear_changed(const std::string& gear_str) {
        if (const auto gear = parse_gear(gear_str)) {
            m_shared_gear->store(*gear);
        }
    }

private:
    // DTT is probed once per client
    void ensure_gear_tracking_registered() {
        if (m_gear_tracking_initialized.load(std::memory_order_acquire)) {
            return;
        }
        std::lock_guard<std::mutex> lock(m_low_power_init_mutex);
        if (m_gear_tracking_initialized.load(std::memory_order_relaxed)) {
            return;
        }
        initialize_gear_tracking();
        m_gear_tracking_initialized.store(true, std::memory_order_release);
    }

    // Must be called while holding m_low_power_init_mutex.
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
        // Context is kept alive here; raw pointer passed to IPF and freed only after confirmed unregister.
        m_callback_context = std::make_unique<std::shared_ptr<std::atomic<int>>>(m_shared_gear);
        const ipf_err_t reg_status =
            IpfRegisterEvent(m_handle, k_dtt_gear_changed_path, gear_changed_callback, m_callback_context.get());
        if (reg_status != IpfError::IPF_ERR_OK) {
            LOG_WARNING_TAG("TelemetryClient: failed to register for %s: %s: %s",
                            k_dtt_gear_changed_path,
                            ipf_ef_error_str(reg_status),
                            IpfGetLastErrorMessage());
            m_callback_context.reset();
            return;
        }
        m_gear_event_registered = true;
        LOG_INFO_TAG("TelemetryClient: registered for %s", k_dtt_gear_changed_path);
    }

    using IpfQueryFn = ipf_err_t (*)(void*, const char*, char*, size_t*);

    // Query IPF node/value data with the two-call buffer-size protocol.
    std::string query_ipf_string(IpfQueryFn query_fn, const char* path) {
        size_t len = 0;
        ipf_err_t status = query_fn(m_handle, path, nullptr, &len);
        if (status != IpfError::IPF_ERR_BUFFERTOOSMALL) {
            // IpfGetLastErrorMessage() is only meaningful for a real failure; do not report it otherwise.
            LOG_WARNING_TAG("TelemetryClient: IPF query(%s) size query failed: %s: %s",
                            path,
                            ipf_ef_error_str(status),
                            IpfGetLastErrorMessage());
            return {};
        }
        if (len == 0) {
            LOG_WARNING_TAG("TelemetryClient: IPF query(%s) returned an empty value", path);
            return {};
        }
        std::vector<char> buf(len);
        status = query_fn(m_handle, path, buf.data(), &len);
        if (status != IpfError::IPF_ERR_OK) {
            LOG_WARNING_TAG("TelemetryClient: IPF query(%s) failed: %s: %s",
                            path,
                            ipf_ef_error_str(status),
                            IpfGetLastErrorMessage());
            return {};
        }
        std::string result(buf.data(), len);
        if (!result.empty() && result.back() == '\0') {
            result.pop_back();
        }
        return result;
    }

    std::string get_node(const char* path) {
        return query_ipf_string(&IpfGetNode, path);
    }

    std::string get_value(const char* path) {
        return query_ipf_string(&IpfGetValue, path);
    }

    // DTT requires reading its root node once to refresh the subtree before individual value queries.
    void refresh_dtt_nodes() {
        const std::string json_str = get_node(k_dtt_root_path);
        LOG_DEBUG_TAG("TelemetryClient: DTT root node refresh %s", json_str.empty() ? "failed" : "succeeded");
    }

    // DTT values arrive as JSON; non-string nodes are dumped verbatim so they stay loggable.
    std::optional<std::string> get_value_as_string(const char* path) {
        const std::string json_str = get_value(path);
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

    // Best-effort one-shot reads; failures are already logged by query_ipf_string/get_value_as_string.
    void log_dtt_version() {
        const auto version = get_value_as_string(k_dtt_version_path);
        if (version.has_value()) {
            LOG_INFO_TAG("TelemetryClient: DTT version = %s", version->c_str());
        }
    }

    void log_current_gear() {
        const auto gear_str = get_value_as_string(k_dtt_current_gear_path);
        if (!gear_str.has_value()) {
            LOG_WARNING_TAG("TelemetryClient: current EPO gear unavailable");
            return;
        }
        LOG_INFO_TAG("TelemetryClient: current EPO gear = %s", gear_str->c_str());
        on_gear_changed(*gear_str);
    }

    void* m_handle = nullptr;
    std::atomic<bool> m_gear_tracking_initialized{false};
    bool m_gear_event_registered = false;
    std::mutex m_low_power_init_mutex;
    std::shared_ptr<std::atomic<int>> m_shared_gear;
    // Owns the heap-allocated shared_ptr passed to IpfRegisterEvent; freed only after confirmed unregister.
    std::unique_ptr<std::shared_ptr<std::atomic<int>>> m_callback_context;
};

void gear_changed_callback(const char* path, const char* event, void* context) {
    if (event == nullptr) {
        LOG_WARNING_TAG("TelemetryClient: received null event payload from %s", path != nullptr ? path : "<null>");
        return;
    }
    LOG_DEBUG_TAG("TelemetryClient: received event from %s. Event data: %s",
                  path != nullptr ? path : "<null>",
                  event);
    try {
        const auto data = nlohmann::json::parse(event);
        if (!data.is_object() || data.empty()) {
            LOG_WARNING_TAG("TelemetryClient: gear-changed payload must be a non-empty JSON object");
            return;
        }
        const auto event_name = data.begin().key();
        const auto& event_value = data.begin().value();
        const std::string gear_str = event_value.is_string() ? event_value.get<std::string>() : event_value.dump();
        LOG_DEBUG_TAG("TelemetryClient: event name=%s, EPO gear=%s", event_name.c_str(), gear_str.c_str());
        TelemetryClient::Impl::handle_gear_changed_event(context, gear_str);
    } catch (const nlohmann::json::exception& e) {
        LOG_WARNING_TAG("TelemetryClient: failed to parse gear-changed event data: %s", e.what());
    }
}

TelemetryClient::TelemetryClient() : m_impl(std::make_unique<Impl>()) {}

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
