// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core_impl.hpp"

#include <memory>
#include <optional>
#include <variant>

#include "check_network_batchable.hpp"
#include "itt.hpp"
#include "model_reader.hpp"
#include "openvino/core/any.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/model_util.hpp"
#include "openvino/core/op_extension.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/core/so_extension.hpp"
#include "openvino/core/version.hpp"
#include "openvino/opsets/opset.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/compilation_context.hpp"
#include "openvino/runtime/device_id_parser.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/remote_context.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/runtime/single_file_storage.hpp"
#include "openvino/runtime/threading/executor_manager.hpp"
#include "openvino/util/container_util.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/log.hpp"
#include "openvino/util/ov_version.hpp"
#include "openvino/util/shared_object.hpp"
#include "openvino/util/variant_visitor.hpp"
#include "openvino/util/xml_parse_utils.hpp"
#include "ov_plugins.hpp"
#include "shared_context_manager.hpp"
#ifdef PROXY_PLUGIN_ENABLED
#    include "openvino/proxy/plugin.hpp"
#    include "openvino/proxy/properties.hpp"
#endif

ov::ICore::~ICore() = default;

namespace {

// Probe each present candidate library and merge the devices they report into one canonical
// device list, deduped by opaque fingerprint. Probe only (no plugin engine constructed).
std::vector<ov::DispatchEntry> build_dispatch_entries(const std::vector<std::filesystem::path>& candidate_libs) {
    std::vector<ov::DispatchEntry> entries;
    for (size_t idx = 0; idx < candidate_libs.size(); ++idx) {
        const auto& lib = candidate_libs[idx];
        // Skip a candidate whose library is absent (missing-plugin fallback).
        if (lib.empty() || !ov::util::file_exists(lib))
            continue;

        // Only called for a dispatch group (>1 candidate), so a present candidate MUST export the
        // probe; a missing symbol is a hard error (an unscoreable library can't share a name).
        std::vector<ov::EnumeratedDevice> enumerated;
        auto probe_so = ov::util::load_shared_object(lib);  // load failures propagate as-is
        ov::EnumerateDevicesFunc* probe = nullptr;
        try {
            probe = reinterpret_cast<ov::EnumerateDevicesFunc*>(
                ov::util::get_symbol(probe_so, ov::enumerate_devices_function));
        } catch (const std::exception& ex) {
            OPENVINO_THROW("Library \"",
                           lib.string(),
                           "\" is registered as one of several candidates for a device but does not "
                           "export the device-enumeration probe (",
                           ov::enumerate_devices_function,
                           "), so it cannot be scored: ",
                           ex.what());
        }
        probe(enumerated);
        // probe_so is released here; the winner is re-opened lazily on construction.

        // An INCOMPATIBLE view is recorded, not skipped: the candidate still enumerates that
        // device once constructed, so the id map pushed to it must cover that device too.
        for (auto& dev : enumerated) {
            // An empty fingerprint has no identity: it would merge unrelated devices into one
            // entry (wrong winner / lost devices). Reject it - dispatch members must fingerprint.
            OPENVINO_ASSERT(!dev.fingerprint.empty(),
                            "Library \"",
                            lib.string(),
                            "\" enumerated a dispatch-group device with an empty fingerprint; a "
                            "scoreable candidate must return a non-empty identity per device.");
            // Merge by fingerprint equality: a device several candidates see is listed once.
            auto it = std::find_if(entries.begin(), entries.end(), [&](const ov::DispatchEntry& e) {
                return e.fingerprint == dev.fingerprint;
            });
            if (it == entries.end()) {
                ov::DispatchEntry entry;
                entry.fingerprint = dev.fingerprint;
                entry.per_lib[idx] = {dev.internal_id, dev.score};
                entries.push_back(std::move(entry));
            } else {
                // One candidate reporting a fingerprint twice cannot be disambiguated: merging
                // would drop a device and its id, silently hiding it. Fail loudly instead.
                OPENVINO_ASSERT(it->per_lib.find(idx) == it->per_lib.end(),
                                "Library \"",
                                lib.string(),
                                "\" enumerated devices \"",
                                it->per_lib.at(idx).internal_id,
                                "\" and \"",
                                dev.internal_id,
                                "\" with the same fingerprint, so they cannot be told apart. The "
                                "driver may not report the PCI bus info the identity is built from.");
                it->per_lib[idx] = {dev.internal_id, dev.score};
            }
        }
    }

    // Assign canonical ".N" ids (as in "<device>.N") in stable enumeration order.
    for (size_t n = 0; n < entries.size(); ++n)
        entries[n].canonical_id = std::to_string(n);
    return entries;
}

#ifdef PROXY_PLUGIN_ENABLED
std::string get_internal_plugin_name(const std::string& device_name, const ov::AnyMap& properties) {
    static constexpr const char* internal_plugin_suffix = "_ov_internal";
    auto it = properties.find(ov::proxy::configuration::internal_name.name());
    if (it != properties.end())
        return it->second.as<std::string>();
    return device_name + internal_plugin_suffix;
}
#endif

template <typename F>
void allowNotImplemented(F&& f) {
    try {
        f();
    } catch (const ov::NotImplemented&) {
    }
}

void stripDeviceName(std::string& device, const std::string& substr) {
    auto pos = device.find(substr);
    if (pos == 0) {
        device.erase(pos, substr.length());
    }
}

/**
 * @brief Converts / flattens ov::device::properties from
 * @code
 * core.compile_model(model, "GPU", ov::device::properties("GPU", ov::cache_path("/tmp")));
 * // or
 * core.compile_model(model, "GPU", ov::device::properties({
 *   { "GPU", ov::cache_path("/tmp") },
 *   { "CPU", ov::cache_path("") }
 * }));
 * @endcode
 * To the form:
 * @code
 * core.compile_model(model, "GPU", ov::cache_path("/tmp"));
 * @endcode
 *
 * @param user_device_name A device name for which properties flattening is performed
 * @param user_properties Original set of properties
 * @return ov::AnyMap Flattened ov::AnyMap with properties
 */
ov::AnyMap flatten_sub_properties(const std::string& user_device_name, const ov::AnyMap& user_properties) {
    ov::AnyMap result_properties = user_properties;

    // puts sub-property to result_properties if it's not there yet
    auto update_result_properties = [&result_properties](const ov::AnyMap& sub_properties) -> void {
        for (auto&& sub_property : sub_properties)
            result_properties[sub_property.first] = sub_property.second;
    };

    // First search for ov::device::properties(DEVICE, ...), which has higher
    for (auto secondary_property = result_properties.begin(); secondary_property != result_properties.end();) {
        auto subprop_device_name_pos = secondary_property->first.find(ov::device::properties.name() + std::string("_"));
        if (subprop_device_name_pos == std::string::npos) {
            // 1. Skip non-matching properties
            secondary_property++;
            continue;
        }

        // 2. device properties DEVICE_PROPERTIES_<device_name_with_id> are found
        auto subprop_device_name =
            secondary_property->first.substr(subprop_device_name_pos + std::strlen(ov::device::properties.name()) + 1);
        // flattening is performed only when config is applicable (see docs for ov::is_config_applicable)
        if (ov::is_config_applicable(user_device_name, subprop_device_name) ||
            ov::is_virtual_device(user_device_name)) {
            // 2.1. keep the secondary property for the other virtual devices, but repack them
            auto device_properties = result_properties.find(ov::device::properties.name());
            if (device_properties == result_properties.end()) {
                result_properties[ov::device::properties.name()] = ov::AnyMap{};
            }
            auto& secondary_properties = result_properties[ov::device::properties.name()].as<ov::AnyMap>();
            auto secondary_properties_it = secondary_properties.find(subprop_device_name);
            if (secondary_properties_it == secondary_properties.end()) {
                // 2.1.1. No device name in map yet, insert all config as is
                secondary_properties[subprop_device_name] = secondary_property->second;
            } else {
                // 2.1.2. Device name is present in config file, merge properties according to:
                // ov::device::properties(<device_name>) overrides ov::device::properties(ov::AnyMap{})
                auto& secondary_device_properties = secondary_properties_it->second.as<ov::AnyMap>();
                for (auto& item : secondary_property->second.as<ov::AnyMap>()) {
                    secondary_device_properties[item.first] = item.second;
                }
            }
        }

        // 3. since the sub-property is flattened, we need to drop it
        secondary_property = result_properties.erase(secondary_property);
    }

    // Second search for ov::device::properties(ov::AnyMap{...})
    for (auto property = result_properties.begin(); property != result_properties.end();) {
        if (property->first != ov::device::properties.name()) {
            // 1. Skip non-matching properties
            property++;
            continue;
        }

        // 2. device properties DEVICE_PROPERTIES are found
        auto& secondary_properties = property->second.as<ov::AnyMap>();

        for (auto secondary_property = secondary_properties.begin();
             secondary_property != secondary_properties.end();) {
            // flattening is performed only when config is applicable (see docs for ov::is_config_applicable)
            if (ov::is_config_applicable(user_device_name, secondary_property->first)) {
                // 2.1. flatten the secondary property for target device
                // example: core.compile_model("GPU", ov::device::properties("GPU", ov::prop1));
                // example: core.compile_model("GPU.1", ov::device::properties("GPU", ov::prop1));
                update_result_properties(secondary_property->second.as<ov::AnyMap>());
                secondary_property = secondary_properties.erase(secondary_property);
            } else if (ov::is_virtual_device(user_device_name)) {
                // 2.2. keep the secondary property for the other virtual devices
                secondary_property++;
                continue;
            } else {
                // 2.3. remove the secondary property setting for other hardware device
                // example: core.compile_model("GPU", ov::device::properties("CPU", ov::prop1));
                secondary_property = secondary_properties.erase(secondary_property);
            }
        }

        // 3. go to the next property
        if (secondary_properties.empty()) {
            // 3.1. since the sub-property is flattened, we need to drop it
            property = result_properties.erase(property);
        } else {
            // 3.2. some properties are still in ov::device::properties(ov::AnyMap{}), abort loop
            break;
        }
    }

    return result_properties;
}

enum class MatchType { EXACT = 0, SUBSTR };

struct DevicePriority {
    std::string prop_name;
    MatchType match_type;
};

DevicePriority get_device_priority_property(const std::string& device_name) {
    return ov::is_virtual_device(device_name)
               ? DevicePriority{ov::device::priorities.name(), MatchType::EXACT}
               :
               // ov::device::properties(GPU.0) can be applied for GPU tile identified by GPU.0.0
               DevicePriority{ov::device::id.name(), MatchType::SUBSTR};
}

void clean_batch_properties(const std::string& device_name, ov::AnyMap& config, const ov::PropertyName& property_name) {
    // auto-batching is not applicable, if there is auto_batch_timeout, delete it
    if (device_name.find("BATCH") == std::string::npos) {
        const auto& batch_timeout_mode = config.find(property_name);
        if (batch_timeout_mode != config.end()) {
            if (!ov::is_virtual_device(device_name))
                config.erase(batch_timeout_mode);
        }
    }
}

static const auto core_properties_names = ov::util::make_array(ov::cache_dir.name(),
                                                               ov::cache_path.name(),
                                                               ov::cache_model_path.name(),
                                                               ov::cache_blob_id.name(),
                                                               ov::enable_mmap.name(),
                                                               ov::force_tbb_terminate.name());

static const auto auto_batch_properties_names =
    ov::util::make_array(ov::auto_batch_timeout.name(), ov::hint::allow_auto_batching.name());

std::filesystem::path extract_weight_path(const std::string& compiled_properties) {
    if (auto start = compiled_properties.find(ov::weights_path.name()); start != std::string::npos) {
        start += std::string_view{ov::weights_path.name()}.size() + 1;
        auto length = compiled_properties.find(",", start);
        if (length != std::string::npos) {
            length -= start;
        }
        return {compiled_properties.substr(start, length)};
    } else {
        return {};
    }
}

using model_hint_t = std::variant<std::shared_ptr<const ov::Model>, std::filesystem::path>;

ov::SoPtr<ov::ICompiledModel> import_compiled_model(const ov::Plugin& plugin,
                                                    const ov::SoPtr<ov::IRemoteContext>& context,
                                                    const ov::AnyMap& config) {
    ov::SoPtr<ov::ICompiledModel> compiled_model;
    if (auto blob_hint = config.find(ov::hint::compiled_blob.name()); blob_hint != config.end()) {
        try {
            const auto& compiled_blob = blob_hint->second.as<ov::Tensor>();
            compiled_model = context ? plugin.import_model(compiled_blob, context, config)
                                     : plugin.import_model(compiled_blob, config);
        } catch (...) {
        }
    }
    return compiled_model;
}

ov::SoPtr<ov::ICompiledModel> import_compiled_model(const ov::Plugin& plugin,
                                                    const ov::SoPtr<ov::IRemoteContext>& context,
                                                    const ov::AnyMap& config,
                                                    const model_hint_t& model_hint) {
    auto cfg = config;
    const auto apply_model_hint = ov::util::VariantVisitor{
        [&cfg, &plugin](const std::shared_ptr<const ov::Model>& model_ptr) {
            if (model_ptr != nullptr && plugin.is_property_supported(ov::hint::model.name())) {
                cfg[ov::hint::model.name()] = model_ptr;
            }
        },
        [&cfg, &plugin](const std::filesystem::path& model_path) {
            if (cfg.count(ov::weights_path.name()) == 0 && plugin.is_property_supported(ov::weights_path.name())) {
                std::filesystem::path weights_path{model_path};
                weights_path.replace_extension(".bin");
                if (ov::util::file_exists(weights_path)) {
                    cfg[ov::weights_path.name()] = ov::util::path_to_string(weights_path);
                }
            }
        }};
    std::visit(apply_model_hint, model_hint);
    return import_compiled_model(plugin, context, cfg);
}

std::filesystem::path get_cache_model_path(const ov::AnyMap& config) {
    const auto it = config.find(ov::cache_model_path.name());
    return it == config.end() ? std::filesystem::path{} : it->second.as<std::filesystem::path>();
}

std::vector<ov::Extension::Ptr> try_get_extensions(std::shared_ptr<void>& so) {
    try {
        return ov::detail::load_extensions(so);
    } catch (const std::runtime_error&) {
        return {};
    }
}

std::string get_blob_id_or_compute(const ov::AnyMap& user_config, std::function<std::string()>&& calculate_blob_id) {
    if (auto blob_id_hint = user_config.find(ov::cache_blob_id.name()); blob_id_hint != user_config.end()) {
        return blob_id_hint->second.as<std::string>();
    } else {
        return calculate_blob_id();
    }
}

ov::SharedContextManager& get_cache_wsh_ctx_manager() {
    static ov::SharedContextManager s_cache_wsh_ctx_manager;
    return s_cache_wsh_ctx_manager;
}
}  // namespace

bool ov::is_config_applicable(const std::string& user_device_name, const std::string& subprop_device_name) {
    // full match
    if (user_device_name == subprop_device_name)
        return true;

    auto parsed_user_device_name = ov::parse_device_name_into_config(user_device_name);
    auto parsed_subprop_device_name = ov::parse_device_name_into_config(subprop_device_name);

    // if device name is matched, check additional condition
    auto is_matched = [&](const std::string& key, MatchType match_type) -> bool {
        const auto& user_value = parsed_user_device_name.m_config.count(key)
                                     ? parsed_user_device_name.m_config.at(key).as<std::string>()
                                     : "";
        const auto& subprop_value = parsed_subprop_device_name.m_config.count(key)
                                        ? parsed_subprop_device_name.m_config.at(key).as<std::string>()
                                        : "";

        if (!user_value.empty() && subprop_value.empty()) {
            // property without additional limitation can be applied
            return true;
        }
        return match_type == MatchType::EXACT ? (user_value == subprop_value) : (user_value.find(subprop_value) == 0);
        return false;
    };

    if (parsed_user_device_name.m_device_name == parsed_subprop_device_name.m_device_name) {
        auto device_priority = get_device_priority_property(parsed_user_device_name.m_device_name);
        return is_matched(device_priority.prop_name, device_priority.match_type);
    }

    return false;
}

bool ov::is_virtual_device(const std::string& device_name) {
    return (device_name.find("AUTO") == 0 || device_name.find("MULTI") == 0 || device_name.find("HETERO") == 0 ||
            device_name.find("BATCH") == 0);
};

namespace {
ov::Parsed parse_device_config(const std::string& device_name,
                               const ov::CoreConfig& core_config,
                               const ov::AnyMap& properties,
                               const bool keep_auto_batch_property) {
    // check to the validity of device name
    auto bracket_pos = device_name.find(")");
    while (bracket_pos != std::string::npos) {
        if (bracket_pos < device_name.length() - 1 &&
            (device_name[bracket_pos + 1] != ',' || bracket_pos + 1 == device_name.length() - 1)) {
            OPENVINO_THROW("Device with \"", device_name, "\" name is illegal in the OpenVINO Runtime");
        }
        bracket_pos = device_name.find(")", bracket_pos + 1);
    }

    /** Note: auto-batching is already applied by this time, so the call:
     * core.compile_model("GPU", ov::device::properties("BATCH", ov::auto_batch_timeout(400)));
     * is transformed and we have here:
     * ov::parseDeviceNameIntoConfig("BATCH", ov::device::priorities("GPU"),
     *                                        ov::device::properties("BATCH",
     *                                        ov::auto_batch_timeout(400)));
     * so, after 'flatten_sub_properties' we will have:
     * core.compile_model("BATCH", ov::auto_batch_timeout(400),
     *                             ov::device::priorities("GPU"));
     *
     * So, if one day, we want to add more options in form of ov::allow_<hetero, etc>, we need to apply it before
     * 'flatten_sub_properties' call to have proper behavior
     */
    ov::Parsed parsed{device_name, flatten_sub_properties(device_name, properties), core_config};
    auto& updated_device_name = parsed.m_device_name;
    auto& updated_config = parsed.m_config;

    std::string parsed_device_priority;

    // try to find ':' to extract name of virtual device
    auto pos = device_name.find_first_of(':');
    if (pos != std::string::npos) {
        updated_device_name = device_name.substr(0, pos);
        parsed_device_priority = device_name.substr(pos + 1);
    } else {
        ov::DeviceIDParser parser(device_name);
        updated_device_name = parser.get_device_name();
        parsed_device_priority = parser.get_device_id();
    }

    // checks and updates device priority
    if (!parsed_device_priority.empty()) {
        const auto priority_prop_name = get_device_priority_property(updated_device_name).prop_name;
        const auto it = updated_config.find(priority_prop_name);
        if (it == updated_config.end())
            updated_config[priority_prop_name] = parsed_device_priority;
        else if (it->second == parsed_device_priority) {
            // do nothing
        } else {
            OPENVINO_THROW("Device priority / ID mismatch: ",
                           parsed_device_priority,
                           " (from ",
                           device_name,
                           ") vs ",
                           it->second.as<std::string>(),
                           " (from config)");
        }
    };

    parsed.m_core_config.set(updated_config, updated_device_name);
    // keep batch property only when called from query_supported_property
    if (!keep_auto_batch_property) {
        for (const auto& name : auto_batch_properties_names) {
            clean_batch_properties(updated_device_name, updated_config, name);
        }
    }
    return parsed;
}

std::optional<std::filesystem::path> get_cache_path_from_config(const ov::AnyMap& config) {
    const auto cache_dir_it = config.find(ov::cache_dir.name());
    const auto cache_path_it = config.find(ov::cache_path.name());

    if (cache_dir_it != config.end() && cache_path_it != config.end()) {
        OPENVINO_THROW("Set cache path by '",
                       ov::cache_path.name(),
                       "' or '",
                       ov::cache_dir.name(),
                       "' property. Both set in configuration.");
    } else if (cache_dir_it != config.end()) {
        return std::make_optional(ov::util::make_path(cache_dir_it->second.as<std::string>()));
    } else if (cache_path_it != config.end()) {
        return std::make_optional(cache_path_it->second.as<std::filesystem::path>());
    } else {
        return std::nullopt;
    }
}

void emplace_cache_dir_if_supported(ov::AnyMap& config,
                                    const ov::Plugin& plugin,
                                    const std::filesystem::path& cache_dir) {
    if (plugin.is_property_supported(ov::cache_dir.name())) {
        config.emplace(ov::cache_dir(ov::util::path_to_string(cache_dir)));
    }
}
}  // namespace

ov::Parsed ov::parse_device_name_into_config(const std::string& device_name,
                                             const AnyMap& config,
                                             const bool keep_auto_batch_property) {
    return parse_device_name_into_config(device_name, CoreConfig{}, config, keep_auto_batch_property);
}

ov::Parsed ov::parse_device_name_into_config(const std::string& device_name,
                                             const CoreConfig& core_config,
                                             const AnyMap& config,
                                             const bool keep_auto_batch_property) {
    auto parsed = parse_device_config(device_name, core_config, config, keep_auto_batch_property);
    // remove core properties for HW devices
    if (!ov::is_virtual_device(parsed.m_device_name)) {
        CoreConfig::remove_core(parsed.m_config);
    }
    return parsed;
}

ov::CoreImpl::CoreImpl() {
    add_mutex("");  // Register global mutex
    m_executor_manager = ov::threading::executor_manager();
    for (const auto& it : ov::get_available_opsets()) {
        m_opset_names.insert(it.first);
    }
}

bool ov::CoreImpl::is_proxy_device(const ov::Plugin& plugin) const {
    return is_proxy_device(plugin.get_name());
}
bool ov::CoreImpl::is_proxy_device(const std::string& dev_name) const {
#ifdef PROXY_PLUGIN_ENABLED
    std::string real_name = ov::parse_device_name_into_config(dev_name).m_device_name;
    return m_plugin_registry.find(real_name) != m_plugin_registry.end() &&
           m_plugin_registry.at(real_name).primary().m_plugin_create_func == ov::proxy::create_plugin;
#else
    return false;
#endif
}

void ov::CoreImpl::register_plugin_in_registry_unsafe(const std::string& device_name, PluginDescriptor& desc) {
#ifdef PROXY_PLUGIN_ENABLED
    // Update proxy plugin config
    const auto& fill_config = [](ov::AnyMap& m_default_config, const ov::AnyMap& config, const std::string& dev_name) {
        // Configure aliases for proxy plugin
        auto it = config.find(ov::proxy::configuration::alias.name());
        std::string alias;
        if (it != config.end()) {
            alias = it->second.as<std::string>();
            if (m_default_config.find(ov::proxy::alias_for.name()) == m_default_config.end()) {
                m_default_config[ov::proxy::alias_for.name()] = std::vector<std::string>();
            }
            m_default_config[ov::proxy::alias_for.name()].as<std::vector<std::string>>().emplace_back(dev_name);
        }

        // Configure device order for proxy_plugin
        it = config.find(ov::proxy::configuration::priority.name());
        if (it != config.end()) {
            if (m_default_config.find(ov::proxy::device_priorities.name()) == m_default_config.end()) {
                m_default_config[ov::proxy::device_priorities.name()] = std::vector<std::string>();
            }
            m_default_config[ov::proxy::device_priorities.name()].as<std::vector<std::string>>().emplace_back(
                dev_name + ":" + it->second.as<std::string>());
        }

        // Configure devices fallback order for proxy_plugin
        // Can use substring to configure the order
        // CUDA iGPU : CUDA iGPU      // just create a new elememnt
        // CPU iGPU : CUDA CPU iGPU   // use substring to find the right place
        it = config.find(ov::proxy::configuration::fallback.name());
        if (it != config.end()) {
            auto fallback = it->second.as<std::string>();
            // Change fallback name if fallback is configured to the HW plugin under the proxy with the same name
            if (m_default_config.find(ov::device::priorities.name()) == m_default_config.end()) {
                m_default_config[ov::device::priorities.name()] =
                    std::vector<std::string>{dev_name, std::move(fallback)};
            } else {
                auto dev_order = m_default_config[ov::device::priorities.name()].as<std::vector<std::string>>();
                auto begin_it = std::find(dev_order.begin(), dev_order.end(), dev_name);
                auto end_it = std::find(dev_order.begin(), dev_order.end(), fallback);
                OPENVINO_ASSERT(begin_it == dev_order.end() && end_it == dev_order.end(),
                                "Cannot restore the fallback order for proxy plugin.");
                if (begin_it != dev_order.end() && end_it != dev_order.end()) {
                    // Nothing to do. Just check that devices have the right order
                    OPENVINO_ASSERT(std::distance(begin_it, end_it) > 0,
                                    "Incorrect order of proxy plugin fallback priority.");
                } else if (begin_it != dev_order.end()) {
                    // Insert fallback device after the primary device
                    dev_order.insert(begin_it + 1, fallback);
                } else if (end_it != dev_order.end()) {
                    // Insert primary device before the fallback device
                    dev_order.insert(end_it, dev_name);
                }
                m_default_config[ov::device::priorities.name()] = dev_order;
            }
        }
    };
#endif

    std::string dev_name = device_name;
#ifdef PROXY_PLUGIN_ENABLED
    auto&& config = desc.primary().m_default_config;
    // Register proxy plugin
    if (config.find(ov::proxy::configuration::alias.name()) != config.end()) {
        // Create proxy plugin for alias
        const auto& alias = config.at(ov::proxy::configuration::alias.name()).as<std::string>();
        if (alias == device_name)
            dev_name = get_internal_plugin_name(dev_name, config);
        // Alias can be registered by several plugins
        if (m_plugin_registry.find(alias) == m_plugin_registry.end()) {
            // Register new plugin
            PluginDescriptor desc = PluginDescriptor(ov::proxy::create_plugin);
            // Add internal name for proxy in order to modify fallback order before the initialization
            if (alias == device_name)
                desc.primary().m_default_config[ov::proxy::configuration::internal_name.name()] = dev_name;

            fill_config(desc.primary().m_default_config, config, dev_name);
            m_plugin_registry[alias] = std::move(desc);
            add_mutex(alias);
        } else {
            // Update registered plugin
            auto& plugin = m_plugin_registry.at(alias);
            // Error if we have an alias for HW plugin
            OPENVINO_ASSERT(plugin.primary().m_plugin_create_func == ov::proxy::create_plugin,
                            "Cannot register plugin for ",
                            dev_name,
                            " plugin with the same name already registered!");
            // Add internal name for proxy in order to modify fallback order before the initialization
            if (alias == device_name)
                plugin.primary().m_default_config[ov::proxy::configuration::internal_name.name()] = dev_name;
            fill_config(plugin.primary().m_default_config, config, dev_name);
        }
    } else if (config.find(ov::proxy::configuration::fallback.name()) != config.end()) {
        // Fallback without alias means that we need to replace original plugin to proxy
        dev_name = get_internal_plugin_name(dev_name, config);
        PluginDescriptor desc = PluginDescriptor(ov::proxy::create_plugin);
        desc.primary().m_default_config[ov::proxy::configuration::internal_name.name()] = dev_name;
        fill_config(desc.primary().m_default_config, config, dev_name);
        m_plugin_registry[device_name] = std::move(desc);
        add_mutex(device_name);
    }

    const static std::vector<ov::PropertyName> proxy_conf_properties = {ov::proxy::configuration::alias,
                                                                        ov::proxy::configuration::fallback,
                                                                        ov::proxy::configuration::internal_name,
                                                                        ov::proxy::configuration::priority};

    // Register real plugin
    for (const auto& proxy_prop : proxy_conf_properties) {
        auto it = desc.primary().m_default_config.find(proxy_prop);
        if (it != desc.primary().m_default_config.end()) {
            desc.primary().m_default_config.erase(it);
        }
    }
#endif

    m_plugin_registry[dev_name] = desc;
    add_mutex(dev_name);
}

void ov::CoreImpl::register_compile_time_plugins() {
    std::lock_guard<std::mutex> lock(get_mutex());

    auto any_copy = [](const std::map<std::string, std::string>& params) -> ov::AnyMap {
        ov::AnyMap result;
        for (auto&& value : params) {
            result.emplace(value.first, value.second);
        }
        return result;
    };

    const decltype(::get_compiled_plugins_registry())& plugins = get_compiled_plugins_registry();
    for (const auto& plugin : plugins) {
        const auto& device_name = plugin.first;
        if (device_name.find('.') != std::string::npos) {
            OPENVINO_THROW("Device name must not contain dot '.' symbol");
        }
#ifdef OPENVINO_STATIC_LIBRARY
        if (m_plugin_registry.find(device_name) == m_plugin_registry.end()) {
            const auto& value = plugin.second;
            ov::AnyMap config = any_copy(value.m_default_config);
            PluginDescriptor desc{value.m_create_plugin_func, config, value.m_create_extensions_func};
            register_plugin_in_registry_unsafe(device_name, desc);
        }
#else
        // A device may map to several candidate libraries (a dispatch group); keep the ones
        // that exist. The first present candidate is primary, the rest are extra candidates.
        if (m_plugin_registry.find(device_name) == m_plugin_registry.end()) {
            ov::AnyMap config = any_copy(plugin.second.m_default_config);
            std::vector<std::filesystem::path> present;
            for (const auto& p : plugin.second.m_plugin_paths) {
                const auto resolved = ov::util::get_compiled_plugin_path(p);
                if (ov::util::file_exists(resolved))
                    present.push_back(resolved);
            }
            if (!present.empty()) {
                PluginDescriptor desc{present.front(), config};
                for (size_t i = 1; i < present.size(); ++i)
                    desc.m_candidates.push_back({present[i], config, nullptr});
                register_plugin_in_registry_unsafe(device_name, desc);
            }
        }
#endif
    }
}

void ov::CoreImpl::register_plugins_in_registry(const std::filesystem::path& xml_config_file, const bool by_abs_path) {
    using namespace ov::util;
    const auto parse_result = pugixml::parse_xml(xml_config_file);
    OPENVINO_ASSERT(parse_result.error_msg.empty(), parse_result.error_msg);

    const auto& xml_doc = *parse_result.xml;
    const auto ie_node = xml_doc.document_element();
    const auto devices_node = ie_node.child("plugins");

    std::lock_guard<std::mutex> lock(get_mutex());

    FOREACH_CHILD (plugin_node, devices_node, "plugin") {
        const auto device_name = pugixml::get_str_attr(plugin_node, "name");
        if (m_plugin_registry.find(device_name) != m_plugin_registry.end()) {
            OPENVINO_THROW("Device with \"", device_name, "\"  is already registered in the OpenVINO Runtime");
        }
        if (device_name.find('.') != std::string::npos) {
            OPENVINO_THROW("Device name must not contain dot '.' symbol");
        }

        // Library path(s): either the legacy single "location" attribute, or ordered
        // <location> children (a dispatch group; first = default). Order is preserved.
        std::vector<std::filesystem::path> locations;
        for (const auto& location_node : plugin_node.children("location")) {
            const auto child_location = std::string{location_node.child_value()};
            OPENVINO_ASSERT(!child_location.empty(),
                            "Empty <location> element for device \"",
                            device_name,
                            "\" in the plugins registry");
            locations.push_back(get_plugin_path(make_path(child_location), xml_config_file, by_abs_path));
        }
        if (const auto attr_location = pugixml::get_str_attr(plugin_node, "location", ""); !attr_location.empty()) {
            OPENVINO_ASSERT(locations.empty(),
                            "Device \"",
                            device_name,
                            "\" declares both a \"location\" attribute and <location> child elements; use one form");
            locations.push_back(get_plugin_path(make_path(attr_location), xml_config_file, by_abs_path));
        }
        OPENVINO_ASSERT(!locations.empty(),
                        "Device \"",
                        device_name,
                        "\" has no library location in the plugins registry");

        PluginDescriptor plugin_desc{locations.front()};
        for (size_t i = 1; i < locations.size(); ++i) {
            plugin_desc.m_candidates.push_back({locations[i], {}, nullptr});
        }

        // check properties
        const auto properties_node = plugin_node.child("properties");
        if (properties_node) {
            FOREACH_CHILD (property_node, properties_node, "property") {
                const auto key = pugixml::get_str_attr(property_node, "key");
                // Default config from <properties> applies to every candidate library.
                const auto value = pugixml::get_str_attr(property_node, "value");
                for (auto& candidate : plugin_desc.m_candidates) {
                    candidate.m_default_config[key] = value;
                }
            }
        }

        // check extensions
        const auto extensions_node = plugin_node.child("extensions");
        if (extensions_node) {
            FOREACH_CHILD (extension_node, extensions_node, "extension") {
                const auto extension_location = pugixml::get_str_attr(extension_node, "location");
                plugin_desc.m_list_of_extensions.push_back(make_path(extension_location));
            }
        }

        // fill value in plugin registry for later lazy initialization
        register_plugin_in_registry_unsafe(device_name, plugin_desc);
    }
}

std::optional<size_t> ov::CoreImpl::resolve_dispatch_winner_unsafe(const std::string& device_name,
                                                                   const PluginDescriptor& desc,
                                                                   const std::string& device_id) const {
    // Lazily build the merged device list for this dispatch group on first use.
    auto map_it = m_dispatch_map.find(device_name);
    if (map_it == m_dispatch_map.end()) {
        std::vector<std::filesystem::path> candidate_libs;
        for (size_t idx = 0; idx < desc.candidate_count(); ++idx)
            candidate_libs.push_back(desc.candidate_lib(idx));
        map_it = m_dispatch_map.emplace(device_name, build_dispatch_entries(candidate_libs)).first;
    }
    auto& entries = map_it->second;
    if (entries.empty())
        return std::nullopt;

    // No id given -> the default device ("0"), mirroring the plugin's m_default_device_id rule.
    const std::string target_id = device_id.empty() ? std::string("0") : device_id;
    DispatchEntry* entry = nullptr;
    for (auto& e : entries) {
        if (e.canonical_id == target_id) {
            entry = &e;
            break;
        }
    }

    if (!entry)
        return std::nullopt;

    if (entry->winner_idx)
        return entry->winner_idx;

    // Highest score wins; ties resolve to registry order (the lowest candidate index).
    // Any runtime override is already baked into the scores by the plugin, so core stays generic.
    std::optional<size_t> winner;
    ov::DeviceCompatibilityScore best = ov::PROBE_SCORE_INCOMPATIBLE;
    for (const auto& [idx, view] : entry->per_lib) {
        if (view.score > best) {
            best = view.score;
            winner = idx;
        }
    }
    entry->winner_idx = winner;
    return winner;
}

std::map<std::string, std::string> ov::CoreImpl::dispatch_device_id_map_unsafe(const std::string& device_name,
                                                                               size_t candidate_idx) const {
    std::map<std::string, std::string> id_map;
    const auto map_it = m_dispatch_map.find(device_name);
    if (map_it == m_dispatch_map.end())
        return id_map;
    // Every device this candidate enumerated, not just the ones it won: it keeps addressing all of
    // them by the canonical ids, so its id set stays as dense as its own enumeration was.
    for (const auto& e : map_it->second) {
        if (const auto v = e.per_lib.find(candidate_idx); v != e.per_lib.end())
            id_map[v->second.internal_id] = e.canonical_id;
    }
    return id_map;
}

std::optional<std::vector<std::string>> ov::CoreImpl::dispatch_group_device_ids(const std::string& device_name) const {
    std::lock_guard<std::mutex> g_lock(get_mutex());
    auto reg_it = m_plugin_registry.find(device_name);
    if (reg_it == m_plugin_registry.end() || !reg_it->second.is_dispatch_group())
        return std::nullopt;

    auto map_it = m_dispatch_map.find(device_name);
    if (map_it == m_dispatch_map.end()) {
        const auto& desc = reg_it->second;
        std::vector<std::filesystem::path> candidate_libs;
        for (size_t idx = 0; idx < desc.candidate_count(); ++idx)
            candidate_libs.push_back(desc.candidate_lib(idx));
        map_it = m_dispatch_map.emplace(device_name, build_dispatch_entries(candidate_libs)).first;
    }

    // A device every candidate scored INCOMPATIBLE is not advertised, but its entry keeps its slot
    // so the canonical ids stay stable and every member's id map stays complete.
    std::vector<std::string> ids;
    for (const auto& entry : map_it->second) {
        const bool servable = std::any_of(entry.per_lib.begin(), entry.per_lib.end(), [](const auto& kv) {
            return kv.second.score != ov::PROBE_SCORE_INCOMPATIBLE;
        });
        if (servable)
            ids.push_back(entry.canonical_id);
    }
    return ids;
}

ov::Plugin ov::CoreImpl::get_plugin(const std::string& plugin_name) const {
    return get_plugin_impl(plugin_name, /*device_id=*/"");
}

ov::Plugin ov::CoreImpl::get_plugin(const std::string& plugin_name, const ov::AnyMap& config) const {
    // The device id (.N) was demoted to config by parse_device_config; route on it.
    std::string device_id;
    if (auto it = config.find(ov::device::id.name()); it != config.end())
        device_id = it->second.as<std::string>();
    return get_plugin_impl(plugin_name, device_id);
}

ov::Plugin ov::CoreImpl::get_plugin_impl(const std::string& plugin_name, const std::string& device_id) const {
    OV_ITT_SCOPE(FIRST_INFERENCE, ov::itt::domains::LoadTime, "CoreImpl::get_plugin");

    auto device_name = plugin_name;
    if (device_name == ov::default_device_name)
        device_name = "AUTO";
    if (device_name == "(CPU)")
        device_name = "CPU";
    stripDeviceName(device_name, "-");
    std::map<std::string, PluginDescriptor>::const_iterator it;
    {
        // Global lock to find plugin.
        // Always use global mutex if iterate over plugins or m_plugin_registry
        std::lock_guard<std::mutex> g_lock(get_mutex());

        // Plugin is not created, check that plugin is registered
        it = m_plugin_registry.find(device_name);
        if (it == m_plugin_registry.end()) {
            if (plugin_name == ov::default_device_name)
                OPENVINO_THROW("No device is provided, so AUTO device is used by default, which is not registered in "
                               "the OpenVINO Runtime.");
            else
                OPENVINO_THROW("Device with \"", device_name, "\" name is not registered in the OpenVINO Runtime");
        }
    }
    std::lock_guard<std::mutex> lock(get_mutex(device_name));

    // The instance cache key: for a dispatch group it is an internal per-winner key
    // (e.g. "<device>#1") so ids routed to different winners coexist; the public name otherwise.
    std::string instance_key = device_name;
    // Set for a dispatch group: the renaming its winner must adopt before anything addresses ids.
    std::optional<std::map<std::string, std::string>> dispatch_id_map;

    PluginDescriptor desc;
    {
        // Global lock to find plugin.
        // Always use global mutex if iterate over plugins or m_plugin_registry
        std::lock_guard<std::mutex> g_lock(get_mutex());

        desc = it->second;

        // Dispatch group: resolve the winner for the requested (or default) id and make it primary
        // (design 3.5/3.7). Ids need no translation - the winner adopts the canonical ones below.
        if (desc.is_dispatch_group()) {
            const auto winner = resolve_dispatch_winner_unsafe(device_name, desc, device_id);
            OPENVINO_ASSERT(winner.has_value(),
                            "No registered candidate library can serve device \"",
                            device_name,
                            device_id.empty() ? "" : "." + device_id,
                            "\". Please check the plugins registry and device drivers.");
            instance_key = device_name + '#' + std::to_string(*winner);
            dispatch_id_map = dispatch_device_id_map_unsafe(device_name, *winner);
            if (*winner != 0)
                std::swap(desc.m_candidates[0], desc.m_candidates[*winner]);
        }

        auto it_plugin = m_plugins.find(instance_key);
        if (it_plugin != m_plugins.end())
            return it_plugin->second;
    }

    if (instance_key != device_name)
        add_mutex(instance_key);
    // Plugin is in registry, but not created, let's create
    std::shared_ptr<void> so;
    try {
        ov::Plugin plugin;

        if (desc.primary().m_plugin_create_func) {  // static OpenVINO case or proxy plugin
            std::shared_ptr<ov::IPlugin> plugin_impl;
            desc.primary().m_plugin_create_func(plugin_impl);
            plugin = Plugin{plugin_impl, {}};
        } else {
            so = ov::util::load_shared_object(desc.primary().m_lib_location);
            std::shared_ptr<ov::IPlugin> plugin_impl;
            reinterpret_cast<ov::CreatePluginFunc*>(ov::util::get_symbol(so, ov::create_plugin_function))(plugin_impl);
            const auto& plugin_name = plugin_impl->get_device_name();

            // Check that device plugin name is the same as requested for HW plugins
            if (!plugin_name.empty() && !ov::is_virtual_device(plugin_name)) {
                OPENVINO_ASSERT(device_name.find(plugin_name) != std::string::npos,
                                desc.primary().m_lib_location,
                                " is used for ",
                                device_name,
                                " , while it contains implementation for ",
                                plugin_name);
            }
            plugin = Plugin{plugin_impl, so};
        }

        {
            plugin.set_name(device_name);

            // Set Core class reference to plugins
            std::weak_ptr<ov::ICore> mutableCore =
                std::const_pointer_cast<ov::ICore>(std::dynamic_pointer_cast<const ov::ICore>(shared_from_this()));
            plugin.set_core(std::move(mutableCore));
        }

        // A group member adopts Core's numbering before anything addresses an id, so every id that
        // crosses this boundary later - in either direction - is already the canonical one.
        if (dispatch_id_map) {
            OPENVINO_ASSERT(device_supports_internal_property(plugin, ov::internal::device_id_map.name()),
                            "Library \"",
                            desc.primary().m_lib_location.string(),
                            "\" is registered as one of several candidates for device \"",
                            device_name,
                            "\" but does not support ",
                            ov::internal::device_id_map.name(),
                            ", so it cannot adopt the device ids Core assigns. Sharing a device "
                            "name requires letting Core name the devices.");
            plugin.set_property({ov::internal::device_id_map(*dispatch_id_map)});
        }

        // configuring
        {
#ifdef PROXY_PLUGIN_ENABLED
            // Initial setup for proxy plugin.
            // It is needed for future initialization to initialize low level plugin
            if (desc.primary().m_plugin_create_func == ov::proxy::create_plugin) {
                ov::AnyMap initial_config;
                auto it = desc.primary().m_default_config.find(ov::proxy::alias_for.name());
                if (it != desc.primary().m_default_config.end()) {
                    initial_config[it->first] = it->second;
                }
                it = desc.primary().m_default_config.find(ov::proxy::device_priorities.name());
                if (it != desc.primary().m_default_config.end()) {
                    initial_config[it->first] = it->second;
                }
                it = desc.primary().m_default_config.find(ov::device::priorities.name());
                if (it != desc.primary().m_default_config.end()) {
                    // Fix fallback names in case if proxy plugin got a conflict in the process of plugins registration
                    auto priorities = it->second.as<std::vector<std::string>>();
                    auto internal_name =
                        desc.primary().m_default_config.find(ov::proxy::configuration::internal_name.name());
                    for (auto&& priority : priorities) {
                        if (priority == device_name) {
                            OPENVINO_ASSERT(internal_name != desc.primary().m_default_config.end(),
                                            "Cannot create proxy device ",
                                            device_name,
                                            ". Device has incorrect configuration.");
                            priority = internal_name->second.as<std::string>();
                        }
                    }
                    initial_config[ov::device::priorities.name()] = priorities;
                }
                plugin.set_property(initial_config);
                try {
                    plugin.get_property(ov::available_devices);
                } catch (const ov::Exception& ex) {
                    OPENVINO_THROW("Failed to create plugin for device ",
                                   device_name,
                                   "\nPlease, check your environment\n",
                                   ex.what());
                }
            }
#endif
            allowNotImplemented([&]() {
                // Add device specific value to support device_name.device_id cases
                {
                    const std::string deviceKey =
                        device_supports_internal_property(plugin, ov::internal::config_device_id.name())
                            ? ov::internal::config_device_id.name()
                            : ov::device::id.name();

                    // here we can store values like GPU.0, GPU.1 and we need to set properties to plugin
                    // for each such .0, .1, .# device to make sure plugin can handle different settings for different
                    // device IDs
                    {
                        std::unique_lock<std::mutex> g_lock(get_mutex());
                        for (auto pluginDesc : m_plugin_registry) {
                            ov::DeviceIDParser parser(pluginDesc.first);
                            if (pluginDesc.first.find(device_name) != std::string::npos &&
                                !parser.get_device_id().empty()) {
                                // A group's per-id entry concerns only a member holding that
                                // device; ids are canonical on both sides now, so none is rewritten.
                                const std::string cfg_id = parser.get_device_id();
                                if (dispatch_id_map &&
                                    std::none_of(dispatch_id_map->begin(), dispatch_id_map->end(), [&](const auto& kv) {
                                        return kv.second == cfg_id;
                                    }))
                                    continue;
                                g_lock.unlock();
                                pluginDesc.second.primary().m_default_config[deviceKey] = cfg_id;
                                plugin.set_property(pluginDesc.second.primary().m_default_config);
                            }
                        }
                    }
                }

                // set global device-id independent settings to plugin
                plugin.set_property(desc.primary().m_default_config);
            });
        }

        // add plugin as extension itself
        std::lock_guard<std::mutex> g_lock(get_mutex());

        std::vector<ov::Extension::Ptr> ext;
        if (desc.m_extension_create_func) {  // static OpenVINO case
            try {
                desc.m_extension_create_func(ext);
            } catch (const ov::Exception&) {
                // the same extension can be registered multiple times - ignore it!
            }
        } else {
            ext = try_get_extensions(so);
        }
        std::move(ext.begin(), ext.end(), std::back_inserter(m_plugin_registry.at(device_name).m_extensions));

        return m_plugins.emplace(instance_key, plugin).first->second;
    } catch (const ov::Exception& ex) {
        OPENVINO_THROW("Failed to create plugin ",
                       desc.primary().m_lib_location,
                       " for device ",
                       device_name,
                       "\n",
                       "Please, check your environment\n",
                       ex.what(),
                       "\n");
    }
}

ov::SoPtr<ov::ICompiledModel> ov::CoreImpl::compile_model(const std::shared_ptr<const ov::Model>& model_,
                                                          const std::string& device_name,
                                                          const ov::AnyMap& config) const {
    OV_ITT_SCOPE(FIRST_INFERENCE, ov::itt::domains::LoadTime, "Core::compile_model::model");
    auto patched_device_name = device_name;
    auto config_with_batch = config;
    // if auto-batching is applicable, the below function will patch the device name and config accordingly:
    const auto model = apply_auto_batching(model_, patched_device_name, config_with_batch);

    auto parsed = parse_device_name_into_config(patched_device_name,
                                                m_core_config,
                                                config_with_batch,
                                                is_proxy_device(patched_device_name));
    auto plugin = get_plugin(parsed.m_device_name, parsed.m_config);
    const auto& [cache_dir, cache_manager] = parsed.m_core_config.get_cache_config_for_device(plugin);
    auto compiled_model = import_compiled_model(plugin, {}, config, model);
    // Skip caching for proxy plugin. HW plugin will load network from the cache
    if (compiled_model) {
        // hint::compiled_blob is set and imported skip compilation
    } else if (cache_manager && device_supports_model_caching(plugin, parsed.m_config) && !is_proxy_device(plugin)) {
        emplace_cache_dir_if_supported(parsed.m_config, plugin, cache_dir);
        CacheContent cache_content{cache_manager, parsed.m_core_config.get_enable_mmap(), get_cache_model_path(config)};
        get_cache_wsh_ctx_manager().init_and_sync_context(std::filesystem::hash_value(cache_dir),
                                                          cache_content.m_shared_ctx);

        const auto compiled_config = create_compile_config(plugin, parsed.m_config);
        cache_content.m_blob_id = get_blob_id_or_compute(config, [&] {
            return ModelCache::compute_hash(model, cache_content.m_model_path, compiled_config);
        });
        cache_content.model = model;
        const auto lock = m_cache_guard.get_hash_lock(cache_content.m_blob_id);
        compiled_model = load_model_from_cache(cache_content, plugin, parsed.m_config, {}, [&]() {
            return compile_model_and_cache(plugin, model, parsed.m_config, {}, cache_content);
        });
    } else {
        compiled_model = plugin.compile_model(model, parsed.m_config);
    }
    return compiled_model;
}

ov::SoPtr<ov::ICompiledModel> ov::CoreImpl::compile_model(const std::shared_ptr<const ov::Model>& model_,
                                                          const ov::SoPtr<ov::IRemoteContext>& context,
                                                          const ov::AnyMap& config) const {
    OV_ITT_SCOPE(FIRST_INFERENCE, ov::itt::domains::LoadTime, "Core::compile_model::RemoteContext");
    if (!context)
        OPENVINO_THROW("Remote context is null");
    auto device_name = context->get_device_name();
    auto config_with_batch = config;
    // if auto-batching is applicable, the below function will patch the device name and config accordingly:
    const auto model = apply_auto_batching(model_, device_name, config_with_batch);

    auto parsed =
        parse_device_name_into_config(device_name, m_core_config, config_with_batch, is_proxy_device(device_name));
    auto plugin = get_plugin(parsed.m_device_name, parsed.m_config);
    const auto& [cache_dir, cache_manager] = parsed.m_core_config.get_cache_config_for_device(plugin);
    auto compiled_model = import_compiled_model(plugin, context, parsed.m_config, model);
    // Skip caching for proxy plugin. HW plugin will load network from the cache
    if (compiled_model) {
        // hint::compiled_blob is set and imported skip compilation
    } else if (cache_manager && device_supports_model_caching(plugin, parsed.m_config) && !is_proxy_device(plugin)) {
        emplace_cache_dir_if_supported(parsed.m_config, plugin, cache_dir);
        CacheContent cache_content{cache_manager, parsed.m_core_config.get_enable_mmap(), get_cache_model_path(config)};
        get_cache_wsh_ctx_manager().init_and_sync_context(std::filesystem::hash_value(cache_dir),
                                                          cache_content.m_shared_ctx);
        const auto compiled_config = create_compile_config(plugin, parsed.m_config);
        cache_content.m_blob_id = get_blob_id_or_compute(config, [&] {
            return ModelCache::compute_hash(model, cache_content.m_model_path, compiled_config);
        });
        cache_content.model = model;

        const auto lock = m_cache_guard.get_hash_lock(cache_content.m_blob_id);
        compiled_model = load_model_from_cache(cache_content, plugin, parsed.m_config, context, [&]() {
            return compile_model_and_cache(plugin, model, parsed.m_config, context, cache_content);
        });
    } else {
        compiled_model = plugin.compile_model(model, context, parsed.m_config);
    }
    return compiled_model;
}

ov::SoPtr<ov::ICompiledModel> ov::CoreImpl::compile_model(const std::filesystem::path& model_path,
                                                          const std::string& device_name,
                                                          const ov::AnyMap& config) const {
    OV_ITT_SCOPE(FIRST_INFERENCE, ov::itt::domains::LoadTime, "Core::compile_model::Path");
    auto parsed = parse_device_config(device_name, m_core_config, config, false);
    // in case of compile_model(file_name), we need to clear-up core-level properties
    auto plugin = get_plugin(parsed.m_device_name, parsed.m_config);
    const auto& [cache_dir, cache_manager] = parsed.m_core_config.get_cache_config_for_device(plugin);
    auto compiled_model = import_compiled_model(plugin, {}, parsed.m_config, model_path);

    if (compiled_model) {
        // hint::compiled_blob is set and imported skip compilation
    } else if (cache_manager && device_supports_model_caching(plugin, parsed.m_config) && !is_proxy_device(plugin)) {
        // Skip caching for proxy plugin. HW plugin will load network from the cache
        CoreConfig::remove_core(parsed.m_config);
        emplace_cache_dir_if_supported(parsed.m_config, plugin, cache_dir);
        CacheContent cache_content{cache_manager, parsed.m_core_config.get_enable_mmap(), model_path};
        get_cache_wsh_ctx_manager().init_and_sync_context(std::filesystem::hash_value(cache_dir),
                                                          cache_content.m_shared_ctx);
        cache_content.m_blob_id = get_blob_id_or_compute(config, [&] {
            return ModelCache::compute_hash(cache_content.m_model_path, create_compile_config(plugin, parsed.m_config));
        });
        const auto lock = m_cache_guard.get_hash_lock(cache_content.m_blob_id);
        compiled_model = load_model_from_cache(cache_content, plugin, parsed.m_config, {}, [&]() {
            const auto model =
                util::read_model(model_path, "", get_extensions_copy(), parsed.m_core_config.get_enable_mmap());
            return compile_model_and_cache(plugin, model, parsed.m_config, {}, cache_content);
        });
    } else {
        compiled_model = plugin.compile_model(model_path, parsed.m_config);
    }
    return compiled_model;
}

ov::SoPtr<ov::ICompiledModel> ov::CoreImpl::compile_model(const std::string& model_str,
                                                          const ov::Tensor& weights,
                                                          const std::string& device_name,
                                                          const ov::AnyMap& config) const {
    OV_ITT_SCOPED_TASK(ov::itt::domains::OV, "Core::compile_model::from_memory");
    auto parsed = parse_device_name_into_config(device_name, m_core_config, config);
    auto plugin = get_plugin(parsed.m_device_name, parsed.m_config);
    const auto& [cache_dir, cache_manager] = parsed.m_core_config.get_cache_config_for_device(plugin);
    auto compiled_model = import_compiled_model(plugin, {}, parsed.m_config);
    // Skip caching for proxy plugin. HW plugin will load network from the cache
    if (compiled_model) {
        // hint::compiled_blob is set and imported skip compilation
    } else if (cache_manager && device_supports_model_caching(plugin, parsed.m_config) && !is_proxy_device(plugin)) {
        emplace_cache_dir_if_supported(parsed.m_config, plugin, cache_dir);
        CacheContent cache_content{cache_manager, parsed.m_core_config.get_enable_mmap()};
        get_cache_wsh_ctx_manager().init_and_sync_context(std::filesystem::hash_value(cache_dir),
                                                          cache_content.m_shared_ctx);
        cache_content.m_blob_id = get_blob_id_or_compute(config, [&] {
            return ModelCache::compute_hash(model_str, weights, create_compile_config(plugin, parsed.m_config));
        });
        const auto lock = m_cache_guard.get_hash_lock(cache_content.m_blob_id);
        compiled_model = load_model_from_cache(cache_content, plugin, parsed.m_config, {}, [&]() {
            const auto model = read_model(model_str, weights);
            return compile_model_and_cache(plugin, model, parsed.m_config, {}, cache_content);
        });
    } else {
        const auto model = read_model(model_str, weights);
        compiled_model = plugin.compile_model(model, parsed.m_config);
    }
    return compiled_model;
}

ov::SoPtr<ov::ICompiledModel> ov::CoreImpl::import_model(std::istream& model,
                                                         const std::string& device_name,
                                                         const ov::AnyMap& config) const {
    OV_ITT_SCOPED_TASK(ov::itt::domains::OV, "Core::import_model");
    auto parsed = parse_device_name_into_config(device_name, config);
    return get_plugin(parsed.m_device_name, parsed.m_config).import_model(model, parsed.m_config);
}

ov::SoPtr<ov::ICompiledModel> ov::CoreImpl::import_model(std::istream& modelStream,
                                                         const ov::SoPtr<ov::IRemoteContext>& context,
                                                         const ov::AnyMap& config) const {
    OV_ITT_SCOPED_TASK(ov::itt::domains::OV, "Core::import_model");
    OPENVINO_ASSERT(context, "Remote context must not be empty.");
    auto parsed = parse_device_name_into_config(context->get_device_name(), config);
    return get_plugin(parsed.m_device_name, parsed.m_config).import_model(modelStream, context, parsed.m_config);
}

ov::SoPtr<ov::ICompiledModel> ov::CoreImpl::import_model(const ov::Tensor& compiled_blob,
                                                         const std::string& device_name,
                                                         const ov::AnyMap& config) const {
    OV_ITT_SCOPED_TASK(ov::itt::domains::OV, "Core::import_model");
    auto parsed = parse_device_name_into_config(device_name, config);
    return get_plugin(parsed.m_device_name, parsed.m_config).import_model(compiled_blob, parsed.m_config);
}

ov::SoPtr<ov::ICompiledModel> ov::CoreImpl::import_model(const ov::Tensor& compiled_blob,
                                                         const ov::SoPtr<ov::IRemoteContext>& context,
                                                         const ov::AnyMap& config) const {
    OV_ITT_SCOPED_TASK(ov::itt::domains::OV, "Core::import_model");
    OPENVINO_ASSERT(context, "Remote context must not be empty.");
    auto parsed = parse_device_name_into_config(context->get_device_name(), config);
    return get_plugin(parsed.m_device_name, parsed.m_config).import_model(compiled_blob, context, parsed.m_config);
}

ov::SupportedOpsMap ov::CoreImpl::query_model(const std::shared_ptr<const ov::Model>& model,
                                              const std::string& device_name,
                                              const ov::AnyMap& config) const {
    OV_ITT_SCOPED_TASK(ov::itt::domains::OV, "Core::query_model");
    auto parsed = parse_device_name_into_config(device_name, config);
    return get_plugin(parsed.m_device_name, parsed.m_config).query_model(model, parsed.m_config);
}

bool ov::CoreImpl::is_hidden_device(const std::string& device_name) const {
#ifdef PROXY_PLUGIN_ENABLED
    std::lock_guard<std::mutex> lock(get_mutex());
    // Alias hides the device
    for (auto&& it : m_plugin_registry) {
        auto it_priority = it.second.primary().m_default_config.find(ov::proxy::alias_for.name());
        if (it.first == device_name || it_priority == it.second.primary().m_default_config.end())
            continue;
        auto devices = it_priority->second.as<std::vector<std::string>>();
        for (const auto& dev : devices) {
            if (dev == device_name)
                return true;
        }
    }
#endif
    return false;
}

std::vector<std::string> ov::CoreImpl::get_available_devices() const {
    std::vector<std::string> devices;
    const std::string propertyName = ov::available_devices.name();

    for (auto&& device_name : get_registered_devices()) {
        std::vector<std::string> devicesIDs;
        // Skip hidden devices
        if (is_hidden_device(device_name))
            continue;
        // A dispatch group's device list is the merged canonical list core owns (each physical
        // device once), not a single candidate's - answer it here rather than delegating.
        if (auto group_ids = dispatch_group_device_ids(device_name)) {
            // The bare form is only usable when the sole servable id is the default ("0"), which is
            // what a bare request resolves to; otherwise the qualified id is the only way in.
            if (group_ids->size() == 1 && group_ids->front() == "0") {
                devices.push_back(std::move(device_name));
            } else {
                for (auto&& id : *group_ids)
                    devices.push_back(device_name + '.' + id);
            }
            continue;
        }
        try {
            devicesIDs = get_property(device_name, ov::available_devices.name(), {}).as<std::vector<std::string>>();
        } catch (const ov::Exception&) {
            // plugin is not created by e.g. invalid env
        } catch (const std::runtime_error&) {
            // plugin is not created by e.g. invalid env
        } catch (const std::exception& ex) {
            OPENVINO_THROW("An exception is thrown while trying to create the ",
                           device_name,
                           " device and call GetMetric: ",
                           ex.what());
        } catch (...) {
            OPENVINO_THROW("Unknown exception is thrown while trying to create the ",
                           device_name,
                           " device and call GetMetric");
        }

        if (devicesIDs.size() > 1) {
            for (auto&& deviceID : devicesIDs) {
                devices.push_back(device_name + '.' + deviceID);
            }
        } else if (!devicesIDs.empty()) {
            devices.push_back(std::move(device_name));
        }
    }

    return devices;
}

ov::SoPtr<ov::IRemoteContext> ov::CoreImpl::create_context(const std::string& device_name, const AnyMap& params) const {
    auto parsed = ov::parse_device_name_into_config(device_name, params);
    return get_plugin(parsed.m_device_name, parsed.m_config).create_context(parsed.m_config);
}

ov::AnyMap ov::CoreImpl::get_supported_property(const std::string& full_device_name,
                                                const ov::AnyMap& user_properties,
                                                const bool keep_core_property) const {
    if (ov::is_virtual_device(full_device_name)) {
        // Considerations:
        // 1. in case of virtual devices all the magic will happen on the level when
        // virtual device calls ICore::get_supported_property for real HW devices
        // so, for now we can return user properties almost as is without any
        // filtering / flattening
        // 2. The only exception here: while common properties like ov::num::streams or
        // ov::hint::performance_mode are shared across all the devices, the
        // ov::device::priority cannot be shared, because it's specific for current virtual
        // plugin. So, we need to remove ov::device::priorities from the list, because it's
        // supposed to be set for current virtual plugin and cannot be propagated down
        auto return_properties = user_properties;
        auto device_priorities_it = return_properties.find(ov::device::priorities.name());
        if (device_priorities_it != return_properties.end()) {
            return_properties.erase(device_priorities_it);
        }

        return return_properties;
    }

    const auto flattened = parse_device_config(full_device_name, {}, user_properties, keep_core_property);
    const auto& flattened_config = flattened.m_config;
    const auto& device_name = flattened.m_device_name;

    // The id was demoted out of the name into the flattened config; pass it back as an argument so
    // a dispatch group answers from the requested device's winner, not the default device's.
    ov::AnyMap device_id_arg;
    if (const auto it = flattened_config.find(ov::device::id.name()); it != flattened_config.end())
        device_id_arg[ov::device::id.name()] = it->second;

    // virtual plugins should bypass core-level properties to HW plugins
    // so, we need to report them as supported
    std::vector<std::string> supported_config_keys;
    auto key_inserter = std::back_inserter(supported_config_keys);
    if (keep_core_property) {
        key_inserter = std::copy(core_properties_names.begin(), core_properties_names.end(), key_inserter);
        key_inserter = std::copy(auto_batch_properties_names.begin(), auto_batch_properties_names.end(), key_inserter);
    }

    // try to search against OV API 2.0' mutable supported_properties
    try {
        for (auto&& property : ICore::get_property(device_name, ov::supported_properties, device_id_arg)) {
            if (property.is_mutable()) {
                *key_inserter = std::move(property);
            }
        }
    } catch (ov::Exception&) {
    }

    // try to search against internal supported_properties
    try {
        for (auto&& property : ICore::get_property(device_name, ov::internal::supported_properties, device_id_arg)) {
            if (property.is_mutable()) {
                *key_inserter = std::move(property);
            }
        }
    } catch (ov::Exception&) {
    }

    // collect supported properties for HW device
    AnyMap supported_config;
    for (auto&& kvp : flattened_config) {
        if (util::contains(supported_config_keys, kvp.first)) {
            supported_config[kvp.first] = kvp.second;
        }
    }

    return supported_config;
}

ov::SoPtr<ov::IRemoteContext> ov::CoreImpl::get_default_context(const std::string& device_name) const {
    auto parsed = ov::parse_device_name_into_config(device_name);
    return get_plugin(parsed.m_device_name, parsed.m_config).get_default_context(parsed.m_config);
}

std::shared_ptr<const ov::Model> ov::CoreImpl::apply_auto_batching(const std::shared_ptr<const ov::Model>& model,
                                                                   std::string& device_name,
                                                                   ov::AnyMap& config) const {
    std::string deviceNameWithBatchSize, deviceNameWithoutBatch;
    // fully strict dims tracking by default (Auto-Batching is enabled implicitly)
    bool strictly_check_dims = true;
    if (device_name.find("BATCH") != std::string::npos) {
        // explicitly enabled Auto-Batching
        auto pos = device_name.find_first_of(":");
        if (pos == std::string::npos)
            return model;  // BATCH device is already configured via the config

        deviceNameWithBatchSize = device_name.substr(pos + 1);
        deviceNameWithoutBatch = ov::DeviceIDParser::get_batch_device(deviceNameWithBatchSize);
        if (device_name.find("(") == std::string::npos) {
            auto parsed = ov::parse_device_name_into_config(deviceNameWithoutBatch);
            // Route by id: for a dispatch group ".N" selects that device's winner, not the default.
            if (!get_plugin(parsed.m_device_name, parsed.m_config)
                     .is_property_supported(ov::optimal_batch_size.name())) {
                return model;
            }
        }
        // when user sets the BATCH device explicitly, we may check the dims less strictly
        // as the result is being checked by the user
        strictly_check_dims = false;
    } else {
        // check if Auto-Batch plugin registered
        try {
            get_plugin("BATCH");
        } catch (const std::runtime_error&) {
            return model;
        }

        // check whether the Auto-Batching is disabled explicitly
        const auto& batch_mode = config.find(ov::hint::allow_auto_batching.name());
        if (batch_mode != config.end()) {
            const auto disabled = !batch_mode->second.as<bool>();
            // virtual plugins like AUTO/MULTI will need the config
            // e.g. to deduce the #requests correctly
            // proxy plugin should also keep the config
            // otherwise, no need for this config key in the rest of loading
            if (!ov::is_virtual_device(device_name) && !is_proxy_device(device_name))
                config.erase(batch_mode);
            if (disabled)
                return model;
        }

        // check whether if the Auto-Batching is applicable to the device
        auto parsed = ov::parse_device_name_into_config(device_name);
        // Do not apply auto batch for proxy device
        if (is_proxy_device(parsed.m_device_name))
            return model;
        deviceNameWithoutBatch = device_name;
        // Route by id: for a dispatch group ".N" selects that device's winner, not the default.
        if (!get_plugin(parsed.m_device_name, parsed.m_config)
                 .is_property_supported(ov::optimal_batch_size.name(), parsed.m_config)) {
            return model;
        }

        // if applicable, the Auto-Batching is implicitly enabled via the performance hints
        bool bTputInPlg = get_plugin(parsed.m_device_name, parsed.m_config)
                              .get_property(ov::hint::performance_mode.name(), parsed.m_config)
                              .as<ov::hint::PerformanceMode>() == ov::hint::PerformanceMode::THROUGHPUT;
        const auto& mode = config.find(ov::hint::performance_mode.name());
        bool bTputInLoadCfg = (mode != config.end() &&
                               mode->second.as<ov::hint::PerformanceMode>() == ov::hint::PerformanceMode::THROUGHPUT);
        const auto& excl = config.find(ov::internal::exclusive_async_requests.name());
        bool bExclReqsEnabled = (excl != config.end() && excl->second.as<bool>() == true);
        if (bExclReqsEnabled || (!bTputInPlg && !bTputInLoadCfg))
            return model;
    }
    auto batchConfig = deviceNameWithBatchSize.empty() ? deviceNameWithoutBatch : deviceNameWithBatchSize;
    auto res = ov::details::is_model_batchable(model, deviceNameWithoutBatch, strictly_check_dims);
    switch (res) {
    case ov::details::NetworkBatchAbility::NO:
        return model;
    case ov::details::NetworkBatchAbility::AS_IS:
        device_name = "BATCH:" + batchConfig;
        break;
    case ov::details::NetworkBatchAbility::WITH_HETERO:
        device_name = "HETERO:BATCH," + deviceNameWithoutBatch;
        config.insert(ov::device::properties("BATCH", ov::device::priorities(batchConfig)));
        break;
    }
    return ov::details::apply_batch_affinity(model, deviceNameWithoutBatch);
}

void ov::CoreImpl::set_property(const std::string& device_name, const AnyMap& properties) {
    OPENVINO_ASSERT(device_name.find("HETERO:") != 0,
                    "set_property is supported only for HETERO itself (without devices). "
                    "You can configure the devices with set_property before creating the HETERO on top.");
    OPENVINO_ASSERT(device_name.find("MULTI:") != 0,
                    "set_property is supported only for MULTI itself (without devices). "
                    "You can configure the devices with set_property before creating the MULTI on top.");
    OPENVINO_ASSERT(device_name.find("AUTO:") != 0,
                    "set_property is supported only for AUTO itself (without devices). "
                    "You can configure the devices with set_property before creating the AUTO on top.");
    OPENVINO_ASSERT(device_name.find("BATCH:") != 0,
                    "set_property is supported only for BATCH itself (without devices). "
                    "You can configure the devices with set_property before creating the BATCH on top.");

    // unsupport to set ov::device::properties to HW device through this function
    auto devices = get_registered_devices();
    for (auto&& config : properties) {
        const auto is_secondary_property = config.first.find(ov::device::properties.name()) != std::string::npos;
        // It is valid change for proxy plugin, proxy plugin allows to set properties for low level fallback devices
        const auto is_proxy = is_proxy_device(ov::parse_device_name_into_config(device_name).m_device_name);
        OPENVINO_ASSERT(!is_secondary_property || is_proxy,
                        "set_property do not support ov::device::propreties. "
                        "You can configure the devices through the compile_model()/query_model() API.");
    }
    set_property_for_device(properties, device_name);
}

ov::Any ov::CoreImpl::get_property_for_core(const std::string& name) const {
    if (name == ov::force_tbb_terminate.name()) {
        const auto flag = ov::threading::executor_manager()->get_property(name).as<bool>();
        return decltype(ov::force_tbb_terminate)::value_type(flag);
    } else if (name == ov::cache_dir.name()) {
        return ov::Any(util::path_to_string(m_core_config.get_cache_dir()));
    } else if (name == ov::cache_path.name()) {
        return {m_core_config.get_cache_dir()};
    } else if (name == ov::enable_mmap.name()) {
        const auto flag = m_core_config.get_enable_mmap();
        return decltype(ov::enable_mmap)::value_type(flag);
    }

    OPENVINO_THROW("Exception is thrown while trying to call get_property with unsupported property: '", name, "'");
}

ov::Any ov::CoreImpl::get_property(const std::string& device_name,
                                   const std::string& name,
                                   const AnyMap& options) const {
    OPENVINO_ASSERT(device_name.find("HETERO:") != 0,
                    "You can only get_property of the HETERO itself (without devices). "
                    "get_property is also possible for the individual devices before creating the HETERO on top.");
    OPENVINO_ASSERT(device_name.find("MULTI:") != 0,
                    "You can only get_property of the MULTI itself (without devices). "
                    "get_property is also possible for the individual devices before creating the MULTI on top.");
    OPENVINO_ASSERT(device_name.find("AUTO:") != 0,
                    "You can only get_property of the AUTO itself (without devices). "
                    "get_property is also possible for the individual devices before creating the AUTO on top.");
    OPENVINO_ASSERT(device_name.find("BATCH:") != 0,
                    "You can only get_property of the BATCH itself (without devices). "
                    "get_property is also possible for the individual devices before creating the BATCH on top.");

    auto parsed = parse_device_name_into_config(device_name, options);

    if (parsed.m_device_name.empty()) {
        return get_property_for_core(name);
    } else if (name == ov::cache_dir.name()) {
        return util::path_to_string(
            m_core_config.get_cache_config_for_device(get_plugin(parsed.m_device_name, parsed.m_config)).m_cache_dir);
    } else if (name == ov::cache_path.name()) {
        return {
            m_core_config.get_cache_config_for_device(get_plugin(parsed.m_device_name, parsed.m_config)).m_cache_dir};
    } else if (name == ov::available_devices.name()) {
        // A group's device list is core's merged canonical list, not a candidate's view; a group
        // serving nothing answers empty, as an ordinary plugin with no devices does.
        if (auto group_ids = dispatch_group_device_ids(parsed.m_device_name))
            return *group_ids;
    }
    // Route by id so a per-device query (e.g. get_property("<device>.1", architecture)) is
    // answered by that device's winner; a bare-name query resolves to the default-device winner.
    return get_plugin(parsed.m_device_name, parsed.m_config).get_property(name, parsed.m_config);
}

void ov::CoreImpl::unload_plugin(const std::string& device_name) {
    std::lock_guard<std::mutex> lock(get_mutex());

    // A dispatch group caches its resolved instances under internal keys (e.g. "GPU#0",
    // "GPU#1"), not under the bare name; erase all of them so unload frees every winner.
    const auto reg_it = m_plugin_registry.find(device_name);
    if (reg_it != m_plugin_registry.end() && reg_it->second.is_dispatch_group()) {
        const std::string prefix = device_name + '#';
        for (auto it = m_plugins.begin(); it != m_plugins.end();) {
            if (it->first.rfind(prefix, 0) == 0)
                it = m_plugins.erase(it);
            else
                ++it;
        }
        // Having nothing to unload is not an error here: a group answers available_devices from the
        // probe, so enumerating it never constructs a winner. An unknown name still throws below.
        m_dispatch_map.erase(device_name);
        reg_it->second.m_extensions.clear();
        return;
    }

    auto it = m_plugins.find(device_name);
    if (it == m_plugins.end()) {
        OPENVINO_THROW("Device with \"", device_name, "\" name is not registered in the OpenVINO Runtime");
    }
    m_plugin_registry[device_name].m_extensions.clear();
    m_plugins.erase(device_name);
}

void ov::CoreImpl::register_plugin(const std::filesystem::path& plugin,
                                   const std::string& device_name,
                                   const ov::AnyMap& properties) {
    if (device_name.find('.') != std::string::npos) {
        OPENVINO_THROW("Device name must not contain dot '.' symbol");
    }

    // Device lock before the global one, in get_plugin_impl's order: appending a candidate must not
    // race a concurrent first load, which would insert a pre-append instance afterwards.
    add_mutex(device_name);
    std::lock_guard<std::mutex> dev_lock(get_mutex(device_name));
    std::lock_guard<std::mutex> lock(get_mutex());

    // A DIFFERENT library under an already-registered name appends a dispatch-group candidate; the
    // enumeration probe picks the winner per device. (Proxy plugins keep their own handling.)
    auto it = m_plugin_registry.find(device_name);
    if (it != m_plugin_registry.end() && !is_proxy_device(device_name)) {
        const auto lib_path = ov::util::get_plugin_path(plugin);
        // A duplicated candidate enumerates the same devices with the same scores as the one
        // already registered, so it could only ever shadow itself: reject it.
        for (size_t i = 0; i < it->second.candidate_count(); ++i) {
            if (it->second.candidate_lib(i) == lib_path)
                OPENVINO_THROW("Library \"",
                               lib_path.string(),
                               "\" is already registered as device \"",
                               device_name,
                               "\". Sharing a device name requires registering a different library under it.");
        }
        it->second.m_candidates.push_back({lib_path, properties, nullptr});
        // Adding a candidate changes resolution: drop the cached merged device list and any
        // instance already created for this name (the bare single-candidate instance, or a
        // stale "<device>#N" winner) so the next request re-probes and re-selects.
        m_dispatch_map.erase(device_name);
        const std::string group_prefix = device_name + '#';
        for (auto p = m_plugins.begin(); p != m_plugins.end();) {
            if (p->first == device_name || p->first.rfind(group_prefix, 0) == 0)
                p = m_plugins.erase(p);
            else
                ++p;
        }
        return;
    }

    PluginDescriptor desc{ov::util::get_plugin_path(plugin), properties};
    register_plugin_in_registry_unsafe(device_name, desc);
}

/**
 * @brief Provides a list of plugin names in registry; physically such plugins may not be created
 * @return A list of plugin names
 */
std::vector<std::string> ov::CoreImpl::get_registered_devices() const {
    std::lock_guard<std::mutex> lock(get_mutex());

    std::vector<std::string> listOfDevices;
    for (auto&& pluginDesc : m_plugin_registry) {
        listOfDevices.push_back(pluginDesc.first);
    }

    return listOfDevices;
}

size_t ov::CoreImpl::get_registered_candidate_count(const std::string& device_name) const {
    std::lock_guard<std::mutex> lock(get_mutex());

    const auto it = m_plugin_registry.find(device_name);
    return it == m_plugin_registry.end() ? 0 : it->second.candidate_count();
}

/**
 * @brief Sets property values for a plugin or set of plugins
 * @param device_name A device name to set config to
 *        If empty, config is set for all the plugins / plugin's meta-data
 * @note  `device_name` is not allowed in form of MULTI:CPU, HETERO:GPU,CPU, AUTO:CPU
 *        just simple forms like CPU, GPU, MULTI, GPU.0, etc
 */
void ov::CoreImpl::set_property_for_device(const ov::AnyMap& config, const std::string& device_name) {
    auto cfg_copy = config;
    if (cfg_copy.empty()) {
        return;
    }

    ov::DeviceIDParser parser(device_name);
    std::string clear_device_name = parser.get_device_name();

    std::vector<std::pair<std::string, ov::Plugin>> created_plugins;
    // Dispatch-group routing: the instance key of the id's winner.
    bool is_group = false, group_select_none = false;
    std::string group_key;
    {
        std::lock_guard<std::mutex> lock(get_mutex());
        created_plugins.reserve(m_plugins.size());
        m_core_config.set_and_update(cfg_copy, clear_device_name);

        if (!cfg_copy.empty()) {
            auto base_desc = m_plugin_registry.find(clear_device_name);
            if (m_plugin_registry.find(device_name) == m_plugin_registry.end() &&
                base_desc != m_plugin_registry.end()) {
                PluginDescriptor desc{base_desc->second.primary().m_lib_location,
                                      cfg_copy,
                                      base_desc->second.m_list_of_extensions};
                m_plugin_registry[device_name] = std::move(desc);
            }

            // set config for plugins in registry
            bool configIsSet = false;
            for (auto& desc : m_plugin_registry) {
                if (device_name.empty() || device_name == desc.first) {
                    for (auto&& conf : cfg_copy) {
                        for (auto& candidate : desc.second.m_candidates) {
                            candidate.m_default_config[conf.first] = conf.second;
                        }
                    }
                    configIsSet = true;
                }
            }

            if (!configIsSet && !device_name.empty()) {
                OPENVINO_THROW("Device with \"", device_name, "\" name is not registered in the OpenVINO Runtime");
            }
        }

        // A group's live instances are keyed "<device>#<candidate>", which the bare-name match
        // below never selects; an id-qualified request needs only that id's winner, by its own id.
        const auto reg_it = m_plugin_registry.find(clear_device_name);
        is_group = reg_it != m_plugin_registry.end() && reg_it->second.is_dispatch_group();
        const std::string group_prefix = clear_device_name + '#';
        // Only a live instance needs routing; resolving otherwise would probe the candidates here.
        const bool group_live = is_group && std::any_of(m_plugins.begin(), m_plugins.end(), [&](const auto& p) {
                                    return p.first.rfind(group_prefix, 0) == 0;
                                });
        if (group_live && !parser.get_device_id().empty()) {
            // Resolve this id's winner as get_plugin would: relying on an already-set winner_idx
            // would leave an unresolved id targeting every group instance with the canonical id.
            const auto winner =
                resolve_dispatch_winner_unsafe(clear_device_name, reg_it->second, parser.get_device_id());
            if (winner) {
                group_key = group_prefix + std::to_string(*winner);
            } else {
                group_select_none = true;  // no candidate serves this id: configure nothing
            }
        }

        // set config for already created plugins
        for (auto& plugin : m_plugins) {
            bool selected = device_name.empty() || clear_device_name == plugin.first;
            if (!selected && is_group && !group_select_none) {
                selected = group_key.empty() ? plugin.first.rfind(group_prefix, 0) == 0 : plugin.first == group_key;
            }
            if (selected) {
                created_plugins.emplace_back(std::pair<std::string, ov::Plugin>{plugin.first, plugin.second});
            }
        }
    }

    for (auto& plugin : created_plugins) {
        allowNotImplemented([&]() {
            std::lock_guard<std::mutex> lock(get_mutex(plugin.first));
            auto config_copy = cfg_copy;
            // Add device specific value to support device_name.device_id cases
            {
                if (!parser.get_device_id().empty()) {
                    const std::string deviceKey =
                        device_supports_internal_property(plugin.second, ov::internal::config_device_id.name())
                            ? ov::internal::config_device_id.name()
                            : ov::device::id.name();
                    config_copy[deviceKey] = parser.get_device_id();
                }
            }
            plugin.second.set_property(config_copy);
        });
    }
}
void ov::CoreImpl::add_extensions_unsafe(const std::vector<ov::Extension::Ptr>& exts) const {
    for (const auto& ext : exts) {
        m_extensions.emplace_back(ext);
        auto ext_obj = ext;
        if (auto so_ext = std::dynamic_pointer_cast<ov::detail::SOExtension>(ext_obj))
            ext_obj = so_ext->extension();
        if (auto op_base_ext = std::dynamic_pointer_cast<ov::BaseOpExtension>(ext_obj)) {
            for (const auto& attached_ext : op_base_ext->get_attached_extensions()) {
                m_extensions.emplace_back(attached_ext);
            }
        }
    }
}

std::vector<ov::Extension::Ptr> ov::CoreImpl::get_extensions_copy() const {
    std::lock_guard<std::mutex> lock(get_mutex());
    auto only_extensions = m_extensions;
    auto ext_it = std::back_inserter(only_extensions);
    for (const auto& [_, plugin_desc] : m_plugin_registry) {
        std::copy(plugin_desc.m_extensions.begin(), plugin_desc.m_extensions.end(), ext_it);
    }
    return only_extensions;
};

void ov::CoreImpl::add_extension(const std::vector<ov::Extension::Ptr>& extensions) {
    std::lock_guard<std::mutex> lock(get_mutex());
    add_extensions_unsafe(extensions);
}

bool ov::CoreImpl::device_supports_model_caching(const std::string& device_name) const {
    auto parsed = parse_device_name_into_config(device_name);
    // Route on the id: for a dispatch group the answer belongs to that device's winner, which
    // need not be the default device's.
    return device_supports_model_caching(get_plugin(parsed.m_device_name, parsed.m_config));
}

bool ov::CoreImpl::device_supports_model_caching(const ov::Plugin& plugin, const ov::AnyMap& arguments) const {
    ov::AnyMap properties_to_virtual_dev = arguments.empty() ? ov::AnyMap{ov::device::priorities("")} : arguments;
    return ov::is_virtual_device(plugin.get_name()) ? plugin.supports_model_caching(properties_to_virtual_dev)
                                                    : plugin.supports_model_caching();
}

bool ov::CoreImpl::device_supports_internal_property(const ov::Plugin& plugin, const ov::PropertyName& key) const {
    return util::contains(plugin.get_property(ov::internal::supported_properties), key);
}

ov::SoPtr<ov::ICompiledModel> ov::CoreImpl::compile_model_and_cache(ov::Plugin& plugin,
                                                                    const std::shared_ptr<const ov::Model>& model,
                                                                    const ov::AnyMap& parsed_config,
                                                                    const ov::SoPtr<ov::IRemoteContext>& context,
                                                                    const CacheContent& cache_content) const {
    OV_ITT_SCOPED_TASK(ov::itt::domains::OV, "CoreImpl::compile_model_and_cache");

    const auto get_cfg_with_shared_ctx = [](auto&& parsed_config, auto&& cache_content) {
        auto config_with_shared_ctx = parsed_config;
        config_with_shared_ctx.emplace(ov::internal::model_sharing_context(cache_content.m_shared_ctx->get_context()));
        return config_with_shared_ctx;
    };
    const auto& cfg =
        cache_content.m_shared_ctx && device_supports_internal_property(plugin, ov::internal::model_sharing_context)
            ? get_cfg_with_shared_ctx(parsed_config, cache_content)
            : parsed_config;

    auto compiled_model = context ? plugin.compile_model(model, context, cfg) : plugin.compile_model(model, cfg);
    if (cache_content.m_cache_manager && device_supports_model_caching(plugin)) {
        try {
            // need to export network for further import from "cache"
            OV_ITT_SCOPE(FIRST_INFERENCE, ov::itt::domains::LoadTime, "Core::compile_model::Export");
            std::string compiled_model_runtime_properties;
            if (device_supports_internal_property(plugin, ov::internal::compiled_model_runtime_properties.name())) {
                compiled_model_runtime_properties =
                    plugin.get_property(ov::internal::compiled_model_runtime_properties.name(), {}).as<std::string>();
            }
            try {
                if (cache_content.m_shared_ctx) {
                    compiled_model->m_weight_context = cache_content.m_shared_ctx->get_context();
                    auto ctx = compiled_model->get_property(ov::internal::model_sharing_context.name())
                                   .as<ov::internal::WeightSharingCtxPtr>();
                    if (ctx) {
                        cache_content.m_shared_ctx->write_context(*ctx);
                    }
                }
            } catch (const ov::Exception&) {
                // do nothing if compile model will not return it
            }
            // write compiled blob
            cache_content.m_cache_manager->write_cache_entry(cache_content.m_blob_id, [&](std::ostream& stream) {
                uint32_t header_size_alignment{};
                if (device_supports_internal_property(plugin, ov::internal::cache_header_alignment.name())) {
                    header_size_alignment =
                        plugin.get_property(ov::internal::cache_header_alignment.name(), {}).as<uint32_t>();
                }

                stream << ov::CompiledBlobHeader(ov::get_openvino_version().buildNumber,
                                                 ov::ModelCache::calculate_file_info(cache_content.m_model_path),
                                                 compiled_model_runtime_properties,
                                                 header_size_alignment);
                compiled_model->export_model(stream);
            });
        } catch (const std::ios_base::failure&) {
            cache_content.m_cache_manager->remove_cache_entry(cache_content.m_blob_id);
        } catch (const ov::Exception&) {
            cache_content.m_cache_manager->remove_cache_entry(cache_content.m_blob_id);
        } catch (...) {
            cache_content.m_cache_manager->remove_cache_entry(cache_content.m_blob_id);
            throw;
        }
    }
    return compiled_model;
}

ov::SoPtr<ov::ICompiledModel> ov::CoreImpl::load_model_from_cache(
    const CacheContent& cache_content,
    ov::Plugin& plugin,
    const ov::AnyMap& config,
    const ov::SoPtr<ov::IRemoteContext>& context,
    std::function<ov::SoPtr<ov::ICompiledModel>()> compile_model_lambda) const {
    ov::SoPtr<ov::ICompiledModel> compiled_model;
    struct HeaderException {};

    OPENVINO_ASSERT(cache_content.m_cache_manager != nullptr);

    try {
        cache_content.m_cache_manager->read_cache_entry(
            cache_content.m_blob_id,
            cache_content.m_mmap_enabled && device_supports_internal_property(plugin, ov::internal::caching_with_mmap),
            [&](ICacheManager::CompiledBlobVariant& compiled_blob) {
                OV_ITT_SCOPE(FIRST_INFERENCE,
                             ov::itt::domains::LoadTime,
                             "Core::load_model_from_cache::ReadStreamAndImport");
                ov::CompiledBlobHeader header;
                size_t compiled_blob_offset = 0;
                try {
                    ov::util::VariantVisitor header_reader{[&](const ov::Tensor& tensor) {
                                                               header.read_from_buffer(
                                                                   static_cast<const char*>(tensor.data()),
                                                                   tensor.get_byte_size(),
                                                                   compiled_blob_offset);
                                                           },
                                                           [&](std::reference_wrapper<std::istream> stream) {
                                                               stream >> header;
                                                           }};
                    std::visit(header_reader, compiled_blob);

                    if (header.get_file_info() != ov::ModelCache::calculate_file_info(cache_content.m_model_path)) {
                        // Original file is changed, don't use cache
                        OPENVINO_THROW("Original model file is changed");
                    }
                    if (device_supports_internal_property(plugin,
                                                          ov::internal::compiled_model_runtime_properties_supported)) {
                        ov::AnyMap compiled_model_runtime_properties = {
                            {ov::internal::compiled_model_runtime_properties.name(),
                             std::string(header.get_runtime_info())}};
                        auto res = plugin.get_property(ov::internal::compiled_model_runtime_properties_supported.name(),
                                                       compiled_model_runtime_properties);
                        if (!res.as<bool>()) {
                            OPENVINO_THROW(
                                "Original model runtime properties have been changed, not supported anymore!");
                        }
                    } else {
                        // Check whether the runtime version is not older than blob version
                        if (!ov::util::is_version_compatible(
                                ov::util::Version(header.get_openvino_version()),
                                ov::util::Version(ov::get_openvino_version().buildNumber))) {
                            // Build number mismatch, don't use this cache
                            OPENVINO_THROW("Version does not match");
                        }
                    }
                } catch (...) {
                    throw HeaderException();
                }

                ov::AnyMap update_config = config;
                update_config[ov::loaded_from_cache.name()] = true;
                if (cache_content.model && plugin.is_property_supported(ov::hint::model.name())) {
                    update_config[ov::hint::model.name()] = cache_content.model;
                }

                if (plugin.is_property_supported(ov::weights_path.name())) {
                    std::filesystem::path weights_path;

                    if (auto&& path_hint = update_config.find(ov::weights_path.name());
                        path_hint != update_config.end()) {
                        weights_path = path_hint->second.as<std::string>();
                    } else if (weights_path = extract_weight_path(header.get_runtime_info()); weights_path.empty()) {
                        weights_path = cache_content.m_model_path;
                    }
                    weights_path.replace_extension(".bin");

                    if (ov::util::file_exists(weights_path)) {
                        update_config[ov::weights_path.name()] = util::path_to_string(weights_path);
                    }
                }
                if (device_supports_internal_property(plugin, ov::internal::model_sharing_context)) {
                    if (cache_content.m_shared_ctx) {
                        update_config.emplace(
                            ov::internal::model_sharing_context(cache_content.m_shared_ctx->get_context()));
                    }
                }

                ov::util::VariantVisitor model_importer{
                    [&](const ov::Tensor& compiled_blob) -> ov::SoPtr<ov::ICompiledModel> {
                        const ov::Tensor compiled_blob_without_header{compiled_blob,
                                                                      {compiled_blob_offset},
                                                                      {compiled_blob.get_size()}};
                        return context ? plugin.import_model(compiled_blob_without_header, context, update_config)
                                       : plugin.import_model(compiled_blob_without_header, update_config);
                    },
                    [&](std::reference_wrapper<std::istream> stream) -> ov::SoPtr<ov::ICompiledModel> {
                        return context ? plugin.import_model(stream, context, update_config)
                                       : plugin.import_model(stream, update_config);
                    }};
                compiled_model = std::visit(model_importer, compiled_blob);
            });
    } catch (const HeaderException&) {
        // For these exceptions just remove old cache and set that import didn't work
        cache_content.m_cache_manager->remove_cache_entry(cache_content.m_blob_id);
    } catch (...) {
        cache_content.m_cache_manager->remove_cache_entry(cache_content.m_blob_id);
        // TODO: temporary disabled by #54335. In future don't throw only for new 'blob_outdated' exception
        // throw;
    }

    // Fallback scenario
    if (!compiled_model) {
        OPENVINO_WARN("Could not load model from cache.");
        compiled_model = compile_model_lambda();
    }
    if (compiled_model && cache_content.m_shared_ctx) {
        compiled_model->m_weight_context = cache_content.m_shared_ctx->get_context();
    }

    return compiled_model;
}

ov::AnyMap ov::CoreImpl::create_compile_config(const ov::Plugin& plugin, const ov::AnyMap& user_config) const {
    ov::AnyMap property_config;
    // 0. Move ov::device::priorities key to property_config
    auto device_priorities_it = user_config.find(ov::device::priorities.name());
    if (device_priorities_it != user_config.end()) {
        property_config[device_priorities_it->first] = device_priorities_it->second.as<std::string>();
    }

    // 1. Move DEVICE_ID key to property_config
    const bool supports_device_id = plugin.is_property_supported(ov::device::id.name());
    auto deviceIt = user_config.find(ov::device::id.name());
    if (deviceIt != user_config.end()) {
        property_config[deviceIt->first] = deviceIt->second.as<std::string>();
    } else if (supports_device_id) {
        property_config[ov::device::id.name()] = plugin.get_property(ov::device::id, {});
    }

    // 2. Extract config keys which affect compilation process
    auto caching_props = plugin.get_property(ov::internal::caching_properties, property_config);
    OPENVINO_ASSERT(!caching_props.empty(),
                    "ov::internal::caching_properties returned by ",
                    plugin.get_name(),
                    " are empty");

    ov::AnyMap compile_config;
    for (const auto& prop : caching_props) {
        // user_config values have higher priority than plugin parameters
        auto it = user_config.find(prop);
        compile_config[prop] = it == user_config.end() ? plugin.get_property(prop, property_config) : it->second;
    }

    return compile_config;
}

ov::CoreConfig::CoreConfig(const CoreConfig& other) {
    {
        std::lock_guard<std::mutex> lock(other.m_cache_config_mutex);
        m_cache_config = other.m_cache_config;
        m_devices_cache_config = other.m_devices_cache_config;
    }
    m_flag_enable_mmap = other.m_flag_enable_mmap;
}

void ov::CoreConfig::set(const ov::AnyMap& config, const std::string& device_name) {
    if (const auto cache_path = get_cache_path_from_config(config); cache_path.has_value()) {
        if (std::lock_guard<std::mutex> lock(m_cache_config_mutex); device_name.empty()) {
            // fill global cache config
            m_cache_config = CoreConfig::CacheConfig::create(*cache_path);
            // sets cache config per-device if it's not set explicitly before
            for (auto& device_cfg : m_devices_cache_config) {
                device_cfg.second = CoreConfig::CacheConfig::create(*cache_path);
            }
        } else {
            m_devices_cache_config[device_name] = CoreConfig::CacheConfig::create(*cache_path);
        }
    }

    if (const auto cfg_entry = config.find(ov::force_tbb_terminate.name()); cfg_entry != config.end()) {
        ov::threading::executor_manager()->set_property({*cfg_entry});
    }

    if (const auto cfg_entry = config.find(ov::enable_mmap.name()); cfg_entry != config.end()) {
        m_flag_enable_mmap = cfg_entry->second.as<bool>();
    }
}

void ov::CoreConfig::set_and_update(ov::AnyMap& config, const std::string& device_name) {
    set(config, device_name);
    remove_core(config);
}

void ov::CoreConfig::remove_core(ov::AnyMap& config) {
    for (const auto& name : core_properties_names) {
        config.erase(name);
    }
}

std::filesystem::path ov::CoreConfig::get_cache_dir() const {
    std::lock_guard<std::mutex> lock(m_cache_config_mutex);
    return m_cache_config.m_cache_dir;
}

bool ov::CoreConfig::get_enable_mmap() const {
    return m_flag_enable_mmap;
}

ov::CoreConfig::CacheConfig ov::CoreConfig::get_cache_config_for_device(const ov::Plugin& plugin) const {
    std::lock_guard<std::mutex> lock(m_cache_config_mutex);
    return m_devices_cache_config.count(plugin.get_name()) ? m_devices_cache_config.at(plugin.get_name())
                                                           : m_cache_config;
}

ov::CoreConfig::CacheConfig ov::CoreConfig::CacheConfig::create(const std::filesystem::path& dir) {
    auto cfg = CacheConfig{dir, nullptr};
    if (dir.extension() == ".bin") {
        cfg.m_cache_manager = std::make_shared<runtime::SingleFileStorage>(dir);
    } else if (!dir.empty()) {
        cfg.m_cache_manager = std::make_shared<FileStorageCacheManager>(dir);
    }
    return cfg;
}

std::mutex& ov::CoreImpl::get_mutex(const std::string& dev_name) const {
    std::lock_guard<std::mutex> lock(m_global_mutex);
    try {
        return m_dev_mutexes.at(dev_name);
    } catch (const std::out_of_range&) {
        OPENVINO_THROW("Cannot get mutex for device: ", dev_name);
    }
}

void ov::CoreImpl::add_mutex(const std::string& dev_name) const {
    std::lock_guard<std::mutex> lock(m_global_mutex);
    m_dev_mutexes[dev_name];
}

std::shared_ptr<ov::Model> ov::CoreImpl::read_model(const std::filesystem::path& model_path,
                                                    const std::filesystem::path& bin_path,
                                                    const AnyMap& properties) const {
    OV_ITT_SCOPE(FIRST_INFERENCE, ov::itt::domains::ReadTime, "CoreImpl::read_model from file");
    auto local_core_config = m_core_config;
    local_core_config.set(properties, {});
    return ov::util::read_model(model_path, bin_path, get_extensions_copy(), local_core_config.get_enable_mmap());
}

std::shared_ptr<ov::Model> ov::CoreImpl::read_model(const std::string& model,
                                                    const ov::Tensor& weights,
                                                    bool frontendMode) const {
    OV_ITT_SCOPE(FIRST_INFERENCE, ov::itt::domains::ReadTime, "CoreImpl::read_model from memory");
    return ov::util::read_model(model, weights, get_extensions_copy(), frontendMode);
}

std::shared_ptr<ov::Model> ov::CoreImpl::read_model(const std::shared_ptr<AlignedBuffer>& model,
                                                    const std::shared_ptr<AlignedBuffer>& weights) const {
    OV_ITT_SCOPE(FIRST_INFERENCE, ov::itt::domains::ReadTime, "CoreImpl::read_model from memory");
    return ov::util::read_model(model, weights, get_extensions_copy());
}

std::map<std::string, ov::Version> ov::CoreImpl::get_versions(const std::string& device_name) const {
    std::map<std::string, ov::Version> versions;
    std::vector<std::string> deviceNames;

    {
        // for compatibility with samples / demo
        if (device_name.find("HETERO") == 0) {
            auto pos = device_name.find_first_of(":");
            if (pos != std::string::npos) {
                deviceNames = ov::DeviceIDParser::get_hetero_devices(device_name.substr(pos + 1));
            }
            deviceNames.push_back("HETERO");
        } else if (device_name.find("MULTI") == 0) {
            auto pos = device_name.find_first_of(":");
            if (pos != std::string::npos) {
                deviceNames = ov::DeviceIDParser::get_multi_devices(device_name.substr(pos + 1));
            }
            deviceNames.push_back("MULTI");
        } else if (device_name.find("AUTO") == 0) {
            auto pos = device_name.find_first_of(":");
            if (pos != std::string::npos) {
                deviceNames = ov::DeviceIDParser::get_multi_devices(device_name.substr(pos + 1));
            }
            deviceNames.emplace_back("AUTO");
        } else if (device_name.find("BATCH") == 0) {
            auto pos = device_name.find_first_of(":");
            if (pos != std::string::npos) {
                deviceNames = {ov::DeviceIDParser::get_batch_device(device_name.substr(pos + 1))};
            }
            deviceNames.push_back("BATCH");
        } else {
            deviceNames.push_back(device_name);
        }
    }

    for (auto&& deviceName_ : deviceNames) {
        ov::DeviceIDParser parser(deviceName_);
        std::string deviceNameLocal = parser.get_device_name();

        try {
            ov::Plugin plugin = get_plugin(deviceNameLocal);
            versions[deviceNameLocal] = plugin.get_version();
        } catch (const ov::Exception& ex) {
            std::string exception(ex.what());
            if (exception.find("not registered in the OpenVINO Runtime") == std::string::npos) {
                throw;
            }
        }
    }

    return versions;
}
