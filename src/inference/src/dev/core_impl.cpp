// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core_impl.hpp"

#include <memory>
#include <variant>

#include "check_network_batchable.hpp"
#include "itt.hpp"
#include "model_reader.hpp"
#include "openvino/core/any.hpp"
#include "openvino/core/except.hpp"
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
#include "openvino/runtime/threading/executor_manager.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"
#include "openvino/util/variant_visitor.hpp"
#include "openvino/util/xml_parse_utils.hpp"
#include "ov_plugins.hpp"
#ifdef PROXY_PLUGIN_ENABLED
#    include "openvino/proxy/plugin.hpp"
#    include "openvino/proxy/properties.hpp"
#endif

ov::ICore::~ICore() = default;

namespace {

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
 * core.compile_model(model, "GPU", ov::device::properties("GPU", ov::cache_dir("/tmp")));
 * // or
 * core.compile_model(model, "GPU", ov::device::properties({
 *   { "GPU", ov::cache_dir("/tmp") },
 *   { "CPU", ov::cache_dir("") }
 * }));
 * @endcode
 * To the form:
 * @code
 * core.compile_model(model, "GPU", ov::cache_dir("/tmp"));
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

void clean_batch_properties(const std::string& deviceName, ov::AnyMap& config, const ov::PropertyName& property_name) {
    // auto-batching is not applicable, if there is auto_batch_timeout, delete it
    if (deviceName.find("BATCH") == std::string::npos) {
        const auto& batch_timeout_mode = config.find(property_name);
        if (batch_timeout_mode != config.end()) {
            if (!ov::is_virtual_device(deviceName))
                config.erase(batch_timeout_mode);
        }
    }
}

static const auto core_properties_names =
    ov::util::make_array(ov::cache_dir.name(), ov::enable_mmap.name(), ov::force_tbb_terminate.name());

static const auto auto_batch_properties_names =
    ov::util::make_array(ov::auto_batch_timeout.name(), ov::hint::allow_auto_batching.name());

ov::util::Path extract_weight_path(const std::string& compiled_properties) {
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

using model_hint_t = std::variant<std::shared_ptr<const ov::Model>, std::string>;

ov::SoPtr<ov::ICompiledModel> import_compiled_model(const ov::Plugin& plugin,
                                                    const ov::SoPtr<ov::IRemoteContext>& context,
                                                    const ov::AnyMap& config) {
    ov::SoPtr<ov::ICompiledModel> compiled_model;
    if (auto blob_hint = config.find(ov::hint::compiled_blob.name()); blob_hint != config.end()) {
        try {
            auto compiled_blob = blob_hint->second.as<ov::Tensor>();
            ov::SharedStreamBuffer buffer{reinterpret_cast<char*>(compiled_blob.data()), compiled_blob.get_byte_size()};
            std::istream stream{&buffer};
            compiled_model =
                context ? plugin.import_model(stream, context, config) : plugin.import_model(stream, config);
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
            if (model_ptr != nullptr &&
                ov::util::contains(plugin.get_property(ov::supported_properties), ov::hint::model)) {
                cfg[ov::hint::model.name()] = model_ptr;
            }
        },
        [&cfg, &plugin](const std::string& model_path) {
            if (cfg.count(ov::weights_path.name()) == 0 &&
                ov::util::contains(plugin.get_property(ov::supported_properties), ov::weights_path)) {
                ov::util::Path weights_path{model_path};
                weights_path.replace_extension(".bin");
                if (ov::util::file_exists(weights_path)) {
                    cfg[ov::weights_path.name()] = weights_path.string();
                }
            }
        }};
    std::visit(apply_model_hint, model_hint);
    return import_compiled_model(plugin, context, cfg);
}
}  // namespace

bool ov::is_config_applicable(const std::string& user_device_name, const std::string& subprop_device_name) {
    // full match
    if (user_device_name == subprop_device_name)
        return true;

    auto parsed_user_device_name = ov::parseDeviceNameIntoConfig(user_device_name);
    auto parsed_subprop_device_name = ov::parseDeviceNameIntoConfig(subprop_device_name);

    // if device name is matched, check additional condition
    auto is_matched = [&](const std::string& key, MatchType match_type) -> bool {
        const auto& user_value =
            parsed_user_device_name._config.count(key) ? parsed_user_device_name._config.at(key).as<std::string>() : "";
        const auto& subprop_value = parsed_subprop_device_name._config.count(key)
                                        ? parsed_subprop_device_name._config.at(key).as<std::string>()
                                        : "";

        if (!user_value.empty() && subprop_value.empty()) {
            // property without additional limitation can be applied
            return true;
        }
        return match_type == MatchType::EXACT ? (user_value == subprop_value) : (user_value.find(subprop_value) == 0);
        return false;
    };

    if (parsed_user_device_name._deviceName == parsed_subprop_device_name._deviceName) {
        auto device_priority = get_device_priority_property(parsed_user_device_name._deviceName);
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
    auto& updated_device_name = parsed._deviceName;
    auto& updated_config = parsed._config;

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

    parsed._core_config.set(updated_config);
    // keep batch property only when called from query_supported_property
    if (!keep_auto_batch_property) {
        for (const auto& name : auto_batch_properties_names) {
            clean_batch_properties(updated_device_name, updated_config, name);
        }
    }
    return parsed;
}
}  // namespace

ov::Parsed ov::parseDeviceNameIntoConfig(const std::string& deviceName,
                                         const AnyMap& config,
                                         const bool keep_auto_batch_property) {
    return parseDeviceNameIntoConfig(deviceName, CoreConfig{}, config, keep_auto_batch_property);
}

ov::Parsed ov::parseDeviceNameIntoConfig(const std::string& deviceName,
                                         const CoreConfig& coreConfig,
                                         const AnyMap& config,
                                         const bool keep_auto_batch_property) {
    auto parsed = parse_device_config(deviceName, coreConfig, config, keep_auto_batch_property);

    // remove core properties for HW devices
    if (!ov::is_virtual_device(parsed._deviceName)) {
        // note: ov::cache_dir kept as plugin may require it
        CoreConfig::remove_core_skip_cache_dir(parsed._config);
    }
    return parsed;
}

ov::CoreImpl::CoreImpl() {
    add_mutex("");  // Register global mutex
    m_executor_manager = ov::threading::executor_manager();
    for (const auto& it : ov::get_available_opsets()) {
        opsetNames.insert(it.first);
    }
}

bool ov::CoreImpl::is_proxy_device(const ov::Plugin& plugin) const {
    return is_proxy_device(plugin.get_name());
}
bool ov::CoreImpl::is_proxy_device(const std::string& dev_name) const {
#ifdef PROXY_PLUGIN_ENABLED
    std::string real_name = ov::parseDeviceNameIntoConfig(dev_name)._deviceName;
    return pluginRegistry.find(real_name) != pluginRegistry.end() &&
           pluginRegistry.at(real_name).pluginCreateFunc == ov::proxy::create_plugin;
#else
    return false;
#endif
}

void ov::CoreImpl::register_plugin_in_registry_unsafe(const std::string& device_name, PluginDescriptor& desc) {
#ifdef PROXY_PLUGIN_ENABLED
    // Update proxy plugin config
    const auto& fill_config = [](ov::AnyMap& defaultConfig, const ov::AnyMap& config, const std::string& dev_name) {
        // Configure aliases for proxy plugin
        auto it = config.find(ov::proxy::configuration::alias.name());
        std::string alias;
        if (it != config.end()) {
            alias = it->second.as<std::string>();
            if (defaultConfig.find(ov::proxy::alias_for.name()) == defaultConfig.end()) {
                defaultConfig[ov::proxy::alias_for.name()] = std::vector<std::string>();
            }
            defaultConfig[ov::proxy::alias_for.name()].as<std::vector<std::string>>().emplace_back(dev_name);
        }

        // Configure device order for proxy_plugin
        it = config.find(ov::proxy::configuration::priority.name());
        if (it != config.end()) {
            if (defaultConfig.find(ov::proxy::device_priorities.name()) == defaultConfig.end()) {
                defaultConfig[ov::proxy::device_priorities.name()] = std::vector<std::string>();
            }
            defaultConfig[ov::proxy::device_priorities.name()].as<std::vector<std::string>>().emplace_back(
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
            if (defaultConfig.find(ov::device::priorities.name()) == defaultConfig.end()) {
                defaultConfig[ov::device::priorities.name()] = std::vector<std::string>{dev_name, std::move(fallback)};
            } else {
                auto dev_order = defaultConfig[ov::device::priorities.name()].as<std::vector<std::string>>();
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
                defaultConfig[ov::device::priorities.name()] = dev_order;
            }
        }
    };
#endif

    std::string dev_name = device_name;
#ifdef PROXY_PLUGIN_ENABLED
    auto&& config = desc.defaultConfig;
    // Register proxy plugin
    if (config.find(ov::proxy::configuration::alias.name()) != config.end()) {
        // Create proxy plugin for alias
        const auto& alias = config.at(ov::proxy::configuration::alias.name()).as<std::string>();
        if (alias == device_name)
            dev_name = get_internal_plugin_name(dev_name, config);
        // Alias can be registered by several plugins
        if (pluginRegistry.find(alias) == pluginRegistry.end()) {
            // Register new plugin
            PluginDescriptor desc = PluginDescriptor(ov::proxy::create_plugin);
            // Add internal name for proxy in order to modify fallback order before the initialization
            if (alias == device_name)
                desc.defaultConfig[ov::proxy::configuration::internal_name.name()] = dev_name;

            fill_config(desc.defaultConfig, config, dev_name);
            pluginRegistry[alias] = std::move(desc);
            add_mutex(alias);
        } else {
            // Update registered plugin
            auto& plugin = pluginRegistry.at(alias);
            // Error if we have an alias for HW plugin
            OPENVINO_ASSERT(plugin.pluginCreateFunc == ov::proxy::create_plugin,
                            "Cannot register plugin for ",
                            dev_name,
                            " plugin with the same name already registered!");
            // Add internal name for proxy in order to modify fallback order before the initialization
            if (alias == device_name)
                plugin.defaultConfig[ov::proxy::configuration::internal_name.name()] = dev_name;
            fill_config(plugin.defaultConfig, config, dev_name);
        }
    } else if (config.find(ov::proxy::configuration::fallback.name()) != config.end()) {
        // Fallback without alias means that we need to replace original plugin to proxy
        dev_name = get_internal_plugin_name(dev_name, config);
        PluginDescriptor desc = PluginDescriptor(ov::proxy::create_plugin);
        desc.defaultConfig[ov::proxy::configuration::internal_name.name()] = dev_name;
        fill_config(desc.defaultConfig, config, dev_name);
        pluginRegistry[device_name] = std::move(desc);
        add_mutex(device_name);
    }

    const static std::vector<ov::PropertyName> proxy_conf_properties = {ov::proxy::configuration::alias,
                                                                        ov::proxy::configuration::fallback,
                                                                        ov::proxy::configuration::internal_name,
                                                                        ov::proxy::configuration::priority};

    // Register real plugin
    for (const auto& proxy_prop : proxy_conf_properties) {
        auto it = desc.defaultConfig.find(proxy_prop);
        if (it != desc.defaultConfig.end()) {
            desc.defaultConfig.erase(it);
        }
    }
#endif

    pluginRegistry[dev_name] = desc;
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
        const auto& deviceName = plugin.first;
        if (deviceName.find('.') != std::string::npos) {
            OPENVINO_THROW("Device name must not contain dot '.' symbol");
        }
#ifdef OPENVINO_STATIC_LIBRARY
        if (pluginRegistry.find(deviceName) == pluginRegistry.end()) {
            const auto& value = plugin.second;
            ov::AnyMap config = any_copy(value.m_default_config);
            PluginDescriptor desc{value.m_create_plugin_func, config, value.m_create_extensions_func};
            register_plugin_in_registry_unsafe(deviceName, desc);
        }
#else
        const auto& pluginPath = ov::util::get_compiled_plugin_path(plugin.second.m_plugin_path);
        if (pluginRegistry.find(deviceName) == pluginRegistry.end() && ov::util::file_exists(pluginPath)) {
            ov::AnyMap config = any_copy(plugin.second.m_default_config);
            PluginDescriptor desc{pluginPath, config};
            register_plugin_in_registry_unsafe(deviceName, desc);
        }
#endif
    }
}

void ov::CoreImpl::register_plugins_in_registry(const std::string& xml_config_file, const bool& by_abs_path) {
    std::lock_guard<std::mutex> lock(get_mutex());

    using namespace ov::util;
    auto parse_result = pugixml::parse_xml(xml_config_file.c_str());
    if (!parse_result.error_msg.empty()) {
        OPENVINO_THROW(parse_result.error_msg);
    }

    pugi::xml_document& xmlDoc = *parse_result.xml;

    pugi::xml_node ieNode = xmlDoc.document_element();
    pugi::xml_node devicesNode = ieNode.child("plugins");

    FOREACH_CHILD (pluginNode, devicesNode, "plugin") {
        std::string deviceName = pugixml::get_str_attr(pluginNode, "name");
        if (pluginRegistry.find(deviceName) != pluginRegistry.end()) {
            OPENVINO_THROW("Device with \"", deviceName, "\"  is already registered in the OpenVINO Runtime");
        }
        if (deviceName.find('.') != std::string::npos) {
            OPENVINO_THROW("Device name must not contain dot '.' symbol");
        }

        ov::util::FilePath pluginPath =
            ov::util::get_plugin_path(pugixml::get_str_attr(pluginNode, "location"), xml_config_file, by_abs_path);

        // check properties
        auto propertiesNode = pluginNode.child("properties");
        ov::AnyMap config;

        if (propertiesNode) {
            FOREACH_CHILD (propertyNode, propertiesNode, "property") {
                std::string key = pugixml::get_str_attr(propertyNode, "key");
                std::string value = pugixml::get_str_attr(propertyNode, "value");
                config[key] = value;
            }
        }

        // check extensions
        auto extensionsNode = pluginNode.child("extensions");
        std::vector<ov::util::FilePath> listOfExtentions;

        if (extensionsNode) {
            FOREACH_CHILD (extensionNode, extensionsNode, "extension") {
                ov::util::FilePath extensionLocation =
                    ov::util::to_file_path(pugixml::get_str_attr(extensionNode, "location").c_str());
                listOfExtentions.push_back(extensionLocation);
            }
        }

        // fill value in plugin registry for later lazy initialization
        {
            PluginDescriptor desc{pluginPath, config, listOfExtentions};
            register_plugin_in_registry_unsafe(deviceName, desc);
        }
    }
}

ov::Plugin ov::CoreImpl::get_plugin(const std::string& pluginName) const {
    OV_ITT_SCOPE(FIRST_INFERENCE, ov::itt::domains::LoadTime, "CoreImpl::get_plugin");

    auto deviceName = pluginName;
    if (deviceName == ov::DEFAULT_DEVICE_NAME)
        deviceName = "AUTO";
    if (deviceName == "(CPU)")
        deviceName = "CPU";
    stripDeviceName(deviceName, "-");
    std::map<std::string, PluginDescriptor>::const_iterator it;
    {
        // Global lock to find plugin.
        // Always use global mutex if iterate over plugins or pluginRegistry
        std::lock_guard<std::mutex> g_lock(get_mutex());

        // Plugin is not created, check that plugin is registered
        it = pluginRegistry.find(deviceName);
        if (it == pluginRegistry.end()) {
            if (pluginName == ov::DEFAULT_DEVICE_NAME)
                OPENVINO_THROW("No device is provided, so AUTO device is used by default, which is not registered in "
                               "the OpenVINO Runtime.");
            else
                OPENVINO_THROW("Device with \"", deviceName, "\" name is not registered in the OpenVINO Runtime");
        }
    }
    std::lock_guard<std::mutex> lock(get_mutex(deviceName));

    PluginDescriptor desc;
    {
        // Global lock to find plugin.
        // Always use global mutex if iterate over plugins or pluginRegistry
        std::lock_guard<std::mutex> g_lock(get_mutex());
        auto it_plugin = plugins.find(deviceName);
        if (it_plugin != plugins.end())
            return it_plugin->second;

        desc = it->second;
    }
    // Plugin is in registry, but not created, let's create
    std::shared_ptr<void> so;
    try {
        ov::Plugin plugin;

        if (desc.pluginCreateFunc) {  // static OpenVINO case or proxy plugin
            std::shared_ptr<ov::IPlugin> plugin_impl;
            desc.pluginCreateFunc(plugin_impl);
            plugin = Plugin{plugin_impl, {}};
        } else {
            so = ov::util::load_shared_object(desc.libraryLocation.c_str());
            std::shared_ptr<ov::IPlugin> plugin_impl;
            reinterpret_cast<ov::CreatePluginFunc*>(ov::util::get_symbol(so, ov::create_plugin_function))(plugin_impl);
            const auto& plugin_name = plugin_impl->get_device_name();

            // Check that device plugin name is the same as requested for HW plugins
            if (!plugin_name.empty() && !ov::is_virtual_device(plugin_name)) {
                OPENVINO_ASSERT(deviceName.find(plugin_name) != std::string::npos,
                                desc.libraryLocation,
                                " is used for ",
                                deviceName,
                                " , while it contains implementation for ",
                                plugin_name);
            }
            plugin = Plugin{plugin_impl, so};
        }

        {
            plugin.set_name(deviceName);

            // Set Core class reference to plugins
            std::weak_ptr<ov::ICore> mutableCore =
                std::const_pointer_cast<ov::ICore>(std::dynamic_pointer_cast<const ov::ICore>(shared_from_this()));
            plugin.set_core(std::move(mutableCore));
        }

        // configuring
        {
#ifdef PROXY_PLUGIN_ENABLED
            // Initial setup for proxy plugin.
            // It is needed for future initialization to initialize low level plugin
            if (desc.pluginCreateFunc == ov::proxy::create_plugin) {
                ov::AnyMap initial_config;
                auto it = desc.defaultConfig.find(ov::proxy::alias_for.name());
                if (it != desc.defaultConfig.end()) {
                    initial_config[it->first] = it->second;
                }
                it = desc.defaultConfig.find(ov::proxy::device_priorities.name());
                if (it != desc.defaultConfig.end()) {
                    initial_config[it->first] = it->second;
                }
                it = desc.defaultConfig.find(ov::device::priorities.name());
                if (it != desc.defaultConfig.end()) {
                    // Fix fallback names in case if proxy plugin got a conflict in the process of plugins registration
                    auto priorities = it->second.as<std::vector<std::string>>();
                    auto internal_name = desc.defaultConfig.find(ov::proxy::configuration::internal_name.name());
                    for (auto&& priority : priorities) {
                        if (priority == deviceName) {
                            OPENVINO_ASSERT(internal_name != desc.defaultConfig.end(),
                                            "Cannot create proxy device ",
                                            deviceName,
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
                                   deviceName,
                                   "\nPlease, check your environment\n",
                                   ex.what());
                }
            }
#endif
            // TODO: remove this block of code once GPU removes support of ov::cache_dir
            // also, remove device_supports_cache_dir at all
            {
                OPENVINO_SUPPRESS_DEPRECATED_START
                if (device_supports_cache_dir(plugin)) {
                    auto cacheConfig = coreConfig.get_cache_config_for_device(plugin);
                    if (cacheConfig._cacheManager) {
                        desc.defaultConfig[ov::cache_dir.name()] = cacheConfig._cacheDir;
                    }
                } else if (desc.defaultConfig.count(ov::cache_dir.name()) > 0) {
                    // Remove "CACHE_DIR" from config if it is not supported by plugin
                    desc.defaultConfig.erase(ov::cache_dir.name());
                }
                OPENVINO_SUPPRESS_DEPRECATED_END
            }

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
                    for (auto pluginDesc : pluginRegistry) {
                        ov::DeviceIDParser parser(pluginDesc.first);
                        if (pluginDesc.first.find(deviceName) != std::string::npos && !parser.get_device_id().empty()) {
                            pluginDesc.second.defaultConfig[deviceKey] = parser.get_device_id();
                            plugin.set_property(pluginDesc.second.defaultConfig);
                        }
                    }
                }

                // set global device-id independent settings to plugin
                plugin.set_property(desc.defaultConfig);
            });
        }

        // add plugin as extension itself
        std::lock_guard<std::mutex> g_lock(get_mutex());

        if (desc.extensionCreateFunc) {  // static OpenVINO case
            try {
                std::vector<ov::Extension::Ptr> ext;
                desc.extensionCreateFunc(ext);
                add_extensions_unsafe(ext);
            } catch (const ov::Exception&) {
                // the same extension can be registered multiple times - ignore it!
            }
        } else {
            try_to_register_plugin_extensions(desc.libraryLocation);
        }

        return plugins.emplace(deviceName, plugin).first->second;
    } catch (const ov::Exception& ex) {
        OPENVINO_THROW("Failed to create plugin ",
                       desc.libraryLocation,
                       " for device ",
                       deviceName,
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
    std::string deviceName = device_name;
    ov::AnyMap config_with_batch = config;
    // if auto-batching is applicable, the below function will patch the device name and config accordingly:
    auto model = apply_auto_batching(model_, deviceName, config_with_batch);

    auto parsed = parseDeviceNameIntoConfig(deviceName, coreConfig, config_with_batch, is_proxy_device(deviceName));
    auto plugin = get_plugin(parsed._deviceName);
    // will consume ov::cache_dir if plugin not support it
    auto cacheManager = parsed._core_config.get_cache_config_for_device(plugin, parsed._config)._cacheManager;
    auto res = import_compiled_model(plugin, {}, parsed._config, model);
    // Skip caching for proxy plugin. HW plugin will load network from the cache
    if (res) {
        // hint::compiled_blob is set and imported skip compilation
    } else if (cacheManager && device_supports_model_caching(plugin, parsed._config) && !is_proxy_device(plugin)) {
        CacheContent cacheContent{cacheManager, parsed._core_config.get_enable_mmap()};
        cacheContent.blobId = ov::ModelCache::compute_hash(model, create_compile_config(plugin, parsed._config));
        cacheContent.model = model;
        std::unique_ptr<CacheGuardEntry> lock = cacheGuard.get_hash_lock(cacheContent.blobId);
        res = load_model_from_cache(cacheContent, plugin, parsed._config, ov::SoPtr<ov::IRemoteContext>{}, [&]() {
            return compile_model_and_cache(plugin,
                                           model,
                                           parsed._config,
                                           ov::SoPtr<ov::IRemoteContext>{},
                                           cacheContent);
        });
    } else {
        res = plugin.compile_model(model, parsed._config);
    }
    return res;
}

ov::SoPtr<ov::ICompiledModel> ov::CoreImpl::compile_model(const std::shared_ptr<const ov::Model>& model_,
                                                          const ov::SoPtr<ov::IRemoteContext>& context,
                                                          const ov::AnyMap& config) const {
    OV_ITT_SCOPE(FIRST_INFERENCE, ov::itt::domains::LoadTime, "Core::compile_model::RemoteContext");
    if (!context)
        OPENVINO_THROW("Remote context is null");
    std::string deviceName = context->get_device_name();
    ov::AnyMap config_with_batch = config;
    // if auto-batching is applicable, the below function will patch the device name and config accordingly:
    auto model = apply_auto_batching(model_, deviceName, config_with_batch);

    auto parsed = parseDeviceNameIntoConfig(deviceName, coreConfig, config_with_batch, is_proxy_device(deviceName));
    auto plugin = get_plugin(parsed._deviceName);
    // will consume ov::cache_dir if plugin not support it
    auto cacheManager = parsed._core_config.get_cache_config_for_device(plugin, parsed._config)._cacheManager;
    auto res = import_compiled_model(plugin, context, parsed._config, model);
    // Skip caching for proxy plugin. HW plugin will load network from the cache
    if (res) {
        // hint::compiled_blob is set and imported skip compilation
    } else if (cacheManager && device_supports_model_caching(plugin, parsed._config) && !is_proxy_device(plugin)) {
        CacheContent cacheContent{cacheManager, parsed._core_config.get_enable_mmap()};
        cacheContent.blobId = ov::ModelCache::compute_hash(model, create_compile_config(plugin, parsed._config));
        std::unique_ptr<CacheGuardEntry> lock = cacheGuard.get_hash_lock(cacheContent.blobId);
        cacheContent.model = model;
        res = load_model_from_cache(cacheContent, plugin, parsed._config, context, [&]() {
            return compile_model_and_cache(plugin, model, parsed._config, context, cacheContent);
        });
    } else {
        res = plugin.compile_model(model, context, parsed._config);
    }
    return res;
}

ov::SoPtr<ov::ICompiledModel> ov::CoreImpl::compile_model(const std::string& model_path,
                                                          const std::string& device_name,
                                                          const ov::AnyMap& config) const {
    OV_ITT_SCOPE(FIRST_INFERENCE, ov::itt::domains::LoadTime, "Core::compile_model::Path");
    auto parsed = parse_device_config(device_name, coreConfig, config, false);
    // in case of compile_model(file_name), we need to clear-up core-level properties
    auto plugin = get_plugin(parsed._deviceName);
    // will consume ov::cache_dir if plugin not support it
    auto cacheManager = parsed._core_config.get_cache_config_for_device(plugin, parsed._config)._cacheManager;
    auto compiled_model = import_compiled_model(plugin, {}, parsed._config, model_path);

    if (compiled_model) {
        // hint::compiled_blob is set and imported skip compilation
    } else if (cacheManager && device_supports_model_caching(plugin, parsed._config) && !is_proxy_device(plugin)) {
        // Skip caching for proxy plugin. HW plugin will load network from the cache
        CoreConfig::remove_core_skip_cache_dir(parsed._config);
        CacheContent cacheContent{cacheManager, parsed._core_config.get_enable_mmap(), model_path};
        cacheContent.blobId = ov::ModelCache::compute_hash(model_path, create_compile_config(plugin, parsed._config));
        std::unique_ptr<CacheGuardEntry> lock = cacheGuard.get_hash_lock(cacheContent.blobId);
        compiled_model =
            load_model_from_cache(cacheContent, plugin, parsed._config, ov::SoPtr<ov::IRemoteContext>{}, [&]() {
                const auto model = util::read_model(model_path, "", extensions, parsed._core_config.get_enable_mmap());
                return compile_model_and_cache(plugin, model, parsed._config, {}, cacheContent);
            });
    } else {
        compiled_model = plugin.compile_model(model_path, parsed._config);
    }
    return compiled_model;
}

ov::SoPtr<ov::ICompiledModel> ov::CoreImpl::compile_model(const std::string& model_str,
                                                          const ov::Tensor& weights,
                                                          const std::string& device_name,
                                                          const ov::AnyMap& config) const {
    OV_ITT_SCOPED_TASK(ov::itt::domains::OV, "Core::compile_model::from_memory");
    auto parsed = parseDeviceNameIntoConfig(device_name, coreConfig, config);
    auto plugin = get_plugin(parsed._deviceName);
    // will consume ov::cache_dir if plugin not support it
    auto cacheManager = parsed._core_config.get_cache_config_for_device(plugin, parsed._config)._cacheManager;
    auto compiled_model = import_compiled_model(plugin, {}, parsed._config);
    // Skip caching for proxy plugin. HW plugin will load network from the cache
    if (compiled_model) {
        // hint::compiled_blob is set and imported skip compilation
    } else if (cacheManager && device_supports_model_caching(plugin, parsed._config) && !is_proxy_device(plugin)) {
        CacheContent cacheContent{cacheManager, parsed._core_config.get_enable_mmap()};
        cacheContent.blobId =
            ov::ModelCache::compute_hash(model_str, weights, create_compile_config(plugin, parsed._config));
        std::unique_ptr<CacheGuardEntry> lock = cacheGuard.get_hash_lock(cacheContent.blobId);
        compiled_model =
            load_model_from_cache(cacheContent, plugin, parsed._config, ov::SoPtr<ov::IRemoteContext>{}, [&]() {
                auto model = read_model(model_str, weights);
                return compile_model_and_cache(plugin,
                                               model,
                                               parsed._config,
                                               ov::SoPtr<ov::IRemoteContext>{},
                                               cacheContent);
            });
    } else {
        auto model = read_model(model_str, weights);
        compiled_model = plugin.compile_model(model, parsed._config);
    }
    return compiled_model;
}

ov::SoPtr<ov::ICompiledModel> ov::CoreImpl::import_model(std::istream& model,
                                                         const std::string& device_name,
                                                         const ov::AnyMap& config) const {
    OV_ITT_SCOPED_TASK(ov::itt::domains::OV, "Core::import_model");
    auto parsed = parseDeviceNameIntoConfig(device_name, config);
    return get_plugin(parsed._deviceName).import_model(model, parsed._config);
}

ov::SoPtr<ov::ICompiledModel> ov::CoreImpl::import_model(std::istream& modelStream,
                                                         const ov::SoPtr<ov::IRemoteContext>& context,
                                                         const ov::AnyMap& config) const {
    OV_ITT_SCOPED_TASK(ov::itt::domains::OV, "Core::import_model");
    OPENVINO_ASSERT(context, "Remote context must not be empty.");
    auto parsed = parseDeviceNameIntoConfig(context->get_device_name(), config);
    return get_plugin(parsed._deviceName).import_model(modelStream, context, parsed._config);
}

ov::SupportedOpsMap ov::CoreImpl::query_model(const std::shared_ptr<const ov::Model>& model,
                                              const std::string& device_name,
                                              const ov::AnyMap& config) const {
    OV_ITT_SCOPED_TASK(ov::itt::domains::OV, "Core::query_model");
    auto parsed = parseDeviceNameIntoConfig(device_name, config);
    return get_plugin(parsed._deviceName).query_model(model, parsed._config);
}

bool ov::CoreImpl::is_hidden_device(const std::string& device_name) const {
#ifdef PROXY_PLUGIN_ENABLED
    std::lock_guard<std::mutex> lock(get_mutex());
    // Alias hides the device
    for (auto&& it : pluginRegistry) {
        auto it_priority = it.second.defaultConfig.find(ov::proxy::alias_for.name());
        if (it.first == device_name || it_priority == it.second.defaultConfig.end())
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

    for (auto&& deviceName : get_registered_devices()) {
        std::vector<std::string> devicesIDs;
        // Skip hidden devices
        if (is_hidden_device(deviceName))
            continue;
        try {
            devicesIDs = get_property(deviceName, ov::available_devices.name(), {}).as<std::vector<std::string>>();
        } catch (const ov::Exception&) {
            // plugin is not created by e.g. invalid env
        } catch (const std::runtime_error&) {
            // plugin is not created by e.g. invalid env
        } catch (const std::exception& ex) {
            OPENVINO_THROW("An exception is thrown while trying to create the ",
                           deviceName,
                           " device and call GetMetric: ",
                           ex.what());
        } catch (...) {
            OPENVINO_THROW("Unknown exception is thrown while trying to create the ",
                           deviceName,
                           " device and call GetMetric");
        }

        if (devicesIDs.size() > 1) {
            for (auto&& deviceID : devicesIDs) {
                devices.push_back(deviceName + '.' + deviceID);
            }
        } else if (!devicesIDs.empty()) {
            devices.push_back(deviceName);
        }
    }

    return devices;
}

ov::SoPtr<ov::IRemoteContext> ov::CoreImpl::create_context(const std::string& device_name, const AnyMap& params) const {
    auto parsed = ov::parseDeviceNameIntoConfig(device_name, params);
    return get_plugin(parsed._deviceName).create_context(parsed._config);
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
    const auto& flattened_config = flattened._config;
    const auto& device_name = flattened._deviceName;

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
        for (auto&& property : ICore::get_property(device_name, ov::supported_properties, {})) {
            if (property.is_mutable()) {
                *key_inserter = std::move(property);
            }
        }
    } catch (ov::Exception&) {
    }

    // try to search against internal supported_properties
    try {
        for (auto&& property : ICore::get_property(device_name, ov::internal::supported_properties, {})) {
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
    auto parsed = ov::parseDeviceNameIntoConfig(device_name);
    return get_plugin(parsed._deviceName).get_default_context(parsed._config);
}

std::shared_ptr<const ov::Model> ov::CoreImpl::apply_auto_batching(const std::shared_ptr<const ov::Model>& model,
                                                                   std::string& deviceName,
                                                                   ov::AnyMap& config) const {
    std::string deviceNameWithBatchSize, deviceNameWithoutBatch;
    // fully strict dims tracking by default (Auto-Batching is enabled implicitly)
    bool strictly_check_dims = true;
    if (deviceName.find("BATCH") != std::string::npos) {
        // explicitly enabled Auto-Batching
        auto pos = deviceName.find_first_of(":");
        if (pos == std::string::npos)
            return model;  // BATCH device is already configured via the config

        deviceNameWithBatchSize = deviceName.substr(pos + 1);
        deviceNameWithoutBatch = ov::DeviceIDParser::get_batch_device(deviceNameWithBatchSize);
        if (deviceName.find("(") == std::string::npos) {
            auto supported_properties = ICore::get_property(deviceNameWithoutBatch, ov::supported_properties, {});
            if (std::find(supported_properties.begin(), supported_properties.end(), ov::optimal_batch_size) ==
                supported_properties.end())
                return model;
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
            if (!ov::is_virtual_device(deviceName) && !is_proxy_device(deviceName))
                config.erase(batch_mode);
            if (disabled)
                return model;
        }

        // check whether if the Auto-Batching is applicable to the device
        auto parsed = ov::parseDeviceNameIntoConfig(deviceName);
        // Do not apply auto batch for proxy device
        if (is_proxy_device(parsed._deviceName))
            return model;
        deviceNameWithoutBatch = deviceName;
        auto metrics = get_plugin(parsed._deviceName)
                           .get_property(ov::supported_properties.name(), parsed._config)
                           .as<std::vector<ov::PropertyName>>();
        auto it = std::find(metrics.begin(), metrics.end(), ov::optimal_batch_size.name());
        if (metrics.end() == it)
            return model;

        // if applicable, the Auto-Batching is implicitly enabled via the performance hints
        bool bTputInPlg = get_plugin(parsed._deviceName)
                              .get_property(ov::hint::performance_mode.name(), parsed._config)
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
        deviceName = "BATCH:" + batchConfig;
        break;
    case ov::details::NetworkBatchAbility::WITH_HETERO:
        deviceName = "HETERO:BATCH," + deviceNameWithoutBatch;
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
        const auto is_proxy = is_proxy_device(ov::parseDeviceNameIntoConfig(device_name)._deviceName);
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
        return ov::Any(coreConfig.get_cache_dir());
    } else if (name == ov::enable_mmap.name()) {
        const auto flag = coreConfig.get_enable_mmap();
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

    auto parsed = parseDeviceNameIntoConfig(device_name, options);

    if (parsed._deviceName.empty()) {
        return get_property_for_core(name);
    } else if (name == ov::cache_dir.name()) {
        return coreConfig.get_cache_config_for_device(get_plugin(parsed._deviceName))._cacheDir;
    }
    return get_plugin(parsed._deviceName).get_property(name, parsed._config);
}

void ov::CoreImpl::unload_plugin(const std::string& deviceName) {
    std::lock_guard<std::mutex> lock(get_mutex());
    auto it = plugins.find(deviceName);
    if (it == plugins.end()) {
        OPENVINO_THROW("Device with \"", deviceName, "\" name is not registered in the OpenVINO Runtime");
    }

    plugins.erase(deviceName);
}

void ov::CoreImpl::register_plugin(const std::string& plugin,
                                   const std::string& device_name,
                                   const ov::AnyMap& properties) {
    std::lock_guard<std::mutex> lock(get_mutex());

    auto it = pluginRegistry.find(device_name);
    // Proxy plugins can be configured in the runtime
    if (it != pluginRegistry.end() && !is_proxy_device(device_name)) {
        OPENVINO_THROW("Device with \"", device_name, "\"  is already registered in the OpenVINO Runtime");
    }

    if (device_name.find('.') != std::string::npos) {
        OPENVINO_THROW("Device name must not contain dot '.' symbol");
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
    for (auto&& pluginDesc : pluginRegistry) {
        listOfDevices.push_back(pluginDesc.first);
    }

    return listOfDevices;
}

/**
 * @brief Sets property values for a plugin or set of plugins
 * @param deviceName A device name to set config to
 *        If empty, config is set for all the plugins / plugin's meta-data
 * @note  `deviceName` is not allowed in form of MULTI:CPU, HETERO:GPU,CPU, AUTO:CPU
 *        just simple forms like CPU, GPU, MULTI, GPU.0, etc
 */
void ov::CoreImpl::set_property_for_device(const ov::AnyMap& configMap, const std::string& deviceName) {
    auto config = configMap;
    if (config.empty()) {
        return;
    }

    ov::DeviceIDParser parser(deviceName);
    std::string clearDeviceName = parser.get_device_name();

    std::vector<std::pair<std::string, ov::Plugin>> created_plugins;
    {
        std::lock_guard<std::mutex> lock(get_mutex());
        created_plugins.reserve(plugins.size());

        // TODO: keep only:
        //    coreConfig.set_and_update(config);
        // once GPU remove support of ov::cache_dir
        // CoreConfg::set_and_update will drop CACHE_DIR from config map
        // and updates core config with new ov::cache_dir
        if (deviceName.empty()) {
            coreConfig.set_and_update(config);
        } else {
            OPENVINO_SUPPRESS_DEPRECATED_START
            auto cache_it = config.find(ov::cache_dir.name());
            if (cache_it != config.end()) {
                coreConfig.set_cache_dir_for_device((cache_it->second).as<std::string>(), clearDeviceName);
                config.erase(cache_it);
            }
            OPENVINO_SUPPRESS_DEPRECATED_END
            // apply and remove core properties
            auto it = config.find(ov::force_tbb_terminate.name());
            if (it != config.end()) {
                auto flag = it->second.as<bool>();
                ov::threading::executor_manager()->set_property({{it->first, flag}});
                config.erase(it);
            }

            it = config.find(ov::enable_mmap.name());
            if (it != config.end()) {
                config.erase(it);
            }
        }

        if (!config.empty()) {
            auto base_desc = pluginRegistry.find(clearDeviceName);
            if (pluginRegistry.find(deviceName) == pluginRegistry.end() && base_desc != pluginRegistry.end()) {
                PluginDescriptor desc{base_desc->second.libraryLocation, config, base_desc->second.listOfExtentions};
                pluginRegistry[deviceName] = std::move(desc);
            }

            // set config for plugins in registry
            bool configIsSet = false;
            for (auto& desc : pluginRegistry) {
                if (deviceName.empty() || deviceName == desc.first) {
                    for (auto&& conf : config) {
                        desc.second.defaultConfig[conf.first] = conf.second;
                    }
                    configIsSet = true;
                }
            }

            if (!configIsSet && !deviceName.empty()) {
                OPENVINO_THROW("Device with \"", deviceName, "\" name is not registered in the OpenVINO Runtime");
            }
        }

        // set config for already created plugins
        for (auto& plugin : plugins) {
            if (deviceName.empty() || clearDeviceName == plugin.first) {
                created_plugins.emplace_back(std::pair<std::string, ov::Plugin>{plugin.first, plugin.second});
            }
        }
    }

    for (auto& plugin : created_plugins) {
        allowNotImplemented([&]() {
            std::lock_guard<std::mutex> lock(get_mutex(plugin.first));
            auto configCopy = config;
            // TODO: remove once GPU remove explicit support of ov::cache_dir
            {
                OPENVINO_SUPPRESS_DEPRECATED_START
                if (device_supports_cache_dir(plugin.second)) {
                    configCopy[ov::cache_dir.name()] = coreConfig.get_cache_config_for_device(plugin.second)._cacheDir;
                } else if (configCopy.count(ov::cache_dir.name()) > 0) {
                    // Remove "CACHE_DIR" from config if it is not supported by plugin
                    configCopy.erase(ov::cache_dir.name());
                }
                OPENVINO_SUPPRESS_DEPRECATED_END
            }
            // Add device specific value to support device_name.device_id cases
            {
                if (!parser.get_device_id().empty()) {
                    const std::string deviceKey =
                        device_supports_internal_property(plugin.second, ov::internal::config_device_id.name())
                            ? ov::internal::config_device_id.name()
                            : ov::device::id.name();
                    configCopy[deviceKey] = parser.get_device_id();
                }
            }
            plugin.second.set_property(configCopy);
        });
    }
}
void ov::CoreImpl::add_extensions_unsafe(const std::vector<ov::Extension::Ptr>& exts) const {
    for (const auto& ext : exts) {
        extensions.emplace_back(ext);
        auto ext_obj = ext;
        if (auto so_ext = std::dynamic_pointer_cast<ov::detail::SOExtension>(ext_obj))
            ext_obj = so_ext->extension();
        if (auto op_base_ext = std::dynamic_pointer_cast<ov::BaseOpExtension>(ext_obj)) {
            for (const auto& attached_ext : op_base_ext->get_attached_extensions()) {
                extensions.emplace_back(attached_ext);
            }
        }
    }
}

void ov::CoreImpl::add_extension(const std::vector<ov::Extension::Ptr>& extensions) {
    std::lock_guard<std::mutex> lock(get_mutex());
    add_extensions_unsafe(extensions);
}

bool ov::CoreImpl::device_supports_model_caching(const std::string& device_name) const {
    auto parsed = parseDeviceNameIntoConfig(device_name);
    return device_supports_model_caching(get_plugin(parsed._deviceName));
}

bool ov::CoreImpl::device_supports_property(const ov::Plugin& plugin, const ov::PropertyName& key) const {
    return util::contains(plugin.get_property(ov::supported_properties), key);
}

bool ov::CoreImpl::device_supports_internal_property(const ov::Plugin& plugin, const ov::PropertyName& key) const {
    return util::contains(plugin.get_property(ov::internal::supported_properties), key);
}

bool ov::CoreImpl::device_supports_model_caching(const ov::Plugin& plugin, const ov::AnyMap& arguments) const {
    ov::AnyMap properties;
    if (arguments.count(ov::device::priorities.name())) {
        properties[ov::device::priorities.name()] = arguments.at(ov::device::priorities.name()).as<std::string>();
    }
    return plugin.supports_model_caching(properties);
}

bool ov::CoreImpl::device_supports_cache_dir(const ov::Plugin& plugin) const {
    try {
        return util::contains(plugin.get_property(ov::supported_properties), ov::cache_dir);
    } catch (const ov::NotImplemented&) {
        return false;
    }
}

ov::SoPtr<ov::ICompiledModel> ov::CoreImpl::compile_model_and_cache(ov::Plugin& plugin,
                                                                    const std::shared_ptr<const ov::Model>& model,
                                                                    const ov::AnyMap& parsedConfig,
                                                                    const ov::SoPtr<ov::IRemoteContext>& context,
                                                                    const CacheContent& cacheContent) const {
    OV_ITT_SCOPED_TASK(ov::itt::domains::OV, "CoreImpl::compile_model_and_cache");
    ov::SoPtr<ov::ICompiledModel> compiled_model =
        context ? plugin.compile_model(model, context, parsedConfig) : plugin.compile_model(model, parsedConfig);
    if (cacheContent.cacheManager && device_supports_model_caching(plugin)) {
        try {
            // need to export network for further import from "cache"
            OV_ITT_SCOPE(FIRST_INFERENCE, ov::itt::domains::LoadTime, "Core::compile_model::Export");
            std::string compiled_model_runtime_properties;
            if (device_supports_internal_property(plugin, ov::internal::compiled_model_runtime_properties.name())) {
                compiled_model_runtime_properties =
                    plugin.get_property(ov::internal::compiled_model_runtime_properties.name(), {}).as<std::string>();
            }
            cacheContent.cacheManager->write_cache_entry(cacheContent.blobId, [&](std::ostream& networkStream) {
                networkStream << ov::CompiledBlobHeader(ov::get_openvino_version().buildNumber,
                                                        ov::ModelCache::calculate_file_info(cacheContent.modelPath),
                                                        compiled_model_runtime_properties);
                compiled_model->export_model(networkStream);
            });
        } catch (...) {
            cacheContent.cacheManager->remove_cache_entry(cacheContent.blobId);
            throw;
        }
    }
    return compiled_model;
}

ov::SoPtr<ov::ICompiledModel> ov::CoreImpl::load_model_from_cache(
    const CacheContent& cacheContent,
    ov::Plugin& plugin,
    const ov::AnyMap& config,
    const ov::SoPtr<ov::IRemoteContext>& context,
    std::function<ov::SoPtr<ov::ICompiledModel>()> compile_model_lambda) const {
    ov::SoPtr<ov::ICompiledModel> compiled_model;
    struct HeaderException {};

    OPENVINO_ASSERT(cacheContent.cacheManager != nullptr);

    try {
        cacheContent.cacheManager->read_cache_entry(
            cacheContent.blobId,
            cacheContent.mmap_enabled && ov::util::contains(plugin.get_property(ov::internal::supported_properties),
                                                            ov::internal::caching_with_mmap),
            [&](std::istream& networkStream, ov::Tensor& compiled_blob) {
                OV_ITT_SCOPE(FIRST_INFERENCE,
                             ov::itt::domains::LoadTime,
                             "Core::load_model_from_cache::ReadStreamAndImport");
                ov::CompiledBlobHeader header;
                try {
                    networkStream >> header;
                    if (header.get_file_info() != ov::ModelCache::calculate_file_info(cacheContent.modelPath)) {
                        // Original file is changed, don't use cache
                        OPENVINO_THROW("Original model file is changed");
                    }
                    if (util::contains(plugin.get_property(ov::internal::supported_properties),
                                       ov::internal::compiled_model_runtime_properties_supported.name())) {
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
                        if (header.get_openvino_version() != ov::get_openvino_version().buildNumber) {
                            // Build number mismatch, don't use this cache
                            OPENVINO_THROW("Version does not match");
                        }
                    }
                } catch (...) {
                    throw HeaderException();
                }

                ov::AnyMap update_config = config;
                update_config[ov::loaded_from_cache.name()] = true;
                if (cacheContent.model &&
                    util::contains(plugin.get_property(ov::supported_properties), ov::hint::model)) {
                    update_config[ov::hint::model.name()] = cacheContent.model;
                }

                if (util::contains(plugin.get_property(ov::supported_properties), ov::hint::model) &&
                    cacheContent.model) {
                    update_config[ov::hint::model.name()] = cacheContent.model;
                }
                if (util::contains(plugin.get_property(ov::supported_properties), ov::weights_path)) {
                    util::Path weights_path;

                    if (auto&& path_hint = update_config.find(ov::weights_path.name());
                        path_hint != update_config.end()) {
                        weights_path = path_hint->second.as<std::string>();
                    } else if (weights_path = extract_weight_path(header.get_runtime_info()); weights_path.empty()) {
                        weights_path = cacheContent.modelPath;
                        weights_path.replace_extension(".bin");
                    }
                    weights_path.replace_extension(".bin");

                    if (ov::util::file_exists(weights_path)) {
                        update_config[ov::weights_path.name()] = weights_path.string();
                    }
                }

                if (compiled_blob) {
                    update_config[ov::hint::compiled_blob.name()] = compiled_blob;
                }
                compiled_model = context ? plugin.import_model(networkStream, context, update_config)
                                         : plugin.import_model(networkStream, update_config);
            });
    } catch (const HeaderException&) {
        // For these exceptions just remove old cache and set that import didn't work
        cacheContent.cacheManager->remove_cache_entry(cacheContent.blobId);
    } catch (...) {
        cacheContent.cacheManager->remove_cache_entry(cacheContent.blobId);
        // TODO: temporary disabled by #54335. In future don't throw only for new 'blob_outdated' exception
        // throw;
    }

    // fallback scenario
    if (!compiled_model)
        compiled_model = compile_model_lambda();

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
    const bool supports_device_id = device_supports_property(plugin, ov::device::id);
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
        std::lock_guard<std::mutex> lock(other._cacheConfigMutex);
        _cacheConfig = other._cacheConfig;
        _cacheConfigPerDevice = other._cacheConfigPerDevice;
    }
    _flag_enable_mmap = other._flag_enable_mmap;
}

void ov::CoreConfig::set(const ov::AnyMap& config) {
    auto it = config.find(ov::cache_dir.name());
    if (it != config.end()) {
        std::lock_guard<std::mutex> lock(_cacheConfigMutex);
        // fill global cache config
        _cacheConfig = CoreConfig::CacheConfig::create(it->second.as<std::string>());
        // sets cache config per-device if it's not set explicitly before
        for (auto& deviceCfg : _cacheConfigPerDevice) {
            deviceCfg.second = CoreConfig::CacheConfig::create(it->second.as<std::string>());
        }
    }

    it = config.find(ov::force_tbb_terminate.name());
    if (it != config.end()) {
        auto flag = it->second.as<bool>();
        ov::threading::executor_manager()->set_property({{it->first, flag}});
    }

    it = config.find(ov::enable_mmap.name());
    if (it != config.end()) {
        auto flag = it->second.as<bool>();
        _flag_enable_mmap = flag;
    }
}

void ov::CoreConfig::set_and_update(ov::AnyMap& config) {
    set(config);
    remove_core(config);
}

void ov::CoreConfig::remove_core(ov::AnyMap& config) {
    for (const auto& name : core_properties_names) {
        config.erase(name);
    }
}

void ov::CoreConfig::remove_core_skip_cache_dir(ov::AnyMap& config) {
    for (const auto& name : {ov::enable_mmap.name(), ov::force_tbb_terminate.name()}) {
        config.erase(name);
    }
}

void ov::CoreConfig::set_cache_dir_for_device(const std::string& dir, const std::string& name) {
    std::lock_guard<std::mutex> lock(_cacheConfigMutex);
    _cacheConfigPerDevice[name] = CoreConfig::CacheConfig::create(dir);
}

std::string ov::CoreConfig::get_cache_dir() const {
    std::lock_guard<std::mutex> lock(_cacheConfigMutex);
    return _cacheConfig._cacheDir;
}

bool ov::CoreConfig::get_enable_mmap() const {
    return _flag_enable_mmap;
}

// Creating thread-safe copy of config including shared_ptr to ICacheManager
// Passing empty or not-existing name will return global cache config
ov::CoreConfig::CacheConfig ov::CoreConfig::get_cache_config_for_device(const ov::Plugin& plugin,
                                                                        ov::AnyMap& parsedConfig) const {
    // cache_dir is enabled locally in compile_model only
    if (parsedConfig.count(ov::cache_dir.name())) {
        const auto& cache_dir_val = parsedConfig.at(ov::cache_dir.name()).as<std::string>();
        const auto& tempConfig = CoreConfig::CacheConfig::create(cache_dir_val);
        // if plugin does not explicitly support cache_dir, and if plugin is not virtual, we need to remove
        // it from config
        if (!util::contains(plugin.get_property(ov::supported_properties), ov::cache_dir) &&
            !ov::is_virtual_device(plugin.get_name())) {
            parsedConfig.erase(ov::cache_dir.name());
        }
        return tempConfig;
    } else {  // cache_dir is set to Core globally or for the specific device
        return get_cache_config_for_device(plugin);
    }
}

ov::CoreConfig::CacheConfig ov::CoreConfig::get_cache_config_for_device(const ov::Plugin& plugin) const {
    std::lock_guard<std::mutex> lock(_cacheConfigMutex);
    return _cacheConfigPerDevice.count(plugin.get_name()) ? _cacheConfigPerDevice.at(plugin.get_name()) : _cacheConfig;
}

ov::CoreConfig::CacheConfig ov::CoreConfig::CacheConfig::create(const std::string& dir) {
    std::shared_ptr<ov::ICacheManager> cache_manager = nullptr;

    if (!dir.empty()) {
#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
        ov::util::create_directory_recursive(ov::util::string_to_wstring(dir));
#else
        ov::util::create_directory_recursive(dir);
#endif
        cache_manager = std::make_shared<ov::FileStorageCacheManager>(dir);
    }

    return {dir, std::move(cache_manager)};
}

std::mutex& ov::CoreImpl::get_mutex(const std::string& dev_name) const {
    std::lock_guard<std::mutex> lock(global_mutex);
    try {
        return dev_mutexes.at(dev_name);
    } catch (const std::out_of_range&) {
        OPENVINO_THROW("Cannot get mutex for device: ", dev_name);
    }
}

void ov::CoreImpl::add_mutex(const std::string& dev_name) {
    std::lock_guard<std::mutex> lock(global_mutex);
    dev_mutexes[dev_name];
}

std::shared_ptr<ov::Model> ov::CoreImpl::read_model(const std::string& modelPath,
                                                    const std::string& binPath,
                                                    const AnyMap& properties) const {
    OV_ITT_SCOPE(FIRST_INFERENCE, ov::itt::domains::ReadTime, "CoreImpl::read_model from file");
    auto local_core_config = coreConfig;
    local_core_config.set(properties);
    return ov::util::read_model(modelPath, binPath, extensions, local_core_config.get_enable_mmap());
}

std::shared_ptr<ov::Model> ov::CoreImpl::read_model(const std::string& model,
                                                    const ov::Tensor& weights,
                                                    bool frontendMode) const {
    OV_ITT_SCOPE(FIRST_INFERENCE, ov::itt::domains::ReadTime, "CoreImpl::read_model from memory");
    return ov::util::read_model(model, weights, extensions, frontendMode);
}

std::shared_ptr<ov::Model> ov::CoreImpl::read_model(const std::shared_ptr<AlignedBuffer>& model,
                                                    const std::shared_ptr<AlignedBuffer>& weights) const {
    OV_ITT_SCOPE(FIRST_INFERENCE, ov::itt::domains::ReadTime, "CoreImpl::read_model from memory");
    return ov::util::read_model(model, weights, extensions);
}

std::map<std::string, ov::Version> ov::CoreImpl::get_versions(const std::string& deviceName) const {
    std::map<std::string, ov::Version> versions;
    std::vector<std::string> deviceNames;

    {
        // for compatibility with samples / demo
        if (deviceName.find("HETERO") == 0) {
            auto pos = deviceName.find_first_of(":");
            if (pos != std::string::npos) {
                deviceNames = ov::DeviceIDParser::get_hetero_devices(deviceName.substr(pos + 1));
            }
            deviceNames.push_back("HETERO");
        } else if (deviceName.find("MULTI") == 0) {
            auto pos = deviceName.find_first_of(":");
            if (pos != std::string::npos) {
                deviceNames = ov::DeviceIDParser::get_multi_devices(deviceName.substr(pos + 1));
            }
            deviceNames.push_back("MULTI");
        } else if (deviceName.find("AUTO") == 0) {
            auto pos = deviceName.find_first_of(":");
            if (pos != std::string::npos) {
                deviceNames = ov::DeviceIDParser::get_multi_devices(deviceName.substr(pos + 1));
            }
            deviceNames.emplace_back("AUTO");
        } else if (deviceName.find("BATCH") == 0) {
            auto pos = deviceName.find_first_of(":");
            if (pos != std::string::npos) {
                deviceNames = {ov::DeviceIDParser::get_batch_device(deviceName.substr(pos + 1))};
            }
            deviceNames.push_back("BATCH");
        } else {
            deviceNames.push_back(deviceName);
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
