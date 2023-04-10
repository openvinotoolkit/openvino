// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core_impl.hpp"

#include <memory>

#include "any_copy.hpp"
#include "check_network_batchable.hpp"
#include "compilation_context.hpp"
#include "cpp_interfaces/interface/ie_iexecutable_network_internal.hpp"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"
#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"
#include "dev/converter_utils.hpp"
#include "dev/icompiled_model_wrapper.hpp"
#include "dev/make_tensor.hpp"
#include "file_utils.h"
#include "ie_itt.hpp"
#include "ie_network_reader.hpp"
#include "ie_ngraph_utils.hpp"
#include "iplugin_wrapper.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/pass/constant_folding.hpp"
#include "openvino/core/any.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/op_extension.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/core/version.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/device_id_parser.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/remote_context.hpp"
#include "openvino/runtime/threading/executor_manager.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"
#include "ov_plugins.hpp"
#include "preprocessing/preprocessing.hpp"
#include "xml_parse_utils.h"

ov::ICore::~ICore() = default;

namespace {

template <typename F>
void allowNotImplemented(F&& f) {
    try {
        f();
    } catch (const InferenceEngine::NotImplemented&) {
    } catch (const ov::NotImplemented&) {
    }
}

void stripDeviceName(std::string& device, const std::string& substr) {
    auto pos = device.find(substr);
    if (pos == 0) {
        device.erase(pos, substr.length());
    }
}

bool is_virtual_device(const std::string& device_name) {
    return (device_name.find("AUTO") != std::string::npos || device_name.find("MULTI") != std::string::npos ||
            device_name.find("HETERO") != std::string::npos || device_name.find("BATCH") != std::string::npos);
};

ov::AnyMap clone_map(const ov::AnyMap& m) {
    ov::AnyMap rm;
    for (auto&& kvp : m) {
        rm[kvp.first] = kvp.second.is<ov::AnyMap>() ? ov::Any(clone_map(kvp.second.as<ov::AnyMap>())) : kvp.second;
    }

    return rm;
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
    ov::AnyMap result_properties = clone_map(user_properties);

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
        if (ov::is_config_applicable(user_device_name, subprop_device_name) || is_virtual_device(user_device_name)) {
            // 2.1. keep the secondary property for the other virtual devices, but repack them
            auto device_properties = result_properties.find(ov::device::properties.name());
            if (device_properties == result_properties.end()) {
                result_properties[ov::device::properties.name()] = ov::AnyMap{};
            } else if (device_properties->second.is<std::string>()) {  // because of legacy API 1.0
                device_properties->second = device_properties->second.as<ov::AnyMap>();
            }
            auto& secondary_properties = result_properties[ov::device::properties.name()].as<ov::AnyMap>();
            auto secondary_properties_it = secondary_properties.find(subprop_device_name);
            if (secondary_properties_it == secondary_properties.end()) {
                // 2.1.1. No device name in map yet, insert all config as is
                secondary_properties[subprop_device_name] = secondary_property->second;
            } else {
                if (secondary_properties_it->second.is<std::string>()) {  // because of legacy API 1.0
                    secondary_properties_it->second = secondary_properties_it->second.as<ov::AnyMap>();
                }
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
        if (property->second.is<std::string>()) {  // because of legacy API 1.0
            property->second = property->second.as<ov::AnyMap>();
        }
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
            } else if (is_virtual_device(user_device_name)) {
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
    return is_virtual_device(device_name)
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
            if (!is_virtual_device(deviceName))
                config.erase(batch_timeout_mode);
        }
    }
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
        auto user_value =
            parsed_user_device_name._config.count(key) ? parsed_user_device_name._config.at(key).as<std::string>() : "";
        auto subprop_value = parsed_subprop_device_name._config.count(key)
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

ov::Parsed ov::parseDeviceNameIntoConfig(const std::string& deviceName, const AnyMap& config) {
    auto updated_config = config;
    auto updated_device_name = deviceName;

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

    updated_config = flatten_sub_properties(deviceName, updated_config);
    std::string parsed_device_priority;

    // try to find ':' to extract name of virtual device
    auto pos = deviceName.find_first_of(':');
    if (pos != std::string::npos) {
        updated_device_name = deviceName.substr(0, pos);
        parsed_device_priority = deviceName.substr(pos + 1);
    } else {
        ov::DeviceIDParser parser(deviceName);
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
            IE_THROW() << "Device priority / ID mismatch: " << parsed_device_priority << " (from " << deviceName
                       << ") vs " << it->second.as<std::string>() << " (from config)";
        }
    };

    // clean-up auto-batch related properties
    clean_batch_properties(updated_device_name, updated_config, ov::hint::allow_auto_batching);
    clean_batch_properties(updated_device_name, updated_config, ov::auto_batch_timeout);

    return {updated_device_name, updated_config};
}

ov::CoreImpl::CoreImpl(bool _newAPI) : m_new_api(_newAPI) {
    add_mutex("");  // Register global mutex
    m_executor_manager = ov::threading::executor_manager();
    for (const auto& it : ov::get_available_opsets()) {
        opsetNames.insert(it.first);
    }
}

void ov::CoreImpl::register_compile_time_plugins() {
    std::lock_guard<std::mutex> lock(get_mutex());

    const decltype(::getCompiledPluginsRegistry())& plugins = getCompiledPluginsRegistry();
    for (const auto& plugin : plugins) {
        const auto& deviceName = plugin.first;
        if (deviceName.find('.') != std::string::npos) {
            OPENVINO_THROW("Device name must not contain dot '.' symbol");
        }
#ifdef OPENVINO_STATIC_LIBRARY
        if (pluginRegistry.find(deviceName) == pluginRegistry.end()) {
            const auto& value = plugin.second;
            ov::AnyMap config = any_copy(value.m_default_config);
            PluginDescriptor desc{value.m_create_plugin_func, config, value.m_create_extension_func};
            pluginRegistry[deviceName] = desc;
            add_mutex(deviceName);
        }
#else
        const auto& pluginPath = ov::util::get_compiled_plugin_path(plugin.second.m_plugin_path);
        if (pluginRegistry.find(deviceName) == pluginRegistry.end() && ov::util::file_exists(pluginPath)) {
            ov::AnyMap config = any_copy(plugin.second.m_default_config);
            PluginDescriptor desc{pluginPath, config};
            pluginRegistry[deviceName] = desc;
            add_mutex(deviceName);
        }
#endif
    }
}

void ov::CoreImpl::register_plugins_in_registry(const std::string& xml_config_file, const bool& by_abs_path) {
    std::lock_guard<std::mutex> lock(get_mutex());

    auto parse_result = ParseXml(xml_config_file.c_str());
    if (!parse_result.error_msg.empty()) {
        IE_THROW() << parse_result.error_msg;
    }

    pugi::xml_document& xmlDoc = *parse_result.xml;

    using namespace pugixml::utils;
    pugi::xml_node ieNode = xmlDoc.document_element();
    pugi::xml_node devicesNode = ieNode.child("plugins");

    FOREACH_CHILD (pluginNode, devicesNode, "plugin") {
        std::string deviceName = GetStrAttr(pluginNode, "name");
        if (pluginRegistry.find(deviceName) != pluginRegistry.end()) {
            IE_THROW() << "Device with \"" << deviceName << "\"  is already registered in the OpenVINO Runtime";
        }
        if (deviceName.find('.') != std::string::npos) {
            IE_THROW() << "Device name must not contain dot '.' symbol";
        }

        ov::util::FilePath pluginPath =
            ov::util::get_plugin_path(GetStrAttr(pluginNode, "location"), xml_config_file, by_abs_path);

        // check properties
        auto propertiesNode = pluginNode.child("properties");
        ov::AnyMap config;

        if (propertiesNode) {
            FOREACH_CHILD (propertyNode, propertiesNode, "property") {
                std::string key = GetStrAttr(propertyNode, "key");
                std::string value = GetStrAttr(propertyNode, "value");
                config[key] = value;
            }
        }

        // check extensions
        auto extensionsNode = pluginNode.child("extensions");
        std::vector<ov::util::FilePath> listOfExtentions;

        if (extensionsNode) {
            FOREACH_CHILD (extensionNode, extensionsNode, "extension") {
                ov::util::FilePath extensionLocation =
                    ov::util::to_file_path(GetStrAttr(extensionNode, "location").c_str());
                listOfExtentions.push_back(extensionLocation);
            }
        }

        // fill value in plugin registry for later lazy initialization
        {
            PluginDescriptor desc{pluginPath, config, listOfExtentions};
            pluginRegistry[deviceName] = desc;
            add_mutex(deviceName);
        }
    }
}

ov::Plugin ov::CoreImpl::get_plugin(const std::string& pluginName) const {
    OV_ITT_SCOPE(FIRST_INFERENCE, InferenceEngine::itt::domains::IE_LT, "CoreImpl::get_plugin");

    auto deviceName = pluginName;
    if (deviceName == ov::DEFAULT_DEVICE_NAME)
        deviceName = "AUTO";
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
                IE_THROW() << "No device is provided, so AUTO device is used by default, which failed loading.";
            else
                IE_THROW() << "Device with \"" << deviceName << "\" name is not registered in the OpenVINO Runtime";
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

        if (desc.pluginCreateFunc) {  // static OpenVINO case
            std::shared_ptr<ov::IPlugin> plugin_impl;
            desc.pluginCreateFunc(plugin_impl);
            plugin = Plugin{plugin_impl, {}};
        } else {
            so = ov::util::load_shared_object(desc.libraryLocation.c_str());
            std::shared_ptr<ov::IPlugin> plugin_impl;
            reinterpret_cast<ov::CreatePluginFunc*>(ov::util::get_symbol(so, ov::create_plugin_function))(plugin_impl);
            plugin = Plugin{plugin_impl, so};
        }

        {
            plugin.set_name(deviceName);

            // Set Core class reference to plugins
            std::weak_ptr<ov::ICore> mutableCore =
                std::const_pointer_cast<ov::ICore>(std::dynamic_pointer_cast<const ov::ICore>(shared_from_this()));
            plugin.set_core(mutableCore);
        }

        // Add registered extensions to new plugin
        allowNotImplemented([&]() {
            for (const auto& ext : extensions) {
                plugin.add_extension(ext);
            }
        });

        // configuring
        {
            // TODO: remove this block of code once GPU removes support of ov::cache_dir
            // also, remove device_supports_cache_dir at all
            {
                OPENVINO_SUPPRESS_DEPRECATED_START
                if (device_supports_cache_dir(plugin)) {
                    ov::AnyMap empty_map;
                    auto cacheConfig = coreConfig.get_cache_config_for_device(plugin, empty_map);
                    if (cacheConfig._cacheManager) {
                        desc.defaultConfig[CONFIG_KEY(CACHE_DIR)] = cacheConfig._cacheDir;
                    }
                } else if (desc.defaultConfig.count(CONFIG_KEY(CACHE_DIR)) > 0) {
                    // Remove "CACHE_DIR" from config if it is not supported by plugin
                    desc.defaultConfig.erase(CONFIG_KEY(CACHE_DIR));
                }
                OPENVINO_SUPPRESS_DEPRECATED_END
            }

            allowNotImplemented([&]() {
                // Add device specific value to support device_name.device_id cases
                {
                    auto supportedConfigKeys =
                        plugin.get_property(METRIC_KEY(SUPPORTED_CONFIG_KEYS), {}).as<std::vector<std::string>>();
                    const bool supportsConfigDeviceID =
                        ov::util::contains(supportedConfigKeys, CONFIG_KEY_INTERNAL(CONFIG_DEVICE_ID));
                    const std::string deviceKey =
                        supportsConfigDeviceID ? CONFIG_KEY_INTERNAL(CONFIG_DEVICE_ID) : CONFIG_KEY(DEVICE_ID);

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

            allowNotImplemented([&]() {
                for (auto&& extensionLocation : desc.listOfExtentions) {
                    plugin.add_extension(std::make_shared<InferenceEngine::Extension>(extensionLocation));
                }
            });
        }

        std::lock_guard<std::mutex> g_lock(get_mutex());
        // add plugin as extension itself
        if (desc.extensionCreateFunc) {  // static OpenVINO case
            try {
                InferenceEngine::IExtensionPtr ext;
                desc.extensionCreateFunc(ext);
                AddExtensionUnsafe(ext);
            } catch (const InferenceEngine::GeneralError&) {
                // the same extension can be registered multiple times - ignore it!
            }
        } else {
            TryToRegisterLibraryAsExtensionUnsafe(desc.libraryLocation);
        }

        return plugins.emplace(deviceName, plugin).first->second;
    } catch (const InferenceEngine::Exception& ex) {
        IE_THROW() << "Failed to create plugin " << ov::util::from_file_path(desc.libraryLocation) << " for device "
                   << deviceName << "\n"
                   << "Please, check your environment\n"
                   << ex.what() << "\n";
    }
}

ov::SoPtr<ov::ICompiledModel> ov::CoreImpl::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                          const std::string& device_name,
                                                          const ov::AnyMap& config) const {
    OV_ITT_SCOPE(FIRST_INFERENCE, ie::itt::domains::IE_LT, "Core::compile_model::model");
    std::string deviceName = device_name;
    ov::AnyMap config_with_batch = config;
    // if auto-batching is applicable, the below function will patch the device name and config accordingly:
    apply_auto_batching(model, deviceName, config_with_batch);

    auto parsed = parseDeviceNameIntoConfig(deviceName, config_with_batch);
    auto plugin = get_plugin(parsed._deviceName);
    ov::SoPtr<ov::ICompiledModel> res;
    auto cacheManager = coreConfig.get_cache_config_for_device(plugin, parsed._config)._cacheManager;
    if (cacheManager && device_supports_model_caching(plugin)) {
        CacheContent cacheContent{cacheManager};
        cacheContent.blobId = ov::ModelCache::compute_hash(model, create_compile_config(plugin, parsed._config));
        auto lock = cacheGuard.get_hash_lock(cacheContent.blobId);
        res = load_model_from_cache(cacheContent, plugin, parsed._config, ov::RemoteContext{}, [&]() {
            return compile_model_and_cache(model, plugin, parsed._config, ov::RemoteContext{}, cacheContent);
        });
    } else {
        res = compile_model_with_preprocess(plugin, model, ov::RemoteContext{}, parsed._config);
    }
    return res;
}

ov::SoPtr<ov::ICompiledModel> ov::CoreImpl::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                          const ov::RemoteContext& context,
                                                          const ov::AnyMap& config) const {
    OV_ITT_SCOPE(FIRST_INFERENCE, ie::itt::domains::IE_LT, "Core::compile_model::RemoteContext");
    if (context._impl == nullptr) {
        IE_THROW() << "Remote context is null";
    }
    std::string deviceName = context.get_device_name();
    ov::AnyMap config_with_batch = config;
    // if auto-batching is applicable, the below function will patch the device name and config accordingly:
    apply_auto_batching(model, deviceName, config_with_batch);

    auto parsed = parseDeviceNameIntoConfig(deviceName, config_with_batch);
    auto plugin = get_plugin(parsed._deviceName);
    ov::SoPtr<ov::ICompiledModel> res;
    auto cacheManager = coreConfig.get_cache_config_for_device(plugin, parsed._config)._cacheManager;
    if (cacheManager && device_supports_model_caching(plugin)) {
        CacheContent cacheContent{cacheManager};
        cacheContent.blobId = ov::ModelCache::compute_hash(model, create_compile_config(plugin, parsed._config));
        auto lock = cacheGuard.get_hash_lock(cacheContent.blobId);
        res = load_model_from_cache(cacheContent, plugin, parsed._config, context, [&]() {
            return compile_model_and_cache(model, plugin, parsed._config, context, cacheContent);
        });
    } else {
        res = compile_model_with_preprocess(plugin, model, context, parsed._config);
    }
    return res;
}

ov::SoPtr<ov::ICompiledModel> ov::CoreImpl::compile_model_with_preprocess(ov::Plugin& plugin,
                                                                          const std::shared_ptr<const ov::Model>& model,
                                                                          const ov::RemoteContext& context,
                                                                          const ov::AnyMap& config) const {
    std::shared_ptr<const ov::Model> preprocessed_model = model;

    if (!is_new_api() && !std::dynamic_pointer_cast<InferenceEngine::IPluginWrapper>(plugin.m_ptr)) {
        ov::pass::Manager manager;
        manager.register_pass<ov::pass::AddPreprocessing>();

        auto cloned_model = model->clone();
        manager.run_passes(cloned_model);
        preprocessed_model = cloned_model;
    }

    return context._impl ? plugin.compile_model(preprocessed_model, context, config)
                         : plugin.compile_model(preprocessed_model, config);
}

ov::SoPtr<ov::ICompiledModel> ov::CoreImpl::compile_model(const std::string& model_path,
                                                          const std::string& device_name,
                                                          const ov::AnyMap& config) const {
    OV_ITT_SCOPE(FIRST_INFERENCE, ie::itt::domains::IE_LT, "Core::compile_model::Path");
    auto parsed = parseDeviceNameIntoConfig(device_name, config);
    // in case of compile_model(file_name), we need to clear-up core-level properties
    auto plugin = get_plugin(parsed._deviceName);
    ov::SoPtr<ov::ICompiledModel> compiled_model;

    auto cacheManager = coreConfig.get_cache_config_for_device(plugin, parsed._config)._cacheManager;
    if (cacheManager && device_supports_model_caching(plugin)) {
        CacheContent cacheContent{cacheManager, model_path};
        cacheContent.blobId = ov::ModelCache::compute_hash(model_path, create_compile_config(plugin, parsed._config));
        auto lock = cacheGuard.get_hash_lock(cacheContent.blobId);
        compiled_model = load_model_from_cache(cacheContent, plugin, parsed._config, ov::RemoteContext{}, [&]() {
            auto cnnNetwork = ReadNetwork(model_path, std::string());
            return compile_model_and_cache(cnnNetwork.getFunction(), plugin, parsed._config, {}, cacheContent);
        });
    } else if (cacheManager) {
        // this code path is enabled for AUTO / MULTI / BATCH devices which don't support
        // import / export explicitly, but can redirect this functionality to actual HW plugin
        compiled_model = plugin.compile_model(model_path, parsed._config);
    } else {
        auto cnnNetwork = ReadNetwork(model_path, std::string());
        compiled_model =
            compile_model_with_preprocess(plugin, cnnNetwork.getFunction(), ov::RemoteContext{}, parsed._config);
    }
    return compiled_model;
}

ov::SoPtr<ov::ICompiledModel> ov::CoreImpl::compile_model(const std::string& model_str,
                                                          const ov::Tensor& weights,
                                                          const std::string& device_name,
                                                          const ov::AnyMap& config) const {
    OV_ITT_SCOPED_TASK(ov::itt::domains::IE, "Core::compile_model::from_memory");
    auto parsed = parseDeviceNameIntoConfig(device_name, config);
    // in case of compile_model(file_name), we need to clear-up core-level properties
    auto plugin = get_plugin(parsed._deviceName);
    ov::SoPtr<ov::ICompiledModel> compiled_model;

    auto cacheManager = coreConfig.get_cache_config_for_device(plugin, parsed._config)._cacheManager;
    if (cacheManager && device_supports_model_caching(plugin)) {
        CacheContent cacheContent{cacheManager};
        cacheContent.blobId =
            ov::ModelCache::compute_hash(model_str, weights, create_compile_config(plugin, parsed._config));
        auto lock = cacheGuard.get_hash_lock(cacheContent.blobId);
        compiled_model = load_model_from_cache(cacheContent, plugin, parsed._config, ov::RemoteContext{}, [&]() {
            auto cnnNetwork = read_model(model_str, weights);
            return compile_model_and_cache(cnnNetwork, plugin, parsed._config, ov::RemoteContext{}, cacheContent);
        });
    } else {
        auto model = read_model(model_str, weights);
        compiled_model = compile_model_with_preprocess(plugin, model, ov::RemoteContext{}, parsed._config);
    }
    return compiled_model;
}

ov::SoPtr<ov::ICompiledModel> ov::CoreImpl::import_model(std::istream& model,
                                                         const std::string& device_name,
                                                         const ov::AnyMap& config) const {
    OV_ITT_SCOPED_TASK(ov::itt::domains::IE, "Core::import_model");
    auto parsed = parseDeviceNameIntoConfig(device_name, config);
    auto compiled_model = get_plugin(parsed._deviceName).import_model(model, parsed._config);
    if (auto wrapper = std::dynamic_pointer_cast<InferenceEngine::ICompiledModelWrapper>(compiled_model._ptr)) {
        wrapper->get_executable_network()->loadedFromCache();
    }

    return compiled_model;
}

ov::SupportedOpsMap ov::CoreImpl::query_model(const std::shared_ptr<const ov::Model>& model,
                                              const std::string& device_name,
                                              const ov::AnyMap& config) const {
    OV_ITT_SCOPED_TASK(ov::itt::domains::IE, "Core::query_model");
    auto parsed = parseDeviceNameIntoConfig(device_name, config);
    return get_plugin(parsed._deviceName).query_model(model, parsed._config);
}

std::vector<std::string> ov::CoreImpl::get_available_devices() const {
    std::vector<std::string> devices;
    const std::string propertyName = METRIC_KEY(AVAILABLE_DEVICES);

    for (auto&& deviceName : get_registered_devices()) {
        std::vector<std::string> devicesIDs;
        try {
            const ie::Parameter p = GetMetric(deviceName, propertyName);
            devicesIDs = p.as<std::vector<std::string>>();
        } catch (const ie::Exception&) {
            // plugin is not created by e.g. invalid env
        } catch (const ov::Exception&) {
            // plugin is not created by e.g. invalid env
        } catch (const std::runtime_error&) {
            // plugin is not created by e.g. invalid env
        } catch (const std::exception& ex) {
            IE_THROW() << "An exception is thrown while trying to create the " << deviceName
                       << " device and call GetMetric: " << ex.what();
        } catch (...) {
            IE_THROW() << "Unknown exception is thrown while trying to create the " << deviceName
                       << " device and call GetMetric";
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

ov::RemoteContext ov::CoreImpl::create_context(const std::string& device_name, const AnyMap& params) const {
    auto parsed = ov::parseDeviceNameIntoConfig(device_name, params);
    return get_plugin(parsed._deviceName).create_context(parsed._config);
}

ov::AnyMap ov::CoreImpl::get_supported_property(const std::string& full_device_name,
                                                const ov::AnyMap& user_properties) const {
    if (is_virtual_device(full_device_name)) {
        // Considerations:
        // 1. in case of virtual devices all the magic will happen on the level when
        // virtual device calls ICore::get_supported_property for real HW devices
        // so, for now we can returns user properties almost as is without any
        // filtering / flattening
        // 2. The only exception here: while common properties like ov::num::streams or
        // ov::hint::performance_mode are shared across all the devices, the
        // ov::device::priority cannot be shared, because it's specific for current virtual
        // plugin. So, we need to remove ov::device::priorities from the list, because it's
        // supposed to be set for current virtual plugin and cannot be propogated down
        ov::AnyMap return_properties = clone_map(user_properties);
        auto device_priorities_it = return_properties.find(ov::device::priorities.name());
        if (device_priorities_it != return_properties.end()) {
            return_properties.erase(device_priorities_it);
        }

        return return_properties;
    }

    static const std::vector<std::string> core_level_properties = {
        ov::cache_dir.name(),
        ov::force_tbb_terminate.name(),
        // auto-batch properties are also treated as core-level
        ov::auto_batch_timeout.name(),
        ov::hint::allow_auto_batching.name(),
    };

    const auto flattened = ov::parseDeviceNameIntoConfig(full_device_name, user_properties);
    const std::string& device_name = flattened._deviceName;
    const auto& flattened_config = flattened._config;

    // virtual plugins should bypass core-level properties to HW plugins
    // so, we need to report them as supported
    std::vector<std::string> supported_config_keys = core_level_properties;

    // try to search against IE API 1.0' SUPPORTED_CONFIG_KEYS
    try {
        const auto supported_keys =
            GetMetric(device_name, METRIC_KEY(SUPPORTED_CONFIG_KEYS), {}).as<std::vector<std::string>>();
        for (auto&& config_key : supported_keys) {
            supported_config_keys.emplace_back(config_key);
        }
    } catch (ov::Exception&) {
    }

    // try to search against OV API 2.0' mutable supported_properties
    try {
        for (auto&& property : ICore::get_property(device_name, ov::supported_properties, {})) {
            if (property.is_mutable()) {
                supported_config_keys.emplace_back(std::move(property));
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

bool ov::CoreImpl::is_new_api() const {
    return m_new_api;
}

ov::RemoteContext ov::CoreImpl::get_default_context(const std::string& device_name) const {
    auto parsed = ov::parseDeviceNameIntoConfig(device_name);
    return get_plugin(parsed._deviceName).get_default_context(parsed._config);
}

void ov::CoreImpl::apply_auto_batching(const std::shared_ptr<const ov::Model>& model,
                                       std::string& deviceName,
                                       ov::AnyMap& config) const {
    std::string deviceNameWithBatchSize, deviceNameWithoutBatch;
    // fully strict dims tracking by default (Auto-Batching is enabled implicitly)
    bool strictly_check_dims = true;
    if (deviceName.find("BATCH") != std::string::npos) {
        // explicitly enabled Auto-Batching
        auto pos = deviceName.find_first_of(":");
        if (pos == std::string::npos)
            return;  // BATCH device is already configured via the config
        deviceNameWithBatchSize = deviceName.substr(pos + 1);
        deviceNameWithoutBatch = ov::DeviceIDParser::get_batch_device(deviceNameWithBatchSize);
        // when user sets the BATCH device explicitly, we may check the dims less strictly
        // as the result is being checked by the user
        strictly_check_dims = false;
    } else {
        // check if Auto-Batch plugin registered
        try {
            get_plugin("BATCH");
        } catch (const std::runtime_error&) {
            return;
        }

        // check whether the Auto-Batching is disabled explicitly
        const auto& batch_mode = config.find(ov::hint::allow_auto_batching.name());
        if (batch_mode != config.end()) {
            const auto disabled = batch_mode->second.as<std::string>() == CONFIG_VALUE(NO);
            // virtual plugins like AUTO/MULTI will need the config
            // e.g to deduce the #requests correctly
            // otherwise, no need for this config key in the rest of loading
            if (!is_virtual_device(deviceName))
                config.erase(batch_mode);
            if (disabled)
                return;
        } else if (!coreConfig.get_allow_auto_batch()) {
            if (is_virtual_device(deviceName)) {
                config[ov::hint::allow_auto_batching.name()] = coreConfig.get_allow_auto_batch();
            }
            return;
        }

        // check whether if the Auto-Batching is applicable to the device
        auto parsed = ov::parseDeviceNameIntoConfig(deviceName);
        deviceNameWithoutBatch = deviceName;
        std::vector<std::string> metrics = get_plugin(parsed._deviceName)
                                               .get_property(METRIC_KEY(SUPPORTED_METRICS), parsed._config)
                                               .as<std::vector<std::string>>();
        auto it = std::find(metrics.begin(), metrics.end(), METRIC_KEY(OPTIMAL_BATCH_SIZE));
        if (metrics.end() == it)
            return;

        // if applicable, the Auto-Batching is implicitly enabled via the performance hints
        bool bTputInPlg =
            GetConfig(parsed._deviceName, CONFIG_KEY(PERFORMANCE_HINT)).as<std::string>() == CONFIG_VALUE(THROUGHPUT);
        const auto& mode = config.find(CONFIG_KEY(PERFORMANCE_HINT));
        bool bTputInLoadCfg = (mode != config.end() && mode->second.as<std::string>() == CONFIG_VALUE(THROUGHPUT));
        const auto& excl = config.find(CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS));
        bool bExclReqsEnabled = (excl != config.end() && excl->second.as<std::string>() == CONFIG_VALUE(YES));
        if (bExclReqsEnabled || (!bTputInPlg && !bTputInLoadCfg))
            return;
    }
    auto batchConfig = deviceNameWithBatchSize.empty() ? deviceNameWithoutBatch : deviceNameWithBatchSize;
    auto res = ov::details::is_model_batchable(model, deviceNameWithoutBatch, strictly_check_dims);
    switch (res) {
    case ov::details::NetworkBatchAbility::NO:
        return;
    case ov::details::NetworkBatchAbility::AS_IS:
        deviceName = "BATCH:" + batchConfig;
        break;
    case ov::details::NetworkBatchAbility::WITH_HETERO:
        deviceName = "HETERO:BATCH," + deviceNameWithoutBatch;
        config[CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG)] = batchConfig;
        break;
    }
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
        OPENVINO_ASSERT(!is_secondary_property,
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
    } else if (name == ov::hint::allow_auto_batching.name()) {
        const auto flag = coreConfig.get_allow_auto_batch();
        return decltype(ov::hint::allow_auto_batching)::value_type(flag);
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
        ov::AnyMap empty_map;
        return coreConfig.get_cache_config_for_device(get_plugin(parsed._deviceName), empty_map)._cacheDir;
    }

    return get_plugin(parsed._deviceName).get_property(name, parsed._config);
}

void ov::CoreImpl::unload_plugin(const std::string& deviceName) {
    std::lock_guard<std::mutex> lock(get_mutex());
    auto it = plugins.find(deviceName);
    if (it == plugins.end()) {
        IE_THROW() << "Device with \"" << deviceName << "\" name is not registered in the OpenVINO Runtime";
    }

    plugins.erase(deviceName);
}

void ov::CoreImpl::register_plugin(const std::string& plugin, const std::string& device_name) {
    std::lock_guard<std::mutex> lock(get_mutex());

    auto it = pluginRegistry.find(device_name);
    if (it != pluginRegistry.end()) {
        IE_THROW() << "Device with \"" << device_name << "\"  is already registered in the OpenVINO Runtime";
    }

    if (device_name.find('.') != std::string::npos) {
        IE_THROW() << "Device name must not contain dot '.' symbol";
    }

    PluginDescriptor desc{ov::util::get_plugin_path(plugin)};
    pluginRegistry[device_name] = desc;
    add_mutex(device_name);
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
            auto cache_it = config.find(CONFIG_KEY(CACHE_DIR));
            if (cache_it != config.end()) {
                coreConfig.set_cache_dir_for_device((cache_it->second).as<std::string>(), clearDeviceName);
            }
            OPENVINO_SUPPRESS_DEPRECATED_END
        }

        auto base_desc = pluginRegistry.find(clearDeviceName);
        if (pluginRegistry.find(deviceName) == pluginRegistry.end() && base_desc != pluginRegistry.end()) {
            PluginDescriptor desc{base_desc->second.libraryLocation, config, base_desc->second.listOfExtentions};
            pluginRegistry[deviceName] = desc;
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
            IE_THROW() << "Device with \"" << deviceName << "\" name is not registered in the OpenVINO Runtime";
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
                    ov::AnyMap empty_map;
                    auto cacheConfig = coreConfig.get_cache_config_for_device(plugin.second, empty_map);
                    if (cacheConfig._cacheManager) {
                        configCopy[CONFIG_KEY(CACHE_DIR)] = cacheConfig._cacheDir;
                    }
                } else if (configCopy.count(CONFIG_KEY(CACHE_DIR)) > 0) {
                    // Remove "CACHE_DIR" from config if it is not supported by plugin
                    configCopy.erase(CONFIG_KEY(CACHE_DIR));
                }
                OPENVINO_SUPPRESS_DEPRECATED_END
            }
            // Add device specific value to support device_name.device_id cases
            {
                if (!parser.get_device_id().empty()) {
                    const std::string deviceKey =
                        device_supports_property(plugin.second, CONFIG_KEY_INTERNAL(CONFIG_DEVICE_ID))
                            ? CONFIG_KEY_INTERNAL(CONFIG_DEVICE_ID)
                            : CONFIG_KEY(DEVICE_ID);
                    configCopy[deviceKey] = parser.get_device_id();
                }
            }
            plugin.second.set_property(configCopy);
        });
    }
}

void ov::CoreImpl::add_extension(const std::vector<ov::Extension::Ptr>& extensions) {
    std::lock_guard<std::mutex> lock(get_mutex());
    for (const auto& ext : extensions) {
        ov_extensions.emplace_back(ext);
        if (auto op_base_ext = std::dynamic_pointer_cast<ov::BaseOpExtension>(ext)) {
            for (const auto& attached_ext : op_base_ext->get_attached_extensions()) {
                ov_extensions.emplace_back(attached_ext);
            }
        }
    }
}

const std::vector<InferenceEngine::IExtensionPtr>& ov::CoreImpl::GetExtensions() const {
    return extensions;
}

bool ov::CoreImpl::device_supports_model_caching(const std::string& deviceName) const {
    auto parsed = parseDeviceNameIntoConfig(deviceName);
    return device_supports_model_caching(get_plugin(parsed._deviceName));
}

bool ov::CoreImpl::device_supports_property(const ov::Plugin& plugin, const ov::PropertyName& key) const {
    return util::contains(plugin.get_property(ov::supported_properties), key);
}

bool ov::CoreImpl::device_supports_model_caching(const ov::Plugin& plugin) const {
    auto supportedMetricKeys = plugin.get_property(METRIC_KEY(SUPPORTED_METRICS), {}).as<std::vector<std::string>>();
    auto supported = util::contains(supportedMetricKeys, METRIC_KEY(IMPORT_EXPORT_SUPPORT)) &&
                     plugin.get_property(METRIC_KEY(IMPORT_EXPORT_SUPPORT), {}).as<bool>();
    if (!supported) {
        supported =
            device_supports_property(plugin, ov::device::capabilities) &&
            util::contains(plugin.get_property(ov::device::capabilities), ov::device::capability::EXPORT_IMPORT);
    }
    if (supported) {
        supported = device_supports_property(plugin, ov::caching_properties);
    }
    return supported;
}

bool ov::CoreImpl::device_supports_cache_dir(const ov::Plugin& plugin) const {
    return util::contains(plugin.get_property(ov::supported_properties), ov::cache_dir);
}

ov::SoPtr<ov::ICompiledModel> ov::CoreImpl::compile_model_and_cache(const std::shared_ptr<const ov::Model>& model,
                                                                    ov::Plugin& plugin,
                                                                    const ov::AnyMap& parsedConfig,
                                                                    const ov::RemoteContext& context,
                                                                    const CacheContent& cacheContent) const {
    OV_ITT_SCOPED_TASK(ov::itt::domains::IE, "CoreImpl::compile_model_and_cache");
    ov::SoPtr<ov::ICompiledModel> execNetwork;
    execNetwork = compile_model_with_preprocess(plugin, model, context, parsedConfig);
    if (cacheContent.cacheManager && device_supports_model_caching(plugin)) {
        try {
            // need to export network for further import from "cache"
            OV_ITT_SCOPE(FIRST_INFERENCE, InferenceEngine::itt::domains::IE_LT, "Core::compile_model::Export");
            cacheContent.cacheManager->write_cache_entry(cacheContent.blobId, [&](std::ostream& networkStream) {
                networkStream << ov::CompiledBlobHeader(InferenceEngine::GetInferenceEngineVersion()->buildNumber,
                                                        ov::ModelCache::calculate_file_info(cacheContent.modelPath));
                execNetwork->export_model(networkStream);
            });
        } catch (...) {
            cacheContent.cacheManager->remove_cache_entry(cacheContent.blobId);
            throw;
        }
    }
    return execNetwork;
}

ov::SoPtr<ov::ICompiledModel> ov::CoreImpl::load_model_from_cache(
    const CacheContent& cacheContent,
    ov::Plugin& plugin,
    const ov::AnyMap& config,
    const ov::RemoteContext& context,
    std::function<ov::SoPtr<ov::ICompiledModel>()> compile_model_lambda) {
    ov::SoPtr<ov::ICompiledModel> compiled_model;
    struct HeaderException {};

    OPENVINO_ASSERT(cacheContent.cacheManager != nullptr);
    try {
        cacheContent.cacheManager->read_cache_entry(cacheContent.blobId, [&](std::istream& networkStream) {
            OV_ITT_SCOPE(FIRST_INFERENCE,
                         InferenceEngine::itt::domains::IE_LT,
                         "Core::load_model_from_cache::ReadStreamAndImport");
            try {
                ov::CompiledBlobHeader header;
                networkStream >> header;
                if (header.getIeVersion() != InferenceEngine::GetInferenceEngineVersion()->buildNumber) {
                    // Build number mismatch, don't use this cache
                    throw InferenceEngine::NetworkNotRead("Version does not match");
                }
                if (header.getFileInfo() != ov::ModelCache::calculate_file_info(cacheContent.modelPath)) {
                    // Original file is changed, don't use cache
                    throw InferenceEngine::NetworkNotRead("Original model file is changed");
                }
            } catch (...) {
                throw HeaderException();
            }

            compiled_model = context._impl ? plugin.import_model(networkStream, context, config)
                                           : plugin.import_model(networkStream, config);
            if (auto wrapper = std::dynamic_pointer_cast<InferenceEngine::ICompiledModelWrapper>(compiled_model._ptr)) {
                wrapper->get_executable_network()->loadedFromCache();
            }
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

    // 0. Move ov::device::priorities / TARGET_FALLBACK key to property_config
    auto device_priorities_it = user_config.find("TARGET_FALLBACK");
    if (device_priorities_it == user_config.end()) {
        device_priorities_it = user_config.find(ov::device::priorities.name());
    }
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
    auto caching_props = plugin.get_property(ov::caching_properties, property_config);
    OPENVINO_ASSERT(!caching_props.empty(), "ov::caching_properties returned by ", plugin.get_name(), " are empty");

    ov::AnyMap compile_config;
    for (const auto& prop : caching_props) {
        // user_config values have higher priority than plugin parameters
        auto it = user_config.find(prop);
        compile_config[prop] = it == user_config.end() ? plugin.get_property(prop, property_config) : it->second;
    }

    return compile_config;
}

void ov::CoreImpl::AddExtensionUnsafe(const InferenceEngine::IExtensionPtr& extension) const {
    std::map<std::string, ngraph::OpSet> opsets = extension->getOpSets();
    for (const auto& it : opsets) {
        if (opsetNames.find(it.first) != opsetNames.end())
            IE_THROW() << "Cannot add opset with name: " << it.first << ". Opset with the same name already exists.";
        opsetNames.insert(it.first);
    }

    // add extensions for already created plugins
    for (auto& plugin : plugins) {
        allowNotImplemented([&]() {
            plugin.second.add_extension(extension);
        });
    }
    extensions.emplace_back(extension);
}

void ov::CoreImpl::CoreConfig::set_and_update(ov::AnyMap& config) {
    auto it = config.find(CONFIG_KEY(CACHE_DIR));
    if (it != config.end()) {
        std::lock_guard<std::mutex> lock(_cacheConfigMutex);
        // fill global cache config
        _cacheConfig = CoreConfig::CacheConfig::create(it->second.as<std::string>());
        // sets cache config per-device if it's not set explicitly before
        for (auto& deviceCfg : _cacheConfigPerDevice) {
            deviceCfg.second = CoreConfig::CacheConfig::create(it->second.as<std::string>());
        }
        config.erase(it);
    }

    it = config.find(ov::force_tbb_terminate.name());
    if (it != config.end()) {
        auto flag = it->second.as<std::string>() == CONFIG_VALUE(YES) ? true : false;
        ov::threading::executor_manager()->set_property({{it->first, flag}});
        config.erase(it);
    }

    it = config.find(ov::hint::allow_auto_batching.name());
    if (it != config.end()) {
        auto flag = it->second.as<bool>();
        _flag_allow_auto_batching = flag;
        config.erase(it);
    }

    it = config.find(ov::enable_mmap.name());
    if (it != config.end()) {
        auto flag = it->second.as<bool>();
        _flag_enable_mmap = flag;
        config.erase(it);
    }
}

void ov::CoreImpl::CoreConfig::set_cache_dir_for_device(const std::string& dir, const std::string& name) {
    std::lock_guard<std::mutex> lock(_cacheConfigMutex);
    _cacheConfigPerDevice[name] = CoreConfig::CacheConfig::create(dir);
}

std::string ov::CoreImpl::CoreConfig::get_cache_dir() const {
    std::lock_guard<std::mutex> lock(_cacheConfigMutex);
    return _cacheConfig._cacheDir;
}

bool ov::CoreImpl::CoreConfig::get_allow_auto_batch() const {
    return _flag_allow_auto_batching;
}

bool ov::CoreImpl::CoreConfig::get_enable_mmap() const {
    return _flag_enable_mmap;
}

// Creating thread-safe copy of config including shared_ptr to ICacheManager
// Passing empty or not-existing name will return global cache config
ov::CoreImpl::CoreConfig::CacheConfig ov::CoreImpl::CoreConfig::get_cache_config_for_device(
    const ov::Plugin& plugin,
    ov::AnyMap& parsedConfig) const {
    // cache_dir is enabled locally in compile_model only
    if (parsedConfig.count(ov::cache_dir.name())) {
        auto cache_dir_val = parsedConfig.at(ov::cache_dir.name()).as<std::string>();
        auto tempConfig = CoreConfig::CacheConfig::create(cache_dir_val);
        // if plugin does not explicitly support cache_dir, and if plugin is not virtual, we need to remove
        // it from config
        if (!util::contains(plugin.get_property(ov::supported_properties), ov::cache_dir) &&
            !is_virtual_device(plugin.get_name())) {
            parsedConfig.erase(ov::cache_dir.name());
        }
        return tempConfig;
    } else {  // cache_dir is set to Core globally or for the specific device
        std::lock_guard<std::mutex> lock(_cacheConfigMutex);
        if (_cacheConfigPerDevice.count(plugin.get_name()) > 0) {
            return _cacheConfigPerDevice.at(plugin.get_name());
        } else {
            return _cacheConfig;
        }
    }
}

ov::CoreImpl::CoreConfig::CacheConfig ov::CoreImpl::CoreConfig::CacheConfig::create(const std::string& dir) {
    std::shared_ptr<ov::ICacheManager> cache_manager = nullptr;

    if (!dir.empty()) {
        FileUtils::createDirectoryRecursive(dir);
        cache_manager = std::make_shared<ov::FileStorageCacheManager>(dir);
    }

    return {dir, cache_manager};
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

std::shared_ptr<ov::Model> ov::CoreImpl::read_model(const std::string& modelPath, const std::string& binPath) const {
    OV_ITT_SCOPE(FIRST_INFERENCE, ov::itt::domains::IE_RT, "CoreImpl::read_model from file");
    return ReadNetwork(modelPath, binPath).getFunction();
}

std::shared_ptr<ov::Model> ov::CoreImpl::read_model(const std::string& model,
                                                    const ov::Tensor& weights,
                                                    bool frontendMode) const {
    InferenceEngine::Blob::Ptr blob;
    if (weights) {
        blob = tensor_to_blob(weights._impl);
    }
    OV_ITT_SCOPE(FIRST_INFERENCE, ov::itt::domains::IE_RT, "CoreImpl::read_model from memory");
    return ReadNetwork(model, blob, frontendMode).getFunction();
}
