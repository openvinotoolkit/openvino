// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "any_copy.hpp"
#include "compilation_context.hpp"
#include "core_impl.hpp"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"
#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"
#include "dev/converter_utils.hpp"
#include "ie_network_reader.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/pass/constant_folding.hpp"
#include "openvino/itt.hpp"
#include "openvino/util/common_util.hpp"

bool ov::CoreImpl::isNewAPI() const {
    return is_new_api();
}

ov::SoPtr<InferenceEngine::IExecutableNetworkInternal> ov::CoreImpl::LoadNetworkImpl(
    const InferenceEngine::CNNNetwork& network,
    ov::Plugin& plugin,
    const std::map<std::string, std::string>& parsedConfig,
    const InferenceEngine::RemoteContext::Ptr& context,
    const CacheContent& cacheContent,
    bool forceDisableCache) {
    OV_ITT_SCOPED_TASK(ov::itt::domains::IE, "CoreImpl::compile_model_impl");
    ov::SoPtr<InferenceEngine::IExecutableNetworkInternal> execNetwork;
    auto wrapper = std::dynamic_pointer_cast<InferenceEngine::IPluginWrapper>(plugin.m_ptr);
    OPENVINO_ASSERT(wrapper);
    auto old_plugin = wrapper->get_plugin();
    execNetwork = {context ? old_plugin->LoadNetwork(network, parsedConfig, context)
                           : old_plugin->LoadNetwork(network, parsedConfig),
                   plugin.m_so};
    if (!forceDisableCache && cacheContent.cacheManager && device_supports_import_export(plugin)) {
        try {
            // need to export network for further import from "cache"
            OV_ITT_SCOPE(FIRST_INFERENCE, InferenceEngine::itt::domains::IE_LT, "Core::LoadNetwork::Export");
            cacheContent.cacheManager->writeCacheEntry(cacheContent.blobId, [&](std::ostream& networkStream) {
                networkStream << InferenceEngine::CompiledBlobHeader(
                    InferenceEngine::GetInferenceEngineVersion()->buildNumber,
                    InferenceEngine::NetworkCompilationContext::calculateFileInfo(cacheContent.modelPath));
                execNetwork->Export(networkStream);
            });
        } catch (...) {
            cacheContent.cacheManager->removeCacheEntry(cacheContent.blobId);
            throw;
        }
    }
    return execNetwork;
}

InferenceEngine::RemoteContext::Ptr ov::CoreImpl::GetDefaultContext(const std::string& deviceName) {
    return get_default_context(deviceName)._impl;
}

InferenceEngine::CNNNetwork ov::CoreImpl::ReadNetwork(const std::string& modelPath, const std::string& binPath) const {
    OV_ITT_SCOPE(FIRST_INFERENCE, ov::itt::domains::IE_RT, "CoreImpl::ReadNetwork from file");
    return InferenceEngine::details::ReadNetwork(modelPath, binPath, extensions, ov_extensions, is_new_api());
}

InferenceEngine::CNNNetwork ov::CoreImpl::ReadNetwork(const std::string& model,
                                                      const InferenceEngine::Blob::CPtr& weights,
                                                      bool frontendMode) const {
    OV_ITT_SCOPE(FIRST_INFERENCE, ov::itt::domains::IE_RT, "CoreImpl::ReadNetwork from memory");
    return InferenceEngine::details::ReadNetwork(model, weights, extensions, ov_extensions, is_new_api(), frontendMode);
}

ov::SoPtr<InferenceEngine::IExecutableNetworkInternal> ov::CoreImpl::LoadNetwork(
    const InferenceEngine::CNNNetwork& network,
    const std::shared_ptr<InferenceEngine::RemoteContext>& context,
    const std::map<std::string, std::string>& config) {
    OV_ITT_SCOPE(FIRST_INFERENCE, InferenceEngine::itt::domains::IE_LT, "Core::LoadNetwork::RemoteContext");
    if (network.getFunction()) {
        ov::RemoteContext ctx{context, {nullptr}};
        auto compiled_model =
            compile_model(ov::legacy_convert::convert_model(network, isNewAPI()), ctx, any_copy(config));
        return {compiled_model._ptr, compiled_model._so};
    }
    if (context == nullptr) {
        IE_THROW() << "Remote context is null";
    }
    // have to deduce the device name/config from the context first
    auto parsed = parseDeviceNameIntoConfig(context->getDeviceName(), config);

    auto plugin = get_plugin(parsed._deviceName);

    ov::SoPtr<InferenceEngine::IExecutableNetworkInternal> res;
    auto conf = ov::any_copy(parsed._config);
    auto cacheManager =
        coreConfig.get_cache_config_for_device(parsed._deviceName, device_supports_cache_dir(plugin), conf)
            ._cacheManager;
    auto cacheContent = CacheContent{cacheManager};
    if (cacheManager && device_supports_import_export(plugin)) {
        cacheContent.blobId = CalculateNetworkHash(network, parsed._deviceName, plugin, ov::any_copy(parsed._config));
        bool loadedFromCache = false;
        auto lock = cacheGuard.getHashLock(cacheContent.blobId);
        res = load_model_from_cache(cacheContent, plugin, conf, {context, {}}, loadedFromCache);
        if (!loadedFromCache) {
            res = LoadNetworkImpl(network, plugin, parsed._config, context, cacheContent);
        } else {
            // Temporary workaround until all plugins support caching of original model inputs
            InferenceEngine::SetExeNetworkInfo(res._ptr, network.getFunction(), isNewAPI());
        }
    } else {
        res = LoadNetworkImpl(network, plugin, parsed._config, context, cacheContent);
    }
    return res;
}

InferenceEngine::SoExecutableNetworkInternal ov::CoreImpl::LoadNetwork(
    const InferenceEngine::CNNNetwork& network,
    const std::string& deviceNameOrig,
    const std::map<std::string, std::string>& config) {
    OV_ITT_SCOPE(FIRST_INFERENCE, InferenceEngine::itt::domains::IE_LT, "Core::LoadNetwork::CNN");
    if (network.getFunction()) {
        auto compiled_model =
            compile_model(ov::legacy_convert::convert_model(network, isNewAPI()), deviceNameOrig, any_copy(config));
        return {compiled_model._ptr, compiled_model._so};
    }
    std::string deviceName = deviceNameOrig;
    std::map<std::string, std::string> config_with_batch = config;
    bool forceDisableCache = config_with_batch.count(CONFIG_KEY_INTERNAL(FORCE_DISABLE_CACHE)) > 0;
    auto parsed = parseDeviceNameIntoConfig(deviceName, config_with_batch);
    if (forceDisableCache) {
        // remove this config key from parsed as plugins can throw unsupported exception
        parsed._config.erase(CONFIG_KEY_INTERNAL(FORCE_DISABLE_CACHE));
    }
    auto plugin = get_plugin(parsed._deviceName);
    ov::SoPtr<InferenceEngine::IExecutableNetworkInternal> res;
    auto conf = ov::any_copy(parsed._config);
    auto cacheManager =
        coreConfig.get_cache_config_for_device(parsed._deviceName, device_supports_cache_dir(plugin), conf)
            ._cacheManager;
    auto cacheContent = CacheContent{cacheManager};
    if (!forceDisableCache && cacheManager && device_supports_import_export(plugin)) {
        cacheContent.blobId = CalculateNetworkHash(network, parsed._deviceName, plugin, ov::any_copy(parsed._config));
        bool loadedFromCache = false;
        auto lock = cacheGuard.getHashLock(cacheContent.blobId);
        res = load_model_from_cache(cacheContent, plugin, conf, {}, loadedFromCache);
        if (!loadedFromCache) {
            res = LoadNetworkImpl(network, plugin, parsed._config, nullptr, cacheContent, forceDisableCache);
        } else {
            // Temporary workaround until all plugins support caching of original model inputs
            InferenceEngine::SetExeNetworkInfo(res._ptr, network.getFunction(), isNewAPI());
        }
    } else {
        res = LoadNetworkImpl(network, plugin, parsed._config, nullptr, cacheContent, forceDisableCache);
    }
    return {res._ptr, res._so};
}

InferenceEngine::SoExecutableNetworkInternal ov::CoreImpl::LoadNetwork(
    const std::string& modelPath,
    const std::string& deviceName,
    const std::map<std::string, std::string>& config,
    const std::function<void(const InferenceEngine::CNNNetwork&)>& val) {
    OV_ITT_SCOPE(FIRST_INFERENCE, ie::itt::domains::IE_LT, "Core::LoadNetwork::Path");
    auto parsed = parseDeviceNameIntoConfig(deviceName, config);
    auto plugin = get_plugin(parsed._deviceName);
    ov::SoPtr<InferenceEngine::IExecutableNetworkInternal> res;
    auto conf = any_copy(parsed._config);
    auto cacheManager =
        coreConfig.get_cache_config_for_device(parsed._deviceName, device_supports_cache_dir(plugin), conf)
            ._cacheManager;
    auto cacheContent = CacheContent{cacheManager, modelPath};
    if (cacheManager && device_supports_import_export(plugin)) {
        bool loadedFromCache = false;
        cacheContent.blobId = calculate_file_hash(modelPath, parsed._deviceName, plugin, conf);
        auto lock = cacheGuard.getHashLock(cacheContent.blobId);
        res = load_model_from_cache(cacheContent, plugin, conf, {}, loadedFromCache);
        if (!loadedFromCache) {
            auto cnnNetwork = ReadNetwork(modelPath, std::string());
            if (val) {
                val(cnnNetwork);
            }
            if (cnnNetwork.getFunction()) {
                res = compile_model_impl(ov::legacy_convert::convert_model(cnnNetwork, isNewAPI()),
                                         plugin,
                                         conf,
                                         {},
                                         cacheContent);
            } else {
                res = LoadNetworkImpl(cnnNetwork, plugin, parsed._config, nullptr, cacheContent);
            }
        }
    } else if (cacheManager) {
        res = plugin.compile_model(modelPath, conf);
    } else {
        auto cnnNetwork = ReadNetwork(modelPath, std::string());
        if (val) {
            val(cnnNetwork);
        }
        if (cnnNetwork.getFunction()) {
            res = compile_model_impl(ov::legacy_convert::convert_model(cnnNetwork, isNewAPI()),
                                     plugin,
                                     conf,
                                     {},
                                     cacheContent);
        } else {
            res = LoadNetworkImpl(cnnNetwork, plugin, parsed._config, nullptr, cacheContent);
        }
    }
    return {res._ptr, res._so};
}

InferenceEngine::SoExecutableNetworkInternal ov::CoreImpl::LoadNetwork(
    const std::string& modelStr,
    const InferenceEngine::Blob::CPtr& weights,
    const std::string& deviceName,
    const std::map<std::string, std::string>& config,
    const std::function<void(const InferenceEngine::CNNNetwork&)>& val) {
    OV_ITT_SCOPE(FIRST_INFERENCE, InferenceEngine::itt::domains::IE_LT, "Core::LoadNetwork::Memory");

    auto compiled_model = compile_model(modelStr,
                                        ov::Tensor{std::const_pointer_cast<InferenceEngine::Blob>(weights), {}},
                                        deviceName,
                                        ov::any_copy(config));
    return {compiled_model._ptr, compiled_model._so};
}

InferenceEngine::SoExecutableNetworkInternal ov::CoreImpl::ImportNetwork(
    std::istream& networkModel,
    const std::string& deviceName,
    const std::map<std::string, std::string>& config) {
    auto compiled_model = import_model(networkModel, deviceName, any_copy(config));
    return {compiled_model._ptr, compiled_model._so};
}

InferenceEngine::QueryNetworkResult ov::CoreImpl::QueryNetwork(const InferenceEngine::CNNNetwork& network,
                                                               const std::string& deviceName,
                                                               const std::map<std::string, std::string>& config) const {
    OV_ITT_SCOPED_TASK(ov::itt::domains::IE, "Core::QueryNetwork");
    auto res = query_model(network.getFunction(), deviceName, any_copy(config));
    ie::QueryNetworkResult ret;
    if (!network.getFunction() || res.empty()) {
        ret.rc = InferenceEngine::GENERAL_ERROR;
        return ret;
    }
    ret.supportedLayersMap = res;

    const auto& func = network.getFunction();
    auto specialized_function = func->clone();

    std::string defDevice = ret.supportedLayersMap.begin()->second;
    ngraph::pass::ConstantFolding().run_on_model(specialized_function);
    std::unordered_set<std::string> opNames;

    for (const auto& op : specialized_function->get_ops())
        opNames.emplace(op->get_friendly_name());

    for (const auto& op : func->get_ops()) {
        if (opNames.find(op->get_friendly_name()) == opNames.end()) {
            ret.supportedLayersMap[op->get_friendly_name()] = defDevice;
        }
    }

    for (const auto& op : func->get_ops()) {
        if (!ret.supportedLayersMap.count(op->get_friendly_name()) &&
            std::dynamic_pointer_cast<ngraph::op::Constant>(op)) {
            bool are_all_users_supported = true;
            for (const auto& user : op->output(0).get_target_inputs()) {
                if (!ret.supportedLayersMap.count(user.get_node()->get_friendly_name())) {
                    are_all_users_supported = false;
                    break;
                }
            }
            if (are_all_users_supported) {
                ret.supportedLayersMap[op->get_friendly_name()] = defDevice;
            }
        }
    }
    return ret;
}

ov::Any ov::CoreImpl::GetMetric(const std::string& deviceName,
                                const std::string& name,
                                const ov::AnyMap& options) const {
    // HETERO case
    {
        if (deviceName.find("HETERO:") == 0) {
            IE_THROW()
                << "You can get specific metrics with the GetMetric only for the HETERO itself (without devices). "
                   "To get individual devices's metrics call GetMetric for each device separately";
        }
    }

    // MULTI case
    {
        if (deviceName.find("MULTI:") == 0) {
            IE_THROW()
                << "You can get specific metrics with the GetMetric only for the MULTI itself (without devices). "
                   "To get individual devices's metrics call GetMetric for each device separately";
        }
    }

    // AUTO case
    {
        if (deviceName.find("AUTO:") == 0) {
            IE_THROW() << "You can get specific metrics with the GetMetric only for the AUTO itself (without devices). "
                          "To get individual devices's metrics call GetMetric for each device separately";
        }
    }

    // BATCH case
    {
        if (deviceName.find("BATCH:") == 0) {
            IE_THROW()
                << "You can get specific metrics with the GetMetric only for the BATCH itself (without devices). "
                   "To get individual devices's metrics call GetMetric for each device separately";
        }
    }

    auto parsed = parseDeviceNameIntoConfig(deviceName);
    for (auto o : options) {
        parsed._config.insert(o);
    }

    return get_plugin(parsed._deviceName).get_property(name, parsed._config);
}

ov::Any ov::CoreImpl::GetConfig(const std::string& deviceName, const std::string& name) const {
    auto parsed = parseDeviceNameIntoConfig(deviceName);
    return get_plugin(parsed._deviceName).get_property(name, parsed._config);
}

std::vector<std::string> ov::CoreImpl::GetAvailableDevices() const {
    std::vector<std::string> devices;
    const std::string propertyName = METRIC_KEY(AVAILABLE_DEVICES);

    for (auto&& deviceName : get_registered_devices()) {
        std::vector<std::string> devicesIDs;
        try {
            const InferenceEngine::Parameter p = GetMetric(deviceName, propertyName);
            devicesIDs = p.as<std::vector<std::string>>();
        } catch (const InferenceEngine::Exception&) {
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

InferenceEngine::RemoteContext::Ptr ov::CoreImpl::CreateContext(const std::string& deviceName,
                                                                const InferenceEngine::ParamMap& params) {
    return create_context(deviceName, params)._impl;
}

/**
 * @brief Registers the extension in a Core object
 *        Such extensions can be used for both CNNNetwork readers and device plugins
 */
void ov::CoreImpl::AddExtension(const InferenceEngine::IExtensionPtr& extension) {
    std::lock_guard<std::mutex> lock(get_mutex());
    AddExtensionUnsafe(extension);
}

bool ov::CoreImpl::DeviceSupportsImportExport(const std::string& deviceName) const {
    return device_supports_import_export(deviceName);
}

std::map<std::string, std::string> ov::CoreImpl::GetSupportedConfig(const std::string& deviceName,
                                                                    const std::map<std::string, std::string>& configs) {
    std::vector<std::string> supportedConfigKeys;
    try {
        supportedConfigKeys = GetMetric(deviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS)).as<std::vector<std::string>>();
    } catch (ov::Exception&) {
    }
    try {
        for (auto&& property : ICore::get_property(deviceName, ov::supported_properties)) {
            if (property.is_mutable()) {
                supportedConfigKeys.emplace_back(std::move(property));
            }
        }
    } catch (ov::Exception&) {
    }
    std::map<std::string, std::string> supportedConfig;
    for (auto&& key : supportedConfigKeys) {
        auto itKey = configs.find(key);
        if (configs.end() != itKey) {
            supportedConfig[key] = itKey->second;
        }
    }
    for (auto&& config : configs) {
        auto parsed = parseDeviceNameIntoConfig(config.first);
        if (deviceName.find(parsed._deviceName) != std::string::npos) {
            std::stringstream strm(config.second);
            std::map<std::string, std::string> device_configs;
            util::Read<std::map<std::string, std::string>>{}(strm, device_configs);
            for (auto&& device_config : device_configs) {
                if (ov::util::contains(supportedConfigKeys, device_config.first)) {
                    supportedConfig[device_config.first] = device_config.second;
                }
            }
            for (auto&& config : parsed._config) {
                supportedConfig[config.first] = config.second.as<std::string>();
            }
        }
    }
    return supportedConfig;
}

std::map<std::string, InferenceEngine::Version> ov::CoreImpl::GetVersions(const std::string& deviceName) const {
    std::map<std::string, InferenceEngine::Version> versions;
    std::vector<std::string> deviceNames;

    {
        // for compatibility with samples / demo
        if (deviceName.find("HETERO") == 0) {
            auto pos = deviceName.find_first_of(":");
            if (pos != std::string::npos) {
                deviceNames = InferenceEngine::DeviceIDParser::getHeteroDevices(deviceName.substr(pos + 1));
            }
            deviceNames.push_back("HETERO");
        } else if (deviceName.find("MULTI") == 0) {
            auto pos = deviceName.find_first_of(":");
            if (pos != std::string::npos) {
                deviceNames = InferenceEngine::DeviceIDParser::getMultiDevices(deviceName.substr(pos + 1));
            }
            deviceNames.push_back("MULTI");
        } else if (deviceName.find("AUTO") == 0) {
            auto pos = deviceName.find_first_of(":");
            if (pos != std::string::npos) {
                deviceNames = InferenceEngine::DeviceIDParser::getMultiDevices(deviceName.substr(pos + 1));
            }
            deviceNames.emplace_back("AUTO");
        } else if (deviceName.find("BATCH") == 0) {
            auto pos = deviceName.find_first_of(":");
            if (pos != std::string::npos) {
                deviceNames = {InferenceEngine::DeviceIDParser::getBatchDevice(deviceName.substr(pos + 1))};
            }
            deviceNames.push_back("BATCH");
        } else {
            deviceNames.push_back(deviceName);
        }
    }

    for (auto&& deviceName_ : deviceNames) {
        ie::DeviceIDParser parser(deviceName_);
        std::string deviceNameLocal = parser.getDeviceName();

        ov::Plugin cppPlugin = get_plugin(deviceNameLocal);

        versions[deviceNameLocal] = ov::legacy_convert::convert_plugin(cppPlugin.m_ptr)->GetVersion();
    }

    return versions;
}
