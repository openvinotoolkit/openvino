// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core_impl.hpp"

#include "check_network_batchable.hpp"
#include "compilation_context.hpp"
#include "cpp/ie_plugin.hpp"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"
#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"
#include "file_utils.h"
#include "ie_itt.hpp"
#include "ie_network_reader.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/pass/constant_folding.hpp"
#include "openvino/core/op_extension.hpp"
#include "openvino/core/version.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/shared_object.hpp"
#include "xml_parse_utils.h"

namespace {

template <typename F>
void allowNotImplemented(F&& f) {
    try {
        f();
    } catch (const InferenceEngine::NotImplemented&) {
    }
}

ov::util::FilePath getPluginPath(const std::string& pluginName, bool needAddSuffixes = false) {
    const auto ieLibraryPath = InferenceEngine::getInferenceEngineLibraryPath();

    auto pluginPath = ov::util::to_file_path(pluginName.c_str());

    // 0. user can provide a full path

#ifndef _WIN32
    try {
        // dlopen works with absolute paths; otherwise searches from LD_LIBRARY_PATH
        pluginPath = ov::util::to_file_path(ov::util::get_absolute_file_path(pluginName));
    } catch (const std::runtime_error&) {
        // failed to resolve absolute path; not critical
    }
#endif  // _WIN32

    if (FileUtils::fileExist(pluginPath))
        return pluginPath;

    // ov::Core::register_plugin(plugin_name, device_name) case
    if (needAddSuffixes)
        pluginPath = FileUtils::makePluginLibraryName({}, pluginPath);

    // plugin can be found either:

    // 1. in openvino-X.Y.Z folder relative to libopenvino.so
    std::ostringstream str;
    str << "openvino-" << OPENVINO_VERSION_MAJOR << "." << OPENVINO_VERSION_MINOR << "." << OPENVINO_VERSION_PATCH;
    const auto subFolder = ov::util::to_file_path(str.str());

    ov::util::FilePath absFilePath = FileUtils::makePath(FileUtils::makePath(ieLibraryPath, subFolder), pluginPath);
    if (FileUtils::fileExist(absFilePath))
        return absFilePath;

    // 2. in the openvino.so location
    absFilePath = FileUtils::makePath(ieLibraryPath, pluginPath);
    if (FileUtils::fileExist(absFilePath))
        return absFilePath;

    // 3. in LD_LIBRARY_PATH on Linux / PATH on Windows
    return pluginPath;
}

void stripDeviceName(std::string& device, const std::string& substr) {
    auto pos = device.find(substr);
    if (pos == 0) {
        device.erase(pos, substr.length());
    }
}

}  // namespace

ov::CoreImpl::CoreImpl(bool _newAPI) : newAPI(_newAPI) {
    add_mutex("");  // Register global mutex
    executorManagerPtr = InferenceEngine::executorManager();
    for (const auto& it : ov::get_available_opsets()) {
        opsetNames.insert(it.first);
    }
}

void ov::CoreImpl::RegisterPluginsInRegistry(const std::string& xmlConfigFile) {
    std::lock_guard<std::mutex> lock(get_mutex());

    auto parse_result = ParseXml(xmlConfigFile.c_str());
    if (!parse_result.error_msg.empty()) {
        IE_THROW() << parse_result.error_msg;
    }

    pugi::xml_document& xmlDoc = *parse_result.xml;

    using namespace XMLParseUtils;
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

        ov::util::FilePath pluginPath = getPluginPath(GetStrAttr(pluginNode, "location"));

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
ov::InferencePlugin ov::CoreImpl::GetCPPPluginByName(const std::string& pluginName) const {
    OV_ITT_SCOPE(FIRST_INFERENCE, InferenceEngine::itt::domains::IE_LT, "CoreImpl::GetCPPPluginByName");

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
        ov::InferencePlugin plugin;

        if (desc.pluginCreateFunc) {  // static OpenVINO case
            std::shared_ptr<InferenceEngine::IInferencePlugin> plugin_impl;
            desc.pluginCreateFunc(plugin_impl);
            plugin = InferencePlugin{plugin_impl, {}};
        } else {
            so = ov::util::load_shared_object(desc.libraryLocation.c_str());
            std::shared_ptr<InferenceEngine::IInferencePlugin> plugin_impl;
            reinterpret_cast<InferenceEngine::CreatePluginEngineFunc*>(
                ov::util::get_symbol(so, InferenceEngine::create_plugin_function))(plugin_impl);
            plugin = InferencePlugin{plugin_impl, so};
        }

        {
            plugin.set_name(deviceName);

            // Set Core class reference to plugins
            std::weak_ptr<InferenceEngine::ICore> mutableCore =
                std::const_pointer_cast<InferenceEngine::ICore>(shared_from_this());
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
            if (DeviceSupportsCacheDir(plugin)) {
                auto cacheConfig = coreConfig.getCacheConfigForDevice(deviceName);
                if (cacheConfig._cacheManager) {
                    desc.defaultConfig[CONFIG_KEY(CACHE_DIR)] = cacheConfig._cacheDir;
                }
            } else if (desc.defaultConfig.count(CONFIG_KEY(CACHE_DIR)) > 0) {
                // Remove "CACHE_DIR" from config if it is not supported by plugin
                desc.defaultConfig.erase(CONFIG_KEY(CACHE_DIR));
            }
            allowNotImplemented([&]() {
                // Add device specific value to support device_name.device_id cases
                std::vector<std::string> supportedConfigKeys = plugin.get_metric(METRIC_KEY(SUPPORTED_CONFIG_KEYS), {});
                auto config_iter = std::find(supportedConfigKeys.begin(),
                                             supportedConfigKeys.end(),
                                             CONFIG_KEY_INTERNAL(CONFIG_DEVICE_ID));
                const bool supportsConfigDeviceID = config_iter != supportedConfigKeys.end();
                const std::string deviceKey =
                    supportsConfigDeviceID ? CONFIG_KEY_INTERNAL(CONFIG_DEVICE_ID) : CONFIG_KEY(DEVICE_ID);

                for (auto pluginDesc : pluginRegistry) {
                    InferenceEngine::DeviceIDParser parser(pluginDesc.first);
                    if (pluginDesc.first.find(deviceName) != std::string::npos && !parser.getDeviceID().empty()) {
                        pluginDesc.second.defaultConfig[deviceKey] = parser.getDeviceID();
                        plugin.set_properties(pluginDesc.second.defaultConfig);
                    }
                }
                plugin.set_properties(desc.defaultConfig);
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

bool ov::CoreImpl::isNewAPI() const {
    return newAPI;
}

InferenceEngine::RemoteContext::Ptr ov::CoreImpl::GetDefaultContext(const std::string& deviceName) {
    auto parsed = parseDeviceNameIntoConfig(deviceName, ov::AnyMap{});
    return GetCPPPluginByName(parsed._deviceName).get_default_context(parsed._config)._ptr;
}

InferenceEngine::CNNNetwork ov::CoreImpl::ReadNetwork(const std::string& modelPath, const std::string& binPath) const {
    OV_ITT_SCOPE(FIRST_INFERENCE, ov::itt::domains::IE_RT, "CoreImpl::ReadNetwork from file");
    return InferenceEngine::details::ReadNetwork(modelPath, binPath, extensions, ov_extensions, newAPI);
}

InferenceEngine::CNNNetwork ov::CoreImpl::ReadNetwork(const std::string& model,
                                                      const InferenceEngine::Blob::CPtr& weights,
                                                      bool frontendMode) const {
    OV_ITT_SCOPE(FIRST_INFERENCE, ov::itt::domains::IE_RT, "CoreImpl::ReadNetwork from memory");
    return InferenceEngine::details::ReadNetwork(model, weights, extensions, ov_extensions, newAPI, frontendMode);
}

ov::SoPtr<InferenceEngine::IExecutableNetworkInternal> ov::CoreImpl::LoadNetwork(
    const InferenceEngine::CNNNetwork& network,
    const std::shared_ptr<InferenceEngine::RemoteContext>& context,
    const std::map<std::string, std::string>& config) {
    OV_ITT_SCOPE(FIRST_INFERENCE, InferenceEngine::itt::domains::IE_LT, "Core::LoadNetwork::RemoteContext");
    if (context == nullptr) {
        IE_THROW() << "Remote context is null";
    }
    // have to deduce the device name/config from the context first
    auto parsed = parseDeviceNameIntoConfig(context->getDeviceName(), config);
    std::string& deviceName = parsed._deviceName;
    std::map<std::string, std::string>& config_with_batch = parsed._config;
    // if auto-batching is applicable, the below function will patch the device name and config accordingly:
    ApplyAutoBatching(network, deviceName, config_with_batch);
    CleanUpProperties(deviceName, config_with_batch, ov::auto_batch_timeout);
    parsed = parseDeviceNameIntoConfig(deviceName, config_with_batch);

    auto plugin = GetCPPPluginByName(parsed._deviceName);
    ov::SoPtr<InferenceEngine::IExecutableNetworkInternal> res;
    auto cacheManager =
        coreConfig.getCacheConfigForDevice(parsed._deviceName, DeviceSupportsCacheDir(plugin), parsed._config)
            ._cacheManager;
    auto cacheContent = CacheContent{cacheManager};
    if (cacheManager && DeviceSupportsImportExport(plugin)) {
        cacheContent.blobId = CalculateNetworkHash(network, parsed._deviceName, plugin, parsed._config);
        bool loadedFromCache = false;
        auto lock = cacheGuard.getHashLock(cacheContent.blobId);
        res = LoadNetworkFromCache(cacheContent, plugin, parsed._config, context, loadedFromCache);
        if (!loadedFromCache) {
            res = compile_model_impl(network, plugin, parsed._config, context, cacheContent);
        } else {
            // Temporary workaround until all plugins support caching of original model inputs
            InferenceEngine::SetExeNetworkInfo(res._ptr, network.getFunction(), isNewAPI());
        }
    } else {
        res = compile_model_impl(network, plugin, parsed._config, context, cacheContent);
    }
    return res;
}
InferenceEngine::SoExecutableNetworkInternal ov::CoreImpl::LoadNetwork(
    const InferenceEngine::CNNNetwork& network,
    const std::string& deviceNameOrig,
    const std::map<std::string, std::string>& config) {
    OV_ITT_SCOPE(FIRST_INFERENCE, InferenceEngine::itt::domains::IE_LT, "Core::LoadNetwork::CNN");
    std::string deviceName = deviceNameOrig;
    std::map<std::string, std::string> config_with_batch = config;
    // if auto-batching is applicable, the below function will patch the device name and config accordingly:
    ApplyAutoBatching(network, deviceName, config_with_batch);
    CleanUpProperties(deviceName, config_with_batch, ov::auto_batch_timeout);

    bool forceDisableCache = config_with_batch.count(CONFIG_KEY_INTERNAL(FORCE_DISABLE_CACHE)) > 0;
    auto parsed = parseDeviceNameIntoConfig(deviceName, config_with_batch);
    if (forceDisableCache) {
        // remove this config key from parsed as plugins can throw unsupported exception
        parsed._config.erase(CONFIG_KEY_INTERNAL(FORCE_DISABLE_CACHE));
    }
    auto plugin = GetCPPPluginByName(parsed._deviceName);
    ov::SoPtr<InferenceEngine::IExecutableNetworkInternal> res;
    auto cacheManager =
        coreConfig.getCacheConfigForDevice(parsed._deviceName, DeviceSupportsCacheDir(plugin), parsed._config)
            ._cacheManager;
    auto cacheContent = CacheContent{cacheManager};
    if (!forceDisableCache && cacheManager && DeviceSupportsImportExport(plugin)) {
        cacheContent.blobId = CalculateNetworkHash(network, parsed._deviceName, plugin, parsed._config);
        bool loadedFromCache = false;
        auto lock = cacheGuard.getHashLock(cacheContent.blobId);
        res = LoadNetworkFromCache(cacheContent, plugin, parsed._config, nullptr, loadedFromCache);
        if (!loadedFromCache) {
            res = compile_model_impl(network, plugin, parsed._config, nullptr, cacheContent, forceDisableCache);
        } else {
            // Temporary workaround until all plugins support caching of original model inputs
            InferenceEngine::SetExeNetworkInfo(res._ptr, network.getFunction(), isNewAPI());
        }
    } else {
        res = compile_model_impl(network, plugin, parsed._config, nullptr, cacheContent, forceDisableCache);
    }
    return {res._ptr, res._so};
}

InferenceEngine::SoExecutableNetworkInternal ov::CoreImpl::LoadNetwork(
    const std::string& modelPath,
    const std::string& deviceName,
    const std::map<std::string, std::string>& config,
    const std::function<void(const InferenceEngine::CNNNetwork&)>& val) {
    OV_ITT_SCOPE(FIRST_INFERENCE, InferenceEngine::itt::domains::IE_LT, "Core::LoadNetwork::Path");
    auto parsed = parseDeviceNameIntoConfig(deviceName, config);
    auto plugin = GetCPPPluginByName(parsed._deviceName);
    ov::SoPtr<InferenceEngine::IExecutableNetworkInternal> res;
    auto cacheManager =
        coreConfig.getCacheConfigForDevice(parsed._deviceName, DeviceSupportsCacheDir(plugin), parsed._config)
            ._cacheManager;
    auto cacheContent = CacheContent{cacheManager, modelPath};
    if (cacheManager && DeviceSupportsImportExport(plugin)) {
        bool loadedFromCache = false;
        cacheContent.blobId = CalculateFileHash(modelPath, parsed._deviceName, plugin, parsed._config);
        auto lock = cacheGuard.getHashLock(cacheContent.blobId);
        res = LoadNetworkFromCache(cacheContent, plugin, parsed._config, nullptr, loadedFromCache);
        if (!loadedFromCache) {
            auto cnnNetwork = ReadNetwork(modelPath, std::string());
            if (val) {
                val(cnnNetwork);
            }
            res = compile_model_impl(cnnNetwork, plugin, parsed._config, nullptr, cacheContent);
        }
    } else if (cacheManager) {
        // TODO: 'validation' for dynamic API doesn't work for this case, as it affects a lot of plugin API
        res = plugin.compile_model(modelPath, parsed._config);
    } else {
        auto cnnNetwork = ReadNetwork(modelPath, std::string());
        if (val) {
            val(cnnNetwork);
        }
        res = compile_model_impl(cnnNetwork, plugin, parsed._config, nullptr, cacheContent);
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
    auto parsed = parseDeviceNameIntoConfig(deviceName, config);
    auto plugin = GetCPPPluginByName(parsed._deviceName);
    ov::SoPtr<InferenceEngine::IExecutableNetworkInternal> res;

    auto cacheManager =
        coreConfig.getCacheConfigForDevice(parsed._deviceName, DeviceSupportsCacheDir(plugin), parsed._config)
            ._cacheManager;
    auto cacheContent = CacheContent{cacheManager};
    if (cacheManager && DeviceSupportsImportExport(plugin)) {
        bool loadedFromCache = false;
        ov::Tensor tensor = ov::Tensor();
        if (weights) {
            tensor = ov::Tensor(element::u8, {weights->byteSize()}, weights->cbuffer().as<uint8_t*>());
        }
        cacheContent.blobId = CalculateMemoryHash(modelStr, tensor, parsed._deviceName, plugin, parsed._config);
        auto lock = cacheGuard.getHashLock(cacheContent.blobId);
        res = LoadNetworkFromCache(cacheContent, plugin, parsed._config, nullptr, loadedFromCache);
        if (!loadedFromCache) {
            auto cnnNetwork = ReadNetwork(modelStr, weights);
            if (val) {
                val(cnnNetwork);
            }
            res = compile_model_impl(cnnNetwork, plugin, parsed._config, nullptr, cacheContent);
        }
    } else {
        auto cnnNetwork = ReadNetwork(modelStr, weights);
        if (val) {
            val(cnnNetwork);
        }
        res = compile_model_impl(cnnNetwork, plugin, parsed._config, nullptr, cacheContent);
    }
    return {res._ptr, res._so};
}

InferenceEngine::SoExecutableNetworkInternal ov::CoreImpl::ImportNetwork(
    std::istream& networkModel,
    const std::string& deviceName,
    const std::map<std::string, std::string>& config) {
    auto parsed = parseDeviceNameIntoConfig(deviceName, config);
    auto exec = GetCPPPluginByName(parsed._deviceName).import_model(networkModel, parsed._config);

    return {exec._ptr, exec._so};
}

InferenceEngine::QueryNetworkResult ov::CoreImpl::QueryNetwork(const InferenceEngine::CNNNetwork& network,
                                                               const std::string& deviceName,
                                                               const std::map<std::string, std::string>& config) const {
    OV_ITT_SCOPED_TASK(ov::itt::domains::IE, "Core::QueryNetwork");
    auto parsed = parseDeviceNameIntoConfig(deviceName, config);
    auto res = GetCPPPluginByName(parsed._deviceName).query_model(network, parsed._config);
    if (!network.getFunction() || res.supportedLayersMap.empty())
        return res;

    const auto& func = network.getFunction();
    auto specialized_function = func->clone();

    std::string defDevice = res.supportedLayersMap.begin()->second;
    ngraph::pass::ConstantFolding().run_on_model(specialized_function);
    std::unordered_set<std::string> opNames;

    for (const auto& op : specialized_function->get_ops())
        opNames.emplace(op->get_friendly_name());

    for (const auto& op : func->get_ops()) {
        if (opNames.find(op->get_friendly_name()) == opNames.end()) {
            res.supportedLayersMap[op->get_friendly_name()] = defDevice;
        }
    }

    for (const auto& op : func->get_ops()) {
        if (!res.supportedLayersMap.count(op->get_friendly_name()) &&
            std::dynamic_pointer_cast<ngraph::op::Constant>(op)) {
            bool are_all_users_supported = true;
            for (const auto& user : op->output(0).get_target_inputs()) {
                if (!res.supportedLayersMap.count(user.get_node()->get_friendly_name())) {
                    are_all_users_supported = false;
                    break;
                }
            }
            if (are_all_users_supported) {
                res.supportedLayersMap[op->get_friendly_name()] = defDevice;
            }
        }
    }
    return res;
}

void ov::CoreImpl::ApplyAutoBatching(const InferenceEngine::CNNNetwork& network,
                                     std::string& deviceName,
                                     std::map<std::string, std::string>& config) {
    std::string deviceNameWithBatchSize, deviceNameWithoutBatch;
    // fully strict dims tracking by default (Auto-Batching is enabled implicitly)
    bool strictly_check_dims = true;
    if (deviceName.find("BATCH") != std::string::npos) {
        // explicitly enabled Auto-Batching
        auto pos = deviceName.find_first_of(":");
        if (pos == std::string::npos)
            return;  // BATCH device is already configured via the config
        deviceNameWithBatchSize = deviceName.substr(pos + 1);
        deviceNameWithoutBatch = InferenceEngine::DeviceIDParser::getBatchDevice(deviceNameWithBatchSize);
        // when user sets the BATCH device explicitly, we may check the dims less strictly
        // as the result is being checked by the user
        strictly_check_dims = false;
    } else {
        // check if Auto-Batch plugin registered
        try {
            GetCPPPluginByName("BATCH");
        } catch (const std::runtime_error&) {
            return;
        }
        // check whether the Auto-Batching is disabled explicitly
        const auto& batch_mode = config.find(ov::hint::allow_auto_batching.name());
        if (batch_mode != config.end()) {
            const auto disabled = batch_mode->second == CONFIG_VALUE(NO);
            // virtual plugins like AUTO/MULTI will need the config
            // e.g to deduce the #requests correctly
            // otherwise, no need for this config key in the rest of loading
            if (deviceName.find("AUTO") == std::string::npos && deviceName.find("MULTI") == std::string::npos)
                config.erase(batch_mode);
            if (disabled)
                return;
        } else if (!coreConfig.flag_allow_auto_batching) {
            return;
        }
        // check whether if the Auto-Batching is applicable to the device
        auto device = parseDeviceNameIntoConfig(deviceName);
        deviceNameWithoutBatch = deviceName;
        auto d = device._deviceName;
        std::vector<std::string> metrics = GetCPPPluginByName(d).get_metric(METRIC_KEY(SUPPORTED_METRICS), {});
        auto it = std::find(metrics.begin(), metrics.end(), METRIC_KEY(OPTIMAL_BATCH_SIZE));
        if (metrics.end() == it)
            return;
        // if applicable, the Auto-Batching is implicitly enabled via the performance hints
        bool bTputInPlg = GetConfig(d, CONFIG_KEY(PERFORMANCE_HINT)).as<std::string>() == CONFIG_VALUE(THROUGHPUT);
        const auto& mode = config.find(CONFIG_KEY(PERFORMANCE_HINT));
        bool bTputInLoadCfg = (mode != config.end() && mode->second == CONFIG_VALUE(THROUGHPUT));
        const auto& excl = config.find(CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS));
        bool bExclReqsEnabled = (excl != config.end() && excl->second == CONFIG_VALUE(YES));
        if (bExclReqsEnabled || (!bTputInPlg && !bTputInLoadCfg))
            return;
    }
    auto batchConfig = deviceNameWithBatchSize.empty() ? deviceNameWithoutBatch : deviceNameWithBatchSize;
    auto res = InferenceEngine::details::isNetworkBatchable(network, deviceNameWithoutBatch, strictly_check_dims);
    switch (res) {
    case InferenceEngine::details::NetworkBatchAbility::NO:
        return;
    case InferenceEngine::details::NetworkBatchAbility::AS_IS:
        deviceName = "BATCH:" + batchConfig;
        break;
    case InferenceEngine::details::NetworkBatchAbility::WITH_HETERO:
        deviceName = "HETERO:BATCH," + deviceNameWithoutBatch;
        config[CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG)] = batchConfig;
        break;
    }
}

void ov::CoreImpl::CleanUpProperties(std::string& deviceName,
                                     std::map<std::string, std::string>& config,
                                     ov::Any property) {
    // auto-batching is not applicable, if there is auto_batch_timeout, delete it
    if (deviceName.find("BATCH") == std::string::npos) {
        const auto& batch_timeout_mode = config.find(property.as<std::string>());
        if (batch_timeout_mode != config.end()) {
            if (deviceName.find("AUTO") == std::string::npos && deviceName.find("MULTI") == std::string::npos)
                config.erase(batch_timeout_mode);
        }
    }
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

    return GetCPPPluginByName(parsed._deviceName).get_metric(name, parsed._config);
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

    bool isMetaDevice = device_name.find("AUTO") != std::string::npos ||
                        device_name.find("MULTI") != std::string::npos ||
                        device_name.find("HETERO") != std::string::npos;
    if (!isMetaDevice) {
        // unsupport to set ov::device::properties to HW device through this function
        auto devices = GetListOfDevicesInRegistry();
        for (auto&& config : properties) {
            auto parsed = parseDeviceNameIntoConfig(config.first);
            auto is_secondary_config_for_hw_device =
                std::any_of(devices.begin(), devices.end(), [&](const std::string& device) {
                    return device == parsed._deviceName;
                });
            OPENVINO_ASSERT(!is_secondary_config_for_hw_device,
                            "set_property only supported ov::device::propreties for Meta device (AUTO/MULTI/HETERO). "
                            "You can configure the devices through the compile_model()/loadNetwork() API.");
        }
    }
    SetConfigForPlugins(properties, device_name);
}

ov::Any ov::CoreImpl::get_property_for_core(const std::string& name) const {
    if (name == ov::force_tbb_terminate.name()) {
        const auto flag = InferenceEngine::executorManager()->getTbbFlag();
        return decltype(ov::force_tbb_terminate)::value_type(flag);
    } else if (name == ov::cache_dir.name()) {
        return ov::Any(coreConfig.get_cache_dir());
    } else if (name == ov::hint::allow_auto_batching.name()) {
        const auto flag = coreConfig.flag_allow_auto_batching;
        return decltype(ov::hint::allow_auto_batching)::value_type(flag);
    }

    IE_THROW() << "Exception is thrown while trying to call get_property with unsupported property: '" << name << "'";
}

ov::Any ov::CoreImpl::get_property(const std::string& device_name,
                                   const std::string& name,
                                   const AnyMap& arguments) const {
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

    if (device_name.empty()) {
        return get_property_for_core(name);
    }

    auto parsed = parseDeviceNameIntoConfig(device_name, arguments);
    return GetCPPPluginByName(parsed._deviceName).get_property(name, parsed._config);
}

ov::Any ov::CoreImpl::GetConfig(const std::string& deviceName, const std::string& name) const {
    auto parsed = parseDeviceNameIntoConfig(deviceName);
    return GetCPPPluginByName(parsed._deviceName).get_config(name, parsed._config);
}

std::vector<std::string> ov::CoreImpl::GetAvailableDevices() const {
    std::vector<std::string> devices;
    const std::string propertyName = METRIC_KEY(AVAILABLE_DEVICES);

    for (auto&& deviceName : GetListOfDevicesInRegistry()) {
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
    auto parsed = parseDeviceNameIntoConfig(deviceName, params);
    return GetCPPPluginByName(parsed._deviceName).create_context(parsed._config)._ptr;
}

void ov::CoreImpl::UnloadPluginByName(const std::string& deviceName) {
    std::lock_guard<std::mutex> lock(get_mutex());
    auto it = plugins.find(deviceName);
    if (it == plugins.end()) {
        IE_THROW() << "Device with \"" << deviceName << "\" name is not registered in the OpenVINO Runtime";
    }

    plugins.erase(deviceName);
}

/**
 * @brief Registers plugin meta-data in registry for specified device
 * @param deviceName A name of device
 */
void ov::CoreImpl::RegisterPluginByName(const std::string& pluginName, const std::string& deviceName) {
    std::lock_guard<std::mutex> lock(get_mutex());

    auto it = pluginRegistry.find(deviceName);
    if (it != pluginRegistry.end()) {
        IE_THROW() << "Device with \"" << deviceName << "\"  is already registered in the OpenVINO Runtime";
    }

    if (deviceName.find('.') != std::string::npos) {
        IE_THROW() << "Device name must not contain dot '.' symbol";
    }

    PluginDescriptor desc{getPluginPath(pluginName, true)};
    pluginRegistry[deviceName] = desc;
    add_mutex(deviceName);
}

/**
 * @brief Provides a list of plugin names in registry; physically such plugins may not be created
 * @return A list of plugin names
 */
std::vector<std::string> ov::CoreImpl::GetListOfDevicesInRegistry() const {
    std::lock_guard<std::mutex> lock(get_mutex());

    std::vector<std::string> listOfDevices;
    for (auto&& pluginDesc : pluginRegistry) {
        listOfDevices.push_back(pluginDesc.first);
    }

    return listOfDevices;
}

/**
 * @brief Sets config values for a plugin or set of plugins
 * @param deviceName A device name to set config to
 *        If empty, config is set for all the plugins / plugin's meta-data
 * @note  `deviceName` is not allowed in form of MULTI:CPU, HETERO:GPU,CPU, AUTO:CPU
 *        just simple forms like CPU, GPU, MULTI, GPU.0, etc
 */
void ov::CoreImpl::SetConfigForPlugins(const ov::AnyMap& configMap, const std::string& deviceName) {
    auto config = configMap;
    if (config.empty()) {
        return;
    }

    InferenceEngine::DeviceIDParser parser(deviceName);
    std::string clearDeviceName = parser.getDeviceName();

    std::vector<std::pair<std::string, ov::InferencePlugin>> created_plugins;
    {
        std::lock_guard<std::mutex> lock(get_mutex());
        created_plugins.reserve(plugins.size());

        if (deviceName.empty()) {
            coreConfig.setAndUpdate(config);
        } else {
            auto cache_it = config.find(CONFIG_KEY(CACHE_DIR));
            if (cache_it != config.end()) {
                coreConfig.setCacheForDevice(cache_it->second, clearDeviceName);
            }
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
                created_plugins.emplace_back(std::pair<std::string, ov::InferencePlugin>{plugin.first, plugin.second});
            }
        }
    }
    for (auto& plugin : created_plugins) {
        allowNotImplemented([&]() {
            std::lock_guard<std::mutex> lock(get_mutex(plugin.first));
            auto configCopy = config;
            if (DeviceSupportsCacheDir(plugin.second)) {
                auto cacheConfig = coreConfig.getCacheConfigForDevice(deviceName);
                if (cacheConfig._cacheManager) {
                    configCopy[CONFIG_KEY(CACHE_DIR)] = cacheConfig._cacheDir;
                }
            } else if (configCopy.count(CONFIG_KEY(CACHE_DIR)) > 0) {
                // Remove "CACHE_DIR" from config if it is not supported by plugin
                configCopy.erase(CONFIG_KEY(CACHE_DIR));
            }
            // Add device specific value to support device_name.device_id cases
            std::vector<std::string> supportedConfigKeys =
                plugin.second.get_metric(METRIC_KEY(SUPPORTED_CONFIG_KEYS), {});
            auto config_iter = std::find(supportedConfigKeys.begin(),
                                         supportedConfigKeys.end(),
                                         CONFIG_KEY_INTERNAL(CONFIG_DEVICE_ID));
            const bool supportsConfigDeviceID = config_iter != supportedConfigKeys.end();
            const std::string deviceKey =
                supportsConfigDeviceID ? CONFIG_KEY_INTERNAL(CONFIG_DEVICE_ID) : CONFIG_KEY(DEVICE_ID);

            if (!parser.getDeviceID().empty()) {
                configCopy[deviceKey] = parser.getDeviceID();
            }
            plugin.second.set_properties(configCopy);
        });
    }
}

/**
 * @brief Get device config it is passed as pair of device_name and `AnyMap`
 * @param configs All set of configs
 * @note  `device_name` is not allowed in form of MULTI:CPU, HETERO:GPU,CPU, AUTO:CPU
 *        just simple forms like CPU, GPU, MULTI, GPU.0, etc
 */
void ov::CoreImpl::ExtractAndSetDeviceConfig(const ov::AnyMap& configs) {
    for (auto&& config : configs) {
        auto parsed = parseDeviceNameIntoConfig(config.first);
        auto devices = GetListOfDevicesInRegistry();
        auto config_is_device_name_in_regestry =
            std::any_of(devices.begin(), devices.end(), [&](const std::string& device) {
                return device == parsed._deviceName;
            });
        if (config_is_device_name_in_regestry) {
            SetConfigForPlugins(config.second.as<ov::AnyMap>(), config.first);
        }
    }
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
                if (util::contains(supportedConfigKeys, device_config.first)) {
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

/**
 * @brief Registers the extension in a Core object
 *        Such extensions can be used for both CNNNetwork readers and device plugins
 */
void ov::CoreImpl::AddExtension(const InferenceEngine::IExtensionPtr& extension) {
    std::lock_guard<std::mutex> lock(get_mutex());
    AddExtensionUnsafe(extension);
}

void ov::CoreImpl::AddOVExtensions(const std::vector<ov::Extension::Ptr>& extensions) {
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

const std::vector<ov::Extension::Ptr>& ov::CoreImpl::GetOVExtensions() const {
    return ov_extensions;
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
        InferenceEngine::DeviceIDParser parser(deviceName_);
        std::string deviceNameLocal = parser.getDeviceName();

        ov::InferencePlugin cppPlugin = GetCPPPluginByName(deviceNameLocal);
        const InferenceEngine::Version version = cppPlugin.get_version();
        versions[deviceNameLocal] = version;
    }

    return versions;
}

bool ov::CoreImpl::DeviceSupportsImportExport(const std::string& deviceName) const {
    auto parsed = parseDeviceNameIntoConfig(deviceName);
    auto plugin = GetCPPPluginByName(parsed._deviceName);
    return DeviceSupportsImportExport(plugin);
}

bool ov::CoreImpl::DeviceSupportsConfigKey(const ov::InferencePlugin& plugin, const std::string& key) const {
    return util::contains(plugin.get_property(ov::supported_properties), key);
}

bool ov::CoreImpl::DeviceSupportsImportExport(const ov::InferencePlugin& plugin) const {
    auto supportedMetricKeys = plugin.get_metric(METRIC_KEY(SUPPORTED_METRICS), {}).as<std::vector<std::string>>();
    auto it = std::find(supportedMetricKeys.begin(), supportedMetricKeys.end(), METRIC_KEY(IMPORT_EXPORT_SUPPORT));
    auto supported =
        (it != supportedMetricKeys.end()) && plugin.get_metric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), {}).as<bool>();
    if (!supported) {
        if (DeviceSupportsConfigKey(plugin, ov::device::capabilities.name())) {
            supported =
                util::contains(plugin.get_property(ov::device::capabilities), ov::device::capability::EXPORT_IMPORT);
        }
    }
    return supported;
}

bool ov::CoreImpl::DeviceSupportsCacheDir(const ov::InferencePlugin& plugin) const {
    return util::contains(plugin.get_property(ov::supported_properties), ov::cache_dir);
}

ov::SoPtr<InferenceEngine::IExecutableNetworkInternal> ov::CoreImpl::compile_model_impl(
    const InferenceEngine::CNNNetwork& network,
    ov::InferencePlugin& plugin,
    const std::map<std::string, std::string>& parsedConfig,
    const InferenceEngine::RemoteContext::Ptr& context,
    const CacheContent& cacheContent,
    bool forceDisableCache) {
    OV_ITT_SCOPED_TASK(ov::itt::domains::IE, "CoreImpl::compile_model_impl");
    ov::SoPtr<InferenceEngine::IExecutableNetworkInternal> execNetwork;
    execNetwork =
        context ? plugin.compile_model(network, context, parsedConfig) : plugin.compile_model(network, parsedConfig);
    if (!forceDisableCache && cacheContent.cacheManager && DeviceSupportsImportExport(plugin)) {
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

ov::SoPtr<InferenceEngine::IExecutableNetworkInternal> ov::CoreImpl::LoadNetworkFromCache(
    const CacheContent& cacheContent,
    ov::InferencePlugin& plugin,
    const std::map<std::string, std::string>& config,
    const std::shared_ptr<InferenceEngine::RemoteContext>& context,
    bool& networkIsImported) {
    ov::SoPtr<InferenceEngine::IExecutableNetworkInternal> execNetwork;
    struct HeaderException {};

    OPENVINO_ASSERT(cacheContent.cacheManager != nullptr);
    try {
        cacheContent.cacheManager->readCacheEntry(cacheContent.blobId, [&](std::istream& networkStream) {
            OV_ITT_SCOPE(FIRST_INFERENCE,
                         InferenceEngine::itt::domains::IE_LT,
                         "Core::LoadNetworkFromCache::ReadStreamAndImport");
            try {
                InferenceEngine::CompiledBlobHeader header;
                networkStream >> header;
                if (header.getIeVersion() != InferenceEngine::GetInferenceEngineVersion()->buildNumber) {
                    // Build number mismatch, don't use this cache
                    throw InferenceEngine::NetworkNotRead("Version does not match");
                }
                if (header.getFileInfo() !=
                    InferenceEngine::NetworkCompilationContext::calculateFileInfo(cacheContent.modelPath)) {
                    // Original file is changed, don't use cache
                    throw InferenceEngine::NetworkNotRead("Original model file is changed");
                }
            } catch (...) {
                throw HeaderException();
            }

            execNetwork = context ? plugin.import_model(networkStream, context, config)
                                  : plugin.import_model(networkStream, config);
            networkIsImported = true;
            execNetwork->loadedFromCache();
        });
    } catch (const HeaderException&) {
        // For these exceptions just remove old cache and set that import didn't work
        cacheContent.cacheManager->removeCacheEntry(cacheContent.blobId);
        networkIsImported = false;
    } catch (...) {
        cacheContent.cacheManager->removeCacheEntry(cacheContent.blobId);
        networkIsImported = false;
        // TODO: temporary disabled by #54335. In future don't throw only for new 'blob_outdated' exception
        // throw;
    }
    return execNetwork;
}

std::map<std::string, std::string> ov::CoreImpl::CreateCompileConfig(
    const ov::InferencePlugin& plugin,
    const std::string& deviceFamily,
    const std::map<std::string, std::string>& origConfig) const {
    std::map<std::string, Any> getMetricConfig;
    std::map<std::string, std::string> compileConfig;

    // 0. Move TARGET_FALLBACK key to getMetricConfig
    auto targetFallbackIt = origConfig.find("TARGET_FALLBACK");
    if (targetFallbackIt == origConfig.end()) {
        targetFallbackIt = origConfig.find(ov::device::priorities.name());
    }
    if (targetFallbackIt != origConfig.end()) {
        getMetricConfig[targetFallbackIt->first] = targetFallbackIt->second;
    }

    // 1. Move DEVICE_ID key to getMetricConfig
    auto deviceIt = origConfig.find(ov::device::id.name());
    if (deviceIt != origConfig.end()) {
        getMetricConfig[deviceIt->first] = deviceIt->second;
    }

    // 2. Replace it with DEVICE_ARCHITECTURE value
    if (DeviceSupportsConfigKey(plugin, ov::device::architecture.name())) {
        compileConfig[ov::device::architecture.name()] = plugin.get_property(ov::device::architecture, getMetricConfig);
    } else {
        // Take device name if device does not support DEVICE_ARCHITECTURE metric
        compileConfig[ov::device::architecture.name()] = deviceFamily;
    }

    // 3. Extract config keys which affect compile config
    if (DeviceSupportsConfigKey(plugin, ov::caching_properties.name())) {
        auto cachingProps = plugin.get_property(ov::caching_properties);
        for (const auto& prop : cachingProps) {
            // origConfig values have higher priority than plugin parameters
            auto it = origConfig.find(prop);
            compileConfig[prop] = it == origConfig.end() ? plugin.get_property(prop, {}).as<std::string>() : it->second;
        }
    }
    return compileConfig;
}

std::string ov::CoreImpl::CalculateNetworkHash(const InferenceEngine::CNNNetwork& network,
                                               const std::string& deviceFamily,
                                               const ov::InferencePlugin& plugin,
                                               const std::map<std::string, std::string>& config) const {
    auto compileConfig = CreateCompileConfig(plugin, deviceFamily, config);
    return InferenceEngine::NetworkCompilationContext::computeHash(network, compileConfig);
}

std::string ov::CoreImpl::CalculateFileHash(const std::string& modelName,
                                            const std::string& deviceFamily,
                                            const ov::InferencePlugin& plugin,
                                            const std::map<std::string, std::string>& config) const {
    auto compileConfig = CreateCompileConfig(plugin, deviceFamily, config);
    return InferenceEngine::NetworkCompilationContext::computeHash(modelName, compileConfig);
}

std::string ov::CoreImpl::CalculateMemoryHash(const std::string& modelStr,
                                              const ov::Tensor& weights,
                                              const std::string& deviceFamily,
                                              const ov::InferencePlugin& plugin,
                                              const std::map<std::string, std::string>& config) const {
    auto compileConfig = CreateCompileConfig(plugin, deviceFamily, config);
    return InferenceEngine::NetworkCompilationContext::computeHash(modelStr, weights, compileConfig);
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
        try {
            plugin.second.add_extension(extension);
        } catch (...) {
        }
    }
    extensions.emplace_back(extension);
}

void ov::CoreImpl::CoreConfig::setAndUpdate(ov::AnyMap& config) {
    auto it = config.find(CONFIG_KEY(CACHE_DIR));
    if (it != config.end()) {
        std::lock_guard<std::mutex> lock(_cacheConfigMutex);
        fillConfig(_cacheConfig, it->second.as<std::string>());
        for (auto& deviceCfg : _cacheConfigPerDevice) {
            fillConfig(deviceCfg.second, it->second.as<std::string>());
        }
        config.erase(it);
    }

    it = config.find(ov::force_tbb_terminate.name());
    if (it != config.end()) {
        auto flag = it->second.as<std::string>() == CONFIG_VALUE(YES) ? true : false;
        InferenceEngine::executorManager()->setTbbFlag(flag);
        config.erase(it);
    }

    it = config.find(ov::hint::allow_auto_batching.name());
    if (it != config.end()) {
        auto flag = it->second.as<bool>();
        flag_allow_auto_batching = flag;
        config.erase(it);
    }
}

void ov::CoreImpl::CoreConfig::setCacheForDevice(const std::string& dir, const std::string& name) {
    std::lock_guard<std::mutex> lock(_cacheConfigMutex);
    fillConfig(_cacheConfigPerDevice[name], dir);
}

std::string ov::CoreImpl::CoreConfig::get_cache_dir() const {
    std::lock_guard<std::mutex> lock(_cacheConfigMutex);
    return _cacheConfig._cacheDir;
}

// Creating thread-safe copy of config including shared_ptr to ICacheManager
// Passing empty or not-existing name will return global cache config
ov::CoreImpl::CoreConfig::CacheConfig ov::CoreImpl::CoreConfig::getCacheConfigForDevice(
    const std::string& device_name,
    bool deviceSupportsCacheDir,
    std::map<std::string, std::string>& parsedConfig) const {
    if (parsedConfig.count(CONFIG_KEY(CACHE_DIR))) {
        CoreConfig::CacheConfig tempConfig;
        CoreConfig::fillConfig(tempConfig, parsedConfig.at(CONFIG_KEY(CACHE_DIR)));
        if (!deviceSupportsCacheDir) {
            parsedConfig.erase(CONFIG_KEY(CACHE_DIR));
        }
        return tempConfig;
    } else {
        std::lock_guard<std::mutex> lock(_cacheConfigMutex);
        if (_cacheConfigPerDevice.count(device_name) > 0) {
            return _cacheConfigPerDevice.at(device_name);
        } else {
            return _cacheConfig;
        }
    }
}

ov::CoreImpl::CoreConfig::CacheConfig ov::CoreImpl::CoreConfig::getCacheConfigForDevice(
    const std::string& device_name) const {
    std::lock_guard<std::mutex> lock(_cacheConfigMutex);
    if (_cacheConfigPerDevice.count(device_name) > 0) {
        return _cacheConfigPerDevice.at(device_name);
    } else {
        return _cacheConfig;
    }
}

void ov::CoreImpl::CoreConfig::fillConfig(CacheConfig& config, const std::string& dir) {
    config._cacheDir = dir;
    if (!dir.empty()) {
        FileUtils::createDirectoryRecursive(dir);
        config._cacheManager = std::make_shared<InferenceEngine::FileStorageCacheManager>(dir);
    } else {
        config._cacheManager = nullptr;
    }
}
std::mutex& ov::CoreImpl::get_mutex(const std::string& dev_name) const {
    std::lock_guard<std::mutex> lock(global_mutex);
    try {
        return dev_mutexes.at(dev_name);
    } catch (const std::out_of_range&) {
        throw ov::Exception("Cannot get mutex for device: " + dev_name);
    }
}
void ov::CoreImpl::add_mutex(const std::string& dev_name) {
    std::lock_guard<std::mutex> lock(global_mutex);
    dev_mutexes[dev_name];
}

std::tuple<bool, std::string> ov::CoreImpl::CheckStatic(const InferenceEngine::CNNNetwork& network) {
    bool res = true;
    std::stringstream errMsg;
    auto model = network.getFunction();
    if (model) {
        for (const auto& input : model->inputs()) {
            if (input.get_partial_shape().is_dynamic()) {
                errMsg << "{ input:'";
                for (const auto& name : input.get_names()) {
                    errMsg << name << ",";
                }
                if (auto node = input.get_node_shared_ptr()) {
                    errMsg << node->get_friendly_name();
                }
                errMsg << "', shape=" << input.get_partial_shape() << "} ";
                res = false;
            }
        }
    }
    return {res, errMsg.str()};
}

#ifndef OPENVINO_STATIC_LIBRARY

std::string ov::findPluginXML(const std::string& xmlFile) {
    std::string xmlConfigFile_ = xmlFile;
    if (xmlConfigFile_.empty()) {
        const auto ielibraryDir = ie::getInferenceEngineLibraryPath();

        // plugins.xml can be found in either:

        // 1. openvino-X.Y.Z relative to libopenvino.so folder
        std::ostringstream str;
        str << "openvino-" << OPENVINO_VERSION_MAJOR << "." << OPENVINO_VERSION_MINOR << "." << OPENVINO_VERSION_PATCH;
        const auto subFolder = ov::util::to_file_path(str.str());

        // register plugins from default openvino-<openvino version>/plugins.xml config
        ov::util::FilePath xmlConfigFileDefault =
            FileUtils::makePath(FileUtils::makePath(ielibraryDir, subFolder), ov::util::to_file_path("plugins.xml"));
        if (FileUtils::fileExist(xmlConfigFileDefault))
            return xmlConfigFile_ = ov::util::from_file_path(xmlConfigFileDefault);

        // 2. in folder with libopenvino.so
        xmlConfigFileDefault = FileUtils::makePath(ielibraryDir, ov::util::to_file_path("plugins.xml"));
        if (FileUtils::fileExist(xmlConfigFileDefault))
            return xmlConfigFile_ = ov::util::from_file_path(xmlConfigFileDefault);

        throw ov::Exception("Failed to find plugins.xml file");
    }
    return xmlConfigFile_;
}

#endif

ov::AnyMap ov::flatten_sub_properties(const std::string& device, const ov::AnyMap& properties) {
    ov::AnyMap result = properties;
    bool isVirtualDev = device.find("AUTO") != std::string::npos || device.find("MULTI") != std::string::npos ||
                        device.find("HETERO") != std::string::npos;
    for (auto item = result.begin(); item != result.end();) {
        auto parsed = parseDeviceNameIntoConfig(item->first);
        if (!item->second.is<ov::AnyMap>()) {
            item++;
            continue;
        }
        if (device == parsed._deviceName) {
            // 1. flatten the scondary property for target device
            for (auto&& sub_property : item->second.as<ov::AnyMap>()) {
                // 1.1 1st level property overides 2nd level property
                if (result.find(sub_property.first) != result.end())
                    continue;
                result[sub_property.first] = sub_property.second;
            }
            item = result.erase(item);
        } else if (isVirtualDev) {
            // 2. keep the secondary property for the other virtual devices
            item++;
        } else {
            // 3. remove the secondary property setting for other hardware device
            item = result.erase(item);
        }
    }
    return result;
}
