// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_core.hpp"

#include <sys/stat.h>

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <threading/ie_executor_manager.hpp>
#include <vector>

#include "any_copy.hpp"
#include "check_network_batchable.hpp"
#include "cnn_network_ngraph_impl.hpp"
#include "compilation_context.hpp"
#include "cpp/ie_cnn_network.h"
#include "cpp/ie_plugin.hpp"
#include "cpp_interfaces/interface/ie_iexecutable_network_internal.hpp"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"
#include "file_utils.h"
#include "ie_cache_guard.hpp"
#include "ie_cache_manager.hpp"
#include "ie_icore.hpp"
#include "ie_itt.hpp"
#include "ie_network_reader.hpp"
#include "ie_ngraph_utils.hpp"
#include "ie_plugin_config.hpp"
#include "ie_remote_context.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset.hpp"
#include "ngraph/pass/constant_folding.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"
#include "so_extension.hpp"
#include "xml_parse_utils.h"

#ifdef OPENVINO_STATIC_LIBRARY
#    include "ie_plugins.hpp"
#endif

using namespace InferenceEngine::PluginConfigParams;
using namespace InferenceEngine;
using namespace std::placeholders;

namespace ov {

// Specify the default device when no device name is provided.
const std::string DEFAULT_DEVICE_NAME = "DEFAULT_DEVICE";

template <typename T>
struct Parsed {
    std::string _deviceName;
    std::map<std::string, T> _config;
};

namespace {

#ifndef OPENVINO_STATIC_LIBRARY

std::string findPluginXML(const std::string& xmlFile) {
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

std::string resolve_extension_path(const std::string& path) {
    std::string retvalue;
    try {
        const std::string absolute_path = ov::util::get_absolute_file_path(path);
        retvalue = FileUtils::fileExist(absolute_path) ? absolute_path : path;
    } catch (const std::runtime_error&) {
        retvalue = path;
    }
    return retvalue;
}

template <typename T = ie::Parameter>
Parsed<T> parseDeviceNameIntoConfig(const std::string& deviceName, const std::map<std::string, T>& config = {}) {
    auto config_ = config;
    auto deviceName_ = deviceName;
    if (deviceName_.find("HETERO:") == 0) {
        deviceName_ = "HETERO";
        config_["TARGET_FALLBACK"] = deviceName.substr(7);
    } else if (deviceName_.find("MULTI:") == 0) {
        deviceName_ = "MULTI";
        config_[ie::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES] = deviceName.substr(6);
    } else if (deviceName == "AUTO" || deviceName.find("AUTO:") == 0) {
        deviceName_ = "AUTO";
        if (deviceName.find("AUTO:") == 0) {
            config_[ie::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES] =
                deviceName.substr(std::string("AUTO:").size());
        }
    } else if (deviceName_.find("BATCH:") == 0) {
        deviceName_ = "BATCH";
        config_[CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG)] = deviceName.substr(6);
    } else {
        ie::DeviceIDParser parser(deviceName_);
        deviceName_ = parser.getDeviceName();
        std::string deviceIDLocal = parser.getDeviceID();

        if (!deviceIDLocal.empty()) {
            config_[KEY_DEVICE_ID] = deviceIDLocal;
        }
    }
    return {deviceName_, config_};
}

template <typename F>
void allowNotImplemented(F&& f) {
    try {
        f();
    } catch (const ie::NotImplemented&) {
    }
}

ov::AnyMap flatten_sub_properties(const std::string& device, const ov::AnyMap& properties) {
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

void stripDeviceName(std::string& device, const std::string& substr) {
    auto pos = device.find(substr);
    if (pos == 0) {
        device.erase(pos, substr.length());
    }
}
}  // namespace

class CoreImpl : public ie::ICore, public std::enable_shared_from_this<ie::ICore> {
    mutable std::map<std::string, ov::InferencePlugin> plugins;
    // Mutex is needed to prevent changes of dev mutexes map from different threads
    mutable std::mutex global_mutex;
    // Global mutex "" locks parallel access to pluginRegistry and plugins
    // Plugin mutexes "plugin_name" lock access to code which changes configuration of particular plugin
    mutable std::unordered_map<std::string, std::mutex> dev_mutexes;
    std::mutex& get_mutex(const std::string& dev_name = "") const {
        std::lock_guard<std::mutex> lock(global_mutex);
        try {
            return dev_mutexes.at(dev_name);
        } catch (const std::out_of_range&) {
            throw ov::Exception("Cannot get mutex for device: " + dev_name);
        }
    }
    void add_mutex(const std::string& dev_name) {
        std::lock_guard<std::mutex> lock(global_mutex);
        dev_mutexes[dev_name];
    }

    class CoreConfig final {
    public:
        struct CacheConfig {
            std::string _cacheDir;
            std::shared_ptr<ie::ICacheManager> _cacheManager;
        };

        bool flag_allow_auto_batching = true;

        void setAndUpdate(ov::AnyMap& config) {
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
                executorManager()->setTbbFlag(flag);
                config.erase(it);
            }

            it = config.find(ov::hint::allow_auto_batching.name());
            if (it != config.end()) {
                auto flag = it->second.as<bool>();
                flag_allow_auto_batching = flag;
                config.erase(it);
            }
        }

        void setCacheForDevice(const std::string& dir, const std::string& name) {
            std::lock_guard<std::mutex> lock(_cacheConfigMutex);
            fillConfig(_cacheConfigPerDevice[name], dir);
        }

        std::string get_cache_dir() const {
            std::lock_guard<std::mutex> lock(_cacheConfigMutex);
            return _cacheConfig._cacheDir;
        }

        // Creating thread-safe copy of config including shared_ptr to ICacheManager
        // Passing empty or not-existing name will return global cache config
        CacheConfig getCacheConfigForDevice(const std::string& device_name,
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

        CacheConfig getCacheConfigForDevice(const std::string& device_name) const {
            std::lock_guard<std::mutex> lock(_cacheConfigMutex);
            if (_cacheConfigPerDevice.count(device_name) > 0) {
                return _cacheConfigPerDevice.at(device_name);
            } else {
                return _cacheConfig;
            }
        }

    private:
        static void fillConfig(CacheConfig& config, const std::string& dir) {
            config._cacheDir = dir;
            if (!dir.empty()) {
                FileUtils::createDirectoryRecursive(dir);
                config._cacheManager = std::make_shared<ie::FileStorageCacheManager>(dir);
            } else {
                config._cacheManager = nullptr;
            }
        }

    private:
        mutable std::mutex _cacheConfigMutex;
        CacheConfig _cacheConfig;
        std::map<std::string, CacheConfig> _cacheConfigPerDevice;
    };

    struct CacheContent {
        explicit CacheContent(const std::shared_ptr<ie::ICacheManager>& cache_manager,
                              const std::string model_path = {})
            : cacheManager(cache_manager),
              modelPath(model_path) {}
        std::shared_ptr<ie::ICacheManager> cacheManager;
        std::string blobId = {};
        std::string modelPath = {};
    };

    // Core settings (cache config, etc)
    CoreConfig coreConfig;

    ie::CacheGuard cacheGuard;

    struct PluginDescriptor {
        ov::util::FilePath libraryLocation;
        ov::AnyMap defaultConfig;
        std::vector<ov::util::FilePath> listOfExtentions;
        InferenceEngine::CreatePluginEngineFunc* pluginCreateFunc = nullptr;
        InferenceEngine::CreateExtensionFunc* extensionCreateFunc = nullptr;

        PluginDescriptor() = default;

        PluginDescriptor(const ov::util::FilePath& libraryLocation,
                         const ov::AnyMap& defaultConfig = {},
                         const std::vector<ov::util::FilePath>& listOfExtentions = {}) {
            this->libraryLocation = libraryLocation;
            this->defaultConfig = defaultConfig;
            this->listOfExtentions = listOfExtentions;
        }

        PluginDescriptor(InferenceEngine::CreatePluginEngineFunc* pluginCreateFunc,
                         const ov::AnyMap& defaultConfig = {},
                         InferenceEngine::CreateExtensionFunc* extensionCreateFunc = nullptr) {
            this->pluginCreateFunc = pluginCreateFunc;
            this->defaultConfig = defaultConfig;
            this->extensionCreateFunc = extensionCreateFunc;
        }
    };

    ExecutorManager::Ptr executorManagerPtr;
    mutable std::unordered_set<std::string> opsetNames;
    // TODO: make extensions to be optional with conditional compilation
    mutable std::vector<ie::IExtensionPtr> extensions;
    std::vector<ov::Extension::Ptr> ov_extensions;

    std::map<std::string, PluginDescriptor> pluginRegistry;

    const bool newAPI;

    bool DeviceSupportsImportExport(const std::string& deviceName) const override {
        auto parsed = parseDeviceNameIntoConfig(deviceName);
        auto plugin = GetCPPPluginByName(parsed._deviceName);
        return DeviceSupportsImportExport(plugin);
    }

    bool DeviceSupportsConfigKey(const ov::InferencePlugin& plugin, const std::string& key) const {
        return util::contains(plugin.get_property(ov::supported_properties), key);
    }

    bool DeviceSupportsImportExport(const ov::InferencePlugin& plugin) const {
        auto supportedMetricKeys = plugin.get_metric(METRIC_KEY(SUPPORTED_METRICS), {}).as<std::vector<std::string>>();
        auto it = std::find(supportedMetricKeys.begin(), supportedMetricKeys.end(), METRIC_KEY(IMPORT_EXPORT_SUPPORT));
        auto supported =
            (it != supportedMetricKeys.end()) && plugin.get_metric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), {}).as<bool>();
        if (!supported) {
            if (DeviceSupportsConfigKey(plugin, ov::device::capabilities.name())) {
                supported = util::contains(plugin.get_property(ov::device::capabilities),
                                           ov::device::capability::EXPORT_IMPORT);
            }
        }
        return supported;
    }

    bool DeviceSupportsCacheDir(const ov::InferencePlugin& plugin) const {
        return util::contains(plugin.get_property(ov::supported_properties), ov::cache_dir);
    }

    ov::SoPtr<ie::IExecutableNetworkInternal> compile_model_impl(const InferenceEngine::CNNNetwork& network,
                                                                 ov::InferencePlugin& plugin,
                                                                 const std::map<std::string, std::string>& parsedConfig,
                                                                 const ie::RemoteContext::Ptr& context,
                                                                 const CacheContent& cacheContent,
                                                                 bool forceDisableCache = false) {
        OV_ITT_SCOPED_TASK(ov::itt::domains::IE, "CoreImpl::compile_model_impl");
        ov::SoPtr<ie::IExecutableNetworkInternal> execNetwork;
        execNetwork = context ? plugin.compile_model(network, context, parsedConfig)
                              : plugin.compile_model(network, parsedConfig);
        if (!forceDisableCache && cacheContent.cacheManager && DeviceSupportsImportExport(plugin)) {
            try {
                // need to export network for further import from "cache"
                OV_ITT_SCOPE(FIRST_INFERENCE, ie::itt::domains::IE_LT, "Core::LoadNetwork::Export");
                cacheContent.cacheManager->writeCacheEntry(cacheContent.blobId, [&](std::ostream& networkStream) {
                    networkStream << ie::CompiledBlobHeader(
                        ie::GetInferenceEngineVersion()->buildNumber,
                        ie::NetworkCompilationContext::calculateFileInfo(cacheContent.modelPath));
                    execNetwork->Export(networkStream);
                });
            } catch (...) {
                cacheContent.cacheManager->removeCacheEntry(cacheContent.blobId);
                throw;
            }
        }
        return execNetwork;
    }

    static ov::SoPtr<ie::IExecutableNetworkInternal> LoadNetworkFromCache(
        const CacheContent& cacheContent,
        ov::InferencePlugin& plugin,
        const std::map<std::string, std::string>& config,
        const std::shared_ptr<ie::RemoteContext>& context,
        bool& networkIsImported) {
        ov::SoPtr<ie::IExecutableNetworkInternal> execNetwork;
        struct HeaderException {};

        OPENVINO_ASSERT(cacheContent.cacheManager != nullptr);
        try {
            cacheContent.cacheManager->readCacheEntry(cacheContent.blobId, [&](std::istream& networkStream) {
                OV_ITT_SCOPE(FIRST_INFERENCE,
                             ie::itt::domains::IE_LT,
                             "Core::LoadNetworkFromCache::ReadStreamAndImport");
                try {
                    ie::CompiledBlobHeader header;
                    networkStream >> header;
                    if (header.getIeVersion() != ie::GetInferenceEngineVersion()->buildNumber) {
                        // Build number mismatch, don't use this cache
                        throw ie::NetworkNotRead("Version does not match");
                    }
                    if (header.getFileInfo() !=
                        ie::NetworkCompilationContext::calculateFileInfo(cacheContent.modelPath)) {
                        // Original file is changed, don't use cache
                        throw ie::NetworkNotRead("Original model file is changed");
                    }
                } catch (...) {
                    throw HeaderException();
                }

                execNetwork = context ? plugin.import_model(networkStream, context, config)
                                      : plugin.import_model(networkStream, config);
                networkIsImported = true;
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

    std::map<std::string, std::string> CreateCompileConfig(const ov::InferencePlugin& plugin,
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
            compileConfig[ov::device::architecture.name()] =
                plugin.get_property(ov::device::architecture, getMetricConfig);
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
                compileConfig[prop] =
                    it == origConfig.end() ? plugin.get_property(prop, {}).as<std::string>() : it->second;
            }
        }
        return compileConfig;
    }

    std::string CalculateNetworkHash(const ie::CNNNetwork& network,
                                     const std::string& deviceFamily,
                                     const ov::InferencePlugin& plugin,
                                     const std::map<std::string, std::string>& config) const {
        auto compileConfig = CreateCompileConfig(plugin, deviceFamily, config);
        return ie::NetworkCompilationContext::computeHash(network, compileConfig);
    }

    std::string CalculateFileHash(const std::string& modelName,
                                  const std::string& deviceFamily,
                                  const ov::InferencePlugin& plugin,
                                  const std::map<std::string, std::string>& config) const {
        auto compileConfig = CreateCompileConfig(plugin, deviceFamily, config);
        return ie::NetworkCompilationContext::computeHash(modelName, compileConfig);
    }

public:
    CoreImpl(bool _newAPI) : newAPI(_newAPI) {
        add_mutex("");  // Register global mutex
        executorManagerPtr = executorManager();
        for (const auto& it : ov::get_available_opsets()) {
            opsetNames.insert(it.first);
        }
    }

    ~CoreImpl() override = default;

    /**
     * @brief Register plugins for devices which are located in .xml configuration file.
     * @note The function supports UNICODE path
     * @param xmlConfigFile An .xml configuraion with device / plugin information
     * @param ByAbsPath A boolean value - register plugins by absolute file path or not
     */
    void RegisterPluginsInRegistry(const std::string& xmlConfigFile, const bool& ByAbsPath = false) {
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
            ov::util::FilePath pluginPath =
                ov::util::get_plugin_path(GetStrAttr(pluginNode, "location"), xmlConfigFile, ByAbsPath);

            if (deviceName.find('.') != std::string::npos) {
                IE_THROW() << "Device name must not contain dot '.' symbol";
            }

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

#ifdef OPENVINO_STATIC_LIBRARY

    /**
     * @brief Register plugins for devices using statically defined configuration
     * @note The function supports UNICODE path
     * @param static_registry a statically defined configuration with device / plugin information
     */
    void RegisterPluginsInRegistry(const decltype(::getStaticPluginsRegistry())& static_registry) {
        std::lock_guard<std::mutex> lock(get_mutex());

        for (const auto& plugin : static_registry) {
            const auto& deviceName = plugin.first;
            if (deviceName.find('.') != std::string::npos) {
                IE_THROW() << "Device name must not contain dot '.' symbol";
            }
            const auto& value = plugin.second;
            ov::AnyMap config = any_copy(value.m_default_config);
            PluginDescriptor desc{value.m_create_plugin_func, config, value.m_create_extension_func};
            pluginRegistry[deviceName] = desc;
            add_mutex(deviceName);
        }
    }

#endif

    //
    // ICore public API
    //

    ie::CNNNetwork ReadNetwork(const std::string& modelPath, const std::string& binPath) const override {
        OV_ITT_SCOPE(FIRST_INFERENCE, ov::itt::domains::IE_RT, "CoreImpl::ReadNetwork from file");
        return InferenceEngine::details::ReadNetwork(modelPath, binPath, extensions, ov_extensions, newAPI);
    }

    ie::CNNNetwork ReadNetwork(const std::string& model,
                               const ie::Blob::CPtr& weights,
                               bool frontendMode = false) const override {
        OV_ITT_SCOPE(FIRST_INFERENCE, ov::itt::domains::IE_RT, "CoreImpl::ReadNetwork from memory");
        return InferenceEngine::details::ReadNetwork(model, weights, extensions, ov_extensions, newAPI, frontendMode);
    }

    bool isNewAPI() const override {
        return newAPI;
    }

    static std::tuple<bool, std::string> CheckStatic(const ie::CNNNetwork& network) {
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

    ie::RemoteContext::Ptr GetDefaultContext(const std::string& deviceName) override {
        auto parsed = ov::parseDeviceNameIntoConfig(deviceName, ParamMap{});
        return GetCPPPluginByName(parsed._deviceName).get_default_context(parsed._config)._ptr;
    }

    ov::SoPtr<ie::IExecutableNetworkInternal> LoadNetwork(const ie::CNNNetwork& network,
                                                          const std::shared_ptr<ie::RemoteContext>& context,
                                                          const std::map<std::string, std::string>& config) override {
        OV_ITT_SCOPE(FIRST_INFERENCE, ie::itt::domains::IE_LT, "Core::LoadNetwork::RemoteContext");
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
        ov::SoPtr<ie::IExecutableNetworkInternal> res;
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

    void ApplyAutoBatching(const ie::CNNNetwork& network,
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
            deviceNameWithoutBatch = DeviceIDParser::getBatchDevice(deviceNameWithBatchSize);
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
            auto device = ov::parseDeviceNameIntoConfig(deviceName);
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

    void CleanUpProperties(std::string& deviceName, std::map<std::string, std::string>& config, ov::Any property) {
        // auto-batching is not applicable, if there is auto_batch_timeout, delete it
        if (deviceName.find("BATCH") == std::string::npos) {
            const auto& batch_timeout_mode = config.find(property.as<std::string>());
            if (batch_timeout_mode != config.end()) {
                if (deviceName.find("AUTO") == std::string::npos && deviceName.find("MULTI") == std::string::npos)
                    config.erase(batch_timeout_mode);
            }
        }
    }

    ie::SoExecutableNetworkInternal LoadNetwork(const ie::CNNNetwork& network,
                                                const std::string& deviceNameOrig,
                                                const std::map<std::string, std::string>& config) override {
        OV_ITT_SCOPE(FIRST_INFERENCE, ie::itt::domains::IE_LT, "Core::LoadNetwork::CNN");
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
        ov::SoPtr<ie::IExecutableNetworkInternal> res;
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

    ie::SoExecutableNetworkInternal LoadNetwork(const std::string& modelPath,
                                                const std::string& deviceName,
                                                const std::map<std::string, std::string>& config,
                                                const std::function<void(const CNNNetwork&)>& val = nullptr) override {
        OV_ITT_SCOPE(FIRST_INFERENCE, ie::itt::domains::IE_LT, "Core::LoadNetwork::Path");
        auto parsed = parseDeviceNameIntoConfig(deviceName, config);
        auto plugin = GetCPPPluginByName(parsed._deviceName);
        ov::SoPtr<ie::IExecutableNetworkInternal> res;
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

    ie::SoExecutableNetworkInternal ImportNetwork(std::istream& networkModel,
                                                  const std::string& deviceName,
                                                  const std::map<std::string, std::string>& config) override {
        auto parsed = parseDeviceNameIntoConfig(deviceName, config);
        auto exec = GetCPPPluginByName(parsed._deviceName).import_model(networkModel, parsed._config);

        return {exec._ptr, exec._so};
    }

    ie::QueryNetworkResult QueryNetwork(const ie::CNNNetwork& network,
                                        const std::string& deviceName,
                                        const std::map<std::string, std::string>& config) const override {
        OV_ITT_SCOPED_TASK(ov::itt::domains::IE, "Core::QueryNetwork");
        auto parsed = parseDeviceNameIntoConfig(deviceName, config);
        auto res = GetCPPPluginByName(parsed._deviceName).query_model(network, parsed._config);
        if (!network.getFunction() || res.supportedLayersMap.empty())
            return res;

        const auto& func = network.getFunction();
        auto specialized_function = ngraph::clone_function(*func);

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

    Any GetMetric(const std::string& deviceName, const std::string& name, const AnyMap& options = {}) const override {
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
                IE_THROW()
                    << "You can get specific metrics with the GetMetric only for the AUTO itself (without devices). "
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

    void set_property(const std::string& device_name, const AnyMap& properties) override {
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
                OPENVINO_ASSERT(
                    !is_secondary_config_for_hw_device,
                    "set_property only supported ov::device::propreties for Meta device (AUTO/MULTI/HETERO). "
                    "You can configure the devices through the compile_model()/loadNetwork() API.");
            }
        }
        SetConfigForPlugins(properties, device_name);
    }

    Any get_property_for_core(const std::string& name) const {
        if (name == ov::force_tbb_terminate.name()) {
            const auto flag = executorManager()->getTbbFlag();
            return decltype(ov::force_tbb_terminate)::value_type(flag);
        } else if (name == ov::cache_dir.name()) {
            return ov::Any(coreConfig.get_cache_dir());
        } else if (name == ov::hint::allow_auto_batching.name()) {
            const auto flag = coreConfig.flag_allow_auto_batching;
            return decltype(ov::hint::allow_auto_batching)::value_type(flag);
        }

        IE_THROW() << "Exception is thrown while trying to call get_property with unsupported property: '" << name
                   << "'";
    }

    Any get_property(const std::string& device_name, const std::string& name, const AnyMap& arguments) const override {
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

    Any GetConfig(const std::string& deviceName, const std::string& name) const override {
        auto parsed = parseDeviceNameIntoConfig(deviceName);
        return GetCPPPluginByName(parsed._deviceName).get_config(name, parsed._config);
    }

    /**
     * @brief Returns devices available for neural networks inference
     *
     * @return A vector of devices. The devices are returned as { CPU, GPU.0, GPU.1, MYRIAD }
     * If there more than one device of specific type, they are enumerated with .# suffix.
     */
    std::vector<std::string> GetAvailableDevices() const override {
        std::vector<std::string> devices;
        const std::string propertyName = METRIC_KEY(AVAILABLE_DEVICES);

        for (auto&& deviceName : GetListOfDevicesInRegistry()) {
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

    /**
     * @brief Create a new shared context object on specified accelerator device
     * using specified plugin-specific low level device API parameters (device handle, pointer, etc.)
     * @param deviceName Name of a device to create new shared context on.
     * @param params Map of device-specific shared context parameters.
     * @return A shared pointer to a created remote context.
     */
    InferenceEngine::RemoteContext::Ptr CreateContext(const std::string& deviceName,
                                                      const InferenceEngine::ParamMap& params) override {
        auto parsed = ov::parseDeviceNameIntoConfig(deviceName, params);
        return GetCPPPluginByName(parsed._deviceName).create_context(parsed._config)._ptr;
    }

    /**
     * @brief Returns reference to CPP plugin wrapper by a device name
     * @param deviceName A name of device
     * @return Reference to a CPP plugin wrapper
     */
    ov::InferencePlugin GetCPPPluginByName(const std::string& pluginName) const {
        OV_ITT_SCOPE(FIRST_INFERENCE, ie::itt::domains::IE_LT, "CoreImpl::GetCPPPluginByName");

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
                std::shared_ptr<ie::IInferencePlugin> plugin_impl;
                desc.pluginCreateFunc(plugin_impl);
                plugin = InferencePlugin{plugin_impl, {}};
            } else {
                so = ov::util::load_shared_object(desc.libraryLocation.c_str());
                std::shared_ptr<ie::IInferencePlugin> plugin_impl;
                reinterpret_cast<InferenceEngine::CreatePluginEngineFunc*>(
                    ov::util::get_symbol(so, InferenceEngine::create_plugin_function))(plugin_impl);
                plugin = InferencePlugin{plugin_impl, so};
            }

            {
                plugin.set_name(deviceName);

                // Set Core class reference to plugins
                std::weak_ptr<ie::ICore> mutableCore = std::const_pointer_cast<ie::ICore>(shared_from_this());
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
                    std::vector<std::string> supportedConfigKeys =
                        plugin.get_metric(METRIC_KEY(SUPPORTED_CONFIG_KEYS), {});
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
                        plugin.add_extension(std::make_shared<ie::Extension>(extensionLocation));
                    }
                });
            }

            std::lock_guard<std::mutex> g_lock(get_mutex());
            // add plugin as extension itself
            if (desc.extensionCreateFunc) {  // static OpenVINO case
                try {
                    ie::IExtensionPtr ext;
                    desc.extensionCreateFunc(ext);
                    AddExtensionUnsafe(ext);
                } catch (const ie::GeneralError&) {
                    // the same extension can be registered multiple times - ignore it!
                }
            } else {
                TryToRegisterLibraryAsExtensionUnsafe(desc.libraryLocation);
            }

            return plugins.emplace(deviceName, plugin).first->second;
        } catch (const ie::Exception& ex) {
            IE_THROW() << "Failed to create plugin " << ov::util::from_file_path(desc.libraryLocation) << " for device "
                       << deviceName << "\n"
                       << "Please, check your environment\n"
                       << ex.what() << "\n";
        }
    }

    /**
     * @brief Unload plugin for specified device, but plugin meta-data is still in plugin registry
     * @param deviceName A name of device
     */
    void UnloadPluginByName(const std::string& deviceName) {
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
    void RegisterPluginByName(const std::string& pluginName, const std::string& deviceName) {
        std::lock_guard<std::mutex> lock(get_mutex());

        auto it = pluginRegistry.find(deviceName);
        if (it != pluginRegistry.end()) {
            IE_THROW() << "Device with \"" << deviceName << "\"  is already registered in the OpenVINO Runtime";
        }

        if (deviceName.find('.') != std::string::npos) {
            IE_THROW() << "Device name must not contain dot '.' symbol";
        }

        PluginDescriptor desc{ov::util::get_plugin_path(pluginName)};
        pluginRegistry[deviceName] = desc;
        add_mutex(deviceName);
    }

    /**
     * @brief Provides a list of plugin names in registry; physically such plugins may not be created
     * @return A list of plugin names
     */
    std::vector<std::string> GetListOfDevicesInRegistry() const {
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
    void SetConfigForPlugins(const ov::AnyMap& configMap, const std::string& deviceName) {
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
                    created_plugins.emplace_back(
                        std::pair<std::string, ov::InferencePlugin>{plugin.first, plugin.second});
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
    void ExtractAndSetDeviceConfig(const ov::AnyMap& configs) {
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

    std::map<std::string, std::string> GetSupportedConfig(const std::string& deviceName,
                                                          const std::map<std::string, std::string>& configs) override {
        std::vector<std::string> supportedConfigKeys;
        try {
            supportedConfigKeys =
                GetMetric(deviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS)).as<std::vector<std::string>>();
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
    void AddExtension(const ie::IExtensionPtr& extension) {
        std::lock_guard<std::mutex> lock(get_mutex());
        AddExtensionUnsafe(extension);
    }

    void AddOVExtensions(const std::vector<ov::Extension::Ptr>& extensions) {
        std::lock_guard<std::mutex> lock(get_mutex());
        for (const auto& ext : extensions) {
            ov_extensions.emplace_back(ext);
            if (auto op_base_ext = std::dynamic_pointer_cast<BaseOpExtension>(ext)) {
                for (const auto& attached_ext : op_base_ext->get_attached_extensions()) {
                    ov_extensions.emplace_back(attached_ext);
                }
            }
        }
    }

    /**
     * @brief Provides a list of extensions
     * @return A list of registered extensions
     */
    const std::vector<ie::IExtensionPtr>& GetExtensions() const {
        return extensions;
    }

    const std::vector<ov::Extension::Ptr>& GetOVExtensions() const {
        return ov_extensions;
    }

    std::map<std::string, ie::Version> GetVersions(const std::string& deviceName) const {
        std::map<std::string, ie::Version> versions;
        std::vector<std::string> deviceNames;

        {
            // for compatibility with samples / demo
            if (deviceName.find("HETERO") == 0) {
                auto pos = deviceName.find_first_of(":");
                if (pos != std::string::npos) {
                    deviceNames = ie::DeviceIDParser::getHeteroDevices(deviceName.substr(pos + 1));
                }
                deviceNames.push_back("HETERO");
            } else if (deviceName.find("MULTI") == 0) {
                auto pos = deviceName.find_first_of(":");
                if (pos != std::string::npos) {
                    deviceNames = ie::DeviceIDParser::getMultiDevices(deviceName.substr(pos + 1));
                }
                deviceNames.push_back("MULTI");
            } else if (deviceName.find("AUTO") == 0) {
                auto pos = deviceName.find_first_of(":");
                if (pos != std::string::npos) {
                    deviceNames = ie::DeviceIDParser::getMultiDevices(deviceName.substr(pos + 1));
                }
                deviceNames.emplace_back("AUTO");
            } else if (deviceName.find("BATCH") == 0) {
                auto pos = deviceName.find_first_of(":");
                if (pos != std::string::npos) {
                    deviceNames = {ie::DeviceIDParser::getBatchDevice(deviceName.substr(pos + 1))};
                }
                deviceNames.push_back("BATCH");
            } else {
                deviceNames.push_back(deviceName);
            }
        }

        for (auto&& deviceName_ : deviceNames) {
            ie::DeviceIDParser parser(deviceName_);
            std::string deviceNameLocal = parser.getDeviceName();

            ov::InferencePlugin cppPlugin = GetCPPPluginByName(deviceNameLocal);
            const ie::Version version = cppPlugin.get_version();
            versions[deviceNameLocal] = version;
        }

        return versions;
    }

private:
    void AddExtensionUnsafe(const ie::IExtensionPtr& extension) const {
        std::map<std::string, ngraph::OpSet> opsets = extension->getOpSets();
        for (const auto& it : opsets) {
            if (opsetNames.find(it.first) != opsetNames.end())
                IE_THROW() << "Cannot add opset with name: " << it.first
                           << ". Opset with the same name already exists.";
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

    template <typename C, typename = FileUtils::enableIfSupportedChar<C>>
    void TryToRegisterLibraryAsExtensionUnsafe(const std::basic_string<C>& path) const {
        try {
            const auto extension_ptr = std::make_shared<InferenceEngine::Extension>(path);
            AddExtensionUnsafe(extension_ptr);
        } catch (const InferenceEngine::GeneralError&) {
            // in case of shared library is not opened
        }
    }
};

}  // namespace ov

namespace InferenceEngine {

DeviceIDParser::DeviceIDParser(const std::string& deviceNameWithID) {
    deviceName = deviceNameWithID;

    auto pos = deviceName.find('.');
    if (pos != std::string::npos) {
        deviceName = deviceNameWithID.substr(0, pos);
        deviceID = deviceNameWithID.substr(pos + 1, deviceNameWithID.size());
    }
}

std::string DeviceIDParser::getDeviceID() const {
    return deviceID;
}

std::string DeviceIDParser::getDeviceName() const {
    return deviceName;
}

std::vector<std::string> DeviceIDParser::getHeteroDevices(std::string fallbackDevice) {
    std::vector<std::string> deviceNames;

    std::string cdevice;
    char delimiter = ',';
    size_t pos = 0;

    while ((pos = fallbackDevice.find(delimiter)) != std::string::npos) {
        deviceNames.push_back(fallbackDevice.substr(0, pos));
        fallbackDevice.erase(0, pos + 1);
    }

    if (!fallbackDevice.empty())
        deviceNames.push_back(fallbackDevice);

    return deviceNames;
}

std::vector<std::string> DeviceIDParser::getMultiDevices(std::string devicesList) {
    std::set<std::string> deviceNames;
    auto trim_request_info = [](const std::string& device_with_requests) {
        auto opening_bracket = device_with_requests.find_first_of('(');
        return device_with_requests.substr(0, opening_bracket);
    };
    std::string device;
    char delimiter = ',';
    size_t pos = 0;
    // in addition to the list of devices, every device can have a #requests in the brackets e.g. "CPU(100)"
    // we skip the #requests info here
    while ((pos = devicesList.find(delimiter)) != std::string::npos) {
        auto d = devicesList.substr(0, pos);
        if (d.find("BATCH") == 0) {
            deviceNames.insert("BATCH");
            auto p = d.find_first_of(":");
            if (p != std::string::npos)
                deviceNames.insert(DeviceIDParser::getBatchDevice(d.substr(p + 1)));
        } else {
            deviceNames.insert(trim_request_info(d));
        }
        devicesList.erase(0, pos + 1);
    }

    if (!devicesList.empty()) {
        if (devicesList.find("BATCH") == 0) {
            deviceNames.insert("BATCH");
            auto p = devicesList.find_first_of(":");
            if (p != std::string::npos)
                deviceNames.insert(DeviceIDParser::getBatchDevice(devicesList.substr(p + 1)));
        } else {
            deviceNames.insert(trim_request_info(devicesList));
        }
    }
    return std::vector<std::string>(deviceNames.begin(), deviceNames.end());
}

std::string DeviceIDParser::getBatchDevice(std::string device) {
    auto trim_request_info = [](const std::string& device_with_requests) {
        auto opening_bracket = device_with_requests.find_first_of('(');
        return device_with_requests.substr(0, opening_bracket);
    };
    return trim_request_info(device);
}

class Core::Impl : public ov::CoreImpl {
public:
    Impl() : ov::CoreImpl(false) {}
};

Core::Core(const std::string& xmlConfigFile) {
    _impl = std::make_shared<Impl>();

#ifdef OPENVINO_STATIC_LIBRARY
    _impl->RegisterPluginsInRegistry(::getStaticPluginsRegistry());
#else
    // If XML is default, load default plugins by absolute paths
    auto loadByAbsPath = xmlConfigFile.empty();
    _impl->RegisterPluginsInRegistry(ov::findPluginXML(xmlConfigFile), loadByAbsPath);
#endif
}

std::map<std::string, Version> Core::GetVersions(const std::string& deviceName) const {
    return _impl->GetVersions(deviceName);
}

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

CNNNetwork Core::ReadNetwork(const std::wstring& modelPath, const std::wstring& binPath) const {
    return ReadNetwork(ov::util::wstring_to_string(modelPath), ov::util::wstring_to_string(binPath));
}

#endif

CNNNetwork Core::ReadNetwork(const std::string& modelPath, const std::string& binPath) const {
    return _impl->ReadNetwork(modelPath, binPath);
}

CNNNetwork Core::ReadNetwork(const std::string& model, const Blob::CPtr& weights) const {
    return _impl->ReadNetwork(model, weights);
}

ExecutableNetwork Core::LoadNetwork(const CNNNetwork& network, const std::map<std::string, std::string>& config) {
    return LoadNetwork(network, ov::DEFAULT_DEVICE_NAME, config);
}

ExecutableNetwork Core::LoadNetwork(const CNNNetwork& network,
                                    const std::string& deviceName,
                                    const std::map<std::string, std::string>& config) {
    auto valid = ov::CoreImpl::CheckStatic(network);
    OPENVINO_ASSERT(std::get<0>(valid),
                    "InferenceEngine::Core::LoadNetwork doesn't support inputs having dynamic shapes. ",
                    "Use ov::Core::compile_model API instead. Dynamic inputs are :",
                    std::get<1>(valid));
    auto exec = _impl->LoadNetwork(network, deviceName, config);
    return {exec._ptr, exec._so};
}

ExecutableNetwork Core::LoadNetwork(const CNNNetwork& network,
                                    RemoteContext::Ptr context,
                                    const std::map<std::string, std::string>& config) {
    auto valid = ov::CoreImpl::CheckStatic(network);
    OPENVINO_ASSERT(std::get<0>(valid),
                    "InferenceEngine::Core::LoadNetwork doesn't support inputs having dynamic shapes. ",
                    "Use ov::Core::compile_model API instead. Dynamic inputs are :",
                    std::get<1>(valid));
    auto exec = _impl->LoadNetwork(network, std::dynamic_pointer_cast<RemoteContext>(context), config);
    return {exec._ptr, exec._so};
}

ExecutableNetwork Core::LoadNetwork(const std::string& modelPath,
                                    const std::string& deviceName,
                                    const std::map<std::string, std::string>& config) {
    auto exec = _impl->LoadNetwork(modelPath, deviceName, config, [](const CNNNetwork& network) {
        auto valid = ov::CoreImpl::CheckStatic(network);
        OPENVINO_ASSERT(std::get<0>(valid),
                        "InferenceEngine::Core::LoadNetwork doesn't support inputs having dynamic shapes. ",
                        "Use ov::Core::compile_model API instead. Dynamic inputs are :",
                        std::get<1>(valid));
    });
    return {exec._ptr, exec._so};
}

ExecutableNetwork Core::LoadNetwork(const std::string& modelPath, const std::map<std::string, std::string>& config) {
    return LoadNetwork(modelPath, ov::DEFAULT_DEVICE_NAME, config);
}

RemoteContext::Ptr Core::CreateContext(const std::string& deviceName, const ParamMap& params) {
    return _impl->CreateContext(deviceName, params);
}

RemoteContext::Ptr Core::GetDefaultContext(const std::string& deviceName) {
    if (deviceName.find("HETERO") == 0) {
        IE_THROW() << "HETERO device does not support remote context";
    }
    if (deviceName.find("MULTI") == 0) {
        IE_THROW() << "MULTI device does not support remote context";
    }
    if (deviceName.find("AUTO") == 0) {
        IE_THROW() << "AUTO device does not support remote context";
    }
    return _impl->GetDefaultContext(deviceName);
}

void Core::AddExtension(IExtensionPtr extension, const std::string& deviceName_) {
    if (deviceName_.find("HETERO") == 0) {
        IE_THROW() << "HETERO device does not support extensions. Please, set extensions directly to fallback devices";
    }
    if (deviceName_.find("MULTI") == 0) {
        IE_THROW() << "MULTI device does not support extensions. Please, set extensions directly to fallback devices";
    }
    if (deviceName_.find("AUTO") == 0) {
        IE_THROW() << "AUTO device does not support extensions. Please, set extensions directly to fallback devices";
    }

    _impl->AddExtension(extension);
}

void Core::AddExtension(const IExtensionPtr& extension) {
    _impl->AddExtension(extension);
}

ExecutableNetwork Core::ImportNetwork(const std::string& modelFileName,
                                      const std::string& deviceName,
                                      const std::map<std::string, std::string>& config) {
    OV_ITT_SCOPED_TASK(ov::itt::domains::IE, "Core::ImportNetwork");
    auto parsed = ov::parseDeviceNameIntoConfig(deviceName, config);
    auto exec = _impl->GetCPPPluginByName(parsed._deviceName).import_model(modelFileName, parsed._config);
    return {exec._ptr, exec._so};
}

ExecutableNetwork Core::ImportNetwork(std::istream& networkModel,
                                      const std::string& deviceName,
                                      const std::map<std::string, std::string>& config) {
    OV_ITT_SCOPED_TASK(ov::itt::domains::IE, "Core::ImportNetwork");
    auto exec = _impl->ImportNetwork(networkModel, deviceName, config);
    return {exec._ptr, exec._so};
}

ExecutableNetwork Core::ImportNetwork(std::istream& networkModel) {
    OV_ITT_SCOPED_TASK(ov::itt::domains::IE, "Core::ImportNetwork");

    using ExportMagic = std::array<char, 4>;
    constexpr static const ExportMagic exportMagic = {{0x1, 0xE, 0xE, 0x1}};

    std::string deviceName;
    ExportMagic magic = {};
    auto currentPos = networkModel.tellg();
    networkModel.read(magic.data(), magic.size());
    if (exportMagic == magic) {
        std::getline(networkModel, deviceName);
    } else {
        IE_THROW() << "Passed compiled stream does not contain device name. "
                      "Please, provide device name manually";
    }
    networkModel.seekg(currentPos, networkModel.beg);

    auto exec = _impl->GetCPPPluginByName(deviceName).import_model(networkModel, {});
    return {exec._ptr, exec._so};
}

ExecutableNetwork Core::ImportNetwork(std::istream& networkModel,
                                      const RemoteContext::Ptr& context,
                                      const std::map<std::string, std::string>& config) {
    OV_ITT_SCOPED_TASK(ov::itt::domains::IE, "Core::ImportNetwork");

    if (context == nullptr) {
        IE_THROW() << "Remote context is null";
    }

    std::string deviceName_ = context->getDeviceName();
    DeviceIDParser device(deviceName_);
    std::string deviceName = device.getDeviceName();

    auto parsed = ov::parseDeviceNameIntoConfig(deviceName, config);
    auto exec = _impl->GetCPPPluginByName(deviceName)
                    .import_model(networkModel, std::dynamic_pointer_cast<RemoteContext>(context), parsed._config);
    return {exec._ptr, exec._so};
}

QueryNetworkResult Core::QueryNetwork(const CNNNetwork& network,
                                      const std::string& deviceName,
                                      const std::map<std::string, std::string>& config) const {
    auto valid = ov::CoreImpl::CheckStatic(network);
    OPENVINO_ASSERT(std::get<0>(valid),
                    "InferenceEngine::Core::QueryNetwork doesn't support inputs having dynamic shapes. ",
                    "Use ov::Core::compile_model API instead. Dynamic inputs are :",
                    std::get<1>(valid));

    return _impl->QueryNetwork(network, deviceName, config);
}

void Core::SetConfig(const std::map<std::string, std::string>& config, const std::string& deviceName) {
    // HETERO case
    if (deviceName.find("HETERO:") == 0) {
        IE_THROW() << "SetConfig is supported only for HETERO itself (without devices). "
                      "You can configure the devices with SetConfig before creating the HETERO on top.";
    }

    // MULTI case
    if (deviceName.find("MULTI:") == 0) {
        IE_THROW() << "SetConfig is supported only for MULTI itself (without devices). "
                      "You can configure the devices with SetConfig before creating the MULTI on top.";
    }

    // AUTO case
    if (deviceName.find("AUTO:") == 0) {
        IE_THROW() << "SetConfig is supported only for AUTO itself (without devices). "
                      "You can configure the devices with SetConfig before creating the AUTO on top.";
    }

    ov::AnyMap conf = ov::any_copy(config);
    if (deviceName.empty()) {
        _impl->SetConfigForPlugins(conf, std::string());
    } else {
        _impl->SetConfigForPlugins(conf, deviceName);
    }
}

Parameter Core::GetConfig(const std::string& deviceName, const std::string& name) const {
    // HETERO case
    {
        if (deviceName.find("HETERO:") == 0) {
            IE_THROW() << "You can only GetConfig of the HETERO itself (without devices). "
                          "GetConfig is also possible for the individual devices before creating the HETERO on top.";
        }
    }
    // MULTI case
    {
        if (deviceName.find("MULTI:") == 0) {
            IE_THROW() << "You can only GetConfig of the MULTI itself (without devices). "
                          "GetConfig is also possible for the individual devices before creating the MULTI on top.";
        }
    }
    // AUTO case
    {
        if (deviceName.find("AUTO:") == 0) {
            IE_THROW() << "You can only GetConfig of the AUTO itself (without devices). "
                          "GetConfig is also possible for the individual devices before creating the AUTO on top.";
        }
    }

    if (name == CONFIG_KEY(FORCE_TBB_TERMINATE)) {
        const auto flag = executorManager()->getTbbFlag();
        return flag ? CONFIG_VALUE(YES) : CONFIG_VALUE(NO);
    }

    auto parsed = ov::parseDeviceNameIntoConfig(deviceName);
    return _impl->GetCPPPluginByName(parsed._deviceName).get_config(name, parsed._config);
}

Parameter Core::GetMetric(const std::string& deviceName, const std::string& name, const ParamMap& options) const {
    return _impl->GetMetric(deviceName, name, options);
}

std::vector<std::string> Core::GetAvailableDevices() const {
    return _impl->GetAvailableDevices();
}

void Core::RegisterPlugin(const std::string& pluginName, const std::string& deviceName) {
    _impl->RegisterPluginByName(pluginName, deviceName);
}

void Core::RegisterPlugins(const std::string& xmlConfigFile) {
    _impl->RegisterPluginsInRegistry(xmlConfigFile);
}

void Core::UnregisterPlugin(const std::string& deviceName_) {
    DeviceIDParser parser(deviceName_);
    std::string deviceName = parser.getDeviceName();

    _impl->UnloadPluginByName(deviceName);
}

}  // namespace InferenceEngine

namespace ov {

#define OV_CORE_CALL_STATEMENT(...)                     \
    try {                                               \
        __VA_ARGS__;                                    \
    } catch (const std::exception& ex) {                \
        throw ov::Exception(ex.what());                 \
    } catch (...) {                                     \
        OPENVINO_ASSERT(false, "Unexpected exception"); \
    }

class Core::Impl : public CoreImpl {
public:
    Impl() : ov::CoreImpl(true) {}
};

Core::Core(const std::string& xmlConfigFile) {
    _impl = std::make_shared<Impl>();

#ifdef OPENVINO_STATIC_LIBRARY
    OV_CORE_CALL_STATEMENT(_impl->RegisterPluginsInRegistry(::getStaticPluginsRegistry());)
#else
    OV_CORE_CALL_STATEMENT({
        // If XML is default, load default plugins by absolute paths
        auto loadByAbsPath = xmlConfigFile.empty();
        _impl->RegisterPluginsInRegistry(findPluginXML(xmlConfigFile), loadByAbsPath);
    })
#endif
}

std::map<std::string, Version> Core::get_versions(const std::string& deviceName) const {
    OV_CORE_CALL_STATEMENT({
        std::map<std::string, Version> versions;
        for (auto&& kvp : _impl->GetVersions(deviceName)) {
            versions[kvp.first] = Version{kvp.second.buildNumber, kvp.second.description};
        }
        return versions;
    })
}
#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
std::shared_ptr<ov::Model> Core::read_model(const std::wstring& modelPath, const std::wstring& binPath) const {
    OV_CORE_CALL_STATEMENT(
        return _impl->ReadNetwork(ov::util::wstring_to_string(modelPath), ov::util::wstring_to_string(binPath))
            .getFunction(););
}
#endif

std::shared_ptr<ov::Model> Core::read_model(const std::string& modelPath, const std::string& binPath) const {
    OV_CORE_CALL_STATEMENT(return _impl->ReadNetwork(modelPath, binPath).getFunction(););
}

std::shared_ptr<ov::Model> Core::read_model(const std::string& model, const ov::Tensor& weights) const {
    InferenceEngine::Blob::Ptr blob;
    if (weights) {
        blob = weights._impl;
    }
    OV_CORE_CALL_STATEMENT(return _impl->ReadNetwork(model, blob).getFunction(););
}

namespace {

ie::CNNNetwork toCNN(const std::shared_ptr<const ngraph::Function>& model) {
    return ie::CNNNetwork(
        std::make_shared<ie::details::CNNNetworkNGraphImpl>(std::const_pointer_cast<ngraph::Function>(model),
                                                            std::vector<ie::IExtensionPtr>{},
                                                            true));
}

}  // namespace

CompiledModel Core::compile_model(const std::shared_ptr<const ov::Model>& model, const AnyMap& config) {
    return compile_model(model, ov::DEFAULT_DEVICE_NAME, config);
}

CompiledModel Core::compile_model(const std::shared_ptr<const ov::Model>& model,
                                  const std::string& deviceName,
                                  const AnyMap& config) {
    OV_CORE_CALL_STATEMENT({
        auto exec = _impl->LoadNetwork(toCNN(model), deviceName, any_copy(flatten_sub_properties(deviceName, config)));
        return {exec._ptr, exec._so};
    });
}

CompiledModel Core::compile_model(const std::string& modelPath, const AnyMap& config) {
    return compile_model(modelPath, ov::DEFAULT_DEVICE_NAME, config);
}

CompiledModel Core::compile_model(const std::string& modelPath, const std::string& deviceName, const AnyMap& config) {
    OV_CORE_CALL_STATEMENT({
        auto exec = _impl->LoadNetwork(modelPath, deviceName, any_copy(flatten_sub_properties(deviceName, config)));
        return {exec._ptr, exec._so};
    });
}

CompiledModel Core::compile_model(const std::shared_ptr<const ov::Model>& model,
                                  const RemoteContext& context,
                                  const AnyMap& config) {
    OV_CORE_CALL_STATEMENT({
        auto exec = _impl->LoadNetwork(toCNN(model),
                                       context._impl,
                                       any_copy(flatten_sub_properties(context.get_device_name(), config)));
        return {exec._ptr, exec._so};
    });
}

void Core::add_extension(const ie::IExtensionPtr& extension) {
    OV_CORE_CALL_STATEMENT(_impl->AddExtension(extension););
}

void Core::add_extension(const std::string& library_path) {
    try {
        const std::string path = resolve_extension_path(library_path);
        add_extension(ov::detail::load_extensions(path));
    } catch (const std::runtime_error&) {
        try {
            // Try to load legacy extension
            const auto extension_ptr = std::make_shared<InferenceEngine::Extension>(library_path);
            OPENVINO_SUPPRESS_DEPRECATED_START
            add_extension(extension_ptr);
            OPENVINO_SUPPRESS_DEPRECATED_END
        } catch (const std::runtime_error&) {
            throw ov::Exception("Cannot add extension. Cannot find entry point to the extension library");
        }
    }
}
#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
void Core::add_extension(const std::wstring& library_path) {
    try {
        const std::string path = resolve_extension_path(ov::util::wstring_to_string(library_path));
        add_extension(ov::detail::load_extensions(ov::util::string_to_wstring(path)));
    } catch (const std::runtime_error&) {
        try {
            // Try to load legacy extension
            const auto extension_ptr = std::make_shared<InferenceEngine::Extension>(library_path);
            OPENVINO_SUPPRESS_DEPRECATED_START
            add_extension(extension_ptr);
            OPENVINO_SUPPRESS_DEPRECATED_END
        } catch (const std::runtime_error&) {
            throw ov::Exception("Cannot add extension. Cannot find entry point to the extension library");
        }
    }
}
#endif

void Core::add_extension(const std::shared_ptr<ov::Extension>& extension) {
    add_extension(std::vector<std::shared_ptr<ov::Extension>>{extension});
}
void Core::add_extension(const std::vector<std::shared_ptr<ov::Extension>>& extensions) {
    OV_CORE_CALL_STATEMENT({ _impl->AddOVExtensions(extensions); });
}

CompiledModel Core::import_model(std::istream& modelStream, const std::string& deviceName, const AnyMap& config) {
    OV_ITT_SCOPED_TASK(ov::itt::domains::IE, "Core::import_model");
    OV_CORE_CALL_STATEMENT({
        auto exec = _impl->ImportNetwork(modelStream, deviceName, any_copy(flatten_sub_properties(deviceName, config)));
        return {exec._ptr, exec._so};
    });
}

CompiledModel Core::import_model(std::istream& modelStream, const RemoteContext& context, const AnyMap& config) {
    OV_ITT_SCOPED_TASK(ov::itt::domains::IE, "Core::import_model");

    using ExportMagic = std::array<char, 4>;
    constexpr static const ExportMagic exportMagic = {{0x1, 0xE, 0xE, 0x1}};

    std::string deviceName;
    ExportMagic magic = {};
    auto currentPos = modelStream.tellg();
    modelStream.read(magic.data(), magic.size());
    if (exportMagic == magic) {
        std::getline(modelStream, deviceName);
    } else {
        OPENVINO_ASSERT(false,
                        "Passed compiled stream does not contain device name. "
                        "Please, provide device name manually");
    }
    modelStream.seekg(currentPos, modelStream.beg);

    OV_CORE_CALL_STATEMENT({
        auto exec = _impl->GetCPPPluginByName(deviceName).import_model(modelStream, {});
        return {exec._ptr, exec._so};
    });
}

SupportedOpsMap Core::query_model(const std::shared_ptr<const ov::Model>& model,
                                  const std::string& deviceName,
                                  const AnyMap& config) const {
    OV_CORE_CALL_STATEMENT({
        auto qnResult =
            _impl->QueryNetwork(toCNN(model), deviceName, any_copy(flatten_sub_properties(deviceName, config)));
        return qnResult.supportedLayersMap;
    });
}

void Core::set_property(const AnyMap& properties) {
    OV_CORE_CALL_STATEMENT(return _impl->set_property({}, properties););
}

void Core::set_property(const std::string& device_name, const AnyMap& properties) {
    OV_CORE_CALL_STATEMENT(return _impl->set_property(device_name, properties););
}

Any Core::get_property(const std::string& deviceName, const std::string& name) const {
    OV_CORE_CALL_STATEMENT(return _impl->get_property(deviceName, name, {}););
}

Any Core::get_property(const std::string& deviceName, const std::string& name, const AnyMap& arguments) const {
    OV_CORE_CALL_STATEMENT(return _impl->get_property(deviceName, name, arguments););
}

std::vector<std::string> Core::get_available_devices() const {
    OV_CORE_CALL_STATEMENT(return _impl->GetAvailableDevices(););
}

void Core::register_plugin(const std::string& pluginName, const std::string& deviceName) {
    OV_CORE_CALL_STATEMENT(_impl->RegisterPluginByName(pluginName, deviceName););
}

void Core::unload_plugin(const std::string& deviceName) {
    OV_CORE_CALL_STATEMENT({
        ie::DeviceIDParser parser(deviceName);
        std::string devName = parser.getDeviceName();

        _impl->UnloadPluginByName(devName);
    });
}

void Core::register_plugins(const std::string& xmlConfigFile) {
    OV_CORE_CALL_STATEMENT(_impl->RegisterPluginsInRegistry(xmlConfigFile););
}

RemoteContext Core::create_context(const std::string& deviceName, const AnyMap& params) {
    OPENVINO_ASSERT(deviceName.find("HETERO") != 0, "HETERO device does not support remote context");
    OPENVINO_ASSERT(deviceName.find("MULTI") != 0, "MULTI device does not support remote context");
    OPENVINO_ASSERT(deviceName.find("AUTO") != 0, "AUTO device does not support remote context");
    OPENVINO_ASSERT(deviceName.find("BATCH") != 0, "BATCH device does not support remote context");

    OV_CORE_CALL_STATEMENT({
        auto parsed = parseDeviceNameIntoConfig(deviceName, flatten_sub_properties(deviceName, params));
        auto remoteContext = _impl->GetCPPPluginByName(parsed._deviceName).create_context(parsed._config);
        return {remoteContext._ptr, {remoteContext._so}};
    });
}

RemoteContext Core::get_default_context(const std::string& deviceName) {
    OPENVINO_ASSERT(deviceName.find("HETERO") != 0, "HETERO device does not support default remote context");
    OPENVINO_ASSERT(deviceName.find("MULTI") != 0, "MULTI device does not support default remote context");
    OPENVINO_ASSERT(deviceName.find("AUTO") != 0, "AUTO device does not support default remote context");
    OPENVINO_ASSERT(deviceName.find("BATCH") != 0, "BATCH device does not support default remote context");

    OV_CORE_CALL_STATEMENT({
        auto parsed = parseDeviceNameIntoConfig(deviceName, AnyMap{});
        auto remoteContext = _impl->GetCPPPluginByName(parsed._deviceName).get_default_context(parsed._config);
        return {remoteContext._ptr, {remoteContext._so}};
    });
}

}  // namespace ov
