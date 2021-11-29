// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <sys/stat.h>

#include <ie_core.hpp>
#include <ie_icore.hpp>
#include <ngraph/opsets/opset.hpp>
#include <ngraph/ngraph.hpp>
#include <ngraph/graph_util.hpp>
#include <ngraph/pass/constant_folding.hpp>

#include "compilation_context.hpp"
#include "cpp/ie_plugin.hpp"
#include "ie_plugin_config.hpp"
#include "ie_cache_manager.hpp"
#include "ie_cache_guard.hpp"
#include "ie_itt.hpp"
#include "file_utils.h"
#include "ie_network_reader.hpp"
#include "xml_parse_utils.h"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"
#include "cpp_interfaces/interface/ie_iexecutable_network_internal.hpp"

using namespace InferenceEngine::PluginConfigParams;
using namespace std::placeholders;

namespace InferenceEngine {

namespace {

template <typename T>
struct Parsed {
    std::string _deviceName;
    std::map<std::string, T> _config;
};

template <typename T = Parameter>
Parsed<T> parseDeviceNameIntoConfig(const std::string& deviceName, const std::map<std::string, T>& config = {}) {
    auto config_ = config;
    auto deviceName_ = deviceName;
    if (deviceName_.find("HETERO:") == 0) {
        deviceName_ = "HETERO";
        config_["TARGET_FALLBACK"] = deviceName.substr(7);
    } else if (deviceName_.find("MULTI:") == 0) {
        deviceName_ = "MULTI";
        config_[InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES] = deviceName.substr(6);
    } else if (deviceName_.find("AUTO") == 0) {
        deviceName_ = "AUTO";
        if (deviceName.size() > std::string("AUTO").size()) {
            std::string deviceList = deviceName.substr(std::string("AUTO:").size());
            if (deviceList.find("AUTO") != std::string::npos) {
                IE_THROW() << "Device list for AUTO should not be AUTO";
            }
            config_[InferenceEngine::KEY_AUTO_DEVICE_LIST] = deviceName.substr(std::string("AUTO:").size());
        }
    } else {
        if (deviceName_.empty()) {
            deviceName_ = "AUTO";
        }
        DeviceIDParser parser(deviceName_);
        deviceName_ = parser.getDeviceName();
        std::string deviceIDLocal = parser.getDeviceID();

        if (!deviceIDLocal.empty()) {
            config_[KEY_DEVICE_ID] = deviceIDLocal;
        }
    }
    return {deviceName_, config_};
}

Parameter copyParameterValue(const Parameter & value) {
    if (value.is<bool>()) {
        return { value.as<bool>() };
    } else if (value.is<int>()) {
        return { value.as<int>() };
    } else if (value.is<unsigned int>()) {
        return { value.as<unsigned int>() };
    } else if (value.is<float>()) {
        return { value.as<float>() };
    } else if (value.is<std::string>()) {
        return { value.as<std::string>() };
    } else if (value.is<std::vector<std::string> >()) {
        return { value.as<std::vector<std::string> >() };
    } else if (value.is<std::vector<int> >()) {
        return { value.as<std::vector<int> >() };
    } else if (value.is<std::vector<float> >()) {
        return { value.as<std::vector<float> >() };
    } else if (value.is<std::vector<unsigned int> >()) {
        return { value.as<std::vector<unsigned int> >() };
    } else if (value.is<std::tuple<unsigned int, unsigned int, unsigned int> >()) {
        return { value.as<std::tuple<unsigned int, unsigned int, unsigned int> >() };
    } else if (value.is<std::tuple<unsigned int, unsigned int> >()) {
        return { value.as<std::tuple<unsigned int, unsigned int> >() };
    }

    return std::move(value);
}

template <typename F>
void allowNotImplemented(F && f) {
    try {
        f();
    } catch (const NotImplemented&) { }
}

}  // namespace

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

    if (!fallbackDevice.empty()) deviceNames.push_back(fallbackDevice);

    return deviceNames;
}

std::vector<std::string> DeviceIDParser::getMultiDevices(std::string devicesList) {
    std::vector<std::string> deviceNames;
    auto trim_request_info = [](std::string device_with_requests) {
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
        deviceNames.push_back(trim_request_info(d));
        devicesList.erase(0, pos + 1);
    }

    if (!devicesList.empty()) deviceNames.push_back(trim_request_info(devicesList));

    return deviceNames;
}

class Core::Impl : public ICore {
    // Fields are ordered by deletion order
    ITaskExecutor::Ptr _taskExecutor = nullptr;

    mutable std::map<std::string, InferencePlugin> plugins;

    class CoreConfig final {
    public:
        struct CacheConfig {
            std::string                    _cacheDir;
            std::shared_ptr<ICacheManager> _cacheManager;
        };

        void setAndUpdate(std::map<std::string, std::string>& config) {
            auto it = config.find(CONFIG_KEY(CACHE_DIR));
            if (it != config.end()) {
                std::lock_guard<std::mutex> lock(_cacheConfigMutex);
                _cacheConfig._cacheDir = it->second;
                if (!it->second.empty()) {
                    FileUtils::createDirectoryRecursive(it->second);
                    _cacheConfig._cacheManager = std::make_shared<FileStorageCacheManager>(std::move(it->second));
                } else {
                    _cacheConfig._cacheManager = nullptr;
                }

                config.erase(it);
            }
        }

        // Creating thread-safe copy of config including shared_ptr to ICacheManager
        CacheConfig getCacheConfig() const {
            std::lock_guard<std::mutex> lock(_cacheConfigMutex);
            return _cacheConfig;
        }

    private:
        mutable std::mutex _cacheConfigMutex;
        CacheConfig _cacheConfig;
    };

    // Core settings (cache config, etc)
    CoreConfig coreConfig;

    CacheGuard cacheGuard;

    struct PluginDescriptor {
        FileUtils::FilePath libraryLocation;
        std::map<std::string, std::string> defaultConfig;
        std::vector<FileUtils::FilePath> listOfExtentions;
    };

    std::unordered_set<std::string> opsetNames;
    std::vector<IExtensionPtr> extensions;

    std::map<std::string, PluginDescriptor> pluginRegistry;
    mutable std::mutex pluginsMutex;  // to lock parallel access to pluginRegistry and plugins

    bool DeviceSupportsImportExport(const std::string& deviceName) const override {
        auto parsed = parseDeviceNameIntoConfig(deviceName);
        auto plugin = GetCPPPluginByName(parsed._deviceName);
        return DeviceSupportsImportExport(plugin);
    }

    bool DeviceSupportsImportExport(const InferencePlugin& plugin) const {
        std::vector<std::string> supportedMetricKeys = plugin.GetMetric(METRIC_KEY(SUPPORTED_METRICS), {});
        auto it = std::find(supportedMetricKeys.begin(), supportedMetricKeys.end(),
                            METRIC_KEY(IMPORT_EXPORT_SUPPORT));
        bool supported = (it != supportedMetricKeys.end()) &&
                    plugin.GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), {});
        return supported;
    }

    bool DeviceSupportsCacheDir(const InferencePlugin& plugin) const {
        return DeviceSupportsConfigKey(plugin, CONFIG_KEY(CACHE_DIR));
    }

    bool DeviceSupportsConfigKey(const InferencePlugin& plugin, const std::string& key) const {
        bool supported = false;
        std::vector<std::string> supportedMetricKeys;
        try {
            // If plugin doesn't support 'SUPPORTED_METRICS' - treat it as config is not supported as well
            supportedMetricKeys =
                    plugin.GetMetric(METRIC_KEY(SUPPORTED_METRICS), {}).as<std::vector<std::string>>();
        } catch(...) {}
        auto it = std::find(supportedMetricKeys.begin(), supportedMetricKeys.end(),
                            METRIC_KEY(SUPPORTED_CONFIG_KEYS));
        if (it != supportedMetricKeys.end()) {
            std::vector<std::string> configKeys = plugin.GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS), {});
            supported = std::find(configKeys.begin(), configKeys.end(), key) != configKeys.end();
        }
        return supported;
    }

    SoExecutableNetworkInternal LoadNetworkImpl(const CNNNetwork& network,
                                                InferencePlugin& plugin,
                                                const std::map<std::string, std::string>& parsedConfig,
                                                const RemoteContext::Ptr& context,
                                                const std::string& blobID,
                                                const std::string& modelPath = std::string(),
                                                bool forceDisableCache = false) {
        OV_ITT_SCOPED_TASK(itt::domains::IE, "Core::Impl::LoadNetworkImpl");
        SoExecutableNetworkInternal execNetwork;
        execNetwork = context ? plugin.LoadNetwork(network, context, parsedConfig) :
                                plugin.LoadNetwork(network, parsedConfig);
        auto cacheManager = coreConfig.getCacheConfig()._cacheManager;
        if (!forceDisableCache && cacheManager && DeviceSupportsImportExport(plugin)) {
            try {
                // need to export network for further import from "cache"
                OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::IE_LT, "Core::LoadNetwork::Export");
                cacheManager->writeCacheEntry(blobID, [&](std::ostream& networkStream) {
                    networkStream << CompiledBlobHeader(GetInferenceEngineVersion()->buildNumber,
                                                        NetworkCompilationContext::calculateFileInfo(modelPath));
                    execNetwork->Export(networkStream);
                });
            } catch (...) {
                cacheManager->removeCacheEntry(blobID);
                throw;
            }
        }
        return execNetwork;
    }

    SoExecutableNetworkInternal LoadNetworkFromCache(const std::shared_ptr<ICacheManager>& cacheManager,
                                                     const std::string& blobId,
                                                     InferencePlugin& plugin,
                                                     const std::map<std::string, std::string>& config,
                                                     const RemoteContext::Ptr& context,
                                                     bool& networkIsImported,
                                                     const std::string& modelPath = std::string()) {
        SoExecutableNetworkInternal execNetwork;
        struct HeaderException {};

        IE_ASSERT(cacheManager != nullptr);
        try {
            cacheManager->readCacheEntry(blobId, [&](std::istream &networkStream) {
                OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::IE_LT, "Core::LoadNetworkFromCache::ReadStreamAndImport");
                try {
                    CompiledBlobHeader header;
                    networkStream >> header;
                    if (header.getIeVersion() != GetInferenceEngineVersion()->buildNumber) {
                        // Build number mismatch, don't use this cache
                        throw NetworkNotRead("Version does not match");
                    }
                    if (header.getFileInfo() != NetworkCompilationContext::calculateFileInfo(modelPath)) {
                        // Original file is changed, don't use cache
                        throw NetworkNotRead("Original model file is changed");
                    }
                } catch (...) {
                    throw HeaderException();
                }

                execNetwork = context ?
                              plugin.ImportNetwork(networkStream, context, config) :
                              plugin.ImportNetwork(networkStream, config);
                networkIsImported = true;
            });
        } catch (const HeaderException& ex) {
            // For these exceptions just remove old cache and set that import didn't work
            cacheManager->removeCacheEntry(blobId);
            networkIsImported = false;
        } catch (...) {
            cacheManager->removeCacheEntry(blobId);
            networkIsImported = false;
            // TODO: temporary disabled by #54335. In future don't throw only for new 'blob_outdated' exception
            // throw;
        }
        return execNetwork;
    }

    std::map<std::string, std::string> CreateCompileConfig(const InferencePlugin& plugin,
                                                           const std::string& deviceFamily,
                                                           const std::map<std::string, std::string>& origConfig) const {
        std::map<std::string, Parameter> getMetricConfig;
        auto compileConfig = origConfig;

        // 0. Remove TARGET_FALLBACK key, move it to getMetricConfig
        auto targetFallbackIt = compileConfig.find("TARGET_FALLBACK");
        if (targetFallbackIt != compileConfig.end()) {
            getMetricConfig[targetFallbackIt->first] = targetFallbackIt->second;
            compileConfig.erase(targetFallbackIt);
        }

        // 1. remove DEVICE_ID key
        auto deviceIt = compileConfig.find(CONFIG_KEY(DEVICE_ID));
        if (deviceIt != compileConfig.end()) {
            getMetricConfig[deviceIt->first] = deviceIt->second;
            compileConfig.erase(deviceIt);
        }

        // 2. replace it with DEVICE_ARCHITECTURE value
        std::vector<std::string> supportedMetricKeys =
            plugin.GetMetric(METRIC_KEY(SUPPORTED_METRICS), getMetricConfig);
        auto archIt = std::find(supportedMetricKeys.begin(), supportedMetricKeys.end(),
            METRIC_KEY(DEVICE_ARCHITECTURE));
        if (archIt != supportedMetricKeys.end()) {
            auto value = plugin.GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), getMetricConfig);
            compileConfig[METRIC_KEY(DEVICE_ARCHITECTURE)] = value.as<std::string>();
        } else {
            // Take device name if device does not support DEVICE_ARCHITECTURE metric
            compileConfig[METRIC_KEY(DEVICE_ARCHITECTURE)] = deviceFamily;
        }
        return compileConfig;
    }

    std::string CalculateNetworkHash(const CNNNetwork& network, const std::string& deviceFamily,
                                     const InferencePlugin& plugin,
                                     const std::map<std::string, std::string>& config) const {
        auto compileConfig = CreateCompileConfig(plugin, deviceFamily, config);
        return NetworkCompilationContext::computeHash(network, compileConfig);
    }

    std::string CalculateFileHash(const std::string& modelName, const std::string& deviceFamily,
                                  const InferencePlugin& plugin,
                                  const std::map<std::string, std::string>& config) const {
        auto compileConfig = CreateCompileConfig(plugin, deviceFamily, config);
        return NetworkCompilationContext::computeHash(modelName, compileConfig);
    }

public:
    Impl() {
        opsetNames.insert("opset1");
        opsetNames.insert("opset2");
        opsetNames.insert("opset3");
        opsetNames.insert("opset4");
        opsetNames.insert("opset5");
        opsetNames.insert("opset6");
        opsetNames.insert("opset7");
    }

    ~Impl() override = default;

    /**
     * @brief Register plugins for devices which are located in .xml configuration file. The function supports UNICODE path
     * @param xmlConfigFile An .xml configuraion with device / plugin information
     */
    void RegisterPluginsInRegistry(const std::string& xmlConfigFile) {
        std::lock_guard<std::mutex> lock(pluginsMutex);

        auto parse_result = ParseXml(xmlConfigFile.c_str());
        if (!parse_result.error_msg.empty()) {
            IE_THROW() << parse_result.error_msg;
        }

        pugi::xml_document& xmlDoc = *parse_result.xml;

        using namespace XMLParseUtils;
        pugi::xml_node ieNode = xmlDoc.document_element();
        pugi::xml_node devicesNode = ieNode.child("plugins");

        FOREACH_CHILD(pluginNode, devicesNode, "plugin") {
            std::string deviceName = GetStrAttr(pluginNode, "name");
            FileUtils::FilePath pluginPath = FileUtils::toFilePath(GetStrAttr(pluginNode, "location").c_str());

            if (deviceName.find('.') != std::string::npos) {
                IE_THROW() << "Device name must not contain dot '.' symbol";
            }

            // append IR library path for default IE plugins
            {
                FileUtils::FilePath absFilePath = FileUtils::makePath(getInferenceEngineLibraryPath(), pluginPath);
                if (FileUtils::fileExist(absFilePath)) pluginPath = absFilePath;
            }

            // check properties
            auto propertiesNode = pluginNode.child("properties");
            std::map<std::string, std::string> config;

            if (propertiesNode) {
                FOREACH_CHILD(propertyNode, propertiesNode, "property") {
                    std::string key = GetStrAttr(propertyNode, "key");
                    std::string value = GetStrAttr(propertyNode, "value");
                    config[key] = value;
                }
            }

            // check extensions
            auto extensionsNode = pluginNode.child("extensions");
            std::vector<FileUtils::FilePath> listOfExtentions;

            if (extensionsNode) {
                FOREACH_CHILD(extensionNode, extensionsNode, "extension") {
                    FileUtils::FilePath extensionLocation = FileUtils::toFilePath(GetStrAttr(extensionNode, "location").c_str());
                    listOfExtentions.push_back(extensionLocation);
                }
            }

            // fill value in plugin registry for later lazy initialization
            {
                PluginDescriptor desc = {pluginPath, config, listOfExtentions};
                pluginRegistry[deviceName] = desc;
            }
        }
    }

    //
    // ICore public API
    //

    /**
     * @brief Returns global task executor
     * @return Reference to task executor
     */
    ITaskExecutor::Ptr GetTaskExecutor() const override {
        return _taskExecutor;
    }

    CNNNetwork ReadNetwork(const std::string& modelPath, const std::string& binPath) const override {
        OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::IE_RT, "Core::Impl::ReadNetwork from file");
        return details::ReadNetwork(modelPath, binPath, extensions);
    }

    CNNNetwork ReadNetwork(const std::string& model, const Blob::CPtr& weights) const override {
        OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::IE_RT, "Core::Impl::ReadNetwork from memory");
        return details::ReadNetwork(model, weights, extensions);
    }

    // TODO: In future this method can be added to ICore interface
    SoExecutableNetworkInternal LoadNetwork(const CNNNetwork& network, const RemoteContext::Ptr& context,
                                            const std::map<std::string, std::string>& config) {
        OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::IE_LT, "Core::LoadNetwork::RemoteContext");
        if (context == nullptr) {
            IE_THROW() << "Remote context is null";
        }
        auto parsed = parseDeviceNameIntoConfig(context->getDeviceName(), config);
        auto plugin = GetCPPPluginByName(parsed._deviceName);
        SoExecutableNetworkInternal res;
        auto cacheManager = coreConfig.getCacheConfig()._cacheManager;
        if (cacheManager && DeviceSupportsImportExport(plugin)) {
            auto hash = CalculateNetworkHash(network, parsed._deviceName, plugin, parsed._config);
            bool loadedFromCache = false;
            auto lock = cacheGuard.getHashLock(hash);
            res = LoadNetworkFromCache(cacheManager, hash, plugin, parsed._config, context, loadedFromCache);
            if (!loadedFromCache) {
                res = LoadNetworkImpl(network, plugin, parsed._config, context, hash);
            }
        } else {
            res = LoadNetworkImpl(network, plugin, parsed._config, context, {});
        }
        return res;
    }

    SoExecutableNetworkInternal LoadNetwork(const CNNNetwork& network,
                                            const std::string& deviceName,
                                            const std::map<std::string, std::string>& config) override {
        OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::IE_LT, "Core::LoadNetwork::CNN");
        bool forceDisableCache = config.count(CONFIG_KEY_INTERNAL(FORCE_DISABLE_CACHE)) > 0;
        auto parsed = parseDeviceNameIntoConfig(deviceName, config);
        if (forceDisableCache) {
            // remove this config key from parsed as plugins can throw unsupported exception
            parsed._config.erase(CONFIG_KEY_INTERNAL(FORCE_DISABLE_CACHE));
        }
        auto plugin = GetCPPPluginByName(parsed._deviceName);
        SoExecutableNetworkInternal res;
        auto cacheManager = coreConfig.getCacheConfig()._cacheManager;
        if (!forceDisableCache && cacheManager && DeviceSupportsImportExport(plugin)) {
            auto hash = CalculateNetworkHash(network, parsed._deviceName, plugin, parsed._config);
            bool loadedFromCache = false;
            auto lock = cacheGuard.getHashLock(hash);
            res = LoadNetworkFromCache(cacheManager, hash, plugin, parsed._config, nullptr, loadedFromCache);
            if (!loadedFromCache) {
                res = LoadNetworkImpl(network, plugin, parsed._config, nullptr, hash, {}, forceDisableCache);
            }
        } else {
            res = LoadNetworkImpl(network, plugin, parsed._config, nullptr, {}, {}, forceDisableCache);
        }
        return res;
    }

    SoExecutableNetworkInternal LoadNetwork(const std::string& modelPath,
                                            const std::string& deviceName,
                                            const std::map<std::string, std::string>& config) override {
        OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::IE_LT, "Core::LoadNetwork::Path");
        auto parsed = parseDeviceNameIntoConfig(deviceName, config);
        auto plugin = GetCPPPluginByName(parsed._deviceName);
        SoExecutableNetworkInternal res;
        auto cacheManager = coreConfig.getCacheConfig()._cacheManager;
        if (cacheManager && DeviceSupportsImportExport(plugin)) {
            bool loadedFromCache = false;
            auto hash = CalculateFileHash(modelPath, parsed._deviceName, plugin, parsed._config);
            auto lock = cacheGuard.getHashLock(hash);
            res = LoadNetworkFromCache(cacheManager, hash, plugin, parsed._config,
                                       nullptr, loadedFromCache, modelPath);
            if (!loadedFromCache) {
                auto cnnNetwork = ReadNetwork(modelPath, std::string());
                res = LoadNetworkImpl(cnnNetwork, plugin, parsed._config, nullptr, hash, modelPath);
            }
        } else if (cacheManager) {
            res = plugin.LoadNetwork(modelPath, parsed._config);
        } else {
            auto cnnNetwork = ReadNetwork(modelPath, std::string());
            res = LoadNetworkImpl(cnnNetwork, plugin, parsed._config, nullptr, {}, modelPath);
        }
        return res;
    }

    SoExecutableNetworkInternal ImportNetwork(std::istream& networkModel, const std::string& deviceName,
                                              const std::map<std::string, std::string>& config) override {
        auto parsed = parseDeviceNameIntoConfig(deviceName, config);
        return GetCPPPluginByName(parsed._deviceName).ImportNetwork(networkModel, parsed._config);
    }

    QueryNetworkResult QueryNetwork(const CNNNetwork& network, const std::string& deviceName,
                                    const std::map<std::string, std::string>& config) const override {
        OV_ITT_SCOPED_TASK(itt::domains::IE, "Core::QueryNetwork");
        auto parsed = parseDeviceNameIntoConfig(deviceName, config);
        auto res = GetCPPPluginByName(parsed._deviceName).QueryNetwork(network, parsed._config);
        if (!network.getFunction() || res.supportedLayersMap.empty())
            return res;

        const auto& func = network.getFunction();
        auto specialized_function = ngraph::clone_function(*func);

        std::string defDevice = res.supportedLayersMap.begin()->second;
        ngraph::pass::ConstantFolding().run_on_function(specialized_function);
        std::unordered_set<std::string> opNames;

        for (const auto& op : specialized_function->get_ops())
            opNames.emplace(op->get_friendly_name());

        for (const auto& op : func->get_ops()) {
            if (opNames.find(op->get_friendly_name()) == opNames.end() ||
                (!res.supportedLayersMap.count(op->get_friendly_name()) &&
                 std::dynamic_pointer_cast<ngraph::op::Constant>(op)))
                res.supportedLayersMap[op->get_friendly_name()] = defDevice;
        }
        return res;
    }

    Parameter GetMetric(const std::string& deviceName, const std::string& name) const override {
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

        auto parsed = parseDeviceNameIntoConfig(deviceName);

        // we need to return a copy of Parameter object which is created on Core side,
        // not in InferenceEngine plugin side, which can be unloaded from Core in a parallel thread
        // TODO: remove this WA after *-31417 is resolved
        return copyParameterValue(GetCPPPluginByName(parsed._deviceName).GetMetric(name, parsed._config));
    }

    /**
     * @brief Returns devices available for neural networks inference
     *
     * @return A vector of devices. The devices are returned as { CPU, FPGA.0, FPGA.1, MYRIAD }
     * If there more than one device of specific type, they are enumerated with .# suffix.
     */
    std::vector<std::string> GetAvailableDevices() const override {
        std::vector<std::string> devices;
        const std::string propertyName = METRIC_KEY(AVAILABLE_DEVICES);

        for (auto&& deviceName : GetListOfDevicesInRegistry()) {
            std::vector<std::string> devicesIDs;
            try {
                const Parameter p = GetMetric(deviceName, propertyName);
                devicesIDs = p.as<std::vector<std::string>>();
            } catch (Exception&) {
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
     * @brief Returns reference to CPP plugin wrapper by a device name
     * @param deviceName A name of device
     * @return Reference to a CPP plugin wrapper
     */
    InferencePlugin GetCPPPluginByName(const std::string& deviceName) const {
        OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::IE_LT, "Core::Impl::GetCPPPluginByName");

        std::lock_guard<std::mutex> lock(pluginsMutex);

        auto it = pluginRegistry.find(deviceName);
        if (it == pluginRegistry.end()) {
            IE_THROW() << "Device with \"" << deviceName << "\" name is not registered in the InferenceEngine";
        }

        // Plugin is in registry, but not created, let's create

        if (plugins.find(deviceName) == plugins.end()) {
            PluginDescriptor desc = it->second;

            try {
                InferencePlugin plugin{desc.libraryLocation};

                {
                    plugin.SetName(deviceName);

                    // Set Inference Engine class reference to plugins
                    ICore* mutableCore = const_cast<ICore*>(static_cast<const ICore*>(this));
                    plugin.SetCore(mutableCore);
                }

                // Add registered extensions to new plugin
                allowNotImplemented([&](){
                    for (const auto& ext : extensions) {
                        plugin.AddExtension(ext);
                    }
                });

                // configuring
                {
                    if (DeviceSupportsCacheDir(plugin)) {
                        auto cacheConfig = coreConfig.getCacheConfig();
                        if (cacheConfig._cacheManager) {
                            desc.defaultConfig[CONFIG_KEY(CACHE_DIR)] = cacheConfig._cacheDir;
                        }
                    }
                    allowNotImplemented([&]() {
                        plugin.SetConfig(desc.defaultConfig);
                    });

                    allowNotImplemented([&]() {
                        for (auto&& extensionLocation : desc.listOfExtentions) {
                            plugin.AddExtension(std::make_shared<Extension>(extensionLocation));
                        }
                    });
                }

                plugins[deviceName] = plugin;
            } catch (const Exception& ex) {
                IE_THROW() << "Failed to create plugin " << FileUtils::fromFilePath(desc.libraryLocation) << " for device " << deviceName
                                   << "\n"
                                   << "Please, check your environment\n"
                                   << ex.what() << "\n";
            }
        }

        return plugins[deviceName];
    }

    /**
     * @brief Unload plugin for specified device, but plugin meta-data is still in plugin registry
     * @param deviceName A name of device
     */
    void UnloadPluginByName(const std::string& deviceName) {
        std::lock_guard<std::mutex> lock(pluginsMutex);
        auto it = plugins.find(deviceName);
        if (it == plugins.end()) {
            IE_THROW() << "Device with \"" << deviceName << "\" name is not registered in the InferenceEngine";
        }

        plugins.erase(deviceName);
    }

    /**
     * @brief Registers plugin meta-data in registry for specified device
     * @param deviceName A name of device
     */
    void RegisterPluginByName(const std::string& pluginName, const std::string& deviceName) {
        std::lock_guard<std::mutex> lock(pluginsMutex);

        auto it = pluginRegistry.find(deviceName);
        if (it != pluginRegistry.end()) {
            IE_THROW() << "Device with \"" << deviceName << "\"  is already registered in the InferenceEngine";
        }

        if (deviceName.find('.') != std::string::npos) {
            IE_THROW() << "Device name must not contain dot '.' symbol";
        }

        // append IR library path for default IE plugins
        FileUtils::FilePath pluginPath;
        {
            pluginPath = FileUtils::makePluginLibraryName({}, FileUtils::toFilePath(pluginName.c_str()));

            FileUtils::FilePath absFilePath = FileUtils::makePath(getInferenceEngineLibraryPath(), pluginPath);
            if (FileUtils::fileExist(absFilePath)) pluginPath = absFilePath;
        }

        PluginDescriptor desc = {pluginPath, {}, {}};
        pluginRegistry[deviceName] = desc;
    }

    /**
     * @brief Porvides a list of plugin names in registry; physically such plugins may not be created
     * @return A list of plugin names
     */
    std::vector<std::string> GetListOfDevicesInRegistry() const {
        std::lock_guard<std::mutex> lock(pluginsMutex);

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
     * @note  `deviceName` is not allowed in form of MULTI:CPU, HETERO:FPGA,CPU, AUTO:CPU
     *        just simple forms like CPU, GPU, MULTU, GPU.0, etc
     */
    void SetConfigForPlugins(const std::map<std::string, std::string>& configMap, const std::string& deviceName) {
        auto config = configMap;

        std::lock_guard<std::mutex> lock(pluginsMutex);

        if (deviceName.empty()) {
            coreConfig.setAndUpdate(config);
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
            IE_THROW() << "Device with \"" << deviceName << "\" name is not registered in the InferenceEngine";
        }

        // set config for already created plugins
        for (auto& plugin : plugins) {
            if (deviceName.empty() || deviceName == plugin.first) {
                allowNotImplemented([&]() {
                    auto configCopy = config;
                    if (DeviceSupportsCacheDir(plugin.second)) {
                        auto cacheConfig = coreConfig.getCacheConfig();
                        if (cacheConfig._cacheManager) {
                            configCopy[CONFIG_KEY(CACHE_DIR)] = cacheConfig._cacheDir;
                        }
                    }
                    plugin.second.SetConfig(configCopy);
                });
            }
        }
    }

    /**
     * @brief Registers the extension in a Core object
     *        Such extensions can be used for both CNNNetwork readers and device plugins
     */
    void AddExtension(const IExtensionPtr& extension) {
        std::lock_guard<std::mutex> lock(pluginsMutex);

        std::map<std::string, ngraph::OpSet> opsets = extension->getOpSets();
        for (const auto& it : opsets) {
            if (opsetNames.find(it.first) != opsetNames.end())
                IE_THROW() << "Cannot add opset with name: " << it.first << ". Opset with the same name already exists.";
            opsetNames.insert(it.first);
        }

        // add extensions for already created plugins
        for (auto& plugin : plugins) {
            try {
                plugin.second.AddExtension(extension);
            } catch (...) {}
        }
        extensions.emplace_back(extension);
    }

    /**
     * @brief Provides a list of extensions
     * @return A list of registered extensions
     */
    const std::vector<IExtensionPtr>& GetExtensions() const {
        return extensions;
    }
};

Core::Core(const std::string& xmlConfigFile) {
    _impl = std::make_shared<Impl>();

    std::string xmlConfigFile_ = xmlConfigFile;
    if (xmlConfigFile_.empty()) {
        // register plugins from default plugins.xml config
        FileUtils::FilePath xmlConfigFileDefault = FileUtils::makePath(getInferenceEngineLibraryPath(), FileUtils::toFilePath("plugins.xml"));
        xmlConfigFile_ = FileUtils::fromFilePath(xmlConfigFileDefault);
    }

    RegisterPlugins(xmlConfigFile_);
}

std::map<std::string, Version> Core::GetVersions(const std::string& deviceName) const {
    std::map<std::string, Version> versions;
    std::vector<std::string> deviceNames;

    {
        // for compatibility with samples / demo
        if (deviceName.find("HETERO") == 0) {
            auto pos = deviceName.find_first_of(":");
            if (pos != std::string::npos) {
                deviceNames = DeviceIDParser::getHeteroDevices(deviceName.substr(pos + 1));
            }
            deviceNames.push_back("HETERO");
        } else if (deviceName.find("MULTI") == 0) {
            auto pos = deviceName.find_first_of(":");
            if (pos != std::string::npos) {
                deviceNames = DeviceIDParser::getMultiDevices(deviceName.substr(pos + 1));
            }
            deviceNames.push_back("MULTI");
        } else if (deviceName.find("AUTO") == 0) {
            auto pos = deviceName.find_first_of(":");
            if (pos != std::string::npos) {
                deviceNames = DeviceIDParser::getHeteroDevices(deviceName.substr(pos + 1));
            }
            deviceNames.emplace_back("AUTO");
        } else {
            deviceNames.push_back(deviceName);
        }
    }

    for (auto&& deviceName_ : deviceNames) {
        DeviceIDParser parser(deviceName_);
        std::string deviceNameLocal = parser.getDeviceName();

        InferenceEngine::InferencePlugin cppPlugin = _impl->GetCPPPluginByName(deviceNameLocal);
        const Version version = cppPlugin.GetVersion();
        versions[deviceNameLocal] = version;
    }

    return versions;
}

#ifdef ENABLE_UNICODE_PATH_SUPPORT

CNNNetwork Core::ReadNetwork(const std::wstring& modelPath, const std::wstring& binPath) const {
    return ReadNetwork(FileUtils::wStringtoMBCSstringChar(modelPath),
                       FileUtils::wStringtoMBCSstringChar(binPath));
}

#endif

CNNNetwork Core::ReadNetwork(const std::string& modelPath, const std::string& binPath) const {
    return _impl->ReadNetwork(modelPath, binPath);
}

CNNNetwork Core::ReadNetwork(const std::string& model, const Blob::CPtr& weights) const {
    return _impl->ReadNetwork(model, weights);
}

ExecutableNetwork Core::LoadNetwork(const CNNNetwork& network, const std::string& deviceName,
                                    const std::map<std::string, std::string>& config) {
    auto exec = _impl->LoadNetwork(network, deviceName, config);
    return { exec, exec };
}

ExecutableNetwork Core::LoadNetwork(const CNNNetwork& network, RemoteContext::Ptr context,
                                    const std::map<std::string, std::string>& config) {
    auto exec = _impl->LoadNetwork(network, context, config);
    return { exec, exec };
}

ExecutableNetwork Core::LoadNetwork(const std::string& modelPath, const std::string& deviceName,
                                    const std::map<std::string, std::string>& config) {
    auto exec = _impl->LoadNetwork(modelPath, deviceName, config);
    return { exec, exec };
}

RemoteContext::Ptr Core::CreateContext(const std::string& deviceName, const ParamMap& params) {
    if (deviceName.find("HETERO") == 0) {
        IE_THROW() << "HETERO device does not support remote context";
    }
    if (deviceName.find("MULTI") == 0) {
        IE_THROW() << "MULTI device does not support remote context";
    }
    if (deviceName.find("AUTO") == 0) {
        IE_THROW() << "AUTO device does not support remote context";
    }

    auto parsed = parseDeviceNameIntoConfig(deviceName, params);
    return _impl->GetCPPPluginByName(parsed._deviceName).CreateContext(parsed._config);
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

    auto parsed = parseDeviceNameIntoConfig(deviceName, ParamMap());
    return _impl->GetCPPPluginByName(parsed._deviceName).GetDefaultContext(parsed._config);
}

void Core::AddExtension(IExtensionPtr extension, const std::string& deviceName_) {
    if (deviceName_.find("HETERO") == 0) {
        IE_THROW()
            << "HETERO device does not support extensions. Please, set extensions directly to fallback devices";
    }
    if (deviceName_.find("MULTI") == 0) {
        IE_THROW()
            << "MULTI device does not support extensions. Please, set extensions directly to fallback devices";
    }
    if (deviceName_.find("AUTO") == 0) {
        IE_THROW()
            << "AUTO device does not support extensions. Please, set extensions directly to fallback devices";
    }

    _impl->AddExtension(extension);
}

void Core::AddExtension(const IExtensionPtr& extension) {
    _impl->AddExtension(extension);
}

ExecutableNetwork Core::ImportNetwork(const std::string& modelFileName, const std::string& deviceName,
                                      const std::map<std::string, std::string>& config) {
    OV_ITT_SCOPED_TASK(itt::domains::IE, "Core::ImportNetwork");
    auto parsed = parseDeviceNameIntoConfig(deviceName, config);
    auto exec = _impl->GetCPPPluginByName(parsed._deviceName).ImportNetwork(modelFileName, parsed._config);
    return { exec, exec };
}

ExecutableNetwork Core::ImportNetwork(std::istream& networkModel, const std::string& deviceName,
                                      const std::map<std::string, std::string>& config) {
    OV_ITT_SCOPED_TASK(itt::domains::IE, "Core::ImportNetwork");
    auto exec = _impl->ImportNetwork(networkModel, deviceName, config);
    return { exec, exec };
}

ExecutableNetwork Core::ImportNetwork(std::istream& networkModel) {
    OV_ITT_SCOPED_TASK(itt::domains::IE, "Core::ImportNetwork");

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

    auto exec = _impl->GetCPPPluginByName(deviceName).ImportNetwork(networkModel, {});
    return { exec, exec };
}

ExecutableNetwork Core::ImportNetwork(std::istream& networkModel,
                                      const RemoteContext::Ptr& context,
                                      const std::map<std::string, std::string>& config) {
    OV_ITT_SCOPED_TASK(itt::domains::IE, "Core::ImportNetwork");

    if (context == nullptr) {
        IE_THROW() << "Remote context is null";
    }

    std::string deviceName_ = context->getDeviceName();
    DeviceIDParser device(deviceName_);
    std::string deviceName = device.getDeviceName();

    auto parsed = parseDeviceNameIntoConfig(deviceName, config);
    auto exec = _impl->GetCPPPluginByName(deviceName).ImportNetwork(networkModel, context, parsed._config);
    return { exec, exec };
}

QueryNetworkResult Core::QueryNetwork(const CNNNetwork& network, const std::string& deviceName,
                                      const std::map<std::string, std::string>& config) const {
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

    // GPU.0, FPGA.1 cases
    if (deviceName.find(".") != std::string::npos) {
        IE_THROW() << "SetConfig is supported only for device family itself (without particular device .#). "
                                "You can pass .# as a particular device instance to QueryNetwork, LoadNetwork, ImportNetwork only";
    }

    if (deviceName.empty()) {
        _impl->SetConfigForPlugins(config, std::string());
    } else {
        auto parsed = parseDeviceNameIntoConfig(deviceName, config);
        _impl->SetConfigForPlugins(parsed._config, parsed._deviceName);
    }
}

Parameter Core::GetConfig(const std::string& deviceName, const std::string& name) const {
    // HETERO case
    {
        if (deviceName.find("HETERO:") == 0) {
            IE_THROW()
                << "You can only GetConfig of the HETERO itself (without devices). "
                   "GetConfig is also possible for the individual devices before creating the HETERO on top.";
        }
    }
    // MULTI case
    {
        if (deviceName.find("MULTI:") == 0) {
            IE_THROW()
                << "You can only GetConfig of the MULTI itself (without devices). "
                   "GetConfig is also possible for the individual devices before creating the MULTI on top.";
        }
    }
    // AUTO case
    {
        if (deviceName.find("AUTO:") == 0) {
            IE_THROW()
                << "You can only GetConfig of the AUTO itself (without devices). "
                   "GetConfig is also possible for the individual devices before creating the AUTO on top.";
        }
    }

    auto parsed = parseDeviceNameIntoConfig(deviceName);

    // we need to return a copy of Parameter object which is created on Core side,
    // not in InferenceEngine plugin side, which can be unloaded from Core in a parallel thread
    // TODO: remove this WA after *-31417 is resolved
    return copyParameterValue(_impl->GetCPPPluginByName(parsed._deviceName).GetConfig(name, parsed._config));
}

Parameter Core::GetMetric(const std::string& deviceName, const std::string& name) const {
    return _impl->GetMetric(deviceName, name);
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
