// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <istream>
#include <mutex>

#include <ie_core.hpp>
#include <multi-device/multi_device_config.hpp>
#include <ngraph/opsets/opset.hpp>
#include <ngraph/ngraph.hpp>
#include <ngraph/graph_util.hpp>

#include <cpp_interfaces/exception2status.hpp>
#include "compilation_context.hpp"
#include "ie_plugin_cpp.hpp"
#include "ie_plugin_config.hpp"
#include "ie_itt.hpp"
#include "file_utils.h"
#include "ie_network_reader.hpp"
#include "xml_parse_utils.h"

using namespace InferenceEngine::PluginConfigParams;

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
    } else {
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
    } catch (const NotImplemented & ex) { }
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

    class CoreConfig {
        std::string _modelCacheDir {};
        std::string _blobFileName {};
        bool _isModelCacheEnabled = false;
        std::map<std::string, std::string> _config;

    public:
        using ConfigMap = std::map<std::string, std::string>;

        CoreConfig() = default;

        CoreConfig(std::map<std::string, std::string> & config,
                   const CoreConfig & globalConfig) {
            // set global config settings
            *this = globalConfig;

            // parse local config
            {
                auto it = config.find(CONFIG_KEY(MODEL_CACHE_DIR));
                if (it != config.end()) {
                    _config[it->first] = it->second;
                    _modelCacheDir = it->second;
                    _isModelCacheEnabled = true;
                    config.erase(it);
                }
            }

            {
                auto it = config.find(CONFIG_KEY(COMPILED_BLOB));
                if (it != config.end()) {
                    _config[it->first] = it->second;
                    _blobFileName = it->second;
                    _isModelCacheEnabled = true;
                    config.erase(it);
                }
            }
        }

        bool isModelCacheEnabled() const { return _isModelCacheEnabled; }
        std::string getModelCacheDir() const { return _modelCacheDir; }
        std::string getBlobFileName() const { return _blobFileName; }
    };

    // Core settings for specific devices
    mutable std::map<std::string, CoreConfig> coreConfig;

    struct PluginDescriptor {
        FileUtils::FilePath libraryLocation;
        std::map<std::string, std::string> defaultConfig;
        std::vector<FileUtils::FilePath> listOfExtentions;
    };

    std::unordered_set<std::string> opsetNames;
    std::vector<IExtensionPtr> extensions;

    std::map<std::string, PluginDescriptor> pluginRegistry;
    mutable std::mutex pluginsMutex;  // to lock parallel access to pluginRegistry and plugins

    ExecutableNetwork LoadNetworkImpl(const CNNNetwork& network, const std::string& deviceName,
                                      std::map<std::string, std::string> config,
                                      const RemoteContext::Ptr & context) {
        OV_ITT_SCOPED_TASK(itt::domains::IE_LT, "Core::Impl::LoadNetwork");

        // check whether the network is already compiled and cached
        auto compileConfig = config;
        compileConfig["DEVICE_NAME"] = deviceName;

        // save inference engine version instead of plugin GUID
        // it does not work only for cases when we locally build the same version
        // multiple times, but change the code frequently
        compileConfig["INFERENCE_ENGINE_VERSION"] = GetInferenceEngineVersion()->buildNumber;

        auto deviceSupportsImport = [&] (ICore * core) -> bool {
            OV_ITT_SCOPED_TASK(itt::domains::IE_LT, "Core::Impl::device_supports_import");
            std::stringstream dummyStream;
            bool supports = true;
            try {
                auto n = context ? core->ImportNetwork(dummyStream, context, config) :
                                   core->ImportNetwork(dummyStream, deviceName, config);
                (void)n;
            } catch (const NetworkNotRead &) {
                supports = true;
            } catch (const NotImplemented &) {
                supports = false;
            } catch (const details::InferenceEngineException & ex) {
                std::string message = ex.what();
                supports = message.find("NOT IMPLEMENTED") == std::string::npos;
#ifdef __APPLE__
            } catch (const std::exception & ex) {
                std::string message = ex.what();
                supports = message.find("NOT IMPLEMENTED") == std::string::npos;
#endif
            }

            if (!supports) {
                std::cout << deviceName << " does not support import" << std::endl;
            }

            return supports;
        };

        std::string deviceFamily; // MULTI:CPU -> MULTI, GPU.0 -> GPU, CPU -> CPU and so on
        {
            auto parsed = parseDeviceNameIntoConfig<std::string>(deviceName);
            deviceFamily = parsed._deviceName;
        }

        // Note:
        //   core.SetConfig({ { DECLARE_CONFIG_KEY(MODEL_CACHE_DIR), "" } }, "MULTI");
        // will cache models only for MULTI itself, but not for MULTI devices
        // To enable caching for MULTI sub-devices, enable it via
        //   core.SetConfig({ { DECLARE_CONFIG_KEY(MODEL_CACHE_DIR), "" } }, "CPU");
        // or using a global version:
        //   core.SetConfig({ { DECLARE_CONFIG_KEY(MODEL_CACHE_DIR), "" } });
        // which tries to use caching for all devices (if import / export is available).

        CoreConfig localCoreConfig(config, coreConfig[deviceFamily]);
        bool modelCacheEnabled = localCoreConfig.isModelCacheEnabled(),
            cachingIsAvailable = false, networkIsImported = false;
        std::string blobFileName = localCoreConfig.getBlobFileName();
        std::string modelCacheDir = localCoreConfig.getModelCacheDir();

        // TEST CODE: FORCE MODEL CACHE
        {
            modelCacheEnabled = true;
            modelCacheDir = getIELibraryPath();
        }

        if (modelCacheEnabled && blobFileName.empty() && deviceSupportsImport(this)) {
            OV_ITT_SCOPED_TASK(itt::domains::IE_LT, "Core::LoadNetwork::hashing");

            // Note: the following information from remote context is taken into account:
            // * device name (part of compileConfig under DEVICE_NAME key)

            NetworkCompilationContext context(network, compileConfig);
            cachingIsAvailable = context.isCachingAvailable();

            // auto-hashing since CONFIG_KEY(COMPILED_BLOB) is not passed
            if (cachingIsAvailable)
                blobFileName = context.computeHash() + ".blob";
        }

        std::cout << (cachingIsAvailable ?
            std::string("caching is available") :
            std::string("caching is not available")) << " for " << deviceName << std::endl;

        auto removeCacheEntry = [&deviceName] (const std::string & blobFileName) {
            std::cout << "Removed cache entry " << blobFileName << " " << deviceName << std::endl;
            if (FileUtils::fileExist(blobFileName))
                std::remove(blobFileName.c_str());
        };

        auto getImportConfig = [] (const std::map<std::string, std::string> & loadConfig) {
            // For import config all options are passed as is
            // Plugin should either:
            // 1. ignore not compatible options (e.g. binary is compiled with 1 option, while provided different value)
            // 2. throw exceptions if values for some options are different
            std::map<std::string, std::string> importConfig;
            auto copyConfigValue = [&] (const std::string & configKey) {
                auto it = loadConfig.find(configKey);
                if (it != loadConfig.end()) {
                    importConfig[configKey] = it->second;
                }
            };

            copyConfigValue(CONFIG_KEY(DEVICE_ID));

            // TODO: return loadConfig as is
            return importConfig;
        };

        ExecutableNetwork execNetwork;

        // make a full path
        blobFileName = FileUtils::makePath(modelCacheDir, blobFileName);

        if (cachingIsAvailable && FileUtils::fileExist(blobFileName)) {
            auto importConfig = parseDeviceNameIntoConfig<std::string>(deviceName, getImportConfig(config));
            try {
                OV_ITT_SCOPED_TASK(itt::domains::IE_LT, "Core::LoadNetwork::ImportNetwork");
                std::cout << "try to import from core to " << deviceName << "\n\n" << std::endl;
                std::ifstream networkStream(blobFileName);
                execNetwork = context ?
                    ImportNetwork(networkStream, context, importConfig._config) :
                    ImportNetwork(networkStream, importConfig._deviceName, importConfig._config);
                networkIsImported = true;
                std::cout << "Network is imported to " << deviceName << std::endl;
            } catch (const NotImplemented &) {
                // 1. Device does not support ImportNetwork / Export flow
                std::cout << "[BUG] Import is not implemented O_o " << importConfig._deviceName << std::endl;
                removeCacheEntry(blobFileName);
            } catch (const NetworkNotRead &) {
                // 2. Device supports this flow, but failed to import network for some reason
                //    (e.g. device arch is not compatible with device arch network compiled for
                //     e.g. compiled for MYX, but current device is M2 stick)
                std::cout << "NetworkNotRead: try to export one more time (remove blob!!) " << importConfig._deviceName << std::endl;
                removeCacheEntry(blobFileName);
            } catch (const std::exception & ex) {
                std::string message = ex.what();
                bool appleRTTI = message.find("NOT IMPLMENENTED") != std::string::npos;

                if (appleRTTI) { // Apple RTTI
                    std::cout << "Apple RTTI: " << ex.what() << std::endl;
                    removeCacheEntry(blobFileName);
                } else { // some issues because of import failed
                    std::cout << "[BUG] Import failed for " << importConfig._deviceName << std::endl;
                    removeCacheEntry(blobFileName);
                }
            }
        }

        if (!networkIsImported) {
            auto parsed = parseDeviceNameIntoConfig(deviceName, config);
            {
                OV_ITT_SCOPED_TASK(itt::domains::IE_LT, "Core::LoadNetwork::LoadNetwork");
                // to limit plugin scope
                auto plugin = GetCPPPluginByName(parsed._deviceName);
                execNetwork = context ? plugin.LoadNetwork(network, parsed._config) :
                                        plugin.LoadNetwork(network, context, parsed._config);
            }

            if (cachingIsAvailable) {
                try {
                    // need to export network for further import from "cache"
                    {
                        OV_ITT_SCOPED_TASK(itt::domains::IE_LT, "Core::LoadNetwork::Export");
                        execNetwork.Export(blobFileName);
                        std::cout << "Network is exported for " << parsed._deviceName
                            << " as " << blobFileName << std::endl;
                    }
                    // {
                    //     OV_ITT_SCOPED_TASK(itt::domains::IE_LT, "Core::LoadNetwork::DesctroyExe");
                    //     execNetwork = {};
                    // }
                    // {
                    //     OV_ITT_SCOPED_TASK(itt::domains::IE_LT, "Core::LoadNetwork::ImportNetwork_fake");
                    //     std::ifstream networkStream(blobFileName);
                    //     auto importConfig = parseDeviceNameIntoConfig<std::string>(deviceName, getImportConfig(config));
                    //     execNetwork = context ?
                    //         ImportNetwork(networkStream, context, importConfig._config) :
                    //         ImportNetwork(networkStream, importConfig._deviceName, importConfig._config);
                    // }
                } catch (const NotImplemented &) {
                    // 1. Network export flow is not implemented in device
                    removeCacheEntry(blobFileName);

                    std::cout << "Export is not implemented " << parsed._deviceName << std::endl;
                } catch (const std::exception & ex) {
                    std::string message = ex.what();
                    bool appleRTTI = message.find("NOT IMPLMENENTED") != std::string::npos;

                    if (appleRTTI) { // APPLE RTTI issue
                        std::cout << "Apple RTTI: " << message << std::endl;
                        removeCacheEntry(blobFileName);
                    } else { // network cannot be exported due to plugin bugs
                        std::cout << "[BUG] Failed to export model " << ex.what() << std::endl;
                        removeCacheEntry(blobFileName);
                    }
                }
            }
        }

        return execNetwork;
    }

public:
    Impl() {
        opsetNames.insert("opset1");
        opsetNames.insert("opset2");
        opsetNames.insert("opset3");
        opsetNames.insert("opset4");
        opsetNames.insert("opset5");
        opsetNames.insert("opset6");
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
            THROW_IE_EXCEPTION << parse_result.error_msg;
        }

        pugi::xml_document& xmlDoc = *parse_result.xml;

        using namespace XMLParseUtils;
        pugi::xml_node ieNode = xmlDoc.document_element();
        pugi::xml_node devicesNode = ieNode.child("plugins");

        FOREACH_CHILD(pluginNode, devicesNode, "plugin") {
            std::string deviceName = GetStrAttr(pluginNode, "name");
            FileUtils::FilePath pluginPath = FileUtils::toFilePath(GetStrAttr(pluginNode, "location").c_str());

            if (deviceName.find('.') != std::string::npos) {
                THROW_IE_EXCEPTION << "Device name must not contain dot '.' symbol";
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
        OV_ITT_SCOPED_TASK(itt::domains::IE, "Core::Impl::ReadNetwork from file");
        return details::ReadNetwork(modelPath, binPath, extensions);
    }

    CNNNetwork ReadNetwork(const std::string& model, const Blob::CPtr& weights) const override {
        OV_ITT_SCOPED_TASK(itt::domains::IE, "Core::Impl::ReadNetwork from memory");
        return details::ReadNetwork(model, weights, extensions);
    }

    ExecutableNetwork LoadNetwork(const CNNNetwork& network, const RemoteContext::Ptr& context,
                                  const std::map<std::string, std::string>& config) override {
        if (context == nullptr) {
            THROW_IE_EXCEPTION << "Remote context is nullptr";
        }

        return LoadNetworkImpl(network, context->getDeviceName(), config, context);
    }

    ExecutableNetwork LoadNetwork(const CNNNetwork& network, const std::string& deviceName,
                                  const std::map<std::string, std::string>& config) override {
        return LoadNetworkImpl(network, deviceName, config, nullptr);
    }

    ExecutableNetwork ImportNetwork(std::istream& networkModel, const std::string& deviceName,
                                    const std::map<std::string, std::string>& config) override {
        OV_ITT_SCOPED_TASK(itt::domains::IE, "Core::ImportNetwork");
        auto parsed = parseDeviceNameIntoConfig(deviceName, config);

        if (parsed._deviceName.empty()) {
            ExportMagic magic = {};
            auto currentPos = networkModel.tellg();
            networkModel.read(magic.data(), magic.size());
            auto exportedWithName = (exportMagic == magic);
            if (exportedWithName) {
                std::getline(networkModel, parsed._deviceName);
            }
            networkModel.seekg(currentPos, networkModel.beg);
        }

        return GetCPPPluginByName(parsed._deviceName).ImportNetwork(networkModel, parsed._config);
    }

    ExecutableNetwork ImportNetwork(std::istream& networkModel,
                                    const RemoteContext::Ptr& context,
                                    const std::map<std::string, std::string>& config) override {
        OV_ITT_SCOPED_TASK(itt::domains::IE, "Core::ImportNetwork");

        if (context == nullptr) {
            THROW_IE_EXCEPTION << "Remote context is null";
        }

        auto parsed = parseDeviceNameIntoConfig(context->getDeviceName(), config);
        return GetCPPPluginByName(parsed._deviceName).ImportNetwork(networkModel, context, parsed._config);
    }

    QueryNetworkResult QueryNetwork(const CNNNetwork& network, const std::string& deviceName,
                                    const std::map<std::string, std::string>& config) const override {
        OV_ITT_SCOPED_TASK(itt::domains::IE, "Core::QueryNetwork");
        auto parsed = parseDeviceNameIntoConfig(deviceName, config);
        return GetCPPPluginByName(parsed._deviceName).QueryNetwork(network, parsed._config);
    }

    Parameter GetMetric(const std::string& deviceName, const std::string& name) const override {
        // HETERO case
        {
            if (deviceName.find("HETERO:") == 0) {
                THROW_IE_EXCEPTION
                    << "You can get specific metrics with the GetMetric only for the HETERO itself (without devices). "
                       "To get individual devices's metrics call GetMetric for each device separately";
            }
        }

        // MULTI case
        {
            if (deviceName.find("MULTI:") == 0) {
                THROW_IE_EXCEPTION
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
     * @brief Returns reference to CPP plugin wrapper by a device name
     * @param deviceName A name of device
     * @return Reference to a CPP plugin wrapper
     */
    InferencePlugin GetCPPPluginByName(const std::string& deviceName) const {
        OV_ITT_SCOPED_TASK(itt::domains::IE_LT, "Core::Impl::GetCPPPluginByName");

        std::lock_guard<std::mutex> lock(pluginsMutex);

        auto it = pluginRegistry.find(deviceName);
        if (it == pluginRegistry.end()) {
            THROW_IE_EXCEPTION << "Device with \"" << deviceName << "\" name is not registered in the InferenceEngine";
        }

        // Plugin is in registry, but not created, let's create

        if (plugins.find(deviceName) == plugins.end()) {
            PluginDescriptor desc = it->second;

            try {
                InferencePlugin plugin(desc.libraryLocation);

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
                    allowNotImplemented([&]() {
                        plugin.SetConfig(desc.defaultConfig);
                    });

                    allowNotImplemented([&]() {
                        for (auto&& extensionLocation : desc.listOfExtentions) {
                            plugin.AddExtension(make_so_pointer<IExtension>(extensionLocation));
                        }
                    });
                }

                plugins[deviceName] = plugin;
            } catch (const details::InferenceEngineException& ex) {
                THROW_IE_EXCEPTION << "Failed to create plugin " << FileUtils::fromFilePath(desc.libraryLocation) << " for device " << deviceName
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
            THROW_IE_EXCEPTION << "Device with \"" << deviceName << "\" name is not registered in the InferenceEngine";
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
            THROW_IE_EXCEPTION << "Device with \"" << deviceName << "\"  is already registered in the InferenceEngine";
        }

        if (deviceName.find('.') != std::string::npos) {
            THROW_IE_EXCEPTION << "Device name must not contain dot '.' symbol";
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
     * @note  `deviceName` is not allowed in form of MULTI:CPU, HETERO:FPGA,CPU
     *        just simple forms like CPU, GPU, MULTU, GPU.0, etc
     */
    void SetConfigForPlugins(const std::map<std::string, std::string>& config, const std::string& deviceName) {
        std::lock_guard<std::mutex> lock(pluginsMutex);

        // set config for plugins in registry (not created plugins)
        bool configIsSet = false;
        for (auto & desc : pluginRegistry) {
            PluginDescriptor & pluginDesc = desc.second;
            if (deviceName.empty() || deviceName == desc.first) {
                // copy config since it's going to be modified
                auto configCopy = config;
                // extract common options to core config
                coreConfig[desc.first] = CoreConfig(configCopy, coreConfig[desc.first]);
                // the rest of the options are to device itself
                for (auto&& conf : configCopy) {
                    pluginDesc.defaultConfig[conf.first] = conf.second;
                }
                configIsSet = true;
            }
        }

        if (!configIsSet && !deviceName.empty()) {
            THROW_IE_EXCEPTION << "Device with \"" << deviceName << "\" name is not registered in the InferenceEngine";
        }

        // set config for already created plugins
        for (auto& plugin : plugins) {
            if (deviceName.empty() || deviceName == plugin.first) {
                allowNotImplemented([&]() {
                    // copy config since it's going to be modified
                    auto configCopy = config;
                    // extract common options to core config
                    coreConfig[plugin.first] = CoreConfig(configCopy, coreConfig[plugin.first]);
                    // the rest of the options are to device itself
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
                THROW_IE_EXCEPTION << "Cannot add opset with name: " << it.first << ". Opset with the same name already exists.";
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
    return _impl->LoadNetwork(network, deviceName, config);
}

ExecutableNetwork Core::LoadNetwork(const CNNNetwork& network, RemoteContext::Ptr context,
                                    const std::map<std::string, std::string>& config) {
    return _impl->LoadNetwork(network, context, config);
}

RemoteContext::Ptr Core::CreateContext(const std::string& deviceName, const ParamMap& params) {
    if (deviceName.find("HETERO") == 0) {
        THROW_IE_EXCEPTION << "HETERO device does not support remote context";
    }
    if (deviceName.find("MULTI") == 0) {
        THROW_IE_EXCEPTION << "MULTI device does not support remote context";
    }

    auto parsed = parseDeviceNameIntoConfig(deviceName, params);
    return _impl->GetCPPPluginByName(parsed._deviceName).CreateContext(parsed._config);
}

RemoteContext::Ptr Core::GetDefaultContext(const std::string& deviceName) {
    if (deviceName.find("HETERO") == 0) {
        THROW_IE_EXCEPTION << "HETERO device does not support remote context";
    }
    if (deviceName.find("MULTI") == 0) {
        THROW_IE_EXCEPTION << "MULTI device does not support remote context";
    }

    auto parsed = parseDeviceNameIntoConfig(deviceName, ParamMap());
    return _impl->GetCPPPluginByName(parsed._deviceName).GetDefaultContext(parsed._config);
}

void Core::AddExtension(IExtensionPtr extension, const std::string& deviceName_) {
    if (deviceName_.find("HETERO") == 0) {
        THROW_IE_EXCEPTION
            << "HETERO device does not support extensions. Please, set extensions directly to fallback devices";
    }
    if (deviceName_.find("MULTI") == 0) {
        THROW_IE_EXCEPTION
            << "MULTI device does not support extensions. Please, set extensions directly to fallback devices";
    }

    _impl->AddExtension(extension);
}

void Core::AddExtension(const IExtensionPtr& extension) {
    _impl->AddExtension(extension);
}

ExecutableNetwork Core::ImportNetwork(const std::string& modelFileName, const std::string& deviceName,
                                      const std::map<std::string, std::string>& config) {
    OV_ITT_SCOPED_TASK(itt::domains::IE, "Core::ImportNetwork");

    // TODO: remove once NotImplemented exception is deprecated and not used
    if (deviceName.find("HETERO") == 0) {
        THROW_IE_EXCEPTION << "HETERO device does not support ImportNetwork";
    }
    if (deviceName.find("MULTI") == 0) {
        THROW_IE_EXCEPTION << "MULTI device does not support ImportNetwork";
    }

    auto parsed = parseDeviceNameIntoConfig(deviceName, config);
    return _impl->GetCPPPluginByName(parsed._deviceName).ImportNetwork(modelFileName, parsed._config);
}

ExecutableNetwork Core::ImportNetwork(std::istream& networkModel, const std::string& deviceName,
                                      const std::map<std::string, std::string>& config) {
    return _impl->ImportNetwork(networkModel, deviceName, config);
}

ExecutableNetwork Core::ImportNetwork(std::istream& networkModel,
                                      const RemoteContext::Ptr& context,
                                      const std::map<std::string, std::string>& config) {
    return _impl->ImportNetwork(networkModel, context, config);
}

QueryNetworkResult Core::QueryNetwork(const CNNNetwork& network, const std::string& deviceName,
                                      const std::map<std::string, std::string>& config) const {
    return _impl->QueryNetwork(network, deviceName, config);
}

void Core::SetConfig(const std::map<std::string, std::string>& config, const std::string& deviceName) {
    // HETERO case
    if (deviceName.find("HETERO:") == 0) {
        THROW_IE_EXCEPTION << "SetConfig is supported only for HETERO itself (without devices). "
                                "You can configure the devices with SetConfig before creating the HETERO on top.";
    }

    // MULTI case
    if (deviceName.find("MULTI:") == 0) {
        THROW_IE_EXCEPTION << "SetConfig is supported only for MULTI itself (without devices). "
                                "You can configure the devices with SetConfig before creating the MULTI on top.";
    }

    // GPU.0, FPGA.1 cases
    if (deviceName.find(".") != std::string::npos) {
        THROW_IE_EXCEPTION << "SetConfig is supported only for device family itself (without particular device .#). "
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
            THROW_IE_EXCEPTION
                << "You can only GetConfig of the HETERO itself (without devices). "
                   "GetConfig is also possible for the individual devices before creating the HETERO on top.";
        }
    }
    // MULTI case
    {
        if (deviceName.find("MULTI:") == 0) {
            THROW_IE_EXCEPTION
                << "You can only GetConfig of the MULTI itself (without devices). "
                   "GetConfig is also possible for the individual devices before creating the MULTI on top.";
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
    std::vector<std::string> devices;

    std::string propertyName = METRIC_KEY(AVAILABLE_DEVICES);

    for (auto&& deviceName : _impl->GetListOfDevicesInRegistry()) {
        std::vector<std::string> devicesIDs;

        try {
            Parameter p = GetMetric(deviceName, propertyName);
            devicesIDs = p.as<std::vector<std::string>>();
        } catch (details::InferenceEngineException&) {
            // plugin is not created by e.g. invalid env
        } catch (const std::exception& ex) {
            THROW_IE_EXCEPTION << "An exception is thrown while trying to create the " << deviceName
                               << " device and call GetMetric: " << ex.what();
        } catch (...) {
            THROW_IE_EXCEPTION << "Unknown exception is thrown while trying to create the " << deviceName
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
