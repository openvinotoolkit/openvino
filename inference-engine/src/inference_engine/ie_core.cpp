// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_core.hpp"

#include <unordered_set>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <ngraph/opsets/opset.hpp>
#include "cpp/ie_cnn_net_reader.h"
#include "cpp/ie_plugin_cpp.hpp"
#include "cpp_interfaces/base/ie_plugin_base.hpp"
#include "details/ie_exception_conversion.hpp"
#include "details/ie_so_pointer.hpp"
#include "file_utils.h"
#include "ie_icore.hpp"
#include "ie_plugin.hpp"
#include "ie_plugin_config.hpp"
#include "ie_profiling.hpp"
#include "ie_util_internal.hpp"
#include "multi-device/multi_device_config.hpp"
#include "xml_parse_utils.h"

using namespace InferenceEngine::PluginConfigParams;

namespace InferenceEngine {

IE_SUPPRESS_DEPRECATED_START

namespace {

std::once_flag flag;
InferenceEngine::details::SharedObjectLoader::Ptr cnnReaderLoader;

InferenceEngine::details::SharedObjectLoader::Ptr createCnnReaderLoader() {
    std::call_once(flag, [&] () {
        FileUtils::FilePath libraryName = FileUtils::toFilePath(std::string("inference_engine_ir_readers") + std::string(IE_BUILD_POSTFIX));
        FileUtils::FilePath irReadersLibraryPath = FileUtils::makeSharedLibraryName(getInferenceEngineLibraryPath(), libraryName);

        if (!FileUtils::fileExist(irReadersLibraryPath)) {
            THROW_IE_EXCEPTION << "Please, make sure that Inference Engine IR readers library "
                << FileUtils::fromFilePath(::FileUtils::makeSharedLibraryName({}, libraryName)) << " is in "
                << getIELibraryPath();
        }
        cnnReaderLoader = std::shared_ptr<InferenceEngine::details::SharedObjectLoader>(
            new InferenceEngine::details::SharedObjectLoader(irReadersLibraryPath.c_str()));
    });

    return cnnReaderLoader;
}

IInferencePluginAPI* getInferencePluginAPIInterface(IInferencePlugin* iplugin) {
    return dynamic_cast<IInferencePluginAPI*>(iplugin);
}

IInferencePluginAPI* getInferencePluginAPIInterface(InferenceEnginePluginPtr iplugin) {
    return getInferencePluginAPIInterface(static_cast<IInferencePlugin*>(iplugin.operator->()));
}

IInferencePluginAPI* getInferencePluginAPIInterface(InferencePlugin plugin) {
    return getInferencePluginAPIInterface(static_cast<InferenceEnginePluginPtr>(plugin));
}

}  // namespace

CNNNetReaderPtr CreateCNNNetReaderPtr() noexcept {
    auto loader = createCnnReaderLoader();
    return CNNNetReaderPtr(loader);
}

IE_SUPPRESS_DEPRECATED_END

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

    IE_SUPPRESS_DEPRECATED_START
    mutable std::map<std::string, InferencePlugin> plugins;
    IE_SUPPRESS_DEPRECATED_END

    struct PluginDescriptor {
        FileUtils::FilePath libraryLocation;
        std::map<std::string, std::string> defaultConfig;
        std::vector<FileUtils::FilePath> listOfExtentions;
    };

    /**
     * Hold original blob in order to avoid situations when original blob is allocated on stack
     */
    class WeightsHolderBlob : public TBlob<uint8_t> {
        Blob::CPtr originBlob;

    public:
        explicit WeightsHolderBlob(const Blob::CPtr& weights) :
            TBlob<uint8_t>(weights->getTensorDesc(),
                           weights->cbuffer().as<uint8_t*>()),
            originBlob(weights) { }
    };

    std::unordered_set<std::string> opsetNames;
    std::vector<IExtensionPtr> extensions;

    std::map<std::string, PluginDescriptor> pluginRegistry;

public:
    Impl();
    ~Impl() override;

    /**
     * @brief Register plugins for devices which are located in .xml configuration file. The function supports UNICODE path
     * @param xmlConfigFile - an .xml configuraion with device / plugin information
     */
    void RegisterPluginsInRegistry(const std::string& xmlConfigFile) {
        auto parse_result = ParseXml(xmlConfigFile.c_str());
        if (!parse_result.error_msg.empty()) {
            THROW_IE_EXCEPTION << parse_result.error_msg;
        }

        pugi::xml_document& xmlDoc = *parse_result.xml;

        using namespace XMLParseUtils;
        pugi::xml_node ieNode = xmlDoc.document_element();
        pugi::xml_node devicesNode = ieNode.child("plugins");

        for (auto pluginNode = devicesNode.child("plugin"); !pluginNode.empty();
             pluginNode = pluginNode.next_sibling("plugin")) {
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
                for (auto propertyNode = propertiesNode.child("property"); !propertyNode.empty();
                     propertyNode = propertyNode.next_sibling("property")) {
                    std::string key = GetStrAttr(propertyNode, "key");
                    std::string value = GetStrAttr(propertyNode, "value");
                    config[key] = value;
                }
            }

            // check extensions
            auto extensionsNode = pluginNode.child("extensions");
            std::vector<FileUtils::FilePath> listOfExtentions;

            if (extensionsNode) {
                for (auto extensionNode = extensionsNode.child("extension"); !extensionNode.empty();
                     extensionNode = extensionNode.next_sibling("extension")) {
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
        IE_PROFILING_AUTO_SCOPE(Core::ReadNetwork)
        IE_SUPPRESS_DEPRECATED_START
        ResponseDesc desc;
        CNNNetReaderPtr cnnReader(createCnnReaderLoader());
        StatusCode rt = cnnReader->ReadNetwork(modelPath.c_str(), &desc);
        if (rt != OK) THROW_IE_EXCEPTION << desc.msg;
        if (cnnReader->getVersion(&desc) >= 10) {
            cnnReader->addExtensions(getExtensions());
        }
        std::string bPath = binPath;
        if (bPath.empty()) {
            bPath = modelPath;
            auto pos = bPath.rfind('.');
            if (pos != std::string::npos) bPath = bPath.substr(0, pos);
            bPath += ".bin";

            if (!FileUtils::fileExist(bPath)) bPath.clear();
        }

        if (!bPath.empty()) {
            rt = cnnReader->ReadWeights(bPath.c_str(), &desc);
            if (rt != OK) THROW_IE_EXCEPTION << desc.msg;
        } else {
            TBlob<uint8_t>::Ptr weights_ptr;
            rt = cnnReader->SetWeights(weights_ptr, &desc);
            if (rt != OK) THROW_IE_EXCEPTION << desc.msg;
        }
        IE_SUPPRESS_DEPRECATED_END

        return CNNNetwork(cnnReader);
    }

    CNNNetwork ReadNetwork(const std::string& model, const Blob::CPtr& weights) const override {
        IE_PROFILING_AUTO_SCOPE(Core::ReadNetwork)
        IE_SUPPRESS_DEPRECATED_START
        ResponseDesc desc;
        CNNNetReaderPtr cnnReader(createCnnReaderLoader());
        StatusCode rt = cnnReader->ReadNetwork(model.data(), model.length(), &desc);
        if (rt != OK) THROW_IE_EXCEPTION << desc.msg;
        if (cnnReader->getVersion(&desc) >= 10) {
            cnnReader->addExtensions(getExtensions());
        }
        TBlob<uint8_t>::Ptr weights_ptr;
        if (weights) {
            weights_ptr = std::make_shared<WeightsHolderBlob>(weights);
        }
        rt = cnnReader->SetWeights(weights_ptr, &desc);
        if (rt != OK) THROW_IE_EXCEPTION << desc.msg;
        IE_SUPPRESS_DEPRECATED_END

        return CNNNetwork(cnnReader);
    }

    IE_SUPPRESS_DEPRECATED_START

    /**
     * @brief Returns reference to plugin by a device name
     * @param deviceName - a name of device
     * @return Reference to a plugin
     */
    InferenceEnginePluginPtr GetPluginByName(const std::string& deviceName) const override {
        return static_cast<InferenceEnginePluginPtr>(GetCPPPluginByName(deviceName));
    }

    /**
     * @brief Returns reference to CPP plugin wrapper by a device name
     * @param deviceName - a name of device
     * @return Reference to a CPP plugin wrapper
     */
    InferencePlugin GetCPPPluginByName(const std::string& deviceName) const {
        IE_SUPPRESS_DEPRECATED_START

        auto it = pluginRegistry.find(deviceName);
        if (it == pluginRegistry.end()) {
            THROW_IE_EXCEPTION << "Device with \"" << deviceName << "\" name is not registered in the InferenceEngine";
        }

        // Plugin is in registry, but not created, let's create

        if (plugins.find(deviceName) == plugins.end()) {
            PluginDescriptor desc = it->second;

            try {
                InferenceEnginePluginPtr plugin(desc.libraryLocation);
                IInferencePlugin* pplugin = static_cast<IInferencePlugin*>(plugin.operator->());
                IInferencePluginAPI* iplugin_api_ptr = dynamic_cast<IInferencePluginAPI*>(pplugin);

                if (iplugin_api_ptr != nullptr) {
                    iplugin_api_ptr->SetName(deviceName);

                    // Set Inference Engine class reference to plugins
                    ICore* mutableCore = const_cast<ICore*>(static_cast<const ICore*>(this));
                    iplugin_api_ptr->SetCore(mutableCore);
                }

                // Add registered extensions to new plugin
                for (const auto& ext : extensions) {
                    plugin->AddExtension(ext, nullptr);
                }

                InferencePlugin cppPlugin(plugin);

                // configuring
                {
                    cppPlugin.SetConfig(desc.defaultConfig);

                    for (auto&& extensionLocation : desc.listOfExtentions) {
                        // TODO: fix once InferenceEngine::Extension can accept FileUtils::FilePath
                        // currently, extensions cannot be loaded using wide path
                        cppPlugin.AddExtension(make_so_pointer<IExtension>(FileUtils::fromFilePath(extensionLocation)));
                    }
                }

                plugins[deviceName] = cppPlugin;
            } catch (const details::InferenceEngineException& ex) {
                THROW_IE_EXCEPTION << "Failed to create plugin " << FileUtils::fromFilePath(desc.libraryLocation) << " for device " << deviceName
                                   << "\n"
                                   << "Please, check your environment\n"
                                   << ex.what() << "\n";
            }
        }

        IE_SUPPRESS_DEPRECATED_END

        return plugins[deviceName];
    }

    IE_SUPPRESS_DEPRECATED_END

    /**
     * @brief Unregisters plugin for specified device
     * @param deviceName - a name of device
     */
    void UnregisterPluginByName(const std::string& deviceName) {
        auto it = plugins.find(deviceName);
        if (it == plugins.end()) {
            THROW_IE_EXCEPTION << "Device with \"" << deviceName << "\" name is not registered in the InferenceEngine";
        }

        plugins.erase(deviceName);
    }

    /**
     * @brief Registers plugin in registry for specified device
     * @param deviceName - a name of device
     */
    void RegisterPluginByName(const std::string& pluginName, const std::string& deviceName) {
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
            pluginPath = FileUtils::makeSharedLibraryName({}, FileUtils::toFilePath(pluginName.c_str()));

            FileUtils::FilePath absFilePath = FileUtils::makePath(getInferenceEngineLibraryPath(), pluginPath);
            if (FileUtils::fileExist(absFilePath)) pluginPath = absFilePath;
        }

        PluginDescriptor desc = {pluginPath, {}, {}};
        pluginRegistry[deviceName] = desc;
    }

    std::vector<std::string> GetListOfDevicesInRegistry() const {
        std::vector<std::string> listOfDevices;
        for (auto&& pluginDesc : pluginRegistry) {
            listOfDevices.push_back(pluginDesc.first);
        }

        return listOfDevices;
    }

    void SetConfigForPlugins(const std::map<std::string, std::string>& config, const std::string& deviceName) {
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
            THROW_IE_EXCEPTION << "Device with \"" << deviceName << "\" name is not registered in the InferenceEngine";
        }

        // set config for already created plugins
        for (auto& plugin : plugins) {
            if (deviceName.empty() || deviceName == plugin.first) {
                IE_SUPPRESS_DEPRECATED_START
                plugin.second.SetConfig(config);
                IE_SUPPRESS_DEPRECATED_END
            }
        }
    }

    void addExtension(const IExtensionPtr& extension) {
        std::map<std::string, ngraph::OpSet> opsets = extension->getOpSets();
        for (const auto& it : opsets) {
            if (opsetNames.find(it.first) != opsetNames.end())
                THROW_IE_EXCEPTION << "Cannot add opset with name: " << it.first << ". Opset with the same name already exists.";
            opsetNames.insert(it.first);
        }

        for (auto& plugin : plugins) {
            IE_SUPPRESS_DEPRECATED_START
            try {
                plugin.second.AddExtension(extension);
            } catch (...) {}
            IE_SUPPRESS_DEPRECATED_END
        }
        extensions.emplace_back(extension);
    }

    const std::vector<IExtensionPtr>& getExtensions() const {
        return extensions;
    }
};

Core::Impl::Impl() {
    opsetNames.insert("opset1");
    opsetNames.insert("opset2");
    opsetNames.insert("opset3");
}

Core::Impl::~Impl() {}

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

        IE_SUPPRESS_DEPRECATED_START
        const Version* version = _impl->GetCPPPluginByName(deviceNameLocal).GetVersion();
        IE_SUPPRESS_DEPRECATED_END
        versions[deviceNameLocal] = *version;
    }

    return versions;
}

IE_SUPPRESS_DEPRECATED_START
void Core::SetLogCallback(IErrorListener&) const {
}
IE_SUPPRESS_DEPRECATED_END

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
}  //  namespace

CNNNetwork Core::ReadNetwork(const std::string& modelPath, const std::string& binPath) const {
    return _impl->ReadNetwork(modelPath, binPath);
}

CNNNetwork Core::ReadNetwork(const std::string& model, const Blob::CPtr& weights) const {
    return _impl->ReadNetwork(model, weights);
}

ExecutableNetwork Core::LoadNetwork(const CNNNetwork network, const std::string& deviceName,
                                    const std::map<std::string, std::string>& config) {
    IE_PROFILING_AUTO_SCOPE(Core::LoadNetwork)
    auto parsed = parseDeviceNameIntoConfig(deviceName, config);
    IE_SUPPRESS_DEPRECATED_START
    return _impl->GetCPPPluginByName(parsed._deviceName).LoadNetwork(network, parsed._config);
    IE_SUPPRESS_DEPRECATED_END
}

void Core::AddExtension(const IExtensionPtr& extension) {
    _impl->addExtension(extension);
}

ExecutableNetwork Core::LoadNetwork(const CNNNetwork network, RemoteContext::Ptr context,
                                    const std::map<std::string, std::string>& config) {
    IE_PROFILING_AUTO_SCOPE(Core::LoadNetwork)
    std::map<std::string, std::string> config_ = config;

    if (context == nullptr) {
        THROW_IE_EXCEPTION << "Remote context is null";
    }

    std::string deviceName_ = context->getDeviceName();
    DeviceIDParser device(deviceName_);
    std::string deviceName = device.getDeviceName();

    IE_SUPPRESS_DEPRECATED_START
    auto pluginAPIInterface = getInferencePluginAPIInterface(_impl->GetCPPPluginByName(deviceName));

    if (pluginAPIInterface == nullptr) {
        THROW_IE_EXCEPTION << deviceName << " does not implement the LoadNetwork method";
    }

    return pluginAPIInterface->LoadNetwork(network, config_, context);
    IE_SUPPRESS_DEPRECATED_END
}

RemoteContext::Ptr Core::CreateContext(const std::string& deviceName_, const ParamMap& params) {
    if (deviceName_.find("HETERO") == 0) {
        THROW_IE_EXCEPTION << "HETERO device does not support remote contexts";
    }
    if (deviceName_.find("MULTI") == 0) {
        THROW_IE_EXCEPTION << "MULTI device does not support remote contexts";
    }

    DeviceIDParser device(deviceName_);
    std::string deviceName = device.getDeviceName();

    IE_SUPPRESS_DEPRECATED_START
    auto pluginAPIInterface = getInferencePluginAPIInterface(_impl->GetCPPPluginByName(deviceName));

    if (pluginAPIInterface == nullptr) {
        THROW_IE_EXCEPTION << deviceName << " does not implement the CreateContext method";
    }

    return pluginAPIInterface->CreateContext(params);
    IE_SUPPRESS_DEPRECATED_END
}

RemoteContext::Ptr Core::GetDefaultContext(const std::string& deviceName_) {
    if (deviceName_.find("HETERO") == 0) {
        THROW_IE_EXCEPTION << "HETERO device does not support remote contexts";
    }
    if (deviceName_.find("MULTI") == 0) {
        THROW_IE_EXCEPTION << "MULTI device does not support remote contexts";
    }

    DeviceIDParser device(deviceName_);
    std::string deviceName = device.getDeviceName();

    IE_SUPPRESS_DEPRECATED_START
    auto pluginAPIInterface = getInferencePluginAPIInterface(_impl->GetCPPPluginByName(deviceName));

    if (pluginAPIInterface == nullptr) {
        THROW_IE_EXCEPTION << deviceName << " does not implement the CreateContext method";
    }

    return pluginAPIInterface->GetDefaultContext();
    IE_SUPPRESS_DEPRECATED_END
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

    DeviceIDParser parser(deviceName_);
    std::string deviceName = parser.getDeviceName();

    IE_SUPPRESS_DEPRECATED_START
    _impl->GetCPPPluginByName(deviceName).AddExtension(extension);
    _impl->addExtension(extension);
    IE_SUPPRESS_DEPRECATED_END
}

ExecutableNetwork Core::ImportNetwork(const std::string& modelFileName, const std::string& deviceName,
                                      const std::map<std::string, std::string>& config) {
    if (deviceName.find("HETERO") == 0) {
        THROW_IE_EXCEPTION << "HETERO device does not support ImportNetwork";
    }
    if (deviceName.find("MULTI") == 0) {
        THROW_IE_EXCEPTION << "MULTI device does not support ImportNetwork";
    }

    auto parsed = parseDeviceNameIntoConfig(deviceName, config);

    IE_SUPPRESS_DEPRECATED_START
    return _impl->GetCPPPluginByName(parsed._deviceName).ImportNetwork(modelFileName, parsed._config);
    IE_SUPPRESS_DEPRECATED_END
}

IE_SUPPRESS_DEPRECATED_START

ExecutableNetwork Core::ImportNetwork(std::istream& networkModel, const std::string& deviceName,
                                      const std::map<std::string, std::string>& config) {
    auto parsed = parseDeviceNameIntoConfig(deviceName, config);

    if (parsed._deviceName.empty()) {
        ExportMagic magic = {};
        networkModel.read(magic.data(), magic.size());
        auto exportedWithName = (exportMagic == magic);
        if (exportedWithName) {
            std::getline(networkModel, parsed._deviceName);
        }
        networkModel.seekg(0, networkModel.beg);
    }

    auto pluginAPIInterface = getInferencePluginAPIInterface(_impl->GetCPPPluginByName(parsed._deviceName));
    if (pluginAPIInterface == nullptr) {
        THROW_IE_EXCEPTION << parsed._deviceName << " does not implement the ImportNetwork method";
    }

    return pluginAPIInterface->ImportNetwork(networkModel, parsed._config);
}

IE_SUPPRESS_DEPRECATED_END

ExecutableNetwork Core::ImportNetwork(std::istream& networkModel,
                                      const RemoteContext::Ptr& context,
                                      const std::map<std::string, std::string>& config) {
    IE_PROFILING_AUTO_SCOPE(Core::ImportNetwork)

    if (context == nullptr) {
        THROW_IE_EXCEPTION << "Remote context is null";
    }

    std::string deviceName_ = context->getDeviceName();
    DeviceIDParser device(deviceName_);
    std::string deviceName = device.getDeviceName();

    auto parsed = parseDeviceNameIntoConfig(deviceName, config);

    IE_SUPPRESS_DEPRECATED_START
    auto pluginAPIInterface = getInferencePluginAPIInterface(_impl->GetCPPPluginByName(parsed._deviceName));

    if (pluginAPIInterface == nullptr) {
        THROW_IE_EXCEPTION << deviceName << " does not implement the ImportNetwork method";
    }
    IE_SUPPRESS_DEPRECATED_END
    return pluginAPIInterface->ImportNetwork(networkModel, context, parsed._config);
}

QueryNetworkResult Core::QueryNetwork(const ICNNNetwork& network, const std::string& deviceName,
                                      const std::map<std::string, std::string>& config) const {
    QueryNetworkResult res;
    auto parsed = parseDeviceNameIntoConfig(deviceName, config);
    IE_SUPPRESS_DEPRECATED_START
    _impl->GetCPPPluginByName(parsed._deviceName).QueryNetwork(network, parsed._config, res);
    IE_SUPPRESS_DEPRECATED_END
    return res;
}

void Core::SetConfig(const std::map<std::string, std::string>& config, const std::string& deviceName) {
    // HETERO case
    {
        if (deviceName.find("HETERO:") == 0) {
            THROW_IE_EXCEPTION << "SetConfig is supported only for HETERO itself (without devices). "
                                  "You can configure the devices with SetConfig before creating the HETERO on top.";
        }
    }

    // MULTI case
    {
        if (deviceName.find("MULTI:") == 0) {
            THROW_IE_EXCEPTION << "SetConfig is supported only for MULTI itself (without devices). "
                                  "You can configure the devices with SetConfig before creating the MULTI on top.";
        }
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
    IE_SUPPRESS_DEPRECATED_START
    auto pluginAPIInterface = getInferencePluginAPIInterface(_impl->GetCPPPluginByName(parsed._deviceName));
    IE_SUPPRESS_DEPRECATED_END
    if (pluginAPIInterface == nullptr) {
        THROW_IE_EXCEPTION << parsed._deviceName << " does not implement the GetConfig method";
    }
    return pluginAPIInterface->GetConfig(name, parsed._config);
}

Parameter Core::GetMetric(const std::string& deviceName, const std::string& name) const {
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
    IE_SUPPRESS_DEPRECATED_START
    auto pluginAPIInterface = getInferencePluginAPIInterface(_impl->GetCPPPluginByName(parsed._deviceName));
    IE_SUPPRESS_DEPRECATED_END
    if (pluginAPIInterface == nullptr) {
        THROW_IE_EXCEPTION << parsed._deviceName << " does not implement the GetMetric method";
    }

    return pluginAPIInterface->GetMetric(name, parsed._config);
}

std::vector<std::string> Core::GetAvailableDevices() const {
    std::vector<std::string> devices;

    std::string propertyName = METRIC_KEY(AVAILABLE_DEVICES);

    for (auto&& deviceName : _impl->GetListOfDevicesInRegistry()) {
        Parameter p;
        std::vector<std::string> devicesIDs;

        try {
            p = GetMetric(deviceName, propertyName);
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

    _impl->UnregisterPluginByName(deviceName);
}

}  // namespace InferenceEngine
