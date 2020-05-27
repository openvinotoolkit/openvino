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
#include <streambuf>
#include <string>
#include <utility>
#include <vector>
#include <istream>
#include <mutex>

#include "ie_blob_stream.hpp"
#include <ie_reader_ptr.hpp>
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
        FileUtils::FilePath libraryName = FileUtils::toFilePath(std::string("inference_engine_ir_reader") + std::string(IE_BUILD_POSTFIX));
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

}  // namespace

class Reader: public IReader {
private:
    InferenceEngine::IReaderPtr ptr;
    std::once_flag readFlag;
    std::string name;
    std::string location;

    InferenceEngine::IReaderPtr getReaderPtr() {
        std::call_once(readFlag, [&] () {
            FileUtils::FilePath libraryName = FileUtils::toFilePath(location);
            FileUtils::FilePath readersLibraryPath = FileUtils::makeSharedLibraryName(getInferenceEngineLibraryPath(), libraryName);

            if (!FileUtils::fileExist(readersLibraryPath)) {
                THROW_IE_EXCEPTION << "Please, make sure that Inference Engine ONNX reader library "
                    << FileUtils::fromFilePath(::FileUtils::makeSharedLibraryName({}, libraryName)) << " is in "
                    << getIELibraryPath();
            }
            ptr = IReaderPtr(readersLibraryPath);
        });

        return ptr;
    }

    InferenceEngine::IReaderPtr getReaderPtr() const {
        return const_cast<Reader*>(this)->getReaderPtr();
    }

    void Release() noexcept override {
        delete this;
    }

public:
    using Ptr = std::shared_ptr<Reader>;
    Reader(const std::string& name, const std::string location): name(name), location(location) {}
    bool supportModel(std::istream& model) const override {
        auto reader = getReaderPtr();
        return reader->supportModel(model);
    }
    CNNNetwork read(std::istream& model, const std::vector<IExtensionPtr>& exts) const override {
        auto reader = getReaderPtr();
        return reader->read(model, exts);
    }
    CNNNetwork read(std::istream& model, std::istream& weights, const std::vector<IExtensionPtr>& exts) const override {
        auto reader = getReaderPtr();
        return reader->read(model, weights, exts);
    }
    std::vector<std::string> getDataFileExtensions() const override {
        auto reader = getReaderPtr();
        return reader->getDataFileExtensions();
    }
    std::string getName() const {
        return name;
    }
};

namespace {

// Extension to plugins creator
std::multimap<std::string, Reader::Ptr> readers;

void registerReaders() {
    static std::mutex readerMutex;
    std::lock_guard<std::mutex> lock(readerMutex);
    // TODO: Read readers info from XML
    auto onnxReader = std::make_shared<Reader>("ONNX", std::string("inference_engine_onnx_reader") + std::string(IE_BUILD_POSTFIX));
    readers.emplace("onnx", onnxReader);
    readers.emplace("prototxt", onnxReader);
    auto irReader = std::make_shared<Reader>("IR", std::string("inference_engine_ir_reader") + std::string(IE_BUILD_POSTFIX));
    readers.emplace("xml", irReader);
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

    std::unordered_set<std::string> opsetNames;
    std::vector<IExtensionPtr> extensions;

    std::map<std::string, PluginDescriptor> pluginRegistry;
    mutable std::mutex pluginsMutex;  // to lock parallel access to pluginRegistry and plugins

public:
    Impl();
    ~Impl() override;

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

        std::ifstream modelStream(modelPath, std::ios::binary);
        if (!modelStream.is_open())
            THROW_IE_EXCEPTION << "Model file " << modelPath << " cannot be opened!";

        auto fileExt = modelPath.substr(modelPath.find_last_of(".") + 1);
        for (auto it = readers.lower_bound(fileExt); it != readers.upper_bound(fileExt); it++) {
            auto reader = it->second;
            if (reader->supportModel(modelStream)) {
                // Find weights
                std::string bPath = binPath;
                if (bPath.empty()) {
                    auto pathWoExt = modelPath;
                    auto pos = modelPath.rfind('.');
                    if (pos != std::string::npos) pathWoExt = modelPath.substr(0, pos);
                    for (const auto& ext : reader->getDataFileExtensions()) {
                        bPath = pathWoExt + "." + ext;
                        if (!FileUtils::fileExist(bPath)) {
                            bPath.clear();
                        } else {
                            break;
                        }
                    }
                }
                if (!bPath.empty()) {
                    std::ifstream binStream;
                    binStream.open(bPath, std::ios::binary);
                    if (!binStream.is_open())
                        THROW_IE_EXCEPTION << "Weights file " << bPath << " cannot be opened!";
                    return reader->read(modelStream, binStream, extensions);
                }
                return reader->read(modelStream, extensions);
            }
        }
        THROW_IE_EXCEPTION << "Unknown model format! Cannot read the model: " << modelPath;
    }

    CNNNetwork ReadNetwork(const std::string& model, const Blob::CPtr& weights) const override {
        IE_PROFILING_AUTO_SCOPE(Core::ReadNetwork)
        std::istringstream modelStream(model);
        details::BlobStream binStream(weights);

        for (auto it = readers.begin(); it != readers.end(); it++) {
            auto reader = it->second;
            if (reader->supportModel(modelStream)) {
                if (weights)
                    return reader->read(modelStream, binStream, extensions);
                return reader->read(modelStream, extensions);
            }
        }
        THROW_IE_EXCEPTION << "Unknown model format! Cannot read the model from string!";
    }

    ExecutableNetwork LoadNetwork(const CNNNetwork& network, const std::string& deviceName,
                                  const std::map<std::string, std::string>& config) override {
        IE_PROFILING_AUTO_SCOPE(Core::LoadNetwork)
        auto parsed = parseDeviceNameIntoConfig(deviceName, config);
        IE_SUPPRESS_DEPRECATED_START
        return GetCPPPluginByName(parsed._deviceName).LoadNetwork(network, parsed._config);
        IE_SUPPRESS_DEPRECATED_END
    }

    IE_SUPPRESS_DEPRECATED_START

    ExecutableNetwork ImportNetwork(std::istream& networkModel, const std::string& deviceName,
                                    const std::map<std::string, std::string>& config) override {
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

        auto cppPlugin = GetCPPPluginByName(parsed._deviceName);
        auto pluginAPIInterface = getInferencePluginAPIInterface(cppPlugin);
        if (pluginAPIInterface == nullptr) {
            THROW_IE_EXCEPTION << parsed._deviceName << " does not implement the ImportNetwork method";
        }

        return pluginAPIInterface->ImportNetwork(networkModel, parsed._config);
    }

    QueryNetworkResult QueryNetwork(const ICNNNetwork& network, const std::string& deviceName,
                                    const std::map<std::string, std::string>& config) const override {
        QueryNetworkResult res;
        auto parsed = parseDeviceNameIntoConfig(deviceName, config);
        IE_SUPPRESS_DEPRECATED_START
        GetCPPPluginByName(parsed._deviceName).QueryNetwork(network, parsed._config, res);
        IE_SUPPRESS_DEPRECATED_END
        return res;
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
        IE_SUPPRESS_DEPRECATED_START
        InferencePlugin cppPlugin = GetCPPPluginByName(parsed._deviceName);
        auto pluginAPIInterface = getInferencePluginAPIInterface(cppPlugin);
        IE_SUPPRESS_DEPRECATED_END

        if (pluginAPIInterface == nullptr) {
            THROW_IE_EXCEPTION << parsed._deviceName << " does not implement the GetMetric method";
        }

        // we need to return a copy of Parameter object which is created on Core side,
        // not in InferenceEngine plugin side, which can be unloaded from Core in a parallel thread
        // TODO: remove this WA after *-31417 is resolved
        return copyParameterValue(pluginAPIInterface->GetMetric(name, parsed._config));
    }

    /**
     * @deprecated
     * @brief Returns reference to CPP plugin wrapper by a device name
     * @param deviceName A name of device
     * @return Reference to a CPP plugin wrapper
     */
    InferencePlugin GetCPPPluginByName(const std::string& deviceName) const {
        std::lock_guard<std::mutex> lock(pluginsMutex);

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
            pluginPath = FileUtils::makeSharedLibraryName({}, FileUtils::toFilePath(pluginName.c_str()));

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
     */
    void SetConfigForPlugins(const std::map<std::string, std::string>& config, const std::string& deviceName) {
        std::lock_guard<std::mutex> lock(pluginsMutex);

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
            IE_SUPPRESS_DEPRECATED_START
            try {
                plugin.second.AddExtension(extension);
            } catch (...) {}
            IE_SUPPRESS_DEPRECATED_END
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

Core::Impl::Impl() {
    opsetNames.insert("opset1");
    opsetNames.insert("opset2");
    opsetNames.insert("opset3");
    registerReaders();
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
        InferenceEngine::InferencePlugin cppPlugin = _impl->GetCPPPluginByName(deviceNameLocal);
        const Version * version = cppPlugin.GetVersion();
        IE_SUPPRESS_DEPRECATED_END
        versions[deviceNameLocal] = *version;
    }

    return versions;
}

IE_SUPPRESS_DEPRECATED_START
void Core::SetLogCallback(IErrorListener&) const {
}
IE_SUPPRESS_DEPRECATED_END

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

void Core::AddExtension(const IExtensionPtr& extension) {
    _impl->AddExtension(extension);
}

ExecutableNetwork Core::LoadNetwork(const CNNNetwork& network, RemoteContext::Ptr context,
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
    auto cppPlugin = _impl->GetCPPPluginByName(deviceName);
    auto pluginAPIInterface = getInferencePluginAPIInterface(cppPlugin);

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
    auto cppPlugin = _impl->GetCPPPluginByName(deviceName);
    auto pluginAPIInterface = getInferencePluginAPIInterface(cppPlugin);

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
    auto cppPlugin = _impl->GetCPPPluginByName(deviceName);
    auto pluginAPIInterface = getInferencePluginAPIInterface(cppPlugin);

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

    _impl->AddExtension(extension);
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

ExecutableNetwork Core::ImportNetwork(std::istream& networkModel, const std::string& deviceName,
                                      const std::map<std::string, std::string>& config) {
    return _impl->ImportNetwork(networkModel, deviceName, config);
}

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
    auto cppPlugin = _impl->GetCPPPluginByName(deviceName);
    auto pluginAPIInterface = getInferencePluginAPIInterface(cppPlugin);

    if (pluginAPIInterface == nullptr) {
        THROW_IE_EXCEPTION << deviceName << " does not implement the ImportNetwork method";
    }
    IE_SUPPRESS_DEPRECATED_END
    return pluginAPIInterface->ImportNetwork(networkModel, context, parsed._config);
}

QueryNetworkResult Core::QueryNetwork(const ICNNNetwork& network, const std::string& deviceName,
                                      const std::map<std::string, std::string>& config) const {
    return _impl->QueryNetwork(network, deviceName, config);
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
    auto cppPlugin = _impl->GetCPPPluginByName(parsed._deviceName);
    auto pluginAPIInterface = getInferencePluginAPIInterface(cppPlugin);
    IE_SUPPRESS_DEPRECATED_END

    if (pluginAPIInterface == nullptr) {
        THROW_IE_EXCEPTION << parsed._deviceName << " does not implement the GetConfig method";
    }

    // we need to return a copy of Parameter object which is created on Core side,
    // not in InferenceEngine plugin side, which can be unloaded from Core in a parallel thread
    // TODO: remove this WA after *-31417 is resolved
    return copyParameterValue(pluginAPIInterface->GetConfig(name, parsed._config));
}

Parameter Core::GetMetric(const std::string& deviceName, const std::string& name) const {
    return _impl->GetMetric(deviceName, name);
}

std::vector<std::string> Core::GetAvailableDevices() const {
    std::vector<std::string> devices;

    std::string propertyName = METRIC_KEY(AVAILABLE_DEVICES);

    for (auto&& deviceName : _impl->GetListOfDevicesInRegistry()) {
        std::vector<std::string> devicesIDs;

        IE_SUPPRESS_DEPRECATED_START
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
        IE_SUPPRESS_DEPRECATED_END

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
