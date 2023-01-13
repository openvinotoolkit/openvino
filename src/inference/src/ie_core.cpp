// Copyright (C) 2018-2022 Intel Corporation
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
#include "dev/core_impl.hpp"
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
}  // namespace

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
    RegisterPlugins(ov::findPluginXML(xmlConfigFile));
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
    _impl->RegisterPluginsInRegistry(::getStaticPluginsRegistry());
#else
    register_plugins(findPluginXML(xmlConfigFile));
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

CompiledModel Core::compile_model(const std::string& model,
                                  const ov::Tensor& weights,
                                  const std::string& deviceName,
                                  const AnyMap& config) {
    InferenceEngine::Blob::Ptr blob;
    if (weights) {
        blob = weights._impl;
    }
    OV_CORE_CALL_STATEMENT({
        auto exec = _impl->LoadNetwork(model, blob, deviceName, any_copy(flatten_sub_properties(deviceName, config)));
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
        add_extension(ov::detail::load_extensions(library_path));
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
        add_extension(ov::detail::load_extensions(library_path));
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
