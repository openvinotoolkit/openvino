// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/core.hpp"

#include "any_copy.hpp"
#include "cnn_network_ngraph_impl.hpp"
#include "cpp/ie_plugin.hpp"
#include "dev/core_impl.hpp"
#include "ie_itt.hpp"
#include "so_extension.hpp"

#ifdef OPENVINO_STATIC_LIBRARY
#    include "ie_plugins.hpp"
#endif

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
