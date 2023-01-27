// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/core.hpp"

#include "any_copy.hpp"
#include "cnn_network_ngraph_impl.hpp"
#include "dev/converter_utils.hpp"
#include "dev/core_impl.hpp"
#include "ie_itt.hpp"
#include "so_extension.hpp"

#ifdef OPENVINO_STATIC_LIBRARY
#    include "ie_plugins.hpp"
#endif

namespace {
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

}  // namespace

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

Core::Core(const std::string& xml_config_file) {
    _impl = std::make_shared<Impl>();

#ifdef OPENVINO_STATIC_LIBRARY
    OV_CORE_CALL_STATEMENT(_impl->register_plugins_in_registry(::getStaticPluginsRegistry());)
#else
    OV_CORE_CALL_STATEMENT(
        // If XML is default, load default plugins by absolute paths
        _impl->register_plugins_in_registry(findPluginXML(xml_config_file), xml_config_file.empty());)
#endif
}

std::map<std::string, Version> Core::get_versions(const std::string& device_name) const {
    OV_CORE_CALL_STATEMENT({
        std::map<std::string, Version> versions;
        for (auto&& kvp : _impl->GetVersions(device_name)) {
            versions[kvp.first] = Version{kvp.second.buildNumber, kvp.second.description};
        }
        return versions;
    })
}
#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
std::shared_ptr<ov::Model> Core::read_model(const std::wstring& model_path, const std::wstring& bin_path) const {
    OV_CORE_CALL_STATEMENT(
        return _impl->read_model(ov::util::wstring_to_string(model_path), ov::util::wstring_to_string(bin_path)););
}
#endif

std::shared_ptr<ov::Model> Core::read_model(const std::string& model_path, const std::string& bin_path) const {
    OV_CORE_CALL_STATEMENT(return _impl->read_model(model_path, bin_path););
}

std::shared_ptr<ov::Model> Core::read_model(const std::string& model, const ov::Tensor& weights) const {
    OV_CORE_CALL_STATEMENT(return _impl->read_model(model, weights););
}

CompiledModel Core::compile_model(const std::shared_ptr<const ov::Model>& model, const AnyMap& config) {
    return compile_model(model, ov::DEFAULT_DEVICE_NAME, config);
}

CompiledModel Core::compile_model(const std::shared_ptr<const ov::Model>& model,
                                  const std::string& device_name,
                                  const AnyMap& config) {
    OV_CORE_CALL_STATEMENT({
        auto exec = _impl->compile_model(model, device_name, flatten_sub_properties(device_name, config));
        return {exec._ptr, exec._so};
    });
}

CompiledModel Core::compile_model(const std::string& model_path, const AnyMap& config) {
    return compile_model(model_path, ov::DEFAULT_DEVICE_NAME, config);
}

CompiledModel Core::compile_model(const std::string& model_path, const std::string& device_name, const AnyMap& config) {
    OV_CORE_CALL_STATEMENT({
        auto exec = _impl->compile_model(model_path, device_name, flatten_sub_properties(device_name, config));
        return {exec._ptr, exec._so};
    });
}

CompiledModel Core::compile_model(const std::string& model,
                                  const ov::Tensor& weights,
                                  const std::string& device_name,
                                  const AnyMap& config) {
    OV_CORE_CALL_STATEMENT({
        auto exec = _impl->compile_model(model, weights, device_name, flatten_sub_properties(device_name, config));
        return {exec._ptr, exec._so};
    });
}

CompiledModel Core::compile_model(const std::shared_ptr<const ov::Model>& model,
                                  const RemoteContext& context,
                                  const AnyMap& config) {
    OV_CORE_CALL_STATEMENT({
        auto exec = _impl->compile_model(model, context, flatten_sub_properties(context.get_device_name(), config));
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
    OV_CORE_CALL_STATEMENT({ _impl->add_extension(extensions); });
}

CompiledModel Core::import_model(std::istream& modelStream, const std::string& device_name, const AnyMap& config) {
    OV_ITT_SCOPED_TASK(ov::itt::domains::IE, "Core::import_model");
    OV_CORE_CALL_STATEMENT({
        auto exec = _impl->import_model(modelStream, device_name, flatten_sub_properties(device_name, config));
        return {exec._ptr, exec._so};
    });
}

CompiledModel Core::import_model(std::istream& modelStream, const RemoteContext& context, const AnyMap& config) {
    OV_ITT_SCOPED_TASK(ov::itt::domains::IE, "Core::import_model");

    auto parsed = parseDeviceNameIntoConfig(context.get_device_name(), config);
    OV_CORE_CALL_STATEMENT({
        auto exec = _impl->get_plugin(parsed._deviceName).import_model(modelStream, context, parsed._config);
        return {exec._ptr, exec._so};
    });
}

SupportedOpsMap Core::query_model(const std::shared_ptr<const ov::Model>& model,
                                  const std::string& device_name,
                                  const AnyMap& config) const {
    OV_CORE_CALL_STATEMENT(return _impl->query_model(model, device_name, flatten_sub_properties(device_name, config)););
}

void Core::set_property(const AnyMap& properties) {
    OV_CORE_CALL_STATEMENT(return _impl->set_property({}, properties););
}

void Core::set_property(const std::string& device_name, const AnyMap& properties) {
    OV_CORE_CALL_STATEMENT(return _impl->set_property(device_name, properties););
}

Any Core::get_property(const std::string& device_name, const std::string& name) const {
    OV_CORE_CALL_STATEMENT(return _impl->get_property(device_name, name, {}););
}

Any Core::get_property(const std::string& device_name, const std::string& name, const AnyMap& arguments) const {
    OV_CORE_CALL_STATEMENT(return _impl->get_property(device_name, name, arguments););
}

std::vector<std::string> Core::get_available_devices() const {
    OV_CORE_CALL_STATEMENT(return _impl->GetAvailableDevices(););
}

void Core::register_plugin(const std::string& plugin, const std::string& device_name) {
    OV_CORE_CALL_STATEMENT(_impl->register_plugin(plugin, device_name););
}

void Core::unload_plugin(const std::string& device_name) {
    OV_CORE_CALL_STATEMENT({
        ie::DeviceIDParser parser(device_name);
        std::string devName = parser.getDeviceName();

        _impl->unload_plugin(devName);
    });
}

void Core::register_plugins(const std::string& xml_config_file) {
    OV_CORE_CALL_STATEMENT(_impl->register_plugins_in_registry(xml_config_file););
}

RemoteContext Core::create_context(const std::string& device_name, const AnyMap& params) {
    OPENVINO_ASSERT(device_name.find("HETERO") != 0, "HETERO device does not support remote context");
    OPENVINO_ASSERT(device_name.find("MULTI") != 0, "MULTI device does not support remote context");
    OPENVINO_ASSERT(device_name.find("AUTO") != 0, "AUTO device does not support remote context");
    OPENVINO_ASSERT(device_name.find("BATCH") != 0, "BATCH device does not support remote context");

    OV_CORE_CALL_STATEMENT({
        auto parsed = parseDeviceNameIntoConfig(device_name, flatten_sub_properties(device_name, params));
        auto remoteContext = _impl->get_plugin(parsed._deviceName).create_context(parsed._config);
        return {remoteContext._impl, {remoteContext._so}};
    });
}

RemoteContext Core::get_default_context(const std::string& device_name) {
    OPENVINO_ASSERT(device_name.find("HETERO") != 0, "HETERO device does not support default remote context");
    OPENVINO_ASSERT(device_name.find("MULTI") != 0, "MULTI device does not support default remote context");
    OPENVINO_ASSERT(device_name.find("AUTO") != 0, "AUTO device does not support default remote context");
    OPENVINO_ASSERT(device_name.find("BATCH") != 0, "BATCH device does not support default remote context");

    OV_CORE_CALL_STATEMENT({
        auto parsed = parseDeviceNameIntoConfig(device_name, AnyMap{});
        auto remoteContext = _impl->get_plugin(parsed._deviceName).get_default_context(parsed._config);
        return {remoteContext._impl, {remoteContext._so}};
    });
}

}  // namespace ov
