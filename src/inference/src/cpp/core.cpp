// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/core.hpp"

#include "dev/core_impl.hpp"
#include "itt.hpp"
#include "openvino/core/so_extension.hpp"
#include "openvino/frontend/manager.hpp"
#include "openvino/runtime/device_id_parser.hpp"
#include "openvino/runtime/iremote_context.hpp"
#include "openvino/util/file_util.hpp"

namespace ov {

namespace {

std::string find_plugins_xml(const std::string& xml_file) {
    std::string xml_file_name = xml_file;
    if (xml_file_name.empty()) {
        // Default plugin xml file name, will search in OV folder.
        xml_file_name = "plugins.xml";
    } else {
        // If file path contains file separator, return file path;
        // Otherwise search it in OV folder with no restriction on file name and extension.
        if (xml_file_name.find(util::FileTraits<char>().file_separator) != xml_file_name.npos) {
            return xml_file_name;
        }
    }
    const auto ov_library_path = ov::util::get_ov_lib_path();

    // plugins xml can be found in either:
    // 1. openvino-X.Y.Z relative to libopenvino.so folder
    std::ostringstream str;
    str << "openvino-" << OPENVINO_VERSION_MAJOR << "." << OPENVINO_VERSION_MINOR << "." << OPENVINO_VERSION_PATCH;
    const auto sub_folder = str.str();

    // register plugins from default openvino-<openvino version>/plugins.xml config
    auto xmlConfigFileDefault = ov::util::path_join({ov_library_path, sub_folder, xml_file_name});
    if (ov::util::file_exists(xmlConfigFileDefault))
        return xmlConfigFileDefault;

    // 2. in folder with libopenvino.so
    xmlConfigFileDefault = ov::util::path_join({std::move(ov_library_path), std::move(xml_file_name)});
    if (ov::util::file_exists(xmlConfigFileDefault))
        return xmlConfigFileDefault;

    return xml_file;
}

}  // namespace

#define OV_CORE_CALL_STATEMENT(...)             \
    try {                                       \
        __VA_ARGS__;                            \
    } catch (const std::exception& ex) {        \
        OPENVINO_THROW(ex.what());              \
    } catch (...) {                             \
        OPENVINO_THROW("Unexpected exception"); \
    }

class Core::Impl : public CoreImpl {
public:
    Impl() : ov::CoreImpl() {}
};

Core::Core(const std::string& xml_config_file) {
    _impl = std::make_shared<Impl>();

    std::string xmlConfigFile = find_plugins_xml(xml_config_file);
    if (!xmlConfigFile.empty())
        OV_CORE_CALL_STATEMENT(
            // If XML is default, load default plugins by absolute paths
            _impl->register_plugins_in_registry(xmlConfigFile, xml_config_file.empty());)
    // Load plugins from the pre-compiled list
    OV_CORE_CALL_STATEMENT(_impl->register_compile_time_plugins();)
}

std::map<std::string, Version> Core::get_versions(const std::string& device_name) const {
    OV_CORE_CALL_STATEMENT({ return _impl->get_versions(device_name); })}
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
        auto exec = _impl->compile_model(model, device_name, config);
        return {exec._ptr, exec._so};
    });
}

CompiledModel Core::compile_model(const std::string& model_path, const AnyMap& config) {
    return compile_model(model_path, ov::DEFAULT_DEVICE_NAME, config);
}

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
CompiledModel Core::compile_model(const std::wstring& model_path, const AnyMap& config) {
    return compile_model(ov::util::wstring_to_string(model_path), config);
}
#endif

CompiledModel Core::compile_model(const std::string& model_path, const std::string& device_name, const AnyMap& config) {
    OV_CORE_CALL_STATEMENT({
        auto exec = _impl->compile_model(model_path, device_name, config);
        return {exec._ptr, exec._so};
    });
}

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
CompiledModel Core::compile_model(const std::wstring& model_path,
                                  const std::string& device_name,
                                  const AnyMap& config) {
    return compile_model(ov::util::wstring_to_string(model_path), device_name, config);
}
#endif

CompiledModel Core::compile_model(const std::string& model,
                                  const ov::Tensor& weights,
                                  const std::string& device_name,
                                  const AnyMap& config) {
    OV_CORE_CALL_STATEMENT({
        auto exec = _impl->compile_model(model, weights, device_name, config);
        return {exec._ptr, exec._so};
    });
}

CompiledModel Core::compile_model(const std::shared_ptr<const ov::Model>& model,
                                  const RemoteContext& context,
                                  const AnyMap& config) {
    OV_CORE_CALL_STATEMENT({
        auto exec = _impl->compile_model(model, ov::SoPtr<ov::IRemoteContext>{context._impl, context._so}, config);
        return {exec._ptr, exec._so};
    });
}

void Core::add_extension(const std::string& library_path) {
    try {
        add_extension(ov::detail::load_extensions(library_path));
    } catch (const std::runtime_error& e) {
        OPENVINO_THROW(
            std::string(
                "Cannot add extension. Cannot find entry point to the extension library. This error happened: ") +
            e.what());
    }
}

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
void Core::add_extension(const std::wstring& library_path) {
    try {
        add_extension(ov::detail::load_extensions(library_path));
    } catch (const std::runtime_error&) {
        OPENVINO_THROW("Cannot add extension. Cannot find entry point to the extension library");
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
    OV_ITT_SCOPED_TASK(ov::itt::domains::OV, "Core::import_model");
    OV_CORE_CALL_STATEMENT({
        auto exec = _impl->import_model(modelStream, device_name, config);
        return {exec._ptr, exec._so};
    });
}

CompiledModel Core::import_model(std::istream& modelStream, const RemoteContext& context, const AnyMap& config) {
    OV_ITT_SCOPED_TASK(ov::itt::domains::OV, "Core::import_model");

    OV_CORE_CALL_STATEMENT({
        auto exec = _impl->import_model(modelStream, ov::SoPtr<ov::IRemoteContext>{context._impl, context._so}, config);
        return {exec._ptr, exec._so};
    });
}

SupportedOpsMap Core::query_model(const std::shared_ptr<const ov::Model>& model,
                                  const std::string& device_name,
                                  const AnyMap& config) const {
    OV_CORE_CALL_STATEMENT(return _impl->query_model(model, device_name, config););
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
    OV_CORE_CALL_STATEMENT(return _impl->get_available_devices(););
}

void Core::register_plugin(const std::string& plugin, const std::string& device_name, const ov::AnyMap& properties) {
    OV_CORE_CALL_STATEMENT(_impl->register_plugin(plugin, device_name, properties););
}

void Core::unload_plugin(const std::string& device_name) {
    OV_CORE_CALL_STATEMENT({
        ov::DeviceIDParser parser(device_name);
        std::string devName = parser.get_device_name();

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
        auto parsed = parseDeviceNameIntoConfig(device_name, params);
        auto remoteContext = _impl->get_plugin(parsed._deviceName).create_context(parsed._config);
        return {remoteContext._ptr, remoteContext._so};
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
        return {remoteContext._ptr, remoteContext._so};
    });
}

void shutdown() {
    frontend::FrontEndManager::shutdown();
}

}  // namespace ov
