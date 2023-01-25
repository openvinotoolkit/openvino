// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "any_copy.hpp"
#include "compilation_context.hpp"
#include "core_impl.hpp"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"
#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"
#include "dev/converter_utils.hpp"
#include "ie_network_reader.hpp"
#include "iplugin_wrapper.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/pass/constant_folding.hpp"
#include "openvino/itt.hpp"
#include "openvino/util/common_util.hpp"

bool ov::CoreImpl::isNewAPI() const {
    return is_new_api();
}

ov::SoPtr<InferenceEngine::IExecutableNetworkInternal> ov::CoreImpl::LoadNetworkImpl(
    const InferenceEngine::CNNNetwork& network,
    ov::Plugin& plugin,
    const std::map<std::string, std::string>& parsedConfig,
    const InferenceEngine::RemoteContext::Ptr& context) {
    OV_ITT_SCOPED_TASK(ov::itt::domains::IE, "CoreImpl::compile_model_impl");
    ov::SoPtr<InferenceEngine::IExecutableNetworkInternal> execNetwork;
    auto wrapper = std::dynamic_pointer_cast<InferenceEngine::IPluginWrapper>(plugin.m_ptr);
    OPENVINO_ASSERT(wrapper);
    auto old_plugin = wrapper->get_plugin();
    execNetwork = {context ? old_plugin->LoadNetwork(network, parsedConfig, context)
                           : old_plugin->LoadNetwork(network, parsedConfig),
                   plugin.m_so};
    return execNetwork;
}

InferenceEngine::RemoteContext::Ptr ov::CoreImpl::GetDefaultContext(const std::string& deviceName) {
    return get_default_context(deviceName)._impl;
}

InferenceEngine::CNNNetwork ov::CoreImpl::ReadNetwork(const std::string& modelPath, const std::string& binPath) const {
    OV_ITT_SCOPE(FIRST_INFERENCE, ov::itt::domains::IE_RT, "CoreImpl::ReadNetwork from file");
    return InferenceEngine::details::ReadNetwork(modelPath, binPath, extensions, ov_extensions, is_new_api());
}

InferenceEngine::CNNNetwork ov::CoreImpl::ReadNetwork(const std::string& model,
                                                      const InferenceEngine::Blob::CPtr& weights,
                                                      bool frontendMode) const {
    OV_ITT_SCOPE(FIRST_INFERENCE, ov::itt::domains::IE_RT, "CoreImpl::ReadNetwork from memory");
    return InferenceEngine::details::ReadNetwork(model, weights, extensions, ov_extensions, is_new_api(), frontendMode);
}

ov::SoPtr<InferenceEngine::IExecutableNetworkInternal> ov::CoreImpl::LoadNetwork(
    const InferenceEngine::CNNNetwork& network,
    const std::shared_ptr<InferenceEngine::RemoteContext>& context,
    const std::map<std::string, std::string>& config) {
    OV_ITT_SCOPE(FIRST_INFERENCE, InferenceEngine::itt::domains::IE_LT, "Core::LoadNetwork::RemoteContext");
    if (network.getFunction()) {
        ov::RemoteContext ctx{context, {nullptr}};
        auto compiled_model =
            compile_model(ov::legacy_convert::convert_model(network, isNewAPI()), ctx, any_copy(config));
        return {compiled_model._ptr, compiled_model._so};
    }
    if (context == nullptr) {
        IE_THROW() << "Remote context is null";
    }
    // have to deduce the device name/config from the context first
    auto parsed = parseDeviceNameIntoConfig(context->getDeviceName(), config);

    auto plugin = get_plugin(parsed._deviceName);

    auto res = LoadNetworkImpl(network, plugin, parsed._config, context);
    return res;
}

InferenceEngine::SoExecutableNetworkInternal ov::CoreImpl::LoadNetwork(
    const InferenceEngine::CNNNetwork& network,
    const std::string& deviceName,
    const std::map<std::string, std::string>& config) {
    OV_ITT_SCOPE(FIRST_INFERENCE, InferenceEngine::itt::domains::IE_LT, "Core::LoadNetwork::CNN");
    if (network.getFunction()) {
        auto compiled_model =
            compile_model(ov::legacy_convert::convert_model(network, isNewAPI()), deviceName, any_copy(config));
        return {compiled_model._ptr, compiled_model._so};
    }
    auto parsed = parseDeviceNameIntoConfig(deviceName, config);
    auto plugin = get_plugin(parsed._deviceName);
    auto res = LoadNetworkImpl(network, plugin, parsed._config, nullptr);
    return {res._ptr, res._so};
}

InferenceEngine::SoExecutableNetworkInternal ov::CoreImpl::LoadNetwork(
    const std::string& modelPath,
    const std::string& deviceName,
    const std::map<std::string, std::string>& config,
    const std::function<void(const InferenceEngine::CNNNetwork&)>& val) {
    OV_ITT_SCOPE(FIRST_INFERENCE, ie::itt::domains::IE_LT, "Core::LoadNetwork::Path");

    auto compiled_model = compile_model(modelPath, deviceName, any_copy(config));
    return {compiled_model._ptr, compiled_model._so};
}

InferenceEngine::SoExecutableNetworkInternal ov::CoreImpl::LoadNetwork(
    const std::string& modelStr,
    const InferenceEngine::Blob::CPtr& weights,
    const std::string& deviceName,
    const std::map<std::string, std::string>& config,
    const std::function<void(const InferenceEngine::CNNNetwork&)>& val) {
    OV_ITT_SCOPE(FIRST_INFERENCE, InferenceEngine::itt::domains::IE_LT, "Core::LoadNetwork::Memory");

    auto compiled_model = compile_model(modelStr,
                                        ov::Tensor{std::const_pointer_cast<InferenceEngine::Blob>(weights), {}},
                                        deviceName,
                                        ov::any_copy(config));
    return {compiled_model._ptr, compiled_model._so};
}

InferenceEngine::SoExecutableNetworkInternal ov::CoreImpl::ImportNetwork(
    std::istream& networkModel,
    const std::string& deviceName,
    const std::map<std::string, std::string>& config) {
    auto compiled_model = import_model(networkModel, deviceName, any_copy(config));
    return {compiled_model._ptr, compiled_model._so};
}

InferenceEngine::QueryNetworkResult ov::CoreImpl::QueryNetwork(const InferenceEngine::CNNNetwork& network,
                                                               const std::string& deviceName,
                                                               const std::map<std::string, std::string>& config) const {
    OV_ITT_SCOPED_TASK(ov::itt::domains::IE, "Core::QueryNetwork");
    ie::QueryNetworkResult ret;
    if (!network.getFunction()) {
        ret.rc = InferenceEngine::GENERAL_ERROR;
        return ret;
    }
    auto res = query_model(network.getFunction(), deviceName, any_copy(config));
    ret.supportedLayersMap = res;

    return ret;
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

    return get_plugin(parsed._deviceName).get_property(name, parsed._config);
}

ov::Any ov::CoreImpl::GetConfig(const std::string& deviceName, const std::string& name) const {
    auto parsed = parseDeviceNameIntoConfig(deviceName);
    return get_plugin(parsed._deviceName).get_property(name, parsed._config);
}

std::vector<std::string> ov::CoreImpl::GetAvailableDevices() const {
    return get_available_devices();
}

InferenceEngine::RemoteContext::Ptr ov::CoreImpl::CreateContext(const std::string& deviceName,
                                                                const InferenceEngine::ParamMap& params) {
    return create_context(deviceName, params)._impl;
}

/**
 * @brief Registers the extension in a Core object
 *        Such extensions can be used for both CNNNetwork readers and device plugins
 */
void ov::CoreImpl::AddExtension(const InferenceEngine::IExtensionPtr& extension) {
    std::lock_guard<std::mutex> lock(get_mutex());
    AddExtensionUnsafe(extension);
}

bool ov::CoreImpl::DeviceSupportsImportExport(const std::string& deviceName) const {
    return device_supports_import_export(deviceName);
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
                if (ov::util::contains(supportedConfigKeys, device_config.first)) {
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
        ie::DeviceIDParser parser(deviceName_);
        std::string deviceNameLocal = parser.getDeviceName();

        ov::Plugin cppPlugin = get_plugin(deviceNameLocal);

        versions[deviceNameLocal] = ov::legacy_convert::convert_plugin(cppPlugin.m_ptr)->GetVersion();
    }

    return versions;
}
