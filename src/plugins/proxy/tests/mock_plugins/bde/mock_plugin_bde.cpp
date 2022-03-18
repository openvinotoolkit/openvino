// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mock_plugin_bde.hpp"

#include <iostream>
#include <map>
#include <string>
#include <utility>

#include "description_buffer.hpp"
#include "openvino/runtime/common.hpp"

using namespace std;
using namespace InferenceEngine;

MockPluginBde::MockPluginBde() {}

void MockPluginBde::SetConfig(const std::map<std::string, std::string>& _config) {
    this->config = _config;
}

Parameter MockPluginBde::GetConfig(const std::string& name,
                                   const std::map<std::string, InferenceEngine::Parameter>& options) const {
    std::string device_id;
    if (options.find(ov::device::id.name()) != options.end()) {
        device_id = options.find(ov::device::id.name())->second.as<std::string>();
    }
    if (name == ov::device::id) {
        return device_id;
    } else if (name == ov::device::full_name) {
        std::string deviceFullName = "";
        if (device_id == "bde_b")
            deviceFullName = "b";
        else if (device_id == "bde_d")
            deviceFullName = "d";
        else if (device_id == "bde_e")
            deviceFullName = "e";
        return decltype(ov::device::full_name)::value_type(deviceFullName);
    }

    IE_THROW(NotImplemented);
}

Parameter MockPluginBde::GetMetric(const std::string& name,
                                   const std::map<std::string, InferenceEngine::Parameter>& options) const {
    auto RO_property = [](const std::string& propertyName) {
        return ov::PropertyName(propertyName, ov::PropertyMutability::RO);
    };
    auto RW_property = [](const std::string& propertyName) {
        return ov::PropertyName(propertyName, ov::PropertyMutability::RW);
    };

    std::string device_id = GetConfig(ov::device::id.name(), options);

    if (name == ov::supported_properties) {
        std::vector<ov::PropertyName> roProperties{
            RO_property(ov::supported_properties.name()),
            RO_property(ov::available_devices.name()),
            RO_property(ov::device::full_name.name()),
        };
        // the whole config is RW before network is loaded.
        std::vector<ov::PropertyName> rwProperties{
            RW_property(ov::num_streams.name()),
            RW_property(ov::affinity.name()),
            RW_property(ov::inference_num_threads.name()),
            RW_property(ov::enable_profiling.name()),
            RW_property(ov::hint::inference_precision.name()),
            RW_property(ov::hint::performance_mode.name()),
            RW_property(ov::hint::num_requests.name()),
        };

        std::vector<ov::PropertyName> supportedProperties;
        supportedProperties.reserve(roProperties.size() + rwProperties.size());
        supportedProperties.insert(supportedProperties.end(), roProperties.begin(), roProperties.end());
        supportedProperties.insert(supportedProperties.end(), rwProperties.begin(), rwProperties.end());

        return decltype(ov::supported_properties)::value_type(supportedProperties);
    } else if (name == ov::device::full_name) {
        std::string deviceFullName = "";
        if (device_id == "bde_b")
            deviceFullName = "b";
        else if (device_id == "bde_d")
            deviceFullName = "d";
        else if (device_id == "bde_e")
            deviceFullName = "e";
        return decltype(ov::device::full_name)::value_type(deviceFullName);
    } else if (name == ov::available_devices) {
        const std::vector<std::string> availableDevices = {"bde_b", "bde_d", "bde_e"};
        return decltype(ov::available_devices)::value_type(availableDevices);
    }

    IE_THROW(NotImplemented);
}

std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> MockPluginBde::LoadNetwork(
    const CNNNetwork& network,
    const std::map<std::string, std::string>& config) {
    IE_THROW(NotImplemented);
}

std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> MockPluginBde::LoadNetwork(
    const CNNNetwork& network,
    const std::map<std::string, std::string>& config,
    const std::shared_ptr<RemoteContext>& context) {
    IE_THROW(NotImplemented);
}

std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> MockPluginBde::LoadNetwork(
    const std::string& modelPath,
    const std::map<std::string, std::string>& config) {
    return InferenceEngine::IInferencePlugin::LoadNetwork(modelPath, config);
}

std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> MockPluginBde::LoadExeNetworkImpl(
    const CNNNetwork& network,
    const std::map<std::string, std::string>& config) {
    return {};
}

std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> MockPluginBde::ImportNetwork(
    std::istream& networkModel,
    const std::map<std::string, std::string>& config) {
    IE_THROW(NotImplemented);
}

std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> MockPluginBde::ImportNetwork(
    std::istream& networkModel,
    const std::shared_ptr<InferenceEngine::RemoteContext>& context,
    const std::map<std::string, std::string>& config) {
    IE_THROW(NotImplemented);
}

std::shared_ptr<InferenceEngine::RemoteContext> MockPluginBde::GetDefaultContext(
    const InferenceEngine::ParamMap& params) {
    IE_THROW(NotImplemented);
}

InferenceEngine::QueryNetworkResult MockPluginBde::QueryNetwork(
    const InferenceEngine::CNNNetwork& network,
    const std::map<std::string, std::string>& config) const {
    IE_THROW(NotImplemented);
}

void MockPluginBde::SetCore(std::weak_ptr<InferenceEngine::ICore> core) noexcept {
    InferenceEngine::IInferencePlugin::SetCore(core);
}

void MockPluginBde::SetName(const std::string& name) noexcept {
    InferenceEngine::IInferencePlugin::SetName(name);
}

std::string MockPluginBde::GetName() const noexcept {
    return InferenceEngine::IInferencePlugin::GetName();
}

OPENVINO_PLUGIN_API void CreatePluginEngine(std::shared_ptr<InferenceEngine::IInferencePlugin>& plugin) {
    plugin = std::make_shared<MockPluginBde>();
}
