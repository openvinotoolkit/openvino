// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mock_plugin_abc.hpp"

#include <iostream>
#include <map>
#include <string>
#include <utility>

#include "description_buffer.hpp"
#include "mock_compiled_model.hpp"
#include "openvino/core/op_extension.hpp"
#include "openvino/runtime/common.hpp"

using namespace std;
using namespace InferenceEngine;

namespace {
bool support_model(const std::shared_ptr<const ov::Model>& model,
                   const InferenceEngine::QueryNetworkResult& supported_ops) {
    for (const auto& op : model->get_ops()) {
        if (supported_ops.supportedLayersMap.find(op->get_friendly_name()) == supported_ops.supportedLayersMap.end())
            return false;
    }
    return true;
}

size_t string_to_size_t(const std::string& s) {
    std::stringstream sstream(s);
    size_t idx;
    sstream >> idx;
    return idx;
}
}  // namespace

MockPluginAbc::MockPluginAbc() {}

void MockPluginAbc::SetConfig(const std::map<std::string, std::string>& _config) {
    if (_config.find("NUM_STREAMS") != _config.end())
        num_streams = string_to_size_t(_config.at("NUM_STREAMS"));
}

void MockPluginAbc::AddExtension(const ov::Extension::Ptr& extension) {
    m_extensions.emplace_back(extension);
}

Parameter MockPluginAbc::GetConfig(const std::string& name,
                                   const std::map<std::string, InferenceEngine::Parameter>& options) const {
    std::string device_id;
    if (options.find(ov::device::id.name()) != options.end()) {
        device_id = options.find(ov::device::id.name())->second.as<std::string>();
    }
    if (name == ov::device::id) {
        return device_id;
    } else if (name == "SUPPORTED_METRICS") {
        std::vector<std::string> metrics;
        metrics.push_back("AVAILABLE_DEVICES");
        metrics.push_back("SUPPORTED_METRICS");
        metrics.push_back("FULL_DEVICE_NAME");
        return metrics;
    } else if (name == "SUPPORTED_CONFIG_KEYS") {
        std::vector<std::string> configs;
        configs.push_back("NUM_STREAMS");
        configs.push_back("PERF_COUNT");
        return configs;
    } else if (name == "NUM_STREAMS") {
        return num_streams;
    } else if (name == ov::device::full_name) {
        std::string deviceFullName = "";
        if (device_id == "abc_a")
            deviceFullName = "a";
        else if (device_id == "abc_b")
            deviceFullName = "b";
        else if (device_id == "abc_c")
            deviceFullName = "c";
        return decltype(ov::device::full_name)::value_type(deviceFullName);
    }

    IE_THROW(NotImplemented) << name;
}

Parameter MockPluginAbc::GetMetric(const std::string& name,
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
        };

        std::vector<ov::PropertyName> supportedProperties;
        supportedProperties.reserve(roProperties.size() + rwProperties.size());
        supportedProperties.insert(supportedProperties.end(), roProperties.begin(), roProperties.end());
        supportedProperties.insert(supportedProperties.end(), rwProperties.begin(), rwProperties.end());

        return decltype(ov::supported_properties)::value_type(supportedProperties);
    } else if (name == "SUPPORTED_METRICS") {
        std::vector<std::string> metrics;
        metrics.push_back("AVAILABLE_DEVICES");
        metrics.push_back("SUPPORTED_METRICS");
        metrics.push_back("FULL_DEVICE_NAME");
        return metrics;
    } else if (name == "SUPPORTED_CONFIG_KEYS") {
        std::vector<std::string> configs;
        configs.push_back("NUM_STREAMS");
        return configs;
    } else if (name == ov::device::full_name) {
        std::string deviceFullName = "";
        if (device_id == "abc_a")
            deviceFullName = "a";
        else if (device_id == "abc_b")
            deviceFullName = "b";
        else if (device_id == "abc_c")
            deviceFullName = "c";
        return decltype(ov::device::full_name)::value_type(deviceFullName);
    } else if (name == ov::available_devices) {
        const std::vector<std::string> availableDevices = {"abc_a", "abc_b", "abc_c"};
        return decltype(ov::available_devices)::value_type(availableDevices);
    }

    IE_THROW(NotImplemented);
}

std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> MockPluginAbc::LoadNetwork(
    const CNNNetwork& network,
    const std::map<std::string, std::string>& config) {
    auto model = network.getFunction();

    OPENVINO_ASSERT(model);
    if (!support_model(model, QueryNetwork(network, config)))
        throw ov::Exception("Unsupported model");

    return std::make_shared<MockCompiledModel>(model, config);
}

std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> MockPluginAbc::LoadNetwork(
    const CNNNetwork& network,
    const std::map<std::string, std::string>& config,
    const std::shared_ptr<RemoteContext>& context) {
    IE_THROW(NotImplemented);
}

std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> MockPluginAbc::LoadNetwork(
    const std::string& modelPath,
    const std::map<std::string, std::string>& config) {
    return InferenceEngine::IInferencePlugin::LoadNetwork(modelPath, config);
}

std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> MockPluginAbc::LoadExeNetworkImpl(
    const CNNNetwork& network,
    const std::map<std::string, std::string>& config) {
    auto model = network.getFunction();

    OPENVINO_ASSERT(model);
    if (!support_model(model, QueryNetwork(network, config)))
        throw ov::Exception("Unsupported model");

    return std::make_shared<MockCompiledModel>(model, config);
}

std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> MockPluginAbc::ImportNetwork(
    std::istream& networkModel,
    const std::map<std::string, std::string>& config) {
    IE_THROW(NotImplemented);
}

std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> MockPluginAbc::ImportNetwork(
    std::istream& networkModel,
    const std::shared_ptr<InferenceEngine::RemoteContext>& context,
    const std::map<std::string, std::string>& config) {
    IE_THROW(NotImplemented);
}

std::shared_ptr<InferenceEngine::RemoteContext> MockPluginAbc::GetDefaultContext(
    const InferenceEngine::ParamMap& params) {
    IE_THROW(NotImplemented);
}

InferenceEngine::QueryNetworkResult MockPluginAbc::QueryNetwork(
    const InferenceEngine::CNNNetwork& network,
    const std::map<std::string, std::string>& config) const {
    auto model = network.getFunction();

    OPENVINO_ASSERT(model);

    std::unordered_set<std::string> supported_ops = {"Parameter", "Result", "Add", "Constant", "Reshape"};

    for (const auto& ext : m_extensions) {
        if (const auto& op_ext = std::dynamic_pointer_cast<ov::BaseOpExtension>(ext)) {
            supported_ops.insert(op_ext->get_type_info().name);
        }
    }
    InferenceEngine::QueryNetworkResult res;
    for (const auto& op : model->get_ordered_ops()) {
        if (supported_ops.find(op->get_type_info().name) == supported_ops.end())
            continue;
        res.supportedLayersMap.emplace(op->get_friendly_name(), GetName());
    }
    return res;
}

void MockPluginAbc::SetCore(std::weak_ptr<InferenceEngine::ICore> core) noexcept {
    InferenceEngine::IInferencePlugin::SetCore(core);
}

void MockPluginAbc::SetName(const std::string& name) noexcept {
    InferenceEngine::IInferencePlugin::SetName(name);
}

std::string MockPluginAbc::GetName() const noexcept {
    return InferenceEngine::IInferencePlugin::GetName();
}

static const Version version = {{2, 1}, "test_plugin", "MockPluginAbc"};
IE_DEFINE_PLUGIN_CREATE_FUNCTION(MockPluginAbc, version)
