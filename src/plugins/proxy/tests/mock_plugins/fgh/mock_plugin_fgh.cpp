// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mock_plugin_fgh.hpp"

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

bool string_to_bool(const std::string& s) {
    return s == "YES";
}
}  // namespace

MockPluginFgh::MockPluginFgh() {}

void MockPluginFgh::SetConfig(const std::map<std::string, std::string>& _config) {
    if (_config.find("PERF_COUNT") != _config.end()) {
        m_profiling = string_to_bool(_config.at("PERF_COUNT"));
    }
}

void MockPluginFgh::AddExtension(const ov::Extension::Ptr& extension) {
    m_extensions.emplace_back(extension);
}

Parameter MockPluginFgh::GetConfig(const std::string& name,
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
        return metrics;
    } else if (name == "SUPPORTED_CONFIG_KEYS") {
        std::vector<std::string> configs;
        configs.push_back("PERF_COUNT");
        return configs;
    }

    IE_THROW(NotImplemented);
}

Parameter MockPluginFgh::GetMetric(const std::string& name,
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
        };
        // the whole config is RW before network is loaded.
        std::vector<ov::PropertyName> rwProperties{
            RW_property(ov::enable_profiling.name()),
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
        return metrics;
    } else if (name == "PERF_COUNT") {
        return m_profiling;
    } else if (name == "SUPPORTED_CONFIG_KEYS") {
        std::vector<std::string> configs;
        configs.push_back("NUM_STREAMS");
        configs.push_back("PERF_COUNT");
        return configs;
    } else if (name == ov::available_devices) {
        const std::vector<std::string> availableDevices = {"fgh_f", "fgh_g", "fgh_h"};
        return decltype(ov::available_devices)::value_type(availableDevices);
    }

    IE_THROW(NotImplemented);
}

std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> MockPluginFgh::LoadNetwork(
    const CNNNetwork& network,
    const std::map<std::string, std::string>& config) {
    auto model = network.getFunction();

    OPENVINO_ASSERT(model);
    if (!support_model(model, QueryNetwork(network, config)))
        throw ov::Exception("Unsupported model");

    return std::make_shared<MockCompiledModel>(model, config);
}

std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> MockPluginFgh::LoadNetwork(
    const CNNNetwork& network,
    const std::map<std::string, std::string>& config,
    const std::shared_ptr<RemoteContext>& context) {
    IE_THROW(NotImplemented);
}

std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> MockPluginFgh::LoadNetwork(
    const std::string& modelPath,
    const std::map<std::string, std::string>& config) {
    return InferenceEngine::IInferencePlugin::LoadNetwork(modelPath, config);
}

std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> MockPluginFgh::LoadExeNetworkImpl(
    const CNNNetwork& network,
    const std::map<std::string, std::string>& config) {
    auto model = network.getFunction();

    OPENVINO_ASSERT(model);
    if (!support_model(model, QueryNetwork(network, config)))
        throw ov::Exception("Unsupported model");

    return std::make_shared<MockCompiledModel>(model, config);
}

std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> MockPluginFgh::ImportNetwork(
    std::istream& networkModel,
    const std::map<std::string, std::string>& config) {
    IE_THROW(NotImplemented);
}

std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> MockPluginFgh::ImportNetwork(
    std::istream& networkModel,
    const std::shared_ptr<InferenceEngine::RemoteContext>& context,
    const std::map<std::string, std::string>& config) {
    IE_THROW(NotImplemented);
}

std::shared_ptr<InferenceEngine::RemoteContext> MockPluginFgh::GetDefaultContext(
    const InferenceEngine::ParamMap& params) {
    IE_THROW(NotImplemented);
}

InferenceEngine::QueryNetworkResult MockPluginFgh::QueryNetwork(
    const InferenceEngine::CNNNetwork& network,
    const std::map<std::string, std::string>& config) const {
    auto model = network.getFunction();

    OPENVINO_ASSERT(model);

    std::unordered_set<std::string> supported_ops = {"Parameter", "Result", "Add", "Constant", "Subtract"};

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

void MockPluginFgh::SetCore(std::weak_ptr<InferenceEngine::ICore> core) noexcept {
    InferenceEngine::IInferencePlugin::SetCore(core);
}

void MockPluginFgh::SetName(const std::string& name) noexcept {
    InferenceEngine::IInferencePlugin::SetName(name);
}

std::string MockPluginFgh::GetName() const noexcept {
    return InferenceEngine::IInferencePlugin::GetName();
}

static const Version version = {{2, 1}, "test_plugin", "MockPluginFgh"};
IE_DEFINE_PLUGIN_CREATE_FUNCTION(MockPluginFgh, version)
