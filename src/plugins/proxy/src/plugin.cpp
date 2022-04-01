// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin.hpp"

#include <sstream>

#include "compiled_model.hpp"
#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"
#include "ie_icore.hpp"
#include "openvino/runtime/common.hpp"
#include "proxy_plugin.hpp"

namespace {
size_t string_to_size_t(const std::string& s) {
    std::stringstream sstream(s);
    size_t idx;
    sstream >> idx;
    return idx;
}
}  // namespace

ov::proxy::Plugin::Plugin() {}
ov::proxy::Plugin::~Plugin() {}

bool ov::proxy::Plugin::has_device_in_config(const std::map<std::string, std::string>& config) const {
    return config.find("DEVICE_ID") != config.end();
}
size_t ov::proxy::Plugin::get_device_from_config(const std::map<std::string, std::string>& config) const {
    OPENVINO_ASSERT(config.find("DEVICE_ID") != config.end());
    return string_to_size_t(config.at("DEVICE_ID"));
}

void ov::proxy::Plugin::SetConfig(const std::map<std::string, std::string>& config) {
    // Set config for primary device
    m_config = config;
    ov::AnyMap property;
    for (const auto it : config) {
        property[it.first] = it.second;
    }
    if (has_device_in_config(config))
        GetCore()->set_property(get_primary_device(get_device_from_config(config)), property);
}

InferenceEngine::QueryNetworkResult ov::proxy::Plugin::QueryNetwork(
    const InferenceEngine::CNNNetwork& network,
    const std::map<std::string, std::string>& config) const {
    // Recall for HW device
    auto dev_id = get_device_from_config(config);
    auto res = GetCore()->QueryNetwork(network, get_fallback_device(dev_id), config);
    // Replace hidden device name
    for (auto&& it : res.supportedLayersMap) {
        it.second = GetName() + "." + std::to_string(dev_id);
    }
    return res;
}

InferenceEngine::IExecutableNetworkInternal::Ptr ov::proxy::Plugin::LoadExeNetworkImpl(
    const InferenceEngine::CNNNetwork& network,
    const std::map<std::string, std::string>& config) {
    auto dev_name = get_fallback_device(get_device_from_config(config));
    return std::make_shared<ov::proxy::CompiledModel>(GetCore()->LoadNetwork(network, dev_name, config));
}

void ov::proxy::Plugin::AddExtension(const std::shared_ptr<InferenceEngine::IExtension>& extension) {
    // Don't need to recall add_extension for hidden plugin, because core objects add extensions for all plugins
    IE_THROW(NotImplemented);
}

InferenceEngine::Parameter ov::proxy::Plugin::GetConfig(
    const std::string& name,
    const std::map<std::string, InferenceEngine::Parameter>& options) const {
    std::string device_id;
    if (options.find(ov::device::id.name()) != options.end()) {
        device_id = options.find(ov::device::id.name())->second.as<std::string>();
    }
    if (name == ov::device::id)
        return device_id;

    if (device_id.empty())
        IE_THROW(NotImplemented);
    size_t idx = string_to_size_t(device_id);
    return GetCore()->GetConfig(get_primary_device(idx), name);
}
InferenceEngine::Parameter ov::proxy::Plugin::GetMetric(
    const std::string& name,
    const std::map<std::string, InferenceEngine::Parameter>& options) const {
    auto RO_property = [](const std::string& propertyName) {
        return ov::PropertyName(propertyName, ov::PropertyMutability::RO);
    };
    std::string device_id = GetConfig(ov::device::id.name(), options);

    // TODO: recall plugin for supported metrics
    if (name == ov::supported_properties) {
        std::vector<ov::PropertyName> roProperties{
            RO_property(ov::supported_properties.name()),
            RO_property(ov::available_devices.name()),
            RO_property(ov::device::full_name.name()),
        };

        std::vector<ov::PropertyName> supportedProperties;
        supportedProperties.reserve(roProperties.size());
        supportedProperties.insert(supportedProperties.end(), roProperties.begin(), roProperties.end());

        return decltype(ov::supported_properties)::value_type(supportedProperties);
    } else if (name == "SUPPORTED_CONFIG_KEYS") {
        std::vector<std::string> configs;
        return configs;
    } else if (name == "SUPPORTED_METRICS") {
        std::vector<std::string> metrics;
        metrics.push_back("AVAILABLE_DEVICES");
        metrics.push_back("SUPPORTED_METRICS");
        metrics.push_back("FULL_DEVICE_NAME");
        return metrics;
    } else if (name == ov::available_devices) {
        auto hidden_devices = get_hidden_devices();
        std::vector<std::string> availableDevices(hidden_devices.size());
        for (size_t i = 0; i < hidden_devices.size(); i++) {
            availableDevices[i] = std::to_string(i);
        }
        return decltype(ov::available_devices)::value_type(availableDevices);
    }

    if (device_id.empty())
        IE_THROW(NotImplemented) << " to call " << name;
    size_t idx = string_to_size_t(device_id);
    return GetCore()->GetMetric(get_primary_device(idx), name, options);
}
InferenceEngine::IExecutableNetworkInternal::Ptr ov::proxy::Plugin::ImportNetwork(
    std::istream& model,
    const std::map<std::string, std::string>& config) {
    // TODO:
    // return GetCore()->ImportNetwork(model, get_fallback_device(get_device_from_config(config)), config);
    IE_THROW(NotImplemented);
}

std::vector<std::pair<std::string, std::vector<std::string>>> ov::proxy::Plugin::get_hidden_devices() const {
    std::map<std::string, std::vector<std::string>> result;

    auto hidden_devices = GetCore()->GetHiddenDevicesFor(this->_pluginName);
    for (const auto& device : hidden_devices) {
        auto full_name = GetCore()->GetConfig(device, ov::device::full_name.name()).as<std::string>();
        result[full_name].emplace_back(device);
    }

    std::vector<std::pair<std::string, std::vector<std::string>>> end_result(result.size());
    size_t i(0);  // Should we use full name?
    for (const auto it : result) {
        end_result[i] = {it.first, it.second};
        i++;
    }
    return end_result;
}

std::vector<std::string> ov::proxy::Plugin::get_primary_devices() const {
    // Return primary devices
    std::vector<std::string> devices;
    const auto all_devices = get_hidden_devices();
    for (const auto& dev : all_devices) {
        devices.emplace_back(dev.second.at(0));
    }

    return devices;
}

std::string ov::proxy::Plugin::get_primary_device(size_t idx) const {
    auto devices = get_primary_devices();

    OPENVINO_ASSERT(devices.size() > idx);
    return devices[idx];
}

std::string ov::proxy::Plugin::get_fallback_device(size_t idx) const {
    const auto all_devices = get_hidden_devices();
    OPENVINO_ASSERT(all_devices.size() > idx);
    if (all_devices[idx].second.size() == 1) {
        return all_devices[idx].second.at(0);
    } else {
        std::string device_concatenation;
        for (const auto& dev : all_devices[idx].second) {
            if (!device_concatenation.empty())
                device_concatenation += ",";
            device_concatenation += dev;
        }
        return "HETERO:" + device_concatenation;
    }
}

void ov::proxy::create_plugin(::std::shared_ptr<::InferenceEngine::IInferencePlugin>& plugin) {
    static const InferenceEngine::Version version = {{2, 1}, CI_BUILD_NUMBER, "openvino_proxy_plugin"};
    try {
        plugin = ::std::make_shared<ov::proxy::Plugin>();
    } catch (const InferenceEngine::Exception&) {
        throw;
    } catch (const std::exception& ex) {
        IE_THROW() << ex.what();
    } catch (...) {
        IE_THROW(Unexpected);
    }
    plugin->SetVersion(version);
}
