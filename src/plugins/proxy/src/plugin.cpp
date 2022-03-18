// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin.hpp"

#include <ie_icore.hpp>

#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"
#include "openvino/runtime/common.hpp"
#include "proxy_plugin.hpp"

ov::proxy::Plugin::Plugin() {}
ov::proxy::Plugin::~Plugin() {}

void ov::proxy::Plugin::SetConfig(const std::map<std::string, std::string>& config) {
    m_config = config;
}
InferenceEngine::QueryNetworkResult ov::proxy::Plugin::QueryNetwork(
    const InferenceEngine::CNNNetwork& network,
    const std::map<std::string, std::string>& config) const {
    IE_THROW(NotImplemented);
}
InferenceEngine::IExecutableNetworkInternal::Ptr ov::proxy::Plugin::LoadExeNetworkImpl(
    const InferenceEngine::CNNNetwork& network,
    const std::map<std::string, std::string>& config) {
    IE_THROW(NotImplemented);
}
void ov::proxy::Plugin::AddExtension(const std::shared_ptr<InferenceEngine::IExtension>& extension) {
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

    IE_THROW(NotImplemented);
}
InferenceEngine::Parameter ov::proxy::Plugin::GetMetric(
    const std::string& name,
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
        if (device_id == "abc_a")
            deviceFullName = "a";
        else if (device_id == "abc_b")
            deviceFullName = "b";
        else if (device_id == "abc_c")
            deviceFullName = "c";
        return decltype(ov::device::full_name)::value_type(deviceFullName);
    } else if (name == ov::available_devices) {
        auto hidden_devices = get_hidden_devices();
        std::vector<std::string> availableDevices(hidden_devices.size());
        size_t i(0);  // Should we use full name?
        for (const auto it : hidden_devices) {
            availableDevices[i] = std::to_string(i);
            i++;
        }
        return decltype(ov::available_devices)::value_type(availableDevices);
    }

    IE_THROW(NotImplemented);
}
InferenceEngine::IExecutableNetworkInternal::Ptr ov::proxy::Plugin::ImportNetwork(
    std::istream& model,
    const std::map<std::string, std::string>& config) {
    IE_THROW(NotImplemented);
}

std::map<std::string, std::vector<std::string>> ov::proxy::Plugin::get_hidden_devices() const {
    std::map<std::string, std::vector<std::string>> result;

    auto hidden_devices = GetCore()->GetHiddenDevicesFor(this->_pluginName);
    for (const auto& device : hidden_devices) {
        auto full_name = GetCore()->GetConfig(device, ov::device::full_name.name()).as<std::string>();
        result[full_name].emplace_back(device);
    }
    return result;
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
