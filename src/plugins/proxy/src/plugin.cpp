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

std::vector<std::string> split(const std::string& str, const std::string& delim = ",") {
    std::vector<std::string> result;
    std::string::size_type start(0);
    std::string::size_type end = str.find(delim);
    while (end != std::string::npos) {
        result.emplace_back(str.substr(start, end - start));
        start = end + delim.size();
        end = str.find(delim, start);
    }
    result.emplace_back(str.substr(start, end - start));
    return result;
}
}  // namespace

ov::proxy::Plugin::Plugin() {}
ov::proxy::Plugin::~Plugin() {}

size_t ov::proxy::Plugin::get_device_from_config(const std::map<std::string, std::string>& config) const {
    if (config.find("DEVICE_ID") != config.end())
        return string_to_size_t(config.at("DEVICE_ID"));
    return 0;
}

void ov::proxy::Plugin::SetConfig(const std::map<std::string, std::string>& config) {
    // Set config for primary device
    ov::AnyMap property;

    // Parse alias config
    auto it = config.find("ALIAS_FOR");
    bool fill_order = config.find("DEVICES_ORDER") == config.end() && device_order.empty();
    if (it != config.end()) {
        for (auto&& dev : split(it->second)) {
            alias_for.emplace(dev);
            if (fill_order)
                device_order.emplace_back(dev);
        }
    }

    // Restore device order
    it = config.find("DEVICES_ORDER");
    if (it != config.end()) {
        std::vector<std::pair<std::string, size_t>> priority_order;
        // Biggest number means minimum priority
        size_t min_priority(0);
        for (auto&& dev_priority : split(it->second)) {
            auto dev_prior = split(dev_priority, ":");
            OPENVINO_ASSERT(dev_prior.size() == 2);
            auto priority = string_to_size_t(dev_prior[1]);
            if (priority > min_priority)
                min_priority = priority;
            priority_order.push_back(std::pair<std::string, size_t>{dev_prior[0], priority});
        }
        // Devices without priority has lower priority
        min_priority++;
        for (const auto& dev : alias_for) {
            if (std::find_if(priority_order.begin(),
                             priority_order.end(),
                             [&](const std::pair<std::string, size_t>& el) {
                                 return el.first == dev;
                             }) == std::end(priority_order)) {
                priority_order.push_back(std::pair<std::string, size_t>{dev, min_priority});
            }
        }
        std::sort(priority_order.begin(),
                  priority_order.end(),
                  [](const std::pair<std::string, size_t>& v1, const std::pair<std::string, size_t>& v2) {
                      return v1.second < v2.second;
                  });
        device_order.reserve(priority_order.size());
        for (const auto& dev : priority_order) {
            device_order.emplace_back(dev.first);
        }
    }

    it = config.find("FALLBACK_PRIORITY");
    if (it != config.end()) {
        fallback_order = split(it->second);
    }
    for (const auto& it : config) {
        // Skip proxy properties
        if (ov::device::id.name() == it.first || it.first == "FALLBACK_PRIORITY" || it.first == "DEVICES_ORDER" ||
            it.first == "ALIAS_FOR")
            continue;
        property[it.first] = it.second;
    }
    GetCore()->set_property(get_primary_device(get_device_from_config(config)), property);
}

InferenceEngine::QueryNetworkResult ov::proxy::Plugin::QueryNetwork(
    const InferenceEngine::CNNNetwork& network,
    const std::map<std::string, std::string>& config) const {
    size_t num_devices = get_hidden_devices().size();
    // Recall for HW device
    auto dev_id = get_device_from_config(config);
    auto res = GetCore()->QueryNetwork(network, get_fallback_device(dev_id), config);
    // Replace hidden device name
    for (auto&& it : res.supportedLayersMap) {
        it.second = GetName();
        if (num_devices > 1)
            it.second += "." + std::to_string(dev_id);
    }
    return res;
}

InferenceEngine::IExecutableNetworkInternal::Ptr ov::proxy::Plugin::LoadExeNetworkImpl(
    const InferenceEngine::CNNNetwork& network,
    const std::map<std::string, std::string>& config) {
    auto dev_name = get_fallback_device(get_device_from_config(config));
    auto device_config = config;
    // Remove proxy properties
    auto it = device_config.find(ov::device::id.name());
    if (it != device_config.end())
        device_config.erase(it);

    return std::make_shared<ov::proxy::CompiledModel>(GetCore()->LoadNetwork(network, dev_name, device_config));
}

void ov::proxy::Plugin::AddExtension(const std::shared_ptr<InferenceEngine::IExtension>& extension) {
    // Don't need to recall add_extension for hidden plugin, because core objects add extensions for all plugins
    IE_THROW(NotImplemented);
}

InferenceEngine::Parameter ov::proxy::Plugin::GetConfig(
    const std::string& name,
    const std::map<std::string, InferenceEngine::Parameter>& options) const {
    std::string device_id = "0";
    if (options.find(ov::device::id.name()) != options.end()) {
        device_id = options.find(ov::device::id.name())->second.as<std::string>();
    }
    if (name == ov::device::id)
        return device_id;

    size_t idx = string_to_size_t(device_id);
    return GetCore()->GetConfig(get_primary_device(idx), name);
}
InferenceEngine::Parameter ov::proxy::Plugin::GetMetric(
    const std::string& name,
    const std::map<std::string, InferenceEngine::Parameter>& options) const {
    std::string device_name = get_primary_device(string_to_size_t(GetConfig(ov::device::id.name(), options)));

    if (name == ov::supported_properties) {
        const static std::unordered_set<std::string> property_names = {ov::supported_properties.name(),
                                                                       ov::available_devices.name()};

        std::vector<ov::PropertyName> supportedProperties;
        supportedProperties.reserve(property_names.size());
        for (const auto& property : property_names) {
            supportedProperties.emplace_back(ov::PropertyName(property, ov::PropertyMutability::RO));
        }

        auto dev_properties = GetCore()->GetMetric(device_name, name, options).as<std::vector<ov::PropertyName>>();

        for (const auto& property : dev_properties) {
            if (property_names.find(property) != property_names.end())
                continue;
            supportedProperties.emplace_back(property);
        }

        return decltype(ov::supported_properties)::value_type(supportedProperties);
    } else if (name == "SUPPORTED_METRICS") {
        const static std::unordered_set<std::string> metric_names = {"SUPPORTED_METRICS", ov::available_devices.name()};

        std::vector<std::string> metrics;
        metrics.reserve(metric_names.size());
        for (const auto& metric : metric_names)
            metrics.emplace_back(metric);

        auto dev_properties = GetCore()->GetMetric(device_name, name, options).as<std::vector<std::string>>();

        for (const auto& property : dev_properties) {
            if (metric_names.find(property) != metric_names.end())
                continue;
            metrics.emplace_back(property);
        }
        return metrics;
    } else if (name == ov::available_devices) {
        auto hidden_devices = get_hidden_devices();
        std::vector<std::string> availableDevices(hidden_devices.size());
        for (size_t i = 0; i < hidden_devices.size(); i++) {
            availableDevices[i] = std::to_string(i);
        }
        return decltype(ov::available_devices)::value_type(availableDevices);
    }

    return GetCore()->GetMetric(device_name, name, options);
}
InferenceEngine::IExecutableNetworkInternal::Ptr ov::proxy::Plugin::ImportNetwork(
    std::istream& model,
    const std::map<std::string, std::string>& config) {
    auto device_config = config;
    // Remove proxy properties
    auto it = device_config.find(ov::device::id.name());
    if (it != device_config.end())
        device_config.erase(it);

    return std::make_shared<ov::proxy::CompiledModel>(
        GetCore()->ImportNetwork(model, get_fallback_device(get_device_from_config(config)), device_config));
}

std::vector<std::vector<std::string>> ov::proxy::Plugin::get_hidden_devices() const {
    // Proxy plugin has 2 modes of matching devices:
    //  * Fallback - in this mode we report devices only for the first hidden plugin
    //  * Alias - Case when we group all devices under one common name
    std::vector<std::vector<std::string>> result;
    const auto core = GetCore();
    OPENVINO_ASSERT(core != nullptr);

    std::map<std::array<uint8_t, ov::device::UUID::MAX_UUID_SIZE>, std::string> first_uuid_dev_map;
    std::vector<std::vector<std::pair<std::string, size_t>>> ordered_result;

    // If we don't have alias we use simple hetero mode
    if (alias_for.empty()) {
        for (const auto& device : fallback_order) {
            if (device != fallback_order[0]) {
                for (auto&& uniq_dev : result) {
                    uniq_dev.emplace_back(device);
                }
                continue;
            }
            const std::vector<std::string> supported_device_ids = core->get_property(device, ov::available_devices);
            for (const auto& device_id : supported_device_ids) {
                const std::string full_device_name = device + '.' + device_id;
                result.emplace_back(std::vector<std::string>{full_device_name});
            }
        }
    } else {
        // First of all create a vector of primary devices with device order.
        // after that use FALLBACK_PRIORITY for right fallback order
        //
        std::unordered_map<std::string, size_t> dev_fallback_priority;
        for (size_t i = 0; i < fallback_order.size(); i++) {
            dev_fallback_priority[fallback_order[i]] = i;
        }
        for (const auto& device : device_order) {
            const std::vector<std::string> supported_device_ids = core->get_property(device, ov::available_devices);
            // Case for fallback without alias
            if (alias_for.find(device) == alias_for.end() && device != device_order[0]) {
                for (auto&& uniq_dev : ordered_result) {
                    uniq_dev.emplace_back(std::pair<std::string, size_t>{device, dev_fallback_priority[device]});
                }
                continue;
            }

            // If we have an alias we need to have all devices and if it possible reduce device duplication
            for (const auto& device_id : supported_device_ids) {
                const std::string full_device_name = device + '.' + device_id;
                try {
                    std::string unique_property_name = ov::device::uuid.name();
                    try {
                        unique_property_name =
                            core->get_property(full_device_name, "DEVICE_ID_PROPERTY", {}).as<std::string>();
                    } catch (...) {
                    }
                    ov::device::UUID uuid = core->get_property(full_device_name, unique_property_name, {});
                    auto it = first_uuid_dev_map.find(uuid.uuid);
                    if (it == first_uuid_dev_map.end()) {
                        // First unique element
                        ordered_result.emplace_back(std::vector<std::pair<std::string, size_t>>{
                            {full_device_name, dev_fallback_priority[device]}});
                        first_uuid_dev_map[uuid.uuid] = full_device_name;
                    } else {
                        for (size_t i = 0; i < ordered_result.size(); i++) {
                            if (ordered_result[i].at(0).first != it->second)
                                continue;
                            ordered_result[i].emplace_back(
                                std::pair<std::string, size_t>{full_device_name, dev_fallback_priority[device]});
                        }
                    }
                } catch (...) {
                    // Device doesn't have UUID, so it means that device is unique
                    ordered_result.emplace_back(
                        std::vector<std::pair<std::string, size_t>>{{full_device_name, dev_fallback_priority[device]}});
                }
            }
        }

        // Restore the right order
        for (auto&& devices : ordered_result) {
            std::sort(devices.begin(),
                      devices.end(),
                      [](const std::pair<std::string, size_t>& v1, const std::pair<std::string, size_t>& v2) {
                          return v1.second < v2.second;
                      });
            std::vector<std::string> ordered_devices(devices.size());

            for (size_t i = 0; i < ordered_devices.size(); i++) {
                ordered_devices[i] = devices[i].first;
            }
            result.emplace_back(ordered_devices);
        }
    }
    return result;
}

std::vector<std::string> ov::proxy::Plugin::get_primary_devices() const {
    // Return primary devices
    std::vector<std::string> devices;
    const auto all_devices = get_hidden_devices();
    for (const auto& dev : all_devices) {
        devices.emplace_back(dev.at(0));
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
    if (all_devices[idx].size() == 1) {
        return all_devices[idx].at(0);
    } else {
        std::string device_concatenation;
        for (const auto& dev : all_devices[idx]) {
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
