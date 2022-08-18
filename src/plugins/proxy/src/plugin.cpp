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

template <class T>
bool is_device_in_config(const std::map<std::string, T>& config) {
    return config.find(ov::device::id.name()) != config.end();
}

template <class T>
size_t get_device_from_config(const std::map<std::string, T>& config) {
    if (is_device_in_config(config))
        return string_to_size_t(config.at(ov::device::id.name()));
    return 0;
}

}  // namespace

std::string ov::proxy::restore_order(const std::string& original_order) {
    std::string result;
    std::vector<std::string> dev_order;
    auto fallback_properties = split(original_order);
    if (fallback_properties.size() == 1) {
        // Simple case I shouldn't restore the right order
        dev_order = split(fallback_properties.at(0), "->");
    } else {
        throw ov::Exception("Cannot restore fallback devices priority from the next config: " + original_order);
    }
    for (const auto& dev : dev_order) {
        if (!result.empty())
            result += ",";
        result += dev;
    }
    return result;
}

ov::proxy::Plugin::Plugin() {
    // Create global config
    configs[""] = {};
}
ov::proxy::Plugin::~Plugin() = default;

std::string ov::proxy::Plugin::get_property(const std::string& property, const std::string& config_name) const {
    std::lock_guard<std::mutex> lock(plugin_mutex);
    std::string result;
    auto name = config_name;
    // If device specific config wasn't found or property in config wasn't found  use global config
    auto it = configs.find(name);
    if (it == configs.end() || it->second.find(property) == it->second.end())
        name = "";

    it = configs.find(name);
    if (it->second.find(property) != it->second.end())
        result = it->second.at(property);

    return result;
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

void ov::proxy::Plugin::SetConfig(const std::map<std::string, std::string>& config) {
    // Cannot change config from different threads
    std::lock_guard<std::mutex> lock(plugin_mutex);
    // Empty config_name means means global config for all devices
    std::string config_name = is_device_in_config(config) ? std::to_string(get_device_from_config(config)) : "";

    // Parse alias config
    auto it = config.find("ALIAS_FOR");
    bool fill_order = config.find("DEVICES_PRIORITY") == config.end() && device_order.empty();
    if (it != config.end()) {
        for (auto&& dev : split(it->second)) {
            alias_for.emplace(dev);
            if (fill_order)
                device_order.emplace_back(dev);
        }
    }

    // Restore device order
    it = config.find("DEVICES_PRIORITY");
    if (it != config.end()) {
        device_order.clear();
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
        // Align sizes of device order with alias
        if (device_order.size() < alias_for.size()) {
            for (const auto& dev : alias_for) {
                if (std::find(std::begin(device_order), std::end(device_order), dev) == std::end(device_order)) {
                    device_order.emplace_back(dev);
                }
            }
        }
    }

    it = config.find(ov::device::priorities.name());
    if (it != config.end()) {
        configs[config_name][ov::device::priorities.name()] = it->second;
        // Main device is needed in case if we don't have alias and would like to be able change fallback order per
        // device
        if (alias_for.empty() && config_name.empty())
            alias_for.insert(split(it->second)[0]);
    }
    for (const auto& it : config) {
        // Skip proxy properties
        if (ov::device::id.name() == it.first || it.first == ov::device::priorities.name() ||
            it.first == "DEVICES_PRIORITY" || it.first == "ALIAS_FOR")
            continue;
        configs[config_name][it.first] = it.second;
    }
}

InferenceEngine::IExecutableNetworkInternal::Ptr ov::proxy::Plugin::LoadExeNetworkImpl(
    const InferenceEngine::CNNNetwork& network,
    const std::map<std::string, std::string>& config) {
    auto dev_name = get_fallback_device(get_device_from_config(config));
    // Initial device config should be equal to default global config
    auto device_config = configs[""];
    bool is_device = is_device_in_config(config);
    if (is_device) {
        // Adds device specific options
        for (const auto& it : configs[dev_name]) {
            device_config[it.first] = it.second;
        }
    }
    // TODO: What if user wants to change fallback_priority for the network
    for (const auto& it : config) {
        device_config[it.first] = it.second;
    }
    // Remove proxy properties
    auto it = device_config.find(ov::device::id.name());
    if (it != device_config.end())
        device_config.erase(it);
    it = device_config.find(ov::device::priorities.name());
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
    size_t device_id = get_device_from_config(options);
    const std::string config_name = is_device_in_config(options) ? std::to_string(device_id) : "";
    if (name == ov::device::id)
        return std::to_string(device_id);

    if (name == ov::device::priorities) {
        return split(get_property(name, config_name));
    }

    return GetCore()->GetConfig(get_primary_device(device_id), name);
}

InferenceEngine::Parameter ov::proxy::Plugin::GetMetric(
    const std::string& name,
    const std::map<std::string, InferenceEngine::Parameter>& options) const {
    std::string device_name = get_primary_device(get_device_from_config(options));

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

        // Add proxy specific options
        {
            // Fallback order
            ov::PropertyName property(ov::device::priorities.name(), ov::device::priorities.mutability);
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
    OPENVINO_ASSERT(!alias_for.empty());  // alias_for cannot be empty. 1 is for fallback mode, >1 in other

    // If we have 1 alias we use simple hetero mode
    if (alias_for.size() == 1) {
        auto device = *alias_for.begin();
        const std::vector<std::string> real_devices_ids = core->get_property(device, ov::available_devices);
        for (const auto& device_id : real_devices_ids) {
            const std::string full_device_name = device + '.' + device_id;
            std::vector<std::string> devices{full_device_name};

            // Add fallback devices use device_id for individual fallback property
            for (const auto& fallback_dev : split(get_property(ov::device::priorities.name(), device_id))) {
                devices.emplace_back(fallback_dev);
            }
            result.emplace_back(devices);
        }
    } else {
        typedef struct DeviceId {
            ov::device::UUID uuid;
            std::unordered_map<std::string, std::string> device_to_full_name;
            bool no_uuid;
        } DeviceID_t;
        OPENVINO_ASSERT(device_order.size() == alias_for.size());

        // 1. Get all available devices
        //   Highlevel devices list contains only unique which:
        //    * don't support uuid
        //    * uuid is unique
        // 2. Use individual fallback priorities to fill each list
        std::vector<DeviceID_t> all_highlevel_devices;
        std::set<std::array<uint8_t, ov::device::UUID::MAX_UUID_SIZE>> unique_devices;
        for (const auto& device : device_order) {
            const std::vector<std::string> supported_device_ids = core->get_property(device, ov::available_devices);
            for (const auto& device_id : supported_device_ids) {
                const std::string full_device_name = device + '.' + device_id;
                try {
                    ov::device::UUID uuid = core->get_property(full_device_name, ov::device::uuid.name(), {});
                    auto it = unique_devices.find(uuid.uuid);
                    if (it == unique_devices.end()) {
                        unique_devices.insert(uuid.uuid);
                        DeviceID_t id;
                        id.no_uuid = false;
                        id.uuid = uuid;
                        id.device_to_full_name[device] = full_device_name;
                        all_highlevel_devices.emplace_back(id);
                    } else {
                        for (auto&& dev_id : all_highlevel_devices) {
                            if (dev_id.uuid.uuid == uuid.uuid) {
                                dev_id.device_to_full_name[device] = full_device_name;
                                break;
                            }
                        }
                    }
                } catch (...) {
                    // Device doesn't have UUID, so it means that device is unique
                    DeviceID_t id;
                    id.no_uuid = false;
                    id.device_to_full_name[device] = full_device_name;
                    all_highlevel_devices.emplace_back(id);
                }
            }
        }

        // Use individual fallback order to generate result list
        for (size_t i = 0; i < all_highlevel_devices.size(); i++) {
            std::vector<std::string> real_fallback_order;
            auto device = all_highlevel_devices[i];
            // In case of aliases use the proxy system of enumeration devices
            const auto fallback_order = split(get_property(ov::device::priorities.name(), std::to_string(i)));

            bool found_primary_device = false;
            bool use_hetero_mode = device.no_uuid ? true : false;
            std::vector<std::string> device_order;
            for (const auto& fallback_dev : fallback_order) {
                if (!found_primary_device) {
                    auto it = device.device_to_full_name.find(fallback_dev);
                    if (it != device.device_to_full_name.end()) {
                        device_order.emplace_back(it->second);
                        real_fallback_order.emplace_back(it->first);
                        found_primary_device = true;
                        continue;
                    } else {
                        continue;
                    }
                }
                // In case of hetero mode just add necessary devices
                if (use_hetero_mode) {
                    device_order.emplace_back(fallback_dev);
                    real_fallback_order.emplace_back(fallback_dev);
                    continue;
                }
                // Try to find unique device
                const std::vector<std::string> supported_device_ids =
                    core->get_property(fallback_dev, ov::available_devices);
                bool found_device = false;
                bool dev_without_uuid = false;
                for (const auto& device_id : supported_device_ids) {
                    const std::string full_device_name = fallback_dev + '.' + device_id;
                    try {
                        ov::device::UUID uuid = core->get_property(full_device_name, ov::device::uuid.name(), {});
                        if (uuid.uuid == device.uuid.uuid) {
                            device_order.emplace_back(full_device_name);
                            real_fallback_order.emplace_back(fallback_dev);
                            found_device = true;
                            break;
                        }
                    } catch (...) {
                        dev_without_uuid = true;
                    }
                }
                // Enable hetero mode if device wasn't found
                if (!found_device && dev_without_uuid) {
                    use_hetero_mode = true;
                    device_order.emplace_back(fallback_dev);
                    real_fallback_order.emplace_back(fallback_dev);
                }
            }
            if (device_order.empty()) {
                device_order.emplace_back(device.device_to_full_name.begin()->second);
                real_fallback_order.emplace_back(device.device_to_full_name.begin()->first);
            }
            result.emplace_back(device_order);
            std::string new_fallback;
            for (const auto& dev : real_fallback_order) {
                if (!new_fallback.empty())
                    new_fallback += ",";
                new_fallback += dev;
            }
            std::lock_guard<std::mutex> lock(plugin_mutex);
            configs[std::to_string(i)][ov::device::priorities.name()] = new_fallback;
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
