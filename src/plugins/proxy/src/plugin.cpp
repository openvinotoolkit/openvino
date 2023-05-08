// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin.hpp"

#include "openvino/core/any.hpp"
#include "openvino/runtime/device_id_parser.hpp"
#include "proxy_plugin.hpp"

namespace {

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

bool is_device_in_config(const ov::AnyMap& config) {
    return config.find(ov::device::id.name()) != config.end();
}

size_t get_device_from_config(const ov::AnyMap& config) {
    if (is_device_in_config(config))
        return config.at(ov::device::id.name()).as<size_t>();
    return 0;
}

ov::AnyMap remove_device_properties(ov::AnyMap& config, const std::vector<std::string>& devices) {
    ov::AnyMap result;
    std::unordered_set<std::string> devs;
    for (const auto& dev : devices)
        devs.insert(dev);

    for (const auto& it : config) {
        ov::DeviceIDParser parser(it.first);
        if (devs.find(it.first) != devs.end() || devs.find(parser.get_device_name()) != devs.end()) {
            // It is a device property
            result[it.first] = it.second;
        }
    }

    // Remove device properties from config
    for (const auto& it : result) {
        auto c_it = config.find(it.first);
        if (c_it != config.end())
            config.erase(c_it);
    }
    return result;
}

ov::AnyMap remove_device_properties(ov::AnyMap& config, const std::string& devices) {
    return remove_device_properties(config, split(devices, " "));
}

ov::AnyMap remove_device_properties(ov::AnyMap& config, const ov::Any& devices) {
    if (devices.is<std::vector<std::string>>())
        return remove_device_properties(config, devices.as<std::vector<std::string>>());
    else
        return remove_device_properties(config, devices.as<std::string>());
}

ov::AnyMap remove_proxy_properties(ov::AnyMap& config, bool rem_device_properties = false) {
    ov::AnyMap dev_properties;
    auto it = config.find(ov::device::id.name());
    if (it != config.end())
        config.erase(it);
    it = config.find(ov::device::priorities.name());
    if (it != config.end()) {
        if (rem_device_properties)
            dev_properties = remove_device_properties(config, it->second);
        config.erase(it);
    }
    it = config.find("ALIAS_FOR");
    if (it != config.end())
        config.erase(it);
    it = config.find("DEVICES_PRIORITY");
    if (it != config.end())
        config.erase(it);
    it = config.find("FALLBACK_PRIORITY");
    if (it != config.end())
        config.erase(it);
    return dev_properties;
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
        OPENVINO_THROW("Cannot restore fallback devices priority from the next config: ", original_order);
    }
    for (const auto& dev : dev_order) {
        if (!result.empty())
            result += " ";
        result += dev;
    }
    return result;
}

ov::proxy::Plugin::Plugin() {
    // Create global config
    m_configs[""] = {};
}
ov::proxy::Plugin::~Plugin() = default;

ov::SupportedOpsMap ov::proxy::Plugin::query_model(const std::shared_ptr<const ov::Model>& model,
                                                   const ov::AnyMap& properties) const {
    size_t num_devices = get_hidden_devices().size();
    // Recall for HW device
    auto dev_id = get_device_from_config(properties);
    auto config_copy = properties;
    remove_proxy_properties(config_copy);
    auto res = get_core()->query_model(model, get_fallback_device(dev_id), config_copy);
    // Replace hidden device name
    for (auto&& it : res) {
        it.second = get_device_name();
        if (num_devices > 1)
            it.second += "." + std::to_string(dev_id);
    }
    return res;
}

void ov::proxy::Plugin::set_property(const ov::AnyMap& properties) {
    OPENVINO_NOT_IMPLEMENTED;
}

ov::Any ov::proxy::Plugin::get_property(const std::string& name, const ov::AnyMap& arguments) const {
    OPENVINO_NOT_IMPLEMENTED;
}
std::shared_ptr<ov::ICompiledModel> ov::proxy::Plugin::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                                     const ov::AnyMap& properties) const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::ICompiledModel> ov::proxy::Plugin::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                                     const ov::AnyMap& properties,
                                                                     const ov::RemoteContext& context) const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::IRemoteContext> ov::proxy::Plugin::create_context(const ov::AnyMap& remote_properties) const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::IRemoteContext> ov::proxy::Plugin::get_default_context(const ov::AnyMap& remote_properties) const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::ICompiledModel> ov::proxy::Plugin::import_model(std::istream& model,
                                                                    const ov::AnyMap& properties) const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::ICompiledModel> ov::proxy::Plugin::import_model(std::istream& model,
                                                                    const ov::RemoteContext& context,
                                                                    const ov::AnyMap& properties) const {
    OPENVINO_NOT_IMPLEMENTED;
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

std::vector<std::vector<std::string>> ov::proxy::Plugin::get_hidden_devices() const {
    // Proxy plugin has 2 modes of matching devices:
    //  * Fallback - in this mode we report devices only for the first hidden plugin
    //  * Alias - Case when we group all devices under one common name
    std::vector<std::vector<std::string>> result;
    const auto core = get_core();
    OPENVINO_ASSERT(core != nullptr);
    OPENVINO_ASSERT(!m_alias_for.empty());  // alias_for cannot be empty. 1 is for fallback mode, >1 in other

    // If we have 1 alias we use simple hetero mode
    if (m_alias_for.size() == 1) {
        auto device = *m_alias_for.begin();
        const std::vector<std::string> real_devices_ids = core->get_property(device, ov::available_devices);
        for (const auto& device_id : real_devices_ids) {
            const std::string full_device_name = device_id.empty() ? device : device + '.' + device_id;
            std::vector<std::string> devices{full_device_name};

            // Add fallback devices use device_id for individual fallback property
            auto fallback = get_internal_property(ov::device::priorities.name(), device_id).as<std::string>();
            if (!fallback.empty()) {
                for (const auto& fallback_dev : split(fallback, " ")) {
                    devices.emplace_back(fallback_dev);
                }
            }
            result.emplace_back(devices);
        }
    } else {
        typedef struct DeviceId {
            ov::device::UUID uuid;
            std::unordered_map<std::string, std::string> device_to_full_name;
            bool no_uuid;
        } DeviceID_t;
        OPENVINO_ASSERT(m_device_order.size() == m_alias_for.size());

        // 1. Get all available devices
        //   Highlevel devices list contains only unique which:
        //    * don't support uuid
        //    * uuid is unique
        // 2. Use individual fallback priorities to fill each list
        std::vector<DeviceID_t> all_highlevel_devices;
        std::set<std::array<uint8_t, ov::device::UUID::MAX_UUID_SIZE>> unique_devices;
        for (const auto& device : m_device_order) {
            const std::vector<std::string> supported_device_ids = core->get_property(device, ov::available_devices);
            for (const auto& device_id : supported_device_ids) {
                const std::string full_device_name = device_id.empty() ? device : device + '.' + device_id;
                try {
                    ov::device::UUID uuid =
                        core->get_property(full_device_name, ov::device::uuid.name(), {}).as<ov::device::UUID>();
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
            const auto fallback_order =
                split(get_internal_property(ov::device::priorities.name(), std::to_string(i)).as<std::string>(), " ");

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
                        ov::device::UUID uuid =
                            core->get_property(full_device_name, ov::device::uuid.name(), {}).as<ov::device::UUID>();
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
                    new_fallback += " ";
                new_fallback += dev;
            }
            std::lock_guard<std::mutex> lock(m_plugin_mutex);
            m_configs[std::to_string(i)][ov::device::priorities.name()] = new_fallback;
        }
    }
    return result;
}

ov::Any ov::proxy::Plugin::get_internal_property(const std::string& property, const std::string& config_name) const {
    std::lock_guard<std::mutex> lock(m_plugin_mutex);
    ov::Any result;
    auto name = config_name;
    // If device specific config wasn't found or property in config wasn't found  use global config
    auto it = m_configs.find(name);
    if (it == m_configs.end() || it->second.find(property) == it->second.end())
        name = "";

    it = m_configs.find(name);
    if (it->second.find(property) != it->second.end())
        result = it->second.at(property);

    return result;
}

void ov::proxy::create_plugin(::std::shared_ptr<::ov::IPlugin>& plugin) {
    static const ov::Version version = {CI_BUILD_NUMBER, "openvino_proxy_plugin"};
    try {
        plugin = ::std::make_shared<ov::proxy::Plugin>();
    } catch (const ov::Exception&) {
        throw;
    } catch (const std::exception& ex) {
        OPENVINO_THROW(ex.what());
    }
    plugin->set_version(version);
}
