// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/proxy/plugin.hpp"

#include <memory>
#include <mutex>
#include <stdexcept>

#include "compiled_model.hpp"
#include "openvino/core/any.hpp"
#include "openvino/core/except.hpp"
#include "openvino/proxy/properties.hpp"
#include "openvino/runtime/device_id_parser.hpp"
#include "openvino/runtime/iinfer_request.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "openvino/util/common_util.hpp"
#include "plugin.hpp"
#include "remote_context.hpp"

#ifdef __GLIBC__
#    include <gnu/libc-version.h>
#    if __GLIBC_MINOR__ < 34
#        define OV_GLIBC_VERSION_LESS_2_34
#    endif
#endif

namespace {

bool compare_containers(const std::vector<std::string>& c1, const std::vector<std::string>& c2) {
    if (c1.size() != c2.size())
        return false;
    for (size_t i = 0; i < c1.size(); i++) {
        if (c1.at(i) != c2.at(i))
            return false;
    }
    return true;
}

bool compare_containers(const std::unordered_set<std::string>& c1, const std::unordered_set<std::string>& c2) {
    if (c1.size() != c2.size())
        return false;
    for (const auto& val : c1) {
        if (c2.find(val) == c2.end())
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

bool is_device_in_config(const ov::AnyMap& config) {
    return config.find(ov::device::id.name()) != config.end();
}

ov::AnyMap remove_device_properties(ov::AnyMap& config, const std::vector<std::string>& devices) {
    ov::AnyMap result;
    std::unordered_set<std::string> devs;
    for (const auto& dev : devices)
        devs.insert(dev);

    for (const auto& it : config) {
        auto subprop_device_name_pos = it.first.find(ov::device::properties.name() + std::string("_"));
        if (subprop_device_name_pos == std::string::npos)
            continue;
        auto subprop_device_name =
            it.first.substr(subprop_device_name_pos + std::strlen(ov::device::properties.name()) + 1);
        ov::DeviceIDParser parser(subprop_device_name);
        if (devs.find(subprop_device_name) != devs.end() || devs.find(parser.get_device_name()) != devs.end()) {
            // It is a device property
            result[subprop_device_name] = it.second;
        }
    }

    // Remove device properties from config
    for (const auto& it : result) {
        auto c_it = config.find(ov::device::properties.name() + std::string("_") + it.first);
        if (c_it != config.end())
            config.erase(c_it);
    }
    return result;
}

ov::AnyMap remove_proxy_properties(ov::AnyMap& config, bool rem_device_properties = false) {
    const static std::vector<ov::PropertyName> proxy_properties = {ov::device::id,
                                                                   ov::internal::config_device_id,
                                                                   ov::proxy::configuration::internal_name,
                                                                   ov::device::priorities,
                                                                   ov::proxy::alias_for,
                                                                   ov::proxy::device_priorities};
    ov::AnyMap dev_properties;
    for (const auto& property : proxy_properties) {
        auto it = config.find(property);
        if (it == config.end())
            continue;
        if (ov::device::priorities == property && rem_device_properties)
            dev_properties = remove_device_properties(config, it->second.as<std::vector<std::string>>());

        config.erase(it);
    }

    return dev_properties;
}

// add cached properties for device configuration
ov::AnyMap construct_device_config(const std::string& dev_name,
                                   const std::unordered_map<std::string, ov::AnyMap>& configs,
                                   const ov::AnyMap& properties) {
    // Initial device config should be equal to default global config
    auto it = configs.find("");
    ov::AnyMap device_config = it != configs.end() ? it->second : ov::AnyMap{};
    it = configs.find(dev_name);
    bool is_device = is_device_in_config(properties) && it != configs.end();
    if (is_device) {
        // Adds device specific options
        for (const auto& it : it->second) {
            device_config[it.first] = it.second;
        }
    }
    for (const auto& it : properties) {
        device_config[it.first] = it.second;
    }
    remove_proxy_properties(device_config);
    return device_config;
}

}  // namespace

size_t ov::proxy::Plugin::get_device_from_config(const ov::AnyMap& config) const {
    if (is_device_in_config(config))
        return config.at(ov::device::id.name()).as<size_t>();
    return m_default_device;
}

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
    auto hw_config = properties;
    // Parse default device ID and remove from config
    auto it = hw_config.find(ov::device::id.name());
    if (it != hw_config.end()) {
        m_default_device = it->second.as<size_t>();
        hw_config.erase(it);
    }

    // Replace device::id by CONFIG_DEVICE_ID
    it = hw_config.find(ov::internal::config_device_id.name());
    if (it != hw_config.end()) {
        hw_config[ov::device::id.name()] = it->second;
        hw_config.erase(it);
    }
    // Empty config_name means means global config for all devices
    std::string config_name = is_device_in_config(hw_config) ? std::to_string(get_device_from_config(hw_config)) : "";

    bool proxy_config_was_changed = false;
    // Parse alias config
    it = hw_config.find(ov::proxy::alias_for.name());
    bool fill_order = hw_config.find(ov::proxy::device_priorities.name()) == hw_config.end() && m_device_order.empty();
    if (it != hw_config.end()) {
        std::unordered_set<std::string> new_alias;
        for (auto&& dev : it->second.as<std::vector<std::string>>()) {
            new_alias.emplace(dev);
            if (fill_order)
                m_device_order.emplace_back(dev);
        }
        if (!compare_containers(m_alias_for, new_alias)) {
            proxy_config_was_changed = true;
            m_alias_for = new_alias;
        }
    }

    // Restore device order
    it = hw_config.find(ov::proxy::device_priorities.name());
    if (it != hw_config.end()) {
        std::vector<std::pair<std::string, size_t>> priority_order;
        // Biggest number means minimum priority
        size_t min_priority(0);
        for (auto&& dev_priority : it->second.as<std::vector<std::string>>()) {
            auto dev_prior = ov::util::split(dev_priority, ':');
            OPENVINO_ASSERT(dev_prior.size() == 2,
                            "Cannot set ov::proxy::device_priorities property. Format is incorrect.");
            auto priority = string_to_size_t(dev_prior[1]);
            if (priority > min_priority)
                min_priority = priority;
            priority_order.push_back(std::pair<std::string, size_t>{dev_prior[0], priority});
        }
        // Devices without priority has lower priority
        min_priority++;
        for (const auto& dev : m_alias_for) {
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
        std::vector<std::string> new_device_order;
        new_device_order.reserve(priority_order.size());
        for (const auto& dev : priority_order) {
            new_device_order.emplace_back(dev.first);
        }
        // Align sizes of device order with alias
        if (new_device_order.size() < m_alias_for.size()) {
            for (const auto& dev : m_alias_for) {
                if (std::find(std::begin(new_device_order), std::end(new_device_order), dev) ==
                    std::end(new_device_order)) {
                    new_device_order.emplace_back(dev);
                }
            }
        }
        if (!compare_containers(m_device_order, new_device_order)) {
            m_device_order = new_device_order;
            proxy_config_was_changed = true;
        }
    }

    {
        // Cannot change config from different threads
        std::lock_guard<std::mutex> lock(m_plugin_mutex);
        it = hw_config.find(ov::device::priorities.name());
        if (it != hw_config.end()) {
            if (m_configs[config_name].find(ov::device::priorities.name()) == m_configs[config_name].end() ||
                !compare_containers(
                    m_configs[config_name][ov::device::priorities.name()].as<std::vector<std::string>>(),
                    it->second.as<std::vector<std::string>>())) {
                proxy_config_was_changed = true;
                m_configs[config_name][ov::device::priorities.name()] = it->second;
            }
            // Main device is needed in case if we don't have alias and would like to be able change fallback order per
            // device
            if (m_alias_for.empty() && config_name.empty()) {
                proxy_config_was_changed = true;
                m_alias_for.insert(it->second.as<std::vector<std::string>>()[0]);
            }
        }
    }
    if (proxy_config_was_changed) {
        // need initialization of hidden devices
        m_init_devs = false;
    }

    // Add fallback priority to detect supported devices in case of HETERO fallback
    auto device_priority = get_internal_property(ov::device::priorities.name(), config_name);
    if (!device_priority.empty())
        hw_config[ov::device::priorities.name()] = device_priority;
    auto dev_id = get_device_from_config(hw_config);
    auto dev_properties = remove_proxy_properties(hw_config, true);

    if (dev_properties.empty() && hw_config.empty())
        // Nothing to do
        return;

    const std::string primary_dev = get_primary_device(dev_id);
    std::string dev_prop_name;
    ov::DeviceIDParser pr_parser(primary_dev);
    for (const auto& it : dev_properties) {
        ov::DeviceIDParser parser(it.first);
        if (parser.get_device_name() == pr_parser.get_device_name()) {
            // Add primary device properties to primary device
            OPENVINO_ASSERT(it.second.is<ov::AnyMap>(),
                            "Internal error. Device properties should be represented as ov::AnyMap.");
            auto dev_map = it.second.as<ov::AnyMap>();
            for (const auto& m_it : dev_map) {
                // Plugin shouldn't contain the different property for the same key
                OPENVINO_ASSERT(
                    hw_config.find(m_it.first) == hw_config.end() || hw_config.at(m_it.first) == m_it.second,
                    "Error found collisions for property: ",
                    m_it.first);
                hw_config[m_it.first] = m_it.second;
            }
            dev_prop_name = it.first;
            break;
        }
    }
    {
        // Cannot change config from different threads
        std::lock_guard<std::mutex> lock(m_plugin_mutex);
        for (const auto& it : hw_config) {
            // Skip proxy and primary device properties
            if (ov::internal::config_device_id.name() == it.first || ov::device::id.name() == it.first ||
                it.first == ov::device::priorities.name() || it.first == ov::proxy::device_priorities.name() ||
                it.first == ov::proxy::alias_for.name() ||
                // Skip options from config for primaty device
                hw_config.find(it.first) != hw_config.end() || (!dev_prop_name.empty() && it.first == dev_prop_name))
                continue;
            // Cache proxy and fallback device options to apply for fallback devices
            m_configs[config_name][it.first] = it.second;
        }
    }
    get_core()->set_property(primary_dev, hw_config);
}

ov::Any ov::proxy::Plugin::get_property(const std::string& name, const ov::AnyMap& arguments) const {
    size_t device_id = get_device_from_config(arguments);
    const std::string config_name = is_device_in_config(arguments) ? std::to_string(device_id) : "";
    if (name == ov::device::id)
        return m_default_device;

    if (name == ov::internal::config_device_id)
        return std::to_string(device_id);

    if (name == ov::device::priorities) {
        return get_internal_property(name, config_name).as<std::vector<std::string>>();
    }
    if (name == ov::available_devices) {
        auto hidden_devices = get_hidden_devices();
        std::vector<std::string> availableDevices(hidden_devices.size());
        for (size_t i = 0; i < hidden_devices.size(); i++) {
            availableDevices[i] = std::to_string(i);
        }
        return decltype(ov::available_devices)::value_type(availableDevices);
    }
    if (name == ov::supported_properties) {
        auto supported_prop =
            get_core()->get_property(get_primary_device(device_id), name, {}).as<std::vector<ov::PropertyName>>();

        // Extend primary device properties by proxy specific property

        // ov::device::id changes the default proxy device
        if (std::find(supported_prop.begin(), supported_prop.end(), ov::device::id) == supported_prop.end())
            supported_prop.emplace_back(ov::device::id);
        return supported_prop;
    } else if (name == ov::internal::supported_properties) {
        auto supported_prop =
            get_core()->get_property(get_primary_device(device_id), name, {}).as<std::vector<ov::PropertyName>>();
        if (std::find(supported_prop.begin(), supported_prop.end(), ov::internal::config_device_id) ==
            supported_prop.end())
            supported_prop.emplace_back(ov::internal::config_device_id);
        return supported_prop;
    }

    if (has_internal_property(name, config_name))
        return get_internal_property(name, config_name);
    return get_core()->get_property(get_primary_device(device_id), name, {});
}

ov::SoPtr<ov::IRemoteContext> ov::proxy::Plugin::create_proxy_context(
    const ov::SoPtr<ov::ICompiledModel>& compiled_model,
    const ov::AnyMap& properties) const {
    auto dev_name = get_device_name();
    auto dev_idx = get_device_from_config(properties);
    auto has_dev_idx = is_device_in_config(properties);
    ov::SoPtr<ov::IRemoteContext> device_context;
    ov::SoPtr<ov::IRemoteContext> remote_context;
    try {
        device_context = compiled_model->get_context();
        if (!device_context._so)
            device_context._so = compiled_model._so;
        remote_context = std::make_shared<ov::proxy::RemoteContext>(device_context, dev_name, dev_idx, has_dev_idx);
    } catch (const ov::NotImplemented&) {
    }
    return remote_context;
}

std::shared_ptr<ov::ICompiledModel> ov::proxy::Plugin::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                                     const ov::AnyMap& properties) const {
    auto dev_name = get_fallback_device(get_device_from_config(properties));
    auto device_config = construct_device_config(dev_name, m_configs, properties);
    std::shared_ptr<const ov::IPlugin> plugin = shared_from_this();

    auto device_model = get_core()->compile_model(model, dev_name, device_config);
    auto remote_context = create_proxy_context(device_model, properties);
    return std::make_shared<ov::proxy::CompiledModel>(device_model, plugin, remote_context);
}

std::shared_ptr<ov::ICompiledModel> ov::proxy::Plugin::compile_model(const std::string& model_path,
                                                                     const ov::AnyMap& properties) const {
    auto dev_name = get_fallback_device(get_device_from_config(properties));
    auto device_config = construct_device_config(dev_name, m_configs, properties);
    std::shared_ptr<const ov::IPlugin> plugin = shared_from_this();

    auto device_model = get_core()->compile_model(model_path, dev_name, device_config);
    auto remote_context = create_proxy_context(device_model, properties);
    return std::make_shared<ov::proxy::CompiledModel>(device_model, plugin, remote_context);
}

std::shared_ptr<ov::ICompiledModel> ov::proxy::Plugin::compile_model(
    const std::shared_ptr<const ov::Model>& model,
    const ov::AnyMap& properties,
    const ov::SoPtr<ov::IRemoteContext>& context) const {
    auto ctx = ov::proxy::RemoteContext::get_hardware_context(context);
    auto dev_name = ctx->get_device_name();
    auto device_config = construct_device_config(dev_name, m_configs, properties);

    std::shared_ptr<const ov::IPlugin> plugin = shared_from_this();
    auto compiled_model =
        std::make_shared<ov::proxy::CompiledModel>(get_core()->compile_model(model, ctx, device_config),
                                                   plugin,
                                                   context);
    return std::dynamic_pointer_cast<ov::ICompiledModel>(compiled_model);
}

ov::SoPtr<ov::IRemoteContext> ov::proxy::Plugin::create_context(const ov::AnyMap& remote_properties) const {
    // TODO: if no device id, try to create context for each plugin
    auto dev_name = get_device_name();
    auto dev_idx = get_device_from_config(remote_properties);
    auto has_dev_idx = is_device_in_config(remote_properties);

    auto device_config = remote_properties;
    remove_proxy_properties(device_config);

    if (has_dev_idx) {
        auto remote_context = std::make_shared<ov::proxy::RemoteContext>(
            get_core()->create_context(get_fallback_device(get_device_from_config(remote_properties)), device_config),
            dev_name,
            dev_idx,
            has_dev_idx);
        return remote_context;
    }
    // Properties doesn't have device id, so try to create context for all devices
    const auto hidden_devices = get_hidden_devices();
    for (size_t i = 0; i < hidden_devices.size(); i++) {
        try {
            auto remote_context = std::make_shared<ov::proxy::RemoteContext>(
                get_core()->create_context(get_fallback_device(get_device_from_config(remote_properties)),
                                           device_config),
                dev_name,
                i,
                has_dev_idx);
            return remote_context;
        } catch (const ov::Exception&) {
        }
    }
    OPENVINO_THROW("Cannot create remote context for provided properties: ",
                   ov::Any(remote_properties).as<std::string>());
}

ov::SoPtr<ov::IRemoteContext> ov::proxy::Plugin::get_default_context(const ov::AnyMap& remote_properties) const {
    auto dev_name = get_device_name();
    auto dev_idx = get_device_from_config(remote_properties);
    auto has_dev_idx = is_device_in_config(remote_properties);

    auto device_config = remote_properties;
    remove_proxy_properties(device_config);

    auto remote_context = std::make_shared<ov::proxy::RemoteContext>(
        get_core()->get_default_context(get_fallback_device(get_device_from_config(remote_properties))),
        dev_name,
        dev_idx,
        has_dev_idx);
    return remote_context;
}

std::shared_ptr<ov::ICompiledModel> ov::proxy::Plugin::import_model(std::istream& model,
                                                                    const ov::AnyMap& properties) const {
    auto dev_name = get_fallback_device(get_device_from_config(properties));
    auto device_config = construct_device_config(dev_name, m_configs, properties);
    auto device_model = get_core()->import_model(model, dev_name, device_config);
    auto remote_context = create_proxy_context(device_model, properties);

    return std::make_shared<ov::proxy::CompiledModel>(device_model, shared_from_this(), remote_context);
}

std::shared_ptr<ov::ICompiledModel> ov::proxy::Plugin::import_model(std::istream& model,
                                                                    const ov::SoPtr<ov::IRemoteContext>& context,
                                                                    const ov::AnyMap& properties) const {
    auto ctx = ov::proxy::RemoteContext::get_hardware_context(context);
    auto dev_name = ctx->get_device_name();
    auto device_config = construct_device_config(dev_name, m_configs, properties);

    return std::make_shared<ov::proxy::CompiledModel>(get_core()->import_model(model, ctx, device_config),
                                                      shared_from_this(),
                                                      context);
}

std::string ov::proxy::Plugin::get_primary_device(size_t idx) const {
    std::vector<std::string> devices;
    const auto all_devices = get_hidden_devices();
    devices.reserve(all_devices.size());
    for (const auto& dev : all_devices) {
        devices.emplace_back(dev.at(0));
    }

    if (devices.empty())
        // Return low level device name in case of no devices wasn't found
        return m_device_order.at(0);
    OPENVINO_ASSERT(devices.size() > idx,
                    "Cannot get primary device for index: ",
                    idx,
                    ". The total number of found devices is ",
                    devices.size());
    return devices[idx];
}

std::string ov::proxy::Plugin::get_fallback_device(size_t idx) const {
    const auto all_devices = get_hidden_devices();
    OPENVINO_ASSERT(all_devices.size() > idx,
                    "Cannot get fallback device for index: ",
                    idx,
                    ". The total number of found devices is ",
                    all_devices.size());
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
    if (m_init_devs)
        return m_hidden_devices;

    std::lock_guard<std::mutex> lock(m_init_devs_mutex);

    if (m_init_devs)
        return m_hidden_devices;

    m_hidden_devices.clear();
    const auto core = get_core();
    OPENVINO_ASSERT(core != nullptr, "Internal error! Plugin doesn't have a pointer to ov::Core");
    OPENVINO_ASSERT(
        !m_alias_for.empty(),
        get_device_name(),
        " cannot find available devices!");  // alias_for cannot be empty. 1 is for fallback mode, >1 in other

    // If we have 1 alias we use simple hetero mode
    if (m_alias_for.size() == 1) {
        auto device = *m_alias_for.begin();
        // Allow to get runtime error, because only one plugin under the alias
        std::vector<std::string> real_devices_ids;
        try {
            real_devices_ids = core->get_property(device, ov::available_devices);
        } catch (const std::runtime_error&) {
            OPENVINO_THROW(get_device_name(), " cannot find available devices!");
        }
        for (const auto& device_id : real_devices_ids) {
            const std::string full_device_name = device_id.empty() ? device : device + '.' + device_id;
            std::vector<std::string> devices;

            // Add fallback devices use device_id for individual fallback property
            auto fallback = get_internal_property(ov::device::priorities.name(), device_id).as<std::string>();
            if (!fallback.empty()) {
                for (const auto& fallback_dev : ov::util::split(fallback, ' ')) {
                    if (fallback_dev != device)
                        devices.emplace_back(fallback_dev);
                    else
                        devices.emplace_back(full_device_name);
                }
            } else {
                devices.emplace_back(full_device_name);
            }
            m_hidden_devices.emplace_back(devices);
        }
    } else {
        typedef struct DeviceId {
            ov::device::UUID uuid;
            std::unordered_map<std::string, std::string> device_to_full_name;
            bool no_uuid;
        } DeviceID_t;
        OPENVINO_ASSERT(m_device_order.size() == m_alias_for.size(),
                        "Internal error! Plugin cannot match device order to devices under the alias");

        // 1. Get all available devices
        //   Highlevel devices list contains only unique which:
        //    * don't support uuid
        //    * uuid is unique
        // 2. Use individual fallback priorities to fill each list
        std::vector<DeviceID_t> all_highlevel_devices;
        std::set<std::array<uint8_t, ov::device::UUID::MAX_UUID_SIZE>> unique_devices;

#ifdef OV_GLIBC_VERSION_LESS_2_34
        // Static unavailable device in order to avoid loading from different ov::Core the same unavailable plugin
        // This issue relates to old libc if we load the same library from different threads
        static std::unordered_set<std::string> unavailable_devices;
        static std::unordered_map<std::string, std::mutex> unavailable_plugin_mutex;
#else
        std::unordered_set<std::string> unavailable_devices;
#endif
        for (const auto& device : m_device_order) {
            // Avoid loading unavailable device for several times
            if (unavailable_devices.count(device))
                continue;
            std::vector<std::string> supported_device_ids;
            {
#ifdef OV_GLIBC_VERSION_LESS_2_34
                std::lock_guard<std::mutex> lock(unavailable_plugin_mutex[device]);
                if (unavailable_devices.count(device))
                    continue;
#endif
                try {
                    supported_device_ids = core->get_property(device, ov::available_devices);
                } catch (const std::runtime_error&) {
                    unavailable_devices.emplace(device);
                    // Device cannot be loaded
                    continue;
                }
            }
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

        OPENVINO_ASSERT(!all_highlevel_devices.empty(),
                        get_device_name(),
                        " cannot find available devices!");  // Devices should be found

        // Use individual fallback order to generate result list
        for (size_t i = 0; i < all_highlevel_devices.size(); i++) {
            std::vector<std::string> real_fallback_order;
            auto device = all_highlevel_devices[i];
            // In case of aliases use the proxy system of enumeration devices
            const auto fallback_order =
                get_internal_property(ov::device::priorities.name(), std::to_string(i)).as<std::vector<std::string>>();

            bool found_primary_device = false;
            bool use_hetero_mode = device.no_uuid ? true : false;
            std::vector<std::string> device_order;
            for (const auto& fallback_dev : fallback_order) {
                if (unavailable_devices.count(fallback_dev))
                    continue;
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
                std::vector<std::string> supported_device_ids = core->get_property(fallback_dev, ov::available_devices);
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
            m_hidden_devices.emplace_back(device_order);
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
    m_init_devs = true;
    return m_hidden_devices;
}

bool ov::proxy::Plugin::has_internal_property(const std::string& property, const std::string& config_name) const {
    std::lock_guard<std::mutex> lock(m_plugin_mutex);
    auto name = config_name;
    // If device specific config wasn't found or property in config wasn't found  use global config
    auto it = m_configs.find(name);
    if (it == m_configs.end() || it->second.find(property) == it->second.end())
        name = "";

    it = m_configs.find(name);
    return (it != m_configs.end() && it->second.find(property) != it->second.end());
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
    if (it != m_configs.end() && it->second.find(property) != it->second.end())
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
