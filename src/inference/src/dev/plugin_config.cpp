// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/plugin_config.hpp"
#include "openvino/core/any.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/device_id_parser.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/env_util.hpp"
#include <fstream>

#ifdef JSON_HEADER
#    include <json.hpp>
#else
#    include <nlohmann/json.hpp>
#endif

namespace ov {

void PluginConfig::set_property(const AnyMap& config) {
    for (auto& kv : config) {
        auto& name = kv.first;
        auto& val = kv.second;

        const auto& known_options = m_options_map;
        auto it = std::find_if(known_options.begin(), known_options.end(), [&](const OptionMapEntry& o) { return o.first == name; });
        OPENVINO_ASSERT(it != known_options.end(), "Option not found: ", name);
        OPENVINO_ASSERT(it->second != nullptr, "Option is invalid: ", name);

        it->second->set_any(val);
    }
}

ov::Any PluginConfig::get_property(const std::string& name) const {
    const auto& known_options = m_options_map;
    auto it = std::find_if(known_options.begin(), known_options.end(), [&](const OptionMapEntry& o) { return o.first == name; });
    OPENVINO_ASSERT(it != known_options.end(), "Option not found: ", name);
    OPENVINO_ASSERT(it->second != nullptr, "Option is invalid: ", name);

    return it->second->get_any();
}

void PluginConfig::set_user_property(const AnyMap& config) {
    for (auto& kv : config) {
        auto& name = kv.first;
        auto& val = kv.second;

        const auto& known_options = m_options_map;
        auto it = std::find_if(known_options.begin(), known_options.end(), [&](const OptionMapEntry& o) { return o.first == name; });
        OPENVINO_ASSERT(it != known_options.end(), "Option not found: ", name);
        OPENVINO_ASSERT(it->second != nullptr, "Option is invalid: ", name);
        OPENVINO_ASSERT(it->second->is_valid_value(val), "Invalid value: ", val.as<std::string>(), " for property: ",  name);

        user_properties[name] = val;
    }
}

void PluginConfig::finalize(std::shared_ptr<IRemoteContext> context, const ov::RTMap& rt_info) {
    apply_rt_info(context, rt_info);
    apply_debug_options(context);
    // Copy internal properties before applying hints to ensure that
    // a property set by hint won't be overriden by a value in user config.
    // E.g num_streams=AUTO && hint=THROUGHPUT
    // If we apply hints first and then copy all values from user config to internal one,
    // then we'll get num_streams=AUTO in final config while some integer number is expected.
    for (const auto& prop : user_properties) {
        auto& option = m_options_map.at(prop.first);
        option->set_any(prop.second);
    }

    finalize_impl(context);

    // Clear properties after finalize_impl to be able to check if a property was set by user during plugin-side finalization
    user_properties.clear();
}

void PluginConfig::apply_debug_options(std::shared_ptr<IRemoteContext> context) {
    if (context) {
        ov::AnyMap config_properties = read_config_file("config.json", context->get_device_name());
        cleanup_unsupported(config_properties);
        set_user_property(config_properties);
    }

    ov::AnyMap env_properties = read_env({"OV_"});
    set_user_property(env_properties);
}

ov::AnyMap PluginConfig::read_config_file(const std::string& filename, const std::string& target_device_name) const {
    ov::AnyMap config;

    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        return config;
    }

    nlohmann::json json_config;
    try {
        ifs >> json_config;
    } catch (const std::exception& e) {
        return config;
    }

    DeviceIDParser parser(target_device_name);
    for (auto item = json_config.cbegin(), end = json_config.cend(); item != end; ++item) {
        const std::string& device_name = item.key();
        if (DeviceIDParser(device_name).get_device_name() != parser.get_device_name())
            continue;

        const auto& item_value = item.value();
        for (auto option = item_value.cbegin(), item_value_end = item_value.cend(); option != item_value_end; ++option) {
            config[option.key()] = option.value().get<std::string>();
        }
    }

    return config;
}

ov::AnyMap PluginConfig::read_env(const std::vector<std::string>& prefixes) const {
    ov::AnyMap config;

    for (auto& kv : m_options_map) {
        for (auto& prefix : prefixes) {
            auto var_name = prefix + kv.first;
            const auto& val = ov::util::getenv_string(var_name.c_str());

            if (!val.empty()) {
                if (dynamic_cast<ConfigOption<bool>*>(kv.second) != nullptr) {
                    const std::set<std::string> off = {"0", "false", "off", "no"};
                    const std::set<std::string> on = {"1", "true", "on", "yes"};

                    const auto& val_lower = ov::util::to_lower(val);
                    if (off.count(val_lower)) {
                        config[kv.first] = false;
                    } else if (on.count(val_lower)) {
                        config[kv.first] = true;
                    } else {
                        OPENVINO_THROW("Unexpected value for boolean property: ", val);
                    }
                } else {
                    config[kv.first] = val;
                }
                break;
            }
        }
    }

    return config;
}

void PluginConfig::cleanup_unsupported(ov::AnyMap& config) const {
    for (auto it = config.begin(); it != config.end();) {
        const auto& known_options = m_options_map;
        auto& name = it->first;
        auto opt_it = std::find_if(known_options.begin(), known_options.end(), [&](const OptionMapEntry& o) { return o.first == name; });
        if (opt_it == known_options.end()) {
            it = config.erase(it);
        } else {
            ++it;
        }
    }
}

std::string PluginConfig::to_string() const {
    std::stringstream s;

    s << "-----------------------------------------\n";
    s << "PROPERTIES:\n";

    for (const auto& option : m_options_map) {
        s << "\t" << option.first << ": " << option.second->get_any().as<std::string>() << std::endl;
    }
    s << "USER PROPERTIES:\n";
    for (const auto& user_prop : user_properties) {
        s << "\t" << user_prop.first << ": " << user_prop.second.as<std::string>() << std::endl;
    }

    return s.str();
}

}  // namespace ov
