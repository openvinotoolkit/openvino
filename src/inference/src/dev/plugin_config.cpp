// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/plugin_config.hpp"
#include "openvino/core/except.hpp"


namespace ov {

void PluginConfig::set_property(const AnyMap& config) {
    for (auto& kv : config) {
        auto& name = kv.first;
        auto& val = kv.second;

        const auto& known_options = m_options_map;
        auto it = std::find_if(known_options.begin(), known_options.end(), [&](const OptionMapEntry& o) { return o.first == name; });
        OPENVINO_ASSERT(it != known_options.end());

        it->second->set_any(val);
    }
}

ov::Any PluginConfig::get_property(const std::string& name) const {
    const auto& known_options = m_options_map;
    auto it = std::find_if(known_options.begin(), known_options.end(), [&](const OptionMapEntry& o) { return o.first == name; });
    OPENVINO_ASSERT(it != known_options.end(), "Option not found: ", name);

    return it->second->get_any();
}

void PluginConfig::set_user_property(const AnyMap& config) {
    for (auto& kv : config) {
        auto& name = kv.first;
        auto& val = kv.second;

        const auto& known_options = m_options_map;
        auto it = std::find_if(known_options.begin(), known_options.end(), [&](const OptionMapEntry& o) { return o.first == name; });
        OPENVINO_ASSERT(it != known_options.end(), "Option not found: ", name);
        OPENVINO_ASSERT(it->second->is_valid_value(val), "Invalid value: ", val.as<std::string>(), " for property: ",  name);

        user_properties[name] = val;
    }
}

void PluginConfig::finalize(std::shared_ptr<IRemoteContext> context, const ov::RTMap& rt_info) {
    // Copy internal properties before applying hints to ensure that
    // a property set by hint won't be overriden by a value in user config.
    // E.g num_streams=AUTO && hint=THROUGHPUT
    // If we apply hints first and then copy all values from user config to internal one,
    // then we'll get num_streams=AUTO in final config while some integer number is expected.
    for (const auto& prop : user_properties) {
        auto& option = m_options_map.at(prop.first);
        option->set_any(prop.second);
    }

    finalize_impl(context, rt_info);

    // Clear properties after finalize_impl to be able to check if a property was set by user during plugin-side finalization
    user_properties.clear();
}

std::string PluginConfig::to_string() const {
    std::stringstream s;

    s << "-----------------------------------------\n";
    s << "PROPERTIES:\n";

    for (const auto& option : m_options_map) {
        s << "\t" << option.first << ":" << option.second->get_any().as<std::string>() << std::endl;
    }
    s << "USER PROPERTIES:\n";
    for (const auto& user_prop : user_properties) {
        s << "\t" << user_prop.first << ": " << user_prop.second.as<std::string>() << std::endl;
    }

    return s.str();
}

}  // namespace ov
