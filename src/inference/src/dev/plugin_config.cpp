// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/plugin_config.hpp"
#include "openvino/core/any.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/device_id_parser.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/env_util.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>

#ifdef JSON_HEADER
#    include <json.hpp>
#else
#    include <nlohmann/json.hpp>
#endif

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <sys/ioctl.h>
#endif

namespace {
size_t get_terminal_width() {
    const size_t default_width = 120;
#ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi)) {
        return csbi.srWindow.Right - csbi.srWindow.Left + 1;
    } else {
        return default_width;
    }
#else
    struct winsize w;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0) {
        return w.ws_col;
    } else {
        return default_width;
    }
#endif  // _WIN32
}
}

namespace ov {

ov::Any PluginConfig::get_property(const std::string& name) const {
    const static std::vector<OptionVisibility> allowed_visibility = {OptionVisibility::RELEASE, OptionVisibility::RELEASE_INTERNAL};
    return get_property(name, allowed_visibility);
}

ov::Any PluginConfig::get_property(const std::string& name, const std::vector<OptionVisibility>& allowed_visibility) const {
    if (m_user_properties.find(name) != m_user_properties.end()) {
        return m_user_properties.at(name);
    }

    auto option = get_option_ptr(name);
     if (std::find(allowed_visibility.begin(), allowed_visibility.end(), option->get_visibility()) == allowed_visibility.end()) {
        OPENVINO_THROW("Couldn't get unknown property: ", name);
    }

    return option->get_any();
}

void PluginConfig::set_property(const AnyMap& config) {
    const static std::vector<OptionVisibility> allowed_visibility = {OptionVisibility::RELEASE,OptionVisibility::RELEASE_INTERNAL, OptionVisibility::DEBUG};
    const bool throw_on_error = true;
    set_property(config, allowed_visibility, throw_on_error);
}

void PluginConfig::set_property(const ov::AnyMap& config, const std::vector<OptionVisibility>& allowed_visibility, bool throw_on_error) {
    OPENVINO_ASSERT(!m_is_finalized, "Setting property after config finalization is prohibited");

    for (auto& kv : config) {
        auto& name = kv.first;
        auto& val = kv.second;

        auto option = get_option_ptr(name);
        if (std::find(allowed_visibility.begin(), allowed_visibility.end(), option->get_visibility()) == allowed_visibility.end()) {
            if (throw_on_error)
                OPENVINO_THROW("Couldn't set unknown property: ", name);
            else
                continue;
        }
        if (!option->is_valid_value(val)) {
            if (throw_on_error)
                OPENVINO_THROW("Invalid value: ", val.as<std::string>(), " for property: ",  name, "\nProperty description: ", get_help_message(name));
            else
                continue;
        }

        m_user_properties[name] = val;
    }
}

void PluginConfig::finalize(std::shared_ptr<IRemoteContext> context, const ov::RTMap& rt_info) {
    if (m_is_finalized)
        return;

    apply_rt_info(context, rt_info);
    apply_debug_options(context);
    // Copy internal properties before applying hints to ensure that
    // a property set by hint won't be overriden by a value in user config.
    // E.g num_streams=AUTO && hint=THROUGHPUT
    // If we apply hints first and then copy all values from user config to internal one,
    // then we'll get num_streams=AUTO in final config while some integer number is expected.
    for (const auto& prop : m_user_properties) {
        auto& option = m_options_map.at(prop.first);
        option->set_any(prop.second);
    }

    finalize_impl(context);

    // Clear properties after finalize_impl to be able to check if a property was set by user during plugin-side finalization
    m_user_properties.clear();

    m_is_finalized = true;
}

bool PluginConfig::visit_attributes(ov::AttributeVisitor& visitor) const {
    // for (const auto& prop : m_user_properties) {
    //     visitor.on_attribute(prop.first + "__user", prop.second.as<std::string>());
    // }
    // for (const auto& prop : m_options_map) {
    //     visitor.on_attribute(prop.first + "__internal", prop.second->get_any().as<std::string>());
    // }
    // visitor.on_attribute("is_finalized", m_is_finalized);

    return true;
}

void PluginConfig::apply_debug_options(std::shared_ptr<IRemoteContext> context) {
    static std::vector<OptionVisibility> allowed_visibility = {
        OptionVisibility::RELEASE,
        OptionVisibility::RELEASE_INTERNAL,
        OptionVisibility::DEBUG
#ifdef ENABLE_DEBUG_CAPS
#endif
    };

    const bool throw_on_error = false;

    if (context) {
        ov::AnyMap config_properties = read_config_file("config.json", context->get_device_name());
        cleanup_unsupported(config_properties);
        set_property(config_properties, allowed_visibility, throw_on_error);
    }

    ov::AnyMap env_properties = read_env({"OV_"});
    cleanup_unsupported(env_properties);
    set_property(env_properties, allowed_visibility, throw_on_error);
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
        auto& name = it->first;
        auto opt_it = std::find_if(m_options_map.begin(), m_options_map.end(), [&](const OptionMapEntry& o) { return o.first == name; });
        if (opt_it == m_options_map.end()) {
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
    for (const auto& user_prop : m_user_properties) {
        s << "\t" << user_prop.first << ": " << user_prop.second.as<std::string>() << std::endl;
    }

    return s.str();
}

void PluginConfig::print_help() const {
    auto format_text = [](const std::string& cpp_name, const std::string& str_name, const std::string& desc, size_t max_name_width, size_t max_width) {
        std::istringstream words(desc);
        std::ostringstream formatted_text;
        std::string word;
        std::vector<std::string> words_vec;

        while (words >> word) {
            words_vec.push_back(word);
        }

        size_t j = 0;
        size_t count_of_desc_lines = (desc.length() + max_width - 1) / max_width;
        for (size_t i = 0 ; i < std::max<size_t>(2, count_of_desc_lines); i++) {
            if (i == 0) {
                formatted_text << std::left << std::setw(max_name_width) << cpp_name;
            } else if (i == 1) {
                formatted_text << std::left << std::setw(max_name_width) << str_name;
            } else {
                formatted_text << std::left << std::setw(max_name_width) << "";
            }

            formatted_text << " | ";

            size_t line_length = max_name_width + 3;
            for (; j < words_vec.size();) {
                line_length += words_vec[j].size() + 1;
                if (line_length > max_width) {
                    break;
                } else {
                    formatted_text << words_vec[j] << " ";
                }
                j++;
            }
            formatted_text << "\n";
        }
        return formatted_text.str();
    };

    const auto& options_desc = get_options_desc();
    std::stringstream ss;
    auto max_name_length_item = std::max_element(options_desc.begin(), options_desc.end(),
        [](const OptionsDesc::value_type& a, const OptionsDesc::value_type& b){
            return std::get<0>(a).size() < std::get<0>(b).size();
    });

    const size_t max_name_width = static_cast<int>(std::get<0>(*max_name_length_item).size() + std::get<1>(*max_name_length_item).size());
    const size_t terminal_width = get_terminal_width();
    ss << std::left << std::setw(max_name_width) << ("Option name") << " | " << " Description " << "\n";
    ss << std::left << std::setw(terminal_width) << std::setfill('-') << "" << "\n";
    for (auto& kv : options_desc) {
        ss << format_text(std::get<0>(kv), std::get<1>(kv), std::get<2>(kv), max_name_width, terminal_width) << "\n";
    }

    std::cout << ss.str();
}

const std::string PluginConfig::get_help_message(const std::string& name) const {
    const auto& options_desc = get_options_desc();
    auto it = std::find_if(options_desc.begin(), options_desc.end(), [&](const OptionsDesc::value_type& v) { return std::get<1>(v) == name; });
    if (it != options_desc.end()) {
        return std::get<2>(*it);
    }

    return "";
}

}  // namespace ov
