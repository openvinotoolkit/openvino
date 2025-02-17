// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/plugin_config.hpp"

#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <string_view>

#include "openvino/core/any.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/device_id_parser.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/env_util.hpp"

#ifdef JSON_HEADER
#    include <json.hpp>
#else
#    include <nlohmann/json.hpp>
#endif

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <sys/ioctl.h>
#    include <unistd.h>
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
#elif defined(__linux__)
    struct winsize w;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0) {
        return w.ws_col;
    } else {
        return default_width;
    }
#else
    return default_width;
#endif
}
}  // namespace

namespace ov {

ov::Any PluginConfig::get_property(const std::string& name, OptionVisibility allowed_visibility) const {
    if (m_user_properties.find(name) != m_user_properties.end()) {
        return m_user_properties.at(name);
    }

    auto option = get_option_ptr(name);
    OPENVINO_ASSERT((allowed_visibility & option->get_visibility()) == option->get_visibility(),
                    "Couldn't get unknown property: ",
                    name);

    return option->get_any();
}

void PluginConfig::set_property(const ov::AnyMap& config) {
    OPENVINO_ASSERT(!m_is_finalized, "Setting property after config finalization is prohibited");

    for (auto& [name, val] : config) {
        get_option_ptr(name)->set_any(val);
    }
}

void PluginConfig::set_user_property(const ov::AnyMap& config, OptionVisibility allowed_visibility) {
    OPENVINO_ASSERT(!m_is_finalized, "Setting property after config finalization is prohibited");

    for (auto& [name, val] : config) {
        auto option = get_option_ptr(name);
        if ((allowed_visibility & option->get_visibility()) != option->get_visibility()) {
            OPENVINO_THROW("Couldn't set unknown property: ", name);
        }
        if (!option->is_valid_value(val)) {
            OPENVINO_THROW("Invalid value: ",
                           val.as<std::string>(),
                           " for property: ",
                           name,
                           "\nProperty description: ",
                           get_help_message(name));
        }

        m_user_properties[name] = val;
    }
}

void PluginConfig::finalize(const IRemoteContext* context, const ov::Model* model) {
    if (m_is_finalized)
        return;

    if (model)
        apply_model_specific_options(context, *model);

    // Copy internal properties before applying hints to ensure that
    // a property set by hint won't be overriden by a value in user config.
    // E.g num_streams=AUTO && hint=THROUGHPUT
    // If we apply hints first and then copy all values from user config to internal one,
    // then we'll get num_streams=AUTO in final config while some integer number is expected.
    for (const auto& [name, value] : m_user_properties) {
        auto& option = m_options_map.at(name);
        option->set_any(value);
    }

    finalize_impl(context);

#ifdef ENABLE_DEBUG_CAPS
    apply_env_options();
#endif

    // Clear properties after finalize_impl to be able to check if a property was set by user during plugin-side
    // finalization
    m_user_properties.clear();

    m_is_finalized = true;
}

bool PluginConfig::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("m_user_properties", m_user_properties);
    for (auto& [name, option] : m_options_map) {
        visitor.on_attribute(name + "__internal", option);
    }

    return true;
}

void PluginConfig::apply_env_options() {
    ov::AnyMap env_properties = read_env();
    cleanup_unsupported(env_properties);
    for (auto& [name, val] : env_properties) {
        std::cout << "Non default env value for " << name << " = " << val.as<std::string>() << std::endl;
    }
    set_property(env_properties);
}

void PluginConfig::apply_config_options(std::string_view device_name, std::filesystem::path config_path) {
    if (!config_path.empty()) {
        ov::AnyMap config_properties = read_config_file(config_path, device_name);
        cleanup_unsupported(config_properties);
        for (auto& [name, val] : config_properties) {
            std::cout << "Non default config value for " << name << " = " << val.as<std::string>() << std::endl;
        }
        set_property(config_properties);
    }
}

ov::AnyMap PluginConfig::read_config_file(std::filesystem::path filename, std::string_view target_device_name) const {
    if (filename.empty())
        return {};

    ov::AnyMap config;

    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        return config;
    }

    nlohmann::json json_config;
    try {
        ifs >> json_config;
    } catch (const std::exception&) {
        return config;
    }

    DeviceIDParser parser(std::string{target_device_name});
    for (auto item = json_config.cbegin(), end = json_config.cend(); item != end; ++item) {
        const std::string& device_name = item.key();
        if (DeviceIDParser(device_name).get_device_name() != parser.get_device_name())
            continue;

        const auto& item_value = item.value();
        for (auto option = item_value.cbegin(), item_value_end = item_value.cend(); option != item_value_end;
             ++option) {
            config[option.key()] = option.value().get<std::string>();
        }
    }

    return config;
}

ov::Any PluginConfig::read_env(const std::string& option_name,
                               std::string_view prefix,
                               const ConfigOptionBase* option) {
    auto var_name = std::string(prefix) + option_name;
    const auto& val = ov::util::getenv_string(var_name.c_str());

    if (!val.empty()) {
        return val;
    } else {
        return ov::Any();
    }
}

ov::AnyMap PluginConfig::read_env() const {
    ov::AnyMap config;

    for (auto& [name, option] : m_options_map) {
        if (auto val = read_env(name, m_allowed_env_prefix, option); !val.empty()) {
            config[name] = val;
        }
    }

    return config;
}

void PluginConfig::cleanup_unsupported(ov::AnyMap& config) const {
    for (auto it = config.begin(); it != config.end();) {
        auto& name = it->first;
        auto opt_it = std::find_if(m_options_map.begin(), m_options_map.end(), [&](const OptionMapEntry& o) {
            return o.first == name;
        });
        if (opt_it == m_options_map.end() || opt_it->second->get_visibility() == OptionVisibility::DEBUG_GLOBAL) {
            it = config.erase(it);
        } else {
            ++it;
        }
    }
}

std::string PluginConfig::to_string() const {
    std::stringstream ss;

    ss << "-----------------------------------------\n";
    ss << "PROPERTIES:\n";

    for (const auto& [name, option] : m_options_map) {
        ss << "\t" << name << ": " << option->get_any().as<std::string>() << std::endl;
    }
    ss << "USER PROPERTIES:\n";
    for (const auto& [name, val] : m_user_properties) {
        ss << "\t" << name << ": " << val.as<std::string>() << std::endl;
    }

    return ss.str();
}

void PluginConfig::print_help() const {
    auto format_text = [](const std::string& cpp_name,
                          std::string_view str_name,
                          std::string_view desc,
                          size_t max_name_width,
                          size_t max_width) {
        std::istringstream words(std::string{desc});
        std::ostringstream formatted_text;
        std::string word;
        std::vector<std::string> words_vec;

        while (words >> word) {
            words_vec.push_back(word);
        }

        size_t j = 0;
        size_t count_of_desc_lines = (desc.length() + max_width - 1) / max_width;
        for (size_t i = 0; i < std::max<size_t>(2, count_of_desc_lines); i++) {
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

    std::stringstream ss;
    auto max_name_length_item = std::max_element(m_options_map.begin(),
                                                 m_options_map.end(),
                                                 [](const OptionMapEntry& a, const OptionMapEntry& b) {
                                                     return std::get<0>(a).size() < std::get<0>(b).size();
                                                 });

    const size_t max_name_width =
        std::max(max_name_length_item->first.size(), max_name_length_item->second->property_name.size()) + 4;
    const size_t terminal_width = get_terminal_width();
    // clang-format off
    ss << std::left << std::setw(max_name_width) << "Option name" << " | Description\n";
    ss << std::left << std::setw(terminal_width) << std::setfill('-') << "" << "\n";
    // clang-format on
    for (auto& [name, option] : m_options_map) {
        ss << format_text(name, option->property_name, option->description, max_name_width, terminal_width) << "\n";
    }

    std::cout << ss.str();
}

std::string_view PluginConfig::get_help_message(const std::string& name) const {
    return get_option_ptr(name)->description;
}

}  // namespace ov
