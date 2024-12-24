// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <map>
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/core/except.hpp"
#include <iostream>

#ifndef COUNT_N
    #define COUNT_N(_1, _2, _3, _4, _5, N, ...) N
#endif

#ifndef COUNT
    #define COUNT(...) EXPAND(COUNT_N(__VA_ARGS__, 5, 4, 3, 2, 1))
#endif

#ifndef CAT
    #define CAT(a, b) a ## b
#endif

#ifndef EXPAND
    #define EXPAND(N) N
#endif

#define GET_EXCEPT_LAST_IMPL(N, ...) CAT(GET_EXCEPT_LAST_IMPL_, N)(__VA_ARGS__)
#define GET_EXCEPT_LAST_IMPL_2(_0, _1) _0
#define GET_EXCEPT_LAST_IMPL_3(_0, _1, _2) _0, _1
#define GET_EXCEPT_LAST_IMPL_4(_0, _1, _2, _3) _0, _1, _2

#define GET_EXCEPT_LAST(...) EXPAND(GET_EXCEPT_LAST_IMPL(COUNT(__VA_ARGS__), __VA_ARGS__))

#define GET_LAST_IMPL(N, ...) CAT(GET_LAST_IMPL_, N)(__VA_ARGS__)
#define GET_LAST_IMPL_0(_0, ...) _0
#define GET_LAST_IMPL_1(_0, _1, ...) _1
#define GET_LAST_IMPL_2(_0, _1, _2, ...) _2
#define GET_LAST_IMPL_3(_0, _1, _2, _3, ...) _3
#define GET_LAST_IMPL_4(_0, _1, _2, _3, _4, ...) _4
#define GET_LAST_IMPL_5(_0, _1, _2, _3, _4, _5, ...) _5
#define GET_LAST_IMPL_6(_0, _1, _2, _3, _4, _5, _6, ...) _6

#define GET_LAST(...) GET_LAST_IMPL(COUNT(__VA_ARGS__), _, __VA_ARGS__ ,,,,,,,,,,,)

#define OV_CONFIG_DECLARE_OPTION(PropertyNamespace, PropertyVar, Visibility, ...) \
    ConfigOption<decltype(PropertyNamespace::PropertyVar)::value_type, Visibility> m_ ## PropertyVar{GET_EXCEPT_LAST(__VA_ARGS__)};

#define OV_CONFIG_DECLARE_GETTERS(PropertyNamespace, PropertyVar, Visibility, ...) \
    const decltype(PropertyNamespace::PropertyVar)::value_type& get_##PropertyVar() const { \
        if (m_is_finalized) { \
            return m_ ## PropertyVar.value; \
        } else { \
            if (m_user_properties.find(PropertyNamespace::PropertyVar.name()) != m_user_properties.end()) { \
                return m_user_properties.at(PropertyNamespace::PropertyVar.name()).as<decltype(PropertyNamespace::PropertyVar)::value_type>(); \
            } else { \
                return m_ ## PropertyVar.value; \
            } \
        } \
    }

#define OV_CONFIG_OPTION_MAPPING(PropertyNamespace, PropertyVar, ...) \
        m_options_map[PropertyNamespace::PropertyVar.name()] = & m_ ## PropertyVar;

#define OV_CONFIG_OPTION_HELP(PropertyNamespace, PropertyVar, Visibility, DefaultValue, ...) \
        { #PropertyNamespace "::" #PropertyVar, PropertyNamespace::PropertyVar.name(), GET_LAST(__VA_ARGS__)},

#define OV_CONFIG_RELEASE_OPTION(PropertyNamespace, PropertyVar, ...) \
    OV_CONFIG_OPTION(PropertyNamespace, PropertyVar, OptionVisibility::RELEASE, __VA_ARGS__)

#define OV_CONFIG_RELEASE_INTERNAL_OPTION(PropertyNamespace, PropertyVar, ...) \
    OV_CONFIG_OPTION(PropertyNamespace, PropertyVar, OptionVisibility::RELEASE_INTERNAL, __VA_ARGS__)

#define OV_CONFIG_DEBUG_OPTION(PropertyNamespace, PropertyVar, ...) \
    OV_CONFIG_OPTION(PropertyNamespace, PropertyVar, OptionVisibility::DEBUG, __VA_ARGS__)

namespace ov {
#define ENABLE_DEBUG_CAPS
enum class OptionVisibility : uint8_t {
    RELEASE = 1 << 0,            // Option can be set for any build type via public interface, environment and config file
    RELEASE_INTERNAL = 1 << 1,   // Option can be set for any build type via environment and config file only
    DEBUG = 1 << 2,              // Option can be set for debug builds only via environment and config file
#ifdef ENABLE_DEBUG_CAPS
    ANY = 0x07,                  // Any visibility is valid including DEBUG
#else
    ANY = 0x03,                  // Any visibility is valid excluding DEBUG
#endif
};

inline OptionVisibility operator&(OptionVisibility a, OptionVisibility b) {
    typedef std::underlying_type<OptionVisibility>::type underlying_type;
    return static_cast<OptionVisibility>(static_cast<underlying_type>(a) & static_cast<underlying_type>(b));
}

inline OptionVisibility operator|(OptionVisibility a, OptionVisibility b) {
    typedef std::underlying_type<OptionVisibility>::type underlying_type;
    return static_cast<OptionVisibility>(static_cast<underlying_type>(a) | static_cast<underlying_type>(b));
}

inline OptionVisibility operator~(OptionVisibility a) {
    typedef std::underlying_type<OptionVisibility>::type underlying_type;
    return static_cast<OptionVisibility>(~static_cast<underlying_type>(a));
}

inline std::ostream& operator<<(std::ostream& os, const OptionVisibility& visibility) {
    switch (visibility) {
    case OptionVisibility::RELEASE: os << "RELEASE"; break;
    case OptionVisibility::RELEASE_INTERNAL: os << "RELEASE_INTERNAL"; break;
    case OptionVisibility::DEBUG: os << "DEBUG"; break;
    default: os << "UNKNOWN"; break;
    }

    return os;
}

struct ConfigOptionBase {
    explicit ConfigOptionBase() {}
    virtual ~ConfigOptionBase() = default;

    virtual void set_any(const ov::Any any) = 0;
    virtual ov::Any get_any() const = 0;
    virtual bool is_valid_value(ov::Any val) = 0;
    virtual OptionVisibility get_visibility() const = 0;
};

template <typename T, OptionVisibility visibility_ = OptionVisibility::DEBUG>
struct ConfigOption : public ConfigOptionBase {
    ConfigOption(const T& default_val, std::function<bool(T)> validator = nullptr)
        : ConfigOptionBase(), value(default_val), validator(validator) {}
    T value;
    constexpr static const auto visibility = visibility_;

    void set_any(const ov::Any any) override {
        if (validator) {
            // TODO: is very any way to print option name here?
            OPENVINO_ASSERT(validator(any.as<T>()), "Invalid value: ", any.as<std::string>());
        }
        value = any.as<T>();
    }

    ov::Any get_any() const override {
        return ov::Any(value);
    }

    bool is_valid_value(ov::Any val) override {
        try {
            auto v = val.as<T>();
            return validator ? validator(v) : true;
        } catch (std::exception&) {
            return false;
        }
    }

    OptionVisibility get_visibility() const override {
        return visibility;
    }

    operator T() const {
        return value;
    }

    ConfigOption& operator=(const T& val) {
        value = val;
        return *this;
    }

    bool operator==(const T& val) const {
        return value == val;
    }

    bool operator!=(const T& val) const {
        return !(*this == val);
    }

private:
    std::function<bool(T)> validator;
};

// Base class for configuration of plugins
// Implementation should provide a list of properties with default values and validators (optional)
// and prepare a map string property name -> ConfigOptionBase pointer
// For the sake of efficiency, we expect that plugin properties are defined as class members of the derived class
// and accessed directly in the plugin's code (i.e. w/o get_property()/set_property() calls)
// get/set property members are provided to handle external property access
// The class provides a helpers to read the properties from configuration file and from environment variables
//
// Expected order of properties resolution:
// 1. Assign default value for each property per device
// 2. Save user properties passed via Core::set_property() call to user_properties
// 3. Save user properties passed via Core::compile_model() call to user_properties
// 4. Apply RT info properties to user_properties if they were not set by user
// 5. Read and apply properties from the config file as user_properties
// 6. Read and apply properties from the the environment variables as user_properties
// 7. Apply user_properties to actual plugin properties
// 8. Update dependant properties if they were not set by user either way
class OPENVINO_RUNTIME_API PluginConfig {
public:
    PluginConfig() {}
    virtual ~PluginConfig() = default;

    // Disable copy and move as we need to setup m_options_map properly and ensure that
    // values are a part of current config object
    PluginConfig(const PluginConfig& other) = delete;
    PluginConfig& operator=(const PluginConfig& other) = delete;
    PluginConfig(PluginConfig&& other) = delete;
    PluginConfig& operator=(PluginConfig&& other) = delete;

    void set_property(const ov::AnyMap& properties);
    void set_user_property(const ov::AnyMap& properties, OptionVisibility allowed_visibility = OptionVisibility::ANY, bool throw_on_error = true);
    Any get_property(const std::string& name, OptionVisibility allowed_visibility = OptionVisibility::ANY) const;

    template <typename... Properties>
    util::EnableIfAllStringAny<void, Properties...> set_property(Properties&&... properties) {
        set_property(ov::AnyMap{std::forward<Properties>(properties)...});
    }

    std::string to_string() const;

    void finalize(std::shared_ptr<IRemoteContext> context, const ov::RTMap& rt_info);

    bool visit_attributes(ov::AttributeVisitor& visitor);

    template <typename T, PropertyMutability mutability>
    bool is_set_by_user(const ov::Property<T, mutability>& property) const {
        return m_user_properties.find(property.name()) != m_user_properties.end();
    }

protected:
    virtual void apply_rt_info(std::shared_ptr<IRemoteContext> context, const ov::RTMap& rt_info) {}
    virtual void apply_debug_options(std::shared_ptr<IRemoteContext> context);
    virtual void finalize_impl(std::shared_ptr<IRemoteContext> context) {}

    ConfigOptionBase* get_option_ptr(const std::string& name) const {
        auto it = m_options_map.find(name);
        // TODO: print more meaningful error message
        OPENVINO_ASSERT(it != m_options_map.end(), "Option not found: ", name);
        OPENVINO_ASSERT(it->second != nullptr, "Option is invalid: ", name);

        return it->second;
    }

    template <typename T, PropertyMutability mutability>
    void apply_rt_info_property(const ov::Property<T, mutability>& property, const ov::RTMap& rt_info) {
        if (!is_set_by_user(property)) {
            auto rt_info_val = rt_info.find(property.name());
            if (rt_info_val != rt_info.end()) {
                set_property(property(rt_info_val->second.template as<T>()));
            }
        }
    }

    ov::AnyMap read_config_file(const std::string& filename, const std::string& target_device_name) const;
    ov::AnyMap read_env(const std::vector<std::string>& prefixes) const;
    void cleanup_unsupported(ov::AnyMap& config) const;

    std::map<std::string, ConfigOptionBase*> m_options_map;

    // List of properties explicitly set by user via Core::set_property() or Core::compile_model() or ov::Model's runtime info
    ov::AnyMap m_user_properties;
    using OptionMapEntry = decltype(m_options_map)::value_type;

    // property variable name, string name, default value, description
    using OptionsDesc = std::vector<std::tuple<std::string, std::string, std::string>>;
    virtual const OptionsDesc& get_options_desc() const { static OptionsDesc empty; return empty; }
    const std::string get_help_message(const std::string& name = "") const;
    void print_help() const;

    bool m_is_finalized = false;
};

}  // namespace ov
