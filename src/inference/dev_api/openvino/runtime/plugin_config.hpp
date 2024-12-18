// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <map>
#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/core/except.hpp"

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

namespace ov {

struct ConfigOptionBase {
    explicit ConfigOptionBase() {}
    virtual ~ConfigOptionBase() = default;

    virtual void set_any(const ov::Any any) = 0;
    virtual ov::Any get_any() const = 0;
    virtual bool is_valid_value(ov::Any val) = 0;
};

template <typename T>
struct ConfigOption : public ConfigOptionBase {
    ConfigOption(const T& default_val, std::function<bool(T)> validator = nullptr)
        : ConfigOptionBase(), value(default_val), validator(validator) {}
    T value;

    void set_any(const ov::Any any) override {
        if (validator)
            OPENVINO_ASSERT(validator(any.as<T>()), "Invalid value: ", any.as<std::string>());
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
    Any get_property(const std::string& name) const;
    void set_user_property(const ov::AnyMap& properties);

    template <typename... Properties>
    util::EnableIfAllStringAny<void, Properties...> set_property(Properties&&... properties) {
        set_property(ov::AnyMap{std::forward<Properties>(properties)...});
    }

    template <typename... Properties>
    util::EnableIfAllStringAny<void, Properties...> set_user_property(Properties&&... properties) {
        set_user_property(ov::AnyMap{std::forward<Properties>(properties)...});
    }

    template <typename T, PropertyMutability mutability>
    T get_property(const ov::Property<T, mutability>& property) const {
        if (is_set_by_user(property)) {
            return user_properties.at(property.name()).template as<T>();
        }
        OPENVINO_ASSERT(m_options_map.find(property.name()) != m_options_map.end(), "Property not found: ", property.name());
        return static_cast<ConfigOption<T>*>(m_options_map.at(property.name()))->value;
    }

    std::string to_string() const;

    void finalize(std::shared_ptr<IRemoteContext> context, const ov::RTMap& rt_info);

protected:
    virtual void apply_rt_info(std::shared_ptr<IRemoteContext> context, const ov::RTMap& rt_info) {}
    virtual void apply_debug_options(std::shared_ptr<IRemoteContext> context);
    virtual void finalize_impl(std::shared_ptr<IRemoteContext> context) {}

    template <typename T, PropertyMutability mutability>
    bool is_set_by_user(const ov::Property<T, mutability>& property) const {
        return user_properties.find(property.name()) != user_properties.end();
    }

    template <typename T, PropertyMutability mutability>
    void apply_rt_info_property(const ov::Property<T, mutability>& property, const ov::RTMap& rt_info) {
        if (!is_set_by_user(property)) {
            auto rt_info_val = rt_info.find(property.name());
            if (rt_info_val != rt_info.end()) {
                set_user_property(property(rt_info_val->second.template as<T>()));
            }
        }
    }

    ov::AnyMap read_config_file(const std::string& filename, const std::string& target_device_name) const;
    ov::AnyMap read_env(const std::vector<std::string>& prefixes) const;
    void cleanup_unsupported(ov::AnyMap& config) const;

    std::map<std::string, ConfigOptionBase*> m_options_map;

    // List of properties explicitly set by user via Core::set_property() or Core::compile_model() or ov::Model's runtime info
    ov::AnyMap user_properties;
    using OptionMapEntry = decltype(m_options_map)::value_type;
};

}  // namespace ov
