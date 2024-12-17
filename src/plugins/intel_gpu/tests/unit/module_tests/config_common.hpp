// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <unordered_map>
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
    std::function<bool(T)> validator;

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
            return validator ? validator(val.as<T>()) : true;
        } catch (std::exception&) {
            return false;
        }

    }
};

class PluginConfig {
public:
    PluginConfig() {}
    PluginConfig(std::initializer_list<ov::AnyMap::value_type> values) : PluginConfig() { set_property(ov::AnyMap(values)); }
    explicit PluginConfig(const ov::AnyMap& properties) : PluginConfig() { set_property(properties); }
    explicit PluginConfig(const ov::AnyMap::value_type& property) : PluginConfig() { set_property(property); }

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
        OPENVINO_ASSERT(m_options_map.find(property.name()) != m_options_map.end(), "Property not found: ", property.name());
        return static_cast<ConfigOption<T>*>(m_options_map.at(property.name()))->value;
    }

    std::string to_string() const;

    void finalize(std::shared_ptr<IRemoteContext> context, const ov::RTMap& rt_info);
    virtual void finalize_impl(std::shared_ptr<IRemoteContext> context, const ov::RTMap& rt_info) = 0;

protected:
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
    std::unordered_map<std::string, ConfigOptionBase*> m_options_map;
    ov::AnyMap user_properties;
    using OptionMapEntry = decltype(m_options_map)::value_type;
};

}  // namespace ov
