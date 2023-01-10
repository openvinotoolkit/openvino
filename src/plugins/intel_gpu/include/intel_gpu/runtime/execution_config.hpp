// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/internal_properties.hpp"
#include "intel_gpu/runtime/device.hpp"

namespace ov {
namespace intel_gpu {

enum class PropertyVisibility {
    INTERNAL = 0,
    PUBLIC = 1
};

inline std::ostream& operator<<(std::ostream& os, const PropertyVisibility& visibility) {
    switch (visibility) {
    case PropertyVisibility::PUBLIC: os << "PUBLIC"; break;
    case PropertyVisibility::INTERNAL: os << "INTERNAL"; break;
    default: os << "UNKNOWN"; break;
    }

    return os;
}

class BaseValidator {
public:
    using Ptr = std::shared_ptr<BaseValidator>;
    virtual ~BaseValidator() = default;
    virtual bool is_valid(const ov::Any& v) const = 0;
};

class FuncValidator : public BaseValidator {
public:
explicit FuncValidator(std::function<bool(const ov::Any)> func) : m_func(func) { }
    bool is_valid(const ov::Any& v) const override {
        return m_func(v);
    }

private:
    std::function<bool(const ov::Any)> m_func;
};

// PropertyTypeValidator ensures that value can be converted to given property type
template<typename T>
class PropertyTypeValidator : public BaseValidator {
public:
    bool is_valid(const ov::Any& v) const override {
        try {
            v.as<T>();
            return true;
        } catch (ov::Exception&) {
            return false;
        }
    }
};

class ExecutionConfig {
public:
    ExecutionConfig();
    ExecutionConfig(std::initializer_list<ov::AnyMap::value_type> values) : ExecutionConfig() { set_property(ov::AnyMap(values)); }
    explicit ExecutionConfig(const ov::AnyMap& properties) : ExecutionConfig() { set_property(properties); }
    explicit ExecutionConfig(const ov::AnyMap::value_type& property) : ExecutionConfig() { set_property(property); }

    void set_default();
    void set_property(const ov::AnyMap& properties);
    void set_user_property(const ov::AnyMap& properties);
    Any get_property(const std::string& name) const;
    bool is_set_by_user(const std::string& name) const;
    bool is_supported(const std::string& name) const;
    void register_property_impl(const std::pair<std::string, ov::Any>& propertiy, PropertyVisibility visibility, BaseValidator::Ptr validator);

    template <PropertyVisibility visibility, typename... PropertyInitializer, typename std::enable_if<(sizeof...(PropertyInitializer) == 0), bool>::type = true>
    void register_property_impl() { }

    template <PropertyVisibility visibility, typename T,  PropertyMutability mutability, typename ValueT, typename... PropertyInitializer>
    void register_property_impl(const std::tuple<ov::Property<T, mutability>, ValueT>& property, PropertyInitializer&&... properties) {
        auto p = std::get<0>(property)(std::get<1>(property));
        auto v = std::dynamic_pointer_cast<BaseValidator>(std::make_shared<PropertyTypeValidator<T>>());
        register_property_impl(std::move(p), visibility, std::move(v));
        register_property_impl<visibility>(properties...);
    }

    template <PropertyVisibility visibility,
              typename T,
              PropertyMutability mutability,
              typename ValueT,
              typename ValidatorT,
              typename... PropertyInitializer>
    typename std::enable_if<std::is_base_of<BaseValidator, ValidatorT>::value, void>::type
    register_property_impl(const std::tuple<ov::Property<T, mutability>, ValueT, ValidatorT>& property, PropertyInitializer&&... properties) {
        auto p = std::get<0>(property)(std::get<1>(property));
        auto v = std::dynamic_pointer_cast<BaseValidator>(std::make_shared<ValidatorT>(std::get<2>(property)));
        register_property_impl(std::move(p), visibility, std::move(v));
        register_property_impl<visibility>(properties...);
    }

    template <PropertyVisibility visibility,
              typename T,
              PropertyMutability mutability,
              typename ValueT,
              typename ValidatorT,
              typename... PropertyInitializer>
    typename std::enable_if<std::is_same<std::function<bool(const ov::Any&)>, ValidatorT>::value, void>::type
    register_property_impl(const std::tuple<ov::Property<T, mutability>, ValueT, ValidatorT>& property, PropertyInitializer&&... properties) {
        auto p = std::get<0>(property)(std::get<1>(property));
        auto v = std::dynamic_pointer_cast<BaseValidator>(std::make_shared<FuncValidator>(std::get<2>(property)));
        register_property_impl(std::move(p), visibility, std::move(v));
        register_property_impl<visibility>(properties...);
    }

    template <PropertyVisibility visibility, typename... PropertyInitializer>
    void register_property(PropertyInitializer&&... properties) {
        register_property_impl<visibility>(properties...);
    }

    template <typename... Properties>
    util::EnableIfAllStringAny<void, Properties...> set_property(Properties&&... properties) {
        set_property(ov::AnyMap{std::forward<Properties>(properties)...});
    }

    template <typename... Properties>
    util::EnableIfAllStringAny<void, Properties...> set_user_property(Properties&&... properties) {
        set_user_property(ov::AnyMap{std::forward<Properties>(properties)...});
    }

    template <typename T, PropertyMutability mutability>
    bool is_set_by_user(const ov::Property<T, mutability>& property) const {
        return is_set_by_user(property.name());
    }

    template <typename T, PropertyMutability mutability>
    T get_property(const ov::Property<T, mutability>& property) const {
        return get_property(property.name()).template as<T>();
    }

    void apply_user_properties(const cldnn::device_info& info);

    std::string to_string() const;

protected:
    void apply_hints(const cldnn::device_info& info);
    void apply_performance_hints(const cldnn::device_info& info);
    void apply_priority_hints(const cldnn::device_info& info);
    void apply_debug_options(const cldnn::device_info& info);

private:
    ov::AnyMap internal_properties;
    ov::AnyMap user_properties;

    std::map<std::string, PropertyVisibility> supported_properties;
    std::map<std::string, BaseValidator::Ptr> property_validators;
};

}  // namespace intel_gpu
}  // namespace ov

namespace cldnn {
using ov::intel_gpu::ExecutionConfig;
}  // namespace cldnn
