// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "ie_parameter.hpp"
#include "ie_performance_hints.hpp"
#include "ie_icore.hpp"
#include "openvino/runtime/auto/properties.hpp"
#include "log.hpp"
#include <string>
#include <map>
#include <vector>
namespace MultiDevicePlugin {
using namespace InferenceEngine;
// legacy config
static constexpr ov::Property<bool, ov::PropertyMutability::RW> exclusive_asyc_requests{"EXCLUSIVE_ASYNC_REQUESTS"};

class BaseValidator {
public:
    using Ptr = std::shared_ptr<BaseValidator>;
    virtual ~BaseValidator() = default;
    virtual bool is_valid(const ov::Any& v) const = 0;
};

class FuncValidator : public BaseValidator {
public:
explicit FuncValidator(std::function<bool(const ov::Any&)> func) : m_func(func) { }
    bool is_valid(const ov::Any& v) const override {
        return m_func(v);
    }
private:
    std::function<bool(const ov::Any&)> m_func;
};

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

class UnsignedTypeValidator : public BaseValidator {
public:
    bool is_valid(const ov::Any& v) const override {
        int val_i = -1;
        try {
            // work around for negative value check (inconsistent behavior on windows/linux)
            auto val = v.as<std::string>();
            val_i = std::stoi(val);
            if (val_i >= 0)
                return true;
            else
                throw std::logic_error("wrong val");
        } catch (std::exception&) {
            return false;
        }
    }
};

class PluginConfig {
public:
    PluginConfig();
    PluginConfig(std::initializer_list<ov::AnyMap::value_type> values) { set_property(ov::AnyMap(values)); }

    void set_default();
    void set_property(const ov::AnyMap& properties);
    void set_user_property(const ov::AnyMap& properties, bool checkfirstlevel = true);
    ov::Any get_property(const std::string& name) const;
    bool is_set_by_user(const std::string& name) const;
    bool is_supported(const std::string& name) const;

    void register_property_impl(const ov::AnyMap::value_type& property, ov::PropertyMutability mutability, BaseValidator::Ptr validator = nullptr);

    template <typename T, ov::PropertyMutability mutability>
    void register_property_impl(const ov::Property<T, mutability>&);

    template <typename... PropertyInitializer, typename std::enable_if<(sizeof...(PropertyInitializer) == 0), bool>::type = true>
    void register_property_impl() { }

    template <typename ValueT, typename... PropertyInitializer>
    void register_property_impl(const std::tuple<ov::device::Priorities, ValueT>& property, PropertyInitializer&&... properties) {
        auto p = std::get<0>(property)(std::get<1>(property));
        auto v = std::dynamic_pointer_cast<BaseValidator>(std::make_shared<PropertyTypeValidator<std::string>>());
        register_property_impl(std::move(p), ov::PropertyMutability::RW, std::move(v));
        register_property_impl(properties...);
    }

    template <typename T,  ov::PropertyMutability mutability, typename... PropertyInitializer>
    void register_property_impl(const std::tuple<ov::Property<T, mutability>>& property, PropertyInitializer&&... properties) {
        auto p = std::get<0>(property);
        register_property_impl(std::move(p));
        register_property_impl(properties...);
    }

    template <typename T,  ov::PropertyMutability mutability, typename ValueT, typename... PropertyInitializer>
    void register_property_impl(const std::tuple<ov::Property<T, mutability>, ValueT>& property, PropertyInitializer&&... properties) {
        auto p = std::get<0>(property)(std::get<1>(property));
        auto v = std::dynamic_pointer_cast<BaseValidator>(std::make_shared<PropertyTypeValidator<T>>());
        register_property_impl(std::move(p), mutability, std::move(v));
        register_property_impl(properties...);
    }

    template <typename T,
              ov::PropertyMutability mutability,
              typename ValueT,
              typename ValidatorT,
              typename... PropertyInitializer>
    typename std::enable_if<std::is_base_of<BaseValidator, ValidatorT>::value, void>::type
    register_property_impl(const std::tuple<ov::Property<T, mutability>, ValueT, ValidatorT>& property, PropertyInitializer&&... properties) {
        auto p = std::get<0>(property)(std::get<1>(property));
        auto v = std::dynamic_pointer_cast<BaseValidator>(std::make_shared<ValidatorT>(std::get<2>(property)));
        register_property_impl(std::move(p), mutability, std::move(v));
        register_property_impl(properties...);
    }

    template <typename... PropertyInitializer>
    void register_property(PropertyInitializer&&... properties) {
        register_property_impl(properties...);
    }

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<void, Properties...> set_property(Properties&&... properties) {
        set_property(ov::AnyMap{std::forward<Properties>(properties)...});
    }

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<void, Properties...> set_user_property(Properties&&... properties) {
        set_user_property(ov::AnyMap{std::forward<Properties>(properties)...});
    }

    template <typename T, ov::PropertyMutability mutability>
    bool is_set_by_user(const ov::Property<T, mutability>& property) const {
        return is_set_by_user(property.name());
    }

    template <typename T, ov::PropertyMutability mutability>
    T get_property(const ov::Property<T, mutability>& property) const {
        return get_property(property.name()).template as<T>();
    }
    void apply_user_properties();
    ov::AnyMap get_full_properties();

    std::vector<std::string> supportedConfigKeys(const std::string& pluginName = "AUTO") const {
        std::vector<std::string> supported_configKeys;
        for (const auto& iter : property_mutabilities) {
            if (iter.second.as<ov::PropertyMutability>() == ov::PropertyMutability::RW)
                supported_configKeys.push_back(iter.first);
        }
        auto multi_supported_configKeys = supported_configKeys;
        multi_supported_configKeys.erase(std::remove(
                                multi_supported_configKeys.begin(), multi_supported_configKeys.end(), ov::intel_auto::enable_startup_fallback.name()),
                                multi_supported_configKeys.end());
        return pluginName == "AUTO" ? supported_configKeys : multi_supported_configKeys;
    }

    std::vector<ov::PropertyName> supportedProperties(const std::string& pluginName = "AUTO") const {
        std::vector<ov::PropertyName> supported_properties;
        for (const auto& iter : property_mutabilities)
            supported_properties.push_back(ov::PropertyName(iter.first, iter.second.as<ov::PropertyMutability>()));

        auto multi_supported_properties = supported_properties;
        multi_supported_properties.erase(std::remove(
                                multi_supported_properties.begin(), multi_supported_properties.end(), ov::intel_auto::enable_startup_fallback),
                                multi_supported_properties.end());
        return pluginName == "AUTO" ? supported_properties : multi_supported_properties;
    }

    std::vector<std::string> supportedMetrics(const std::string& pluginName = "AUTO") const {
        std::vector<std::string> supported_metrics;
        for (const auto& iter : property_mutabilities) {
            if (iter.second.as<ov::PropertyMutability>() == ov::PropertyMutability::RO)
                supported_metrics.push_back(iter.first);
        }
        supported_metrics.push_back(METRIC_KEY(SUPPORTED_METRICS));
        supported_metrics.push_back(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
        auto multi_supported_metrics = supported_metrics;
        return pluginName == "AUTO" ? supported_metrics : multi_supported_metrics;
    }

    bool isSupportedDevice(const std::string& deviceName) const {
        if (deviceName.empty())
            return false;
        auto realDevName = deviceName[0] != '-' ? deviceName : deviceName.substr(1);
        if (realDevName.empty()) {
            return false;
        }
        realDevName = DeviceIDParser(realDevName).getDeviceName();
        std::string::size_type realEndPos = 0;
        if ((realEndPos = realDevName.find('(')) != std::string::npos) {
            realDevName = realDevName.substr(0, realEndPos);
        }
        if (_availableDevices.end() == std::find(_availableDevices.begin(), _availableDevices.end(), realDevName)) {
            return false;
        }
        return true;
    }

    std::vector<std::string> ParsePrioritiesDevices(const std::string& priorities, const char separator = ',') const {
        std::vector<std::string> devices;
        std::string::size_type pos = 0;
        std::string::size_type endpos = 0;
        while ((endpos = priorities.find(separator, pos)) != std::string::npos) {
            auto subStr = priorities.substr(pos, endpos - pos);
            if (!isSupportedDevice(subStr)) {
                IE_THROW() << "Unavailable device name: " << subStr;
            }
            devices.push_back(subStr);
            pos = endpos + 1;
        }
        auto subStr = priorities.substr(pos, priorities.length() - pos);
        if (!isSupportedDevice(subStr)) {
            IE_THROW() << "Unavailable device name: " << subStr;
        }
        devices.push_back(subStr);
        return devices;
    }

private:
    ov::AnyMap internal_properties;   // internal supported properties for auto/multi
    ov::AnyMap user_properties;       // user set properties, including secondary properties
    ov::AnyMap full_properties;       // combined with user set properties, including secondary properties
    ov::AnyMap property_mutabilities; // mutability for supported configs/metrics installation
    std::map<std::string, BaseValidator::Ptr> property_validators;
    BaseValidator::Ptr device_property_validator;
    static const std::set<std::string> _availableDevices;
};
} // namespace MultiDevicePlugin