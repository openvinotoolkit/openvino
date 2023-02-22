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

class PerformanceModeValidator : public BaseValidator {
public:
    bool is_valid(const ov::Any& v) const override {
        auto mode = v.as<ov::hint::PerformanceMode>();
        return mode == ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT ||
               mode == ov::hint::PerformanceMode::THROUGHPUT ||
               mode == ov::hint::PerformanceMode::LATENCY ||
               mode == ov::hint::PerformanceMode::UNDEFINED;
    }
};

class ExecutionConfig {
public:
    ExecutionConfig();
    ExecutionConfig(std::initializer_list<ov::AnyMap::value_type> values) { set_property(ov::AnyMap(values)); }

    void set_default();
    void set_property(const ov::AnyMap& properties);
    void set_user_property(const ov::AnyMap& properties);
    ov::Any get_property(const std::string& name) const;
    bool is_set_by_user(const std::string& name) const;
    bool is_supported(const std::string& name) const;
    void register_property_impl(const ov::AnyMap::value_type& propertiy, BaseValidator::Ptr validator);

    template <typename... PropertyInitializer, typename std::enable_if<(sizeof...(PropertyInitializer) == 0), bool>::type = true>
    void register_property_impl() { }

    template <typename ValueT, typename... PropertyInitializer>
    void register_property_impl(const std::tuple<ov::device::Priorities, ValueT>& property, PropertyInitializer&&... properties) {
        auto p = std::get<0>(property)(std::get<1>(property));
        auto v = std::dynamic_pointer_cast<BaseValidator>(std::make_shared<PropertyTypeValidator<std::string>>());
        register_property_impl(std::move(p), std::move(v));
        register_property_impl(properties...);
    }

    template <typename T,  ov::PropertyMutability mutability, typename ValueT, typename... PropertyInitializer>
    void register_property_impl(const std::tuple<ov::Property<T, mutability>, ValueT>& property, PropertyInitializer&&... properties) {
        auto p = std::get<0>(property)(std::get<1>(property));
        auto v = std::dynamic_pointer_cast<BaseValidator>(std::make_shared<PropertyTypeValidator<T>>());
        register_property_impl(std::move(p), std::move(v));
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
        register_property_impl(std::move(p), std::move(v));
        register_property_impl(properties...);
    }

    template <typename T,
              ov::PropertyMutability mutability,
              typename ValueT,
              typename ValidatorT,
              typename... PropertyInitializer>
    typename std::enable_if<std::is_same<std::function<bool(const ov::Any&)>, ValidatorT>::value, void>::type
    register_property_impl(const std::tuple<ov::Property<T, mutability>, ValueT, ValidatorT>& property, PropertyInitializer&&... properties) {
        auto p = std::get<0>(property)(std::get<1>(property));
        auto v = std::dynamic_pointer_cast<BaseValidator>(std::make_shared<FuncValidator>(std::get<2>(property)));
        register_property_impl(std::move(p), std::move(v));
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
    std::string to_string() const;

    std::vector<std::string> supportedConfigKeys(const std::string& pluginName = "AUTO") const {
        std::vector<std::string> supported_configKeys = []() -> decltype(PerfHintsConfig::SupportedKeys()) {
            auto res = PerfHintsConfig::SupportedKeys();
            res.push_back(ov::device::priorities.name());
            res.push_back(ov::enable_profiling.name());
            res.push_back(PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS);
            res.push_back(ov::hint::model_priority.name());
            res.push_back(ov::hint::allow_auto_batching.name());
            res.push_back(ov::log::level.name());
            res.push_back(ov::intel_auto::device_bind_buffer.name());
            res.push_back(ov::auto_batch_timeout.name());
            return res;
        }();
        auto multi_supported_configKeys = supported_configKeys;
        return pluginName == "AUTO" ? supported_configKeys : multi_supported_configKeys;
    }

    std::vector<ov::PropertyName> supportedProperties(const std::string& pluginName = "AUTO") const {
        std::vector<ov::PropertyName> supported_properties = []() -> std::vector<ov::PropertyName> {
            auto RO_property = [](const std::string& propertyName) {
                return ov::PropertyName(propertyName, ov::PropertyMutability::RO);
            };
            auto RW_property = [](const std::string& propertyName) {
                return ov::PropertyName(propertyName, ov::PropertyMutability::RW);
            };
            std::vector<ov::PropertyName> roProperties{RO_property(ov::supported_properties.name()),
                                                       RO_property(ov::device::full_name.name()),
                                                       RO_property(ov::device::capabilities.name())};
            // the whole config is RW before network is loaded.
            std::vector<ov::PropertyName> rwProperties{RW_property(ov::hint::model_priority.name()),
                                                       RW_property(ov::log::level.name()),
                                                       RW_property(ov::device::priorities.name()),
                                                       RW_property(ov::enable_profiling.name()),
                                                       RW_property(ov::hint::allow_auto_batching.name()),
                                                       RW_property(ov::auto_batch_timeout.name()),
                                                       RW_property(ov::hint::performance_mode.name()),
                                                       RW_property(ov::hint::num_requests.name()),
                                                       RW_property(ov::intel_auto::device_bind_buffer.name()),
                                                       RW_property(ov::cache_dir.name())};
            std::vector<ov::PropertyName> supportedProperties;
            supportedProperties.reserve(roProperties.size() + rwProperties.size());
            supportedProperties.insert(supportedProperties.end(), roProperties.begin(), roProperties.end());
            supportedProperties.insert(supportedProperties.end(), rwProperties.begin(), rwProperties.end());
            return supportedProperties;
        }();
        auto multi_supported_properties = supported_properties;
        return pluginName == "AUTO" ? supported_properties : multi_supported_properties;
    }

    std::vector<std::string> supportedMetrics(const std::string& pluginName = "AUTO") const {
        std::vector<std::string> supported_metrics = []() -> std::vector<std::string> {
            std::vector<std::string> metrics;
            metrics.push_back(METRIC_KEY(SUPPORTED_METRICS));
            metrics.push_back(METRIC_KEY(FULL_DEVICE_NAME));
            metrics.push_back(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
            metrics.push_back(METRIC_KEY(OPTIMIZATION_CAPABILITIES));
            return metrics;
        }();
        auto multi_supported_metrics = supported_metrics;
        return pluginName == "AUTO" ? supported_metrics : multi_supported_metrics;
    }

private:
    ov::AnyMap internal_properties; // internal supported properties for auto/multi
    ov::AnyMap user_properties; // user set properties, including secondary properties

    std::map<std::string, BaseValidator::Ptr> property_validators;
    //std::string plugin_name;
    BaseValidator::Ptr device_property_validator;
    static const std::set<std::string> _availableDevices;
};
} // namespace MultiDevicePlugin