// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "openvino/runtime/auto/properties.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "utils/log.hpp"
#include "utils/log_util.hpp"
#include "openvino/runtime/device_id_parser.hpp"
#include <string>
#include <map>
#include <vector>

namespace ov {
namespace auto_plugin {

class BaseValidator {
public:
    using Ptr = std::shared_ptr<BaseValidator>;
    virtual ~BaseValidator() = default;
    virtual bool is_valid(const ov::Any& v) const = 0;
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
            const auto& val = v.as<std::string>();
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
    void set_user_property(const ov::AnyMap& properties);
    ov::Any get_property(const std::string& name) const;
    bool is_batching_disabled() const;
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

    std::vector<std::string> supported_rw_properties(const std::string& plugin_name = "AUTO") const {
        std::vector<std::string> supported_configKeys;
        for (const auto& iter : property_mutabilities) {
            if (iter.second.as<ov::PropertyMutability>() == ov::PropertyMutability::RW)
                supported_configKeys.push_back(iter.first);
        }
        auto multi_supported_configKeys = supported_configKeys;
        multi_supported_configKeys.erase(std::remove(
                                multi_supported_configKeys.begin(), multi_supported_configKeys.end(), ov::intel_auto::enable_startup_fallback.name()),
                                multi_supported_configKeys.end());
        multi_supported_configKeys.erase(std::remove(
                                multi_supported_configKeys.begin(), multi_supported_configKeys.end(), ov::intel_auto::enable_runtime_fallback.name()),
                                multi_supported_configKeys.end());
        return plugin_name == "AUTO" ? supported_configKeys : multi_supported_configKeys;
    }

    std::vector<ov::PropertyName> supported_properties(const std::string& plugin_name = "AUTO") const {
        std::vector<ov::PropertyName> supported_properties;
        for (const auto& iter : property_mutabilities)
            supported_properties.push_back(ov::PropertyName(iter.first, iter.second.as<ov::PropertyMutability>()));

        auto multi_supported_properties = supported_properties;
        multi_supported_properties.erase(std::remove(
                                multi_supported_properties.begin(), multi_supported_properties.end(), ov::intel_auto::enable_startup_fallback),
                                multi_supported_properties.end());
        multi_supported_properties.erase(std::remove(
                                multi_supported_properties.begin(), multi_supported_properties.end(), ov::intel_auto::enable_runtime_fallback),
                                multi_supported_properties.end());
        return plugin_name == "AUTO" ? supported_properties : multi_supported_properties;
    }

    std::vector<std::string> supported_ro_properties(const std::string& plugin_name = "AUTO") const {
        std::vector<std::string> supported_ro_properties;
        for (const auto& iter : property_mutabilities) {
            if (iter.second.as<ov::PropertyMutability>() == ov::PropertyMutability::RO)
                supported_ro_properties.push_back(iter.first);
        }
        auto multi_supported_ro_properties = supported_ro_properties;
        return plugin_name == "AUTO" ? supported_ro_properties : multi_supported_ro_properties;
    }

    bool is_supported_device(const std::string& device_name, const std::string& option) const {
        if (device_name.empty())
            return false;
        auto real_dev_name = device_name[0] != '-' ? device_name : device_name.substr(1);
        if (real_dev_name.empty()) {
            return false;
        }
        real_dev_name = ov::DeviceIDParser(real_dev_name).get_device_name();
        if (real_dev_name.find("GPU") != std::string::npos) {
            // Check if the device is an Intel device
            if (option.find("vendor=0x8086") == std::string::npos) {
                real_dev_name = "notIntelGPU";
            }
        }
        std::string::size_type real_end_pos = 0;
        if ((real_end_pos = real_dev_name.find('(')) != std::string::npos) {
            real_dev_name = real_dev_name.substr(0, real_end_pos);
        }
        if (device_block_list.end() != std::find(device_block_list.begin(), device_block_list.end(), real_dev_name)) {
            return false;
        }
        return true;
    }

    std::vector<std::string> parse_priorities_devices(const std::string& priorities, const char separator = ',') const {
        std::vector<std::string> devices;
        std::string::size_type pos = 0;
        std::string::size_type endpos = 0;
        while ((endpos = priorities.find(separator, pos)) != std::string::npos) {
            auto substr = priorities.substr(pos, endpos - pos);
            if (!substr.empty())
                devices.push_back(substr);
            pos = endpos + 1;
        }
        auto substr = priorities.substr(pos, priorities.length() - pos);
        if (!substr.empty())
            devices.push_back(substr);
        return devices;
    }

private:
    ov::AnyMap internal_properties;   // internal supported properties for auto/multi
    ov::AnyMap user_properties;       // user set properties, including secondary properties
    ov::AnyMap full_properties;       // combined with user set properties, including secondary properties
    ov::AnyMap property_mutabilities; // mutability for supported properties installation
    std::map<std::string, BaseValidator::Ptr> property_validators;
    static const std::set<std::string> device_block_list;
};
} // namespace auto_plugin
} // namespace ov
