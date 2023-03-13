// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "utils/plugin_config.hpp"

namespace MultiDevicePlugin {
const std::set<std::string> PluginConfig::_availableDevices = {"AUTO", "CPU", "GPU", "TEMPLATE", "NVIDIA", "VPUX", "MULTI", "HETERO", "mock"};

PluginConfig::PluginConfig() {
    set_default();
    device_property_validator = std::dynamic_pointer_cast<BaseValidator>(std::make_shared<FuncValidator>([](const ov::Any& target) -> bool {
        auto deviceName = target.as<std::string>();
        return _availableDevices.end() != std::find(_availableDevices.begin(), _availableDevices.end(),
                                                            DeviceIDParser(deviceName).getDeviceName());
    }));
}

void PluginConfig::set_default() {
    register_property(
        std::make_tuple(ov::enable_profiling, false),
        std::make_tuple(ov::device::priorities, ""),
        std::make_tuple(ov::hint::model_priority, ov::hint::Priority::MEDIUM),
        std::make_tuple(ov::log::level, ov::log::Level::NO),
        std::make_tuple(ov::intel_auto::device_bind_buffer, false),
        std::make_tuple(ov::hint::performance_mode, ov::hint::PerformanceMode::UNDEFINED),
        std::make_tuple(ov::hint::execution_mode, ov::hint::ExecutionMode::UNDEFINED),
        std::make_tuple(ov::hint::num_requests, 0, UnsignedTypeValidator()),
        std::make_tuple(ov::intel_auto::enable_startup_fallback, true),
        // TODO 1) cache_dir 2) allow_auto_batch 3) auto_batch_timeout
        std::make_tuple(ov::cache_dir, ""),
        std::make_tuple(ov::hint::allow_auto_batching, true),
        std::make_tuple(ov::auto_batch_timeout, 1000),
        // Legacy API properties
        std::make_tuple(exclusive_asyc_requests, false),
        // RO for register only
        std::make_tuple(ov::device::full_name),
        std::make_tuple(ov::device::capabilities),
        std::make_tuple(ov::supported_properties));
}
void PluginConfig::register_property_impl(const ov::AnyMap::value_type& property, ov::PropertyMutability mutability, BaseValidator::Ptr validator) {
    property_validators[property.first] = validator;
    internal_properties[property.first] = property.second;
    property_mutabilities[property.first] = mutability;
}

template <typename T, ov::PropertyMutability mutability>
void PluginConfig::register_property_impl(const ov::Property<T, mutability>& property) {
    property_mutabilities[property.name()] = mutability;
}

void PluginConfig::set_property(const ov::AnyMap& properties) {
    for (auto& kv : properties) {
        auto& name = kv.first;
        auto& val = kv.second;
        if (is_supported(kv.first)) {
            OPENVINO_ASSERT(property_validators.at(name)->is_valid(val),
                    "Invalid value for property ", name,  ": ", val.as<std::string>());
            internal_properties[name] = val;
            // when user call set_property to set some config to plugin, we also respect this and pass through the config in this case
            user_properties[name] = val;
        } else {
            OPENVINO_ASSERT(false, "property:", name,  ": not supported");
        }
    }
}

ov::Any PluginConfig::get_property(const std::string& name) const {
    OPENVINO_ASSERT(internal_properties.find(name) != internal_properties.end(), "[AUTO]", "not supported property ", name);
    return internal_properties.at(name);
}

bool PluginConfig::is_supported(const std::string& name) const {
    bool supported = internal_properties.find(name) != internal_properties.end();
    bool has_validator = property_validators.find(name) != property_validators.end();

    return supported && has_validator;
}

bool PluginConfig::is_set_by_user(const std::string& name) const {
    return user_properties.find(name) != user_properties.end();
}

void PluginConfig::set_user_property(const ov::AnyMap& config, bool checkfirstlevel) {
    // user property, accept either internal property, or secondary property for hardware plugin
    // TODO: for multi, other first level property are also accepted
    for (auto& kv : config) {
        auto& name = kv.first;
        auto& val = kv.second;
        if (is_supported(name)) {
            OPENVINO_ASSERT(property_validators.at(name)->is_valid(val),
                        "Invalid value for property ", name,  ": ", val.as<std::string>());
            internal_properties[kv.first] = kv.second;
            user_properties[kv.first] = kv.second;
        } else {
            if (device_property_validator->is_valid(ov::Any(name))) { // if it's a valid secondary, accept it
                user_properties[kv.first] = kv.second;
            } else if (!checkfirstlevel) { // for multi, accept it anyway when compiled model
                user_properties[kv.first] = kv.second;
            } else {
                OPENVINO_ASSERT(false, "property", name,  ": not supported");
            }
        }
    }
}

void PluginConfig::apply_user_properties() {
    full_properties = internal_properties;
    for (auto& kv : user_properties)
        full_properties[kv.first] = kv.second;
}

ov::AnyMap PluginConfig::get_full_properties() {
    return full_properties;
}

} // namespace MultiDevicePlugin