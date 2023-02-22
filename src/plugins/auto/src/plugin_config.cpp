// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "utils/plugin_config.hpp"

namespace MultiDevicePlugin {
const std::set<std::string> ExecutionConfig::_availableDevices = {"AUTO", "CPU", "GPU", "TEMPLATE", "MYRIAD", "VPUX", "MULTI", "HETERO", "mock"};

ExecutionConfig::ExecutionConfig() {
    set_default();
    device_property_validator = std::dynamic_pointer_cast<BaseValidator>(std::make_shared<FuncValidator>([](const ov::Any& target) -> bool {
        auto deviceName = target.as<std::string>();
        return _availableDevices.end() != std::find(_availableDevices.begin(),
                                                            _availableDevices.end(),
                                                            DeviceIDParser(deviceName).getDeviceName());
    }));
}

void ExecutionConfig::set_default() {
    register_property(
        std::make_tuple(ov::enable_profiling, false),
        std::make_tuple(ov::device::priorities, ""),
        std::make_tuple(ov::hint::model_priority, ov::hint::Priority::MEDIUM),
        std::make_tuple(ov::log::level, ov::log::Level::NO),
        std::make_tuple(ov::intel_auto::device_bind_buffer, false),
        std::make_tuple(ov::hint::performance_mode, ov::hint::PerformanceMode::UNDEFINED, PerformanceModeValidator()),
        // TODO 1) cache_dir 2) allow_auto_batch 3) auto_batch_timeout
        // Legacy API properties
        std::make_tuple(exclusive_asyc_requests, false));
}
void ExecutionConfig::register_property_impl(const ov::AnyMap::value_type& property, BaseValidator::Ptr validator) {
    property_validators[property.first] = validator;
    internal_properties[property.first] = property.second;
}

void ExecutionConfig::set_property(const ov::AnyMap& properties) {
    for (auto& kv : properties) {
        auto& name = kv.first;
        auto& val = kv.second;
        if (is_supported(kv.first)) {
            OPENVINO_ASSERT(property_validators.at(name)->is_valid(val),
                    "[AUTO]", "Invalid value for property ", name,  ": ", val.as<std::string>());
            internal_properties[name] = val;
        }
    }
}

ov::Any ExecutionConfig::get_property(const std::string& name) const {
    OPENVINO_ASSERT(internal_properties.find(name) != internal_properties.end(), "[AUTO]", "not supported property ", name);
    return internal_properties.at(name);
}

bool ExecutionConfig::is_supported(const std::string& name) const {
    bool supported = internal_properties.find(name) != internal_properties.end();
    bool has_validator = property_validators.find(name) != property_validators.end();

    return supported && has_validator;
}

bool ExecutionConfig::is_set_by_user(const std::string& name) const {
    return user_properties.find(name) != user_properties.end();
}

void ExecutionConfig::set_user_property(const ov::AnyMap& config) {
    // user property, accept either internal property, or secondary property for hardware plugin
    for (auto& kv : config) {
        auto& name = kv.first;
        auto& val = kv.second;
        if (is_supported(name)) {
            OPENVINO_ASSERT(property_validators.at(name)->is_valid(val),
                        "[AUTO]", "Invalid value for property ", name,  ": ", val.as<std::string>());
            user_properties[kv.first] = kv.second;
        } else {
            OPENVINO_ASSERT(device_property_validator->is_valid(ov::Any(name)), "[AUTO]", "Invalid property name:", name);
            user_properties[kv.first] = kv.second;
        }
    }
}

void ExecutionConfig::apply_user_properties() {
    // update internal if existed
    for (auto& kv : user_properties) {
        internal_properties[kv.first] = kv.second;
    }
    user_properties.clear();
}

std::string ExecutionConfig::to_string() const {
    std::stringstream s;
    s << "internal properties:\n";
    for (auto& kv : internal_properties) {
        s << "\t" << kv.first << ": " << kv.second.as<std::string>() << std::endl;
    }
    s << "user seconadary properties:\n";
    for (auto& kv : user_properties) {
        s << "\t" << kv.first << ": " << kv.second.as<std::string>() << std::endl;
    }
    return s.str();
}
} // namespace MultiDevicePlugin