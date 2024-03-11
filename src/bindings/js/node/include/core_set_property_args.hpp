#pragma once

#include <napi.h>

#include "openvino/runtime/core.hpp"

/**
 * @brief This struct retrieves data from Napi::CallbackInfo.
 */
struct CoreSetPropertyArgs {
    std::string device_name;
    ov::AnyMap parameters;

    CoreSetPropertyArgs() {}
    CoreSetPropertyArgs(const Napi::CallbackInfo& info) {
        const size_t args_length = info.Length();

        if (!is_valid_input(args_length, info))
            throw std::runtime_error("Invalid arguments of set_property function");

        if (args_length > 1) device_name = info[0].ToString();
        
        Napi::Object parameters = info[args_length > 1 ? 1 : 0].ToObject();
        const auto& keys = parameters.GetPropertyNames();

        for (uint32_t i = 0; i < keys.Length(); ++i) {
            auto property_name = static_cast<Napi::Value>(keys[i]).ToString().Utf8Value();
            Napi::Value value = parameters.Get(property_name);

            if (value.IsString()) {
                this->parameters.insert(std::make_pair(property_name, value.ToString().Utf8Value()));
            } else if (value.IsNumber()) {
                this->parameters.insert(std::make_pair(property_name, value.ToNumber()));
            } else if (value.IsBoolean()) {
                this->parameters.insert(std::make_pair(property_name, value.ToBoolean()));
            } else {
                throw std::runtime_error("Unsupported type of parameter value");
            }
        }
    }

    bool is_valid_input(size_t args_length, const Napi::CallbackInfo& info) {
        const bool is_passed_device = info[0].IsString();
        const bool has_params_obj = info[is_passed_device ? 1 : 0];

        return args_length <= (is_passed_device ? 2 : 1) && has_params_obj;
    }
};
