// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

#include "node/include/helper.hpp"
#include "openvino/runtime/core.hpp"

/**
 * @brief This struct retrieves data from Napi::CallbackInfo.
 */
struct CoreSetPropertyArgs {
    std::string device_name;
    ov::AnyMap parameters;

    CoreSetPropertyArgs() {}
    CoreSetPropertyArgs(const Napi::CallbackInfo& info) {
        CoreSetPropertyArgs::validate(info);

        const size_t args_length = info.Length();

        if (args_length > 1)
            device_name = info[0].ToString();

        const size_t parameters_position_index = device_name.empty() ? 0 : 1;
        Napi::Object parameters = info[parameters_position_index].ToObject();
        const auto& keys = parameters.GetPropertyNames();

        for (uint32_t i = 0; i < keys.Length(); ++i) {
            auto property_name = static_cast<Napi::Value>(keys[i]).ToString().Utf8Value();

            ov::Any any_value = js_to_any(info, parameters.Get(property_name));

            this->parameters.insert(std::make_pair(property_name, any_value));
        }
    }

    void static validate(const Napi::CallbackInfo& info) {
        const size_t args_length = info.Length();
        const bool is_device_specified = info[0].IsString();
        const bool has_params_obj = info[is_device_specified ? 1 : 0];

        if (!has_params_obj)
            throw std::runtime_error("Properties parameter must be an object");

        if (args_length > (is_device_specified ? 2 : 1))
            throw std::runtime_error("setProperty applies 1 or 2 arguments only");
    }
};
