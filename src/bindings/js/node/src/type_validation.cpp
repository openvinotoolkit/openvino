// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "node/include/type_validation.hpp"

namespace js {

template <>
bool validate_value<std::string>(const Napi::Env& env, const Napi::Value& value) {
    return napi_string == value.Type();
}

template <>
bool validate_value<int>(const Napi::Env& env, const Napi::Value& value) {
    return value.IsNumber() && env.Global()
                                   .Get("Number")
                                   .ToObject()
                                   .Get("isInteger")
                                   .As<Napi::Function>()
                                   .Call({value.ToNumber()})
                                   .ToBoolean()
                                   .Value();
}

template <>
bool validate_value<ov::Model>(const Napi::Env& env, const Napi::Value& value) {
    const auto& prototype = env.GetInstanceData<AddonData>()->model;
    return value.ToObject().InstanceOf(prototype.Value().As<Napi::Function>());
}

template <>
bool validate_value<ov::Tensor>(const Napi::Env& env, const Napi::Value& value) {
    const auto& prototype = env.GetInstanceData<AddonData>()->tensor;
    return value.ToObject().InstanceOf(prototype.Value().As<Napi::Function>());
}

template <>
bool validate_value<Napi::Object>(const Napi::Env& env, const Napi::Value& value) {
    return napi_object == value.Type();
}

}  // namespace js
